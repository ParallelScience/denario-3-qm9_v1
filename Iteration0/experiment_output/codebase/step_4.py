import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import subprocess

# Ensure RDKit is available
try:
    import rdkit
except ModuleNotFoundError:
    print("rdkit not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", override_dir, "--quiet",
        "rdkit"
    ])
    sys.path.insert(0, override_dir)

import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem

class DenseGINEConv(nn.Module):
    """
    Dense implementation of Graph Isomorphism Network with Edge features (GINE).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.LayerNorm(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_proj = nn.Linear(4, emb_dim, bias=False)

    def forward(self, x, adj, mask):
        B, N, _ = x.shape
        # x_j: (B, N, N, emb_dim) - features of neighbor nodes
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)
        
        # edge_emb: (B, N, N, emb_dim)
        edge_emb = self.edge_proj(adj)
        
        # message: (B, N, N, emb_dim)
        m = F.relu(x_j + edge_emb)
        
        # mask out non-edges
        edge_mask = (adj.sum(dim=-1) > 0).unsqueeze(-1)
        m = m * edge_mask.float()
        
        # Aggregate messages from neighbors
        agg = m.sum(dim=2)
        
        # Update node embeddings
        out = (1 + self.eps) * x + agg
        out = self.mlp(out)
        
        # Mask out padding nodes
        out = out * mask.unsqueeze(-1).float()
        return out

class DenseGlobalAttentionPooling(nn.Module):
    """
    Global Attention Pooling for dense graph tensors.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
        self.feat_nn = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x, mask):
        gate = self.gate_nn(x).squeeze(-1)
        # Mask out padding nodes before softmax
        gate = gate.masked_fill(~mask, -1e9)
        alpha = F.softmax(gate, dim=-1)
        alpha = alpha * mask.float()
        
        feat = self.feat_nn(x)
        # Weighted sum of node features
        out = (alpha.unsqueeze(-1) * feat).sum(dim=1)
        return out

class GNNModel(nn.Module):
    """
    Full GNN architecture with GINE convolutions and Global Attention Pooling.
    """
    def __init__(self, node_dim=16, edge_dim=4, emb_dim=128, num_layers=4):
        super().__init__()
        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )
        self.convs = nn.ModuleList([DenseGINEConv(emb_dim) for _ in range(num_layers)])
        self.pool = DenseGlobalAttentionPooling(emb_dim)
        self.out_nn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )
        
    def forward(self, x, adj, mask):
        h = self.node_emb(x)
        h = h * mask.unsqueeze(-1).float()
        
        for conv in self.convs:
            h = conv(h, adj, mask)
            
        pooled = self.pool(h, mask)
        out = self.out_nn(pooled)
        return out.squeeze(-1)

def process_row(row):
    """
    Converts a SMILES string into dense node features, adjacency matrix, and mask.
    """
    smiles, target = row
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    
    max_N = 9 # QM9 molecules have at most 9 heavy atoms
    x = np.zeros((max_N, 16), dtype=np.float32)
    adj = np.zeros((max_N, max_N, 4), dtype=np.float32)
    mask = np.zeros(max_N, dtype=bool)
    
    atom_mapping = {6: 0, 7: 1, 8: 2, 9: 3} # C, N, O, F
    
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_N:
            break
        mask[i] = True
        
        # Atom type (4 dims)
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_mapping:
            x[i, atom_mapping[atomic_num]] = 1.0
            
        # Degree (5 dims)
        deg = atom.GetDegree()
        if deg <= 4:
            x[i, 4 + deg] = 1.0
            
        # Formal charge (1 dim)
        x[i, 9] = atom.GetFormalCharge()
        
        # Aromaticity (1 dim)
        x[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
        
        # Hydrogen count (5 dims)
        h_count = atom.GetTotalNumHs()
        if h_count <= 4:
            x[i, 11 + h_count] = 1.0
            
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i >= max_N or j >= max_N:
            continue
            
        btype = bond.GetBondType()
        if btype == Chem.BondType.SINGLE:
            idx = 0
        elif btype == Chem.BondType.DOUBLE:
            idx = 1
        elif btype == Chem.BondType.TRIPLE:
            idx = 2
        elif btype == Chem.BondType.AROMATIC:
            idx = 3
        else:
            continue
            
        adj[i, j, idx] = 1.0
        adj[j, i, idx] = 1.0
        
    return x, adj, mask, target

if __name__ == '__main__':
    print("Loading standardized residuals data...")
    df = pd.read_csv('data/standardized_residuals.csv')
    
    print("Processing SMILES to dense graphs...")
    smiles_list = df['smiles'].tolist()
    targets = df['std_residual'].tolist()
    
    num_workers = min(16, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_row, zip(smiles_list, targets))
        
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    df = df.iloc[valid_indices].reset_index(drop=True)
    results = [results[i] for i in valid_indices]
    
    X_all = torch.tensor(np.stack([r[0] for r in results]), dtype=torch.float32)
    Adj_all = torch.tensor(np.stack([r[1] for r in results]), dtype=torch.float32)
    Mask_all = torch.tensor(np.stack([r[2] for r in results]), dtype=torch.bool)
    Y_all = torch.tensor(np.stack([r[3] for r in results]), dtype=torch.float32)
    
    train_idx = df.index[df['split'] == 'train'].tolist()
    val_idx = df.index[df['split'] == 'val'].tolist()
    test_idx = df.index[df['split'] == 'test'].tolist()
    
    train_dataset = TensorDataset(X_all[train_idx], Adj_all[train_idx], Mask_all[train_idx], Y_all[train_idx])
    val_dataset = TensorDataset(X_all[val_idx], Adj_all[val_idx], Mask_all[val_idx], Y_all[val_idx])
    test_dataset = TensorDataset(X_all[test_idx], Adj_all[test_idx], Mask_all[test_idx], Y_all[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"CUDA memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
        device = torch.device('cuda')
        print("Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU.")
        
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GNN Model Parameters: {num_params}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("\nTraining GNN...")
    for epoch in range(200):
        model.train()
        train_loss = 0
        for x, adj, mask, y in train_loader:
            x, adj, mask, y = x.to(device), adj.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, adj, mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, adj, mask, y in val_loader:
                x, adj, mask, y = x.to(device), adj.to(device), mask.to(device), y.to(device)
                out = model(x, adj, mask)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/gnn_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load('data/gnn_model.pt'))
    model.eval()
    
    def evaluate(loader):
        preds = []
        trues = []
        with torch.no_grad():
            for x, adj, mask, y in loader:
                x, adj, mask = x.to(device), adj.to(device), mask.to(device)
                out = model(x, adj, mask)
                preds.append(out.cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mse = np.mean((preds - trues)**2)
        r2 = 1 - np.sum((preds - trues)**2) / np.sum((trues - np.mean(trues))**2)
        return mse, r2, preds

    train_mse, train_r2, _ = evaluate(train_loader)
    val_mse, val_r2, _ = evaluate(val_loader)
    test_mse, test_r2, test_preds = evaluate(test_loader)
    
    print("\nFinal GNN Performance (Standardized Residuals):")
    print(f"  Train - MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    print(f"  Val   - MSE: {val_mse:.4f}, R2: {val_r2:.4f}")
    print(f"  Test  - MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
    
    with open('data/gnn_metrics.txt', 'w') as f:
        f.write(f"Train - MSE: {train_mse:.4f}, R2: {train_r2:.4f}\n")
        f.write(f"Val   - MSE: {val_mse:.4f}, R2: {val_r2:.4f}\n")
        f.write(f"Test  - MSE: {test_mse:.4f}, R2: {test_r2:.4f}\n")
    print("\nsaved data/gnn_metrics.txt")
    
    test_df = df.iloc[test_idx].copy()
    test_df['gnn_pred'] = test_preds
    test_df.to_csv('data/gnn_test_predictions.csv', index=False)
    print("saved data/gnn_test_predictions.csv columns:", list(test_df.columns))
    print("saved data/gnn_model.pt")