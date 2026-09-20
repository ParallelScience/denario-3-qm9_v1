import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE

# Ensure RDKit is available
try:
    import rdkit
    from rdkit import Chem
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
    from rdkit import Chem

# ---------------------------------------------------------
# GNN Architecture Definition (matching Step 4)
# ---------------------------------------------------------
class DenseGINEConv(nn.Module):
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
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)
        edge_emb = self.edge_proj(adj)
        m = F.relu(x_j + edge_emb)
        edge_mask = (adj.sum(dim=-1) > 0).unsqueeze(-1)
        m = m * edge_mask.float()
        agg = m.sum(dim=2)
        out = (1 + self.eps) * x + agg
        out = self.mlp(out)
        out = out * mask.unsqueeze(-1).float()
        return out

class DenseGlobalAttentionPooling(nn.Module):
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
        gate = gate.masked_fill(~mask, -1e9)
        alpha = F.softmax(gate, dim=-1)
        alpha = alpha * mask.float()
        feat = self.feat_nn(x)
        out = (alpha.unsqueeze(-1) * feat).sum(dim=1)
        return out

class GNNModel(nn.Module):
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
        
    def extract_latent(self, x, adj, mask):
        """Extracts the graph-level representation before the final MLP."""
        h = self.node_emb(x)
        h = h * mask.unsqueeze(-1).float()
        for conv in self.convs:
            h = conv(h, adj, mask)
        pooled = self.pool(h, mask)
        return pooled

# ---------------------------------------------------------
# Data Processing Function
# ---------------------------------------------------------
def process_row(row):
    smiles, target = row
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    
    max_N = 9
    x = np.zeros((max_N, 16), dtype=np.float32)
    adj = np.zeros((max_N, max_N, 4), dtype=np.float32)
    mask = np.zeros(max_N, dtype=bool)
    
    atom_mapping = {6: 0, 7: 1, 8: 2, 9: 3}
    
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_N:
            break
        mask[i] = True
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_mapping:
            x[i, atom_mapping[atomic_num]] = 1.0
        deg = atom.GetDegree()
        if deg <= 4:
            x[i, 4 + deg] = 1.0
        x[i, 9] = atom.GetFormalCharge()
        x[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
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
    print("Loading test set data...")
    test_df = pd.read_csv('data/gnn_test_predictions.csv')
    smiles_list = test_df['smiles'].tolist()
    targets = test_df['std_residual'].tolist()
    
    print(f"Processing {len(smiles_list)} SMILES to dense graphs...")
    num_workers = min(16, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_row, zip(smiles_list, targets))
        
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    test_df = test_df.iloc[valid_indices].reset_index(drop=True)
    results = [results[i] for i in valid_indices]
    
    X_test = torch.tensor(np.stack([r[0] for r in results]), dtype=torch.float32)
    Adj_test = torch.tensor(np.stack([r[1] for r in results]), dtype=torch.float32)
    Mask_test = torch.tensor(np.stack([r[2] for r in results]), dtype=torch.bool)
    
    test_dataset = TensorDataset(X_test, Adj_test, Mask_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading trained GNN model...")
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    
    print("Extracting latent representations...")
    latents = []
    with torch.no_grad():
        for x, adj, mask in test_loader:
            x, adj, mask = x.to(device), adj.to(device), mask.to(device)
            latent = model.extract_latent(x, adj, mask)
            latents.append(latent.cpu().numpy())
            
    latents = np.concatenate(latents, axis=0)
    print(f"Extracted latent vectors shape: {latents.shape}")
    
    print("\nLatent Space Summary Statistics:")
    print(f"  Mean: {np.mean(latents):.4f}")
    print(f"  Std:  {np.std(latents):.4f}")
    print(f"  Min:  {np.min(latents):.4f}")
    print(f"  Max:  {np.max(latents):.4f}")
    
    print("\nPerforming t-SNE dimensionality reduction...")
    # Using n_jobs=8 to speed up t-SNE computation
    tsne = TSNE(n_components=2, random_state=42, n_jobs=8)
    latents_2d = tsne.fit_transform(latents)
    print(f"t-SNE embeddings shape: {latents_2d.shape}")
    
    print("\nt-SNE Embeddings Summary Statistics:")
    print(f"  Dim 1 - Mean: {np.mean(latents_2d[:, 0]):.4f}, Std: {np.std(latents_2d[:, 0]):.4f}")
    print(f"  Dim 2 - Mean: {np.mean(latents_2d[:, 1]):.4f}, Std: {np.std(latents_2d[:, 1]):.4f}")
    
    print("\nMerging intensive properties for downstream analysis...")
    qm9_df = pd.read_csv('data/cleaned_qm9.csv')
    test_df = test_df.merge(qm9_df[['smiles', 'gap', 'mu', 'homo', 'lumo']], on='smiles', how='left')
    
    test_df['tsne_1'] = latents_2d[:, 0]
    test_df['tsne_2'] = latents_2d[:, 1]
    
    # Save raw latents
    np.savez_compressed('data/gnn_test_latents.npz', latents=latents, smiles=test_df['smiles'].values)
    print("\nsaved data/gnn_test_latents.npz keys:", list(np.load('data/gnn_test_latents.npz').files))
    
    # Save 2D embeddings and properties
    test_df.to_csv('data/gnn_test_latent_embeddings.csv', index=False)
    print("saved data/gnn_test_latent_embeddings.csv columns:", list(test_df.columns))