import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import multiprocessing as mp
import subprocess

os.environ["OMP_NUM_THREADS"] = "8"

try:
    import torch_geometric
    HAS_PYG = True
except ImportError:
    print("torch_geometric not found. Attempting to install locally...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--target", override_dir, "--quiet",
            "torch_geometric"
        ])
        sys.path.insert(0, override_dir)
        import torch_geometric
        HAS_PYG = True
        print("Successfully installed torch_geometric.")
    except Exception as e:
        print(f"Failed to install torch_geometric: {e}")
        HAS_PYG = False

if HAS_PYG:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import MessagePassing, global_mean_pool
else:
    print("Falling back to MLP on Morgan fingerprints.")

def process_smiles_to_raw(args):
    smiles, y_val = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (np.zeros((1, 5), dtype=np.float32), np.empty((2, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float32), y_val)
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetHybridization()), int(atom.GetIsAromatic()), atom.GetFormalCharge()])
    x = np.array(x, dtype=np.float32)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e_feat = [float(bond.GetBondTypeAsDouble()), int(bond.GetIsConjugated())]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e_feat, e_feat]
    if len(edge_indices) > 0:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 2), dtype=np.float32)
    return (x, edge_index, edge_attr, y_val)

def process_smiles_to_morgan(args):
    smiles, y_val = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fp = np.zeros(2048, dtype=np.float32)
    else:
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
    return fp, y_val

if HAS_PYG:
    class CustomConv(MessagePassing):
        def __init__(self, hidden_dim):
            super().__init__(aggr='mean')
            self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        def message(self, x_i, x_j, edge_attr):
            tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
            return self.mlp(tmp)

    class MPNN(nn.Module):
        def __init__(self, node_dim=5, edge_dim=2, hidden_dim=128):
            super().__init__()
            self.node_emb = nn.Linear(node_dim, hidden_dim)
            self.edge_emb = nn.Linear(edge_dim, hidden_dim)
            self.convs = nn.ModuleList([CustomConv(hidden_dim) for _ in range(3)])
            self.pool = global_mean_pool
            self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = self.node_emb(x)
            if edge_attr.numel() > 0:
                edge_attr = self.edge_emb(edge_attr)
            else:
                edge_attr = torch.empty((0, self.edge_emb.out_features), device=x.device)
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_attr)) + x
            x = self.pool(x, batch)
            return self.out(x).squeeze(-1)

    class QM9GraphDataset(Dataset):
        def __init__(self, df):
            super().__init__()
            args_list = list(zip(df['smiles'], df['u0_res']))
            with mp.Pool(8) as pool:
                raw_data = pool.map(process_smiles_to_raw, args_list)
            self.data_list = [Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr), y=torch.tensor(y_val, dtype=torch.float)) for x, edge_index, edge_attr, y_val in raw_data]
        def len(self):
            return len(self.data_list)
        def get(self, idx):
            return self.data_list[idx]
else:
    class MorganDataset(TorchDataset):
        def __init__(self, df):
            args_list = list(zip(df['smiles'], df['u0_res']))
            with mp.Pool(8) as pool:
                raw_data = pool.map(process_smiles_to_morgan, args_list)
            self.x = torch.tensor(np.array([item[0] for item in raw_data]))
            self.y = torch.tensor(np.array([item[1] for item in raw_data]), dtype=torch.float)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    class MLP(nn.Module):
        def __init__(self, input_dim=2048, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            if HAS_PYG:
                batch = batch.to(device)
                preds = model(batch)
                all_preds.append(preds.cpu().numpy())
                all_y.append(batch.y.cpu().numpy())
            else:
                x, y = batch
                preds = model(x.to(device))
                all_preds.append(preds.cpu().numpy())
                all_y.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_y)

if __name__ == '__main__':
    df = pd.read_csv('data/u0_res_baseline.csv')
    train_df, val_df, test_df = df[df['split'] == 'train'].copy(), df[df['split'] == 'val'].copy(), df[df['split'] == 'test'].copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_workers = 512, 0
    if HAS_PYG:
        train_dataset, val_dataset, test_dataset = QM9GraphDataset(train_df), QM9GraphDataset(val_df), QM9GraphDataset(test_df)
        train_loader, val_loader, test_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True), PyGDataLoader(val_dataset, batch_size=batch_size), PyGDataLoader(test_dataset, batch_size=batch_size)
        model = MPNN(hidden_dim=128).to(device)
    else:
        train_dataset, val_dataset, test_dataset = MorganDataset(train_df), MorganDataset(val_df), MorganDataset(test_df)
        train_loader, val_loader, test_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True), TorchDataLoader(val_dataset, batch_size=batch_size), TorchDataLoader(test_dataset, batch_size=batch_size)
        model = MLP(hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val_loss, patience_counter, model_path = float('inf'), 0, 'data/gnn_model.pt'
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if HAS_PYG:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.y)
            else:
                x, y = batch
                loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if HAS_PYG:
                    batch = batch.to(device)
                    val_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
                else:
                    x, y = batch
                    val_loss += criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= 15: break
    model.load_state_dict(torch.load(model_path))
    metrics = {}
    for name, loader, split_df in [('train', train_loader, train_df), ('val', val_loader, val_df), ('test', test_loader, test_df)]:
        preds, y_true = evaluate_model(model, loader, device)
        metrics[name] = {'RMSE': np.sqrt(mean_squared_error(y_true, preds)), 'MAE': mean_absolute_error(y_true, preds), 'R2': r2_score(y_true, preds)}
        split_df['gnn_pred_u0_res'] = preds
    pd.concat([train_df, val_df, test_df]).sort_index()[['mol_id', 'smiles', 'split', 'u0_res', 'gnn_pred_u0_res']].to_csv('data/gnn_predictions.csv', index=False)
    with open('data/gnn_metrics.json', 'w') as f: json.dump(metrics, f, indent=2)