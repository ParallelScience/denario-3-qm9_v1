import os
import sys
import json
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "8"

# Ensure required packages are available
override_dir = os.path.abspath("./.pip_overrides")
os.makedirs(override_dir, exist_ok=True)
if override_dir not in sys.path:
    sys.path.insert(0, override_dir)

try:
    import torch_geometric
except ImportError:
    print("torch_geometric not found. Attempting to install locally...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", override_dir, "--quiet",
        "torch_geometric"
    ])
    import torch_geometric

try:
    import xgboost as xgb
except ImportError:
    print("xgboost not found. Attempting to install locally...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", override_dir, "--quiet",
        "xgboost"
    ])
    import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- GNN Architecture Definition (Matches Step 3) ---
class CustomConv(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
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
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, 1)
        )
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

# --- Helper Functions ---
def process_smiles_to_raw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
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
    return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr))

def compute_ig_batched(model, loader, steps=50, device='cpu'):
    model.eval()
    all_ig_x = []
    all_ig_edge = []
    
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        x_baseline = torch.zeros_like(x)
        edge_attr_baseline = torch.zeros_like(edge_attr)
        
        ig_x = torch.zeros_like(x)
        ig_edge = torch.zeros_like(edge_attr)
        
        for i in range(1, steps + 1):
            alpha = i / steps
            x_step = (x_baseline + alpha * (x - x_baseline)).clone().detach().requires_grad_(True)
            edge_step = (edge_attr_baseline + alpha * (edge_attr - edge_attr_baseline)).clone().detach().requires_grad_(True)
            
            data_step = Data(x=x_step, edge_index=edge_index, edge_attr=edge_step, batch=batch_idx)
            out = model(data_step)
            loss = out.sum()
            loss.backward()
            
            ig_x += x_step.grad / steps
            if edge_step.grad is not None:
                ig_edge += edge_step.grad / steps
                
        ig_x = ig_x * (x - x_baseline)
        ig_edge = ig_edge * (edge_attr - edge_attr_baseline)
        
        for i in range(batch.num_graphs):
            node_mask = batch_idx == i
            all_ig_x.append(ig_x[node_mask].cpu().numpy())
            
            if edge_index.size(1) > 0:
                edge_batch = batch_idx[edge_index[0]]
                edge_mask = edge_batch == i
                all_ig_edge.append(ig_edge[edge_mask].cpu().numpy())
            else:
                all_ig_edge.append(np.empty((0, edge_attr.shape[1]), dtype=np.float32))
            
    return all_ig_x, all_ig_edge

def mutate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    rwmol = Chem.RWMol(mol)
    mutated = False
    for bond in rwmol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetIsConjugated() and not bond.GetIsAromatic():
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            bond.SetIsConjugated(False)
            mutated = True
            
    if mutated:
        try:
            Chem.SanitizeMol(rwmol)
            return rwmol
        except:
            return None
    return None

def get_gnn_preds(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)

DESC_NAMES = [
    "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", 
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", 
    "BalabanJ", "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3", 
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", 
    "NumHeterocycles", "NumAromaticHeterocycles", "NumAromaticCarbocycles", 
    "NumAliphaticHeterocycles", "NumAliphaticCarbocycles", "NumSpiroAtoms", 
    "NumBridgeheadAtoms", "NumRotatableBonds", "NumHeteroatoms", 
    "TPSA", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount"
]

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(DESC_NAMES)
    
    desc_dict = dict(Descriptors.descList)
    res = []
    for name in DESC_NAMES:
        try:
            if name in desc_dict:
                val = desc_dict[name](mol)
                res.append(float(val))
            else:
                res.append(np.nan)
        except Exception:
            res.append(np.nan)
    return res

if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_csv('data/u0_res_baseline.csv')
    test_df = df[df['split'] == 'test'].copy()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading GNN model...")
    model = MPNN(hidden_dim=128).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device))
    
    print("\n--- Part 1: Integrated Gradients ---")
    print("Selecting 1000 random test molecules for Integrated Gradients...")
    ig_df = test_df.sample(n=1000, random_state=42).copy()
    
    ig_graphs = []
    for smiles in ig_df['smiles']:
        ig_graphs.append(process_smiles_to_raw(smiles))
        
    ig_loader = PyGDataLoader(ig_graphs, batch_size=128, shuffle=False)
    
    print("Computing Integrated Gradients...")
    ig_x, ig_edge = compute_ig_batched(model, ig_loader, steps=50, device=device)
    
    ig_results = {
        'mol_id': ig_df['mol_id'].tolist(),
        'ig_x': ig_x,
        'ig_edge': ig_edge
    }
    torch.save(ig_results, 'data/ig_attributions.pt')
    print("Saved data/ig_attributions.pt")
    
    x_attr_sums = [np.abs(attr).sum() for attr in ig_x]
    edge_attr_sums = [np.abs(attr).sum() for attr in ig_edge]
    print(f"Mean absolute node attribution sum per molecule: {np.mean(x_attr_sums):.4f}")
    print(f"Mean absolute edge attribution sum per molecule: {np.mean(edge_attr_sums):.4f}")
    
    print("\n--- Part 2: In-Silico Mutations ---")
    mutated_data = []
    for idx, row in test_df.iterrows():
        mut_mol = mutate_molecule(row['smiles'])
        if mut_mol is not None:
            mutated_data.append({
                'mol_id': row['mol_id'],
                'original_smiles': row['smiles'],
                'mutated_smiles': Chem.MolToSmiles(mut_mol),
                'original_u0_res': row['u0_res']
            })
            
    mutated_df = pd.DataFrame(mutated_data)
    print(f"Found {len(mutated_df)} molecules with non-aromatic conjugated double bonds for mutation.")
    
    if len(mutated_df) > 0:
        print("Processing mutated graphs for GNN...")
        mutated_graphs = [process_smiles_to_raw(s) for s in mutated_df['mutated_smiles']]
        original_graphs = [process_smiles_to_raw(s) for s in mutated_df['original_smiles']]
        
        mut_loader = PyGDataLoader(mutated_graphs, batch_size=512, shuffle=False)
        orig_loader = PyGDataLoader(original_graphs, batch_size=512, shuffle=False)
        
        gnn_mut_preds = get_gnn_preds(model, mut_loader, device)
        gnn_orig_preds = get_gnn_preds(model, orig_loader, device)
        
        mutated_df['gnn_orig_pred'] = gnn_orig_preds
        mutated_df['gnn_mut_pred'] = gnn_mut_preds
        mutated_df['gnn_shift'] = gnn_mut_preds - gnn_orig_preds
        
        print("Computing descriptors for XGBoost training...")
        train_df = df[df['split'] == 'train']
        with mp.Pool(8) as pool:
            train_descs = pool.map(compute_descriptors, train_df['smiles'].tolist())
            
        train_desc_df = pd.DataFrame(train_descs, columns=DESC_NAMES)
        medians = train_desc_df.median()
        X_train = train_desc_df.fillna(medians).values
        y_train = train_df['u0_res'].values
        
        print("Retraining XGBoost model (not saved in Step 2)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8
        )
        xgb_model.fit(X_train, y_train)
        
        print("Computing descriptors for mutated molecules...")
        with mp.Pool(8) as pool:
            orig_descs = pool.map(compute_descriptors, mutated_df['original_smiles'].tolist())
            mut_descs = pool.map(compute_descriptors, mutated_df['mutated_smiles'].tolist())
            
        X_orig = pd.DataFrame(orig_descs, columns=DESC_NAMES).fillna(medians).values
        X_mut = pd.DataFrame(mut_descs, columns=DESC_NAMES).fillna(medians).values
        
        xgb_orig_preds = xgb_model.predict(X_orig)
        xgb_mut_preds = xgb_model.predict(X_mut)
        
        mutated_df['xgb_orig_pred'] = xgb_orig_preds
        mutated_df['xgb_mut_pred'] = xgb_mut_preds
        mutated_df['xgb_shift'] = xgb_mut_preds - xgb_orig_preds
        
        mean_gnn_shift = mutated_df['gnn_shift'].mean()
        mean_xgb_shift = mutated_df['xgb_shift'].mean()
        std_gnn_shift = mutated_df['gnn_shift'].std()
        std_xgb_shift = mutated_df['xgb_shift'].std()
        
        print("\n--- Mutation Shift Analysis Results ---")
        print(f"Sample size: {len(mutated_df)}")
        print(f"Mean GNN Shift (Mutated - Original): {mean_gnn_shift:.6f} +/- {std_gnn_shift:.6f} Hartree")
        print(f"Mean XGB Shift (Mutated - Original): {mean_xgb_shift:.6f} +/- {std_xgb_shift:.6f} Hartree")
        
        if len(mutated_df) > 1:
            corr, p_val = stats.pearsonr(mutated_df['gnn_shift'], mutated_df['xgb_shift'])
            t_stat, p_val_t = stats.ttest_rel(mutated_df['gnn_shift'], mutated_df['xgb_shift'])
            print(f"Pearson correlation (GNN vs XGB shifts): {corr:.4f} (p-value: {p_val:.2e})")
            print(f"Paired t-test (GNN vs XGB shifts): t = {t_stat:.4f}, p-value = {p_val_t:.2e}")
        
        mutated_df.to_csv('data/mutation_shifts.csv', index=False)
        print(f"\nSaved data/mutation_shifts.csv columns: {list(mutated_df.columns)}")
        
        plt.rcParams['text.usetex'] = False
        plt.figure(figsize=(8, 6))
        plt.scatter(mutated_df['xgb_shift'], mutated_df['gnn_shift'], alpha=0.6, edgecolors='k')
        
        min_val = min(mutated_df['xgb_shift'].min(), mutated_df['gnn_shift'].min())
        max_val = max(mutated_df['xgb_shift'].max(), mutated_df['gnn_shift'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
        
        plt.axvline(0, color='k', linestyle=':', alpha=0.5)
        plt.axhline(0, color='k', linestyle=':', alpha=0.5)
        plt.xlabel('XGBoost Predicted Shift (Hartree)')
        plt.ylabel('GNN Predicted Shift (Hartree)')
        plt.title('Predicted Energy Shift upon Breaking Conjugation')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/mutation_shifts_scatter.png', dpi=300)
        print("Saved data/mutation_shifts_scatter.png")