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
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader

try:
    import rdkit
    from rdkit import Chem
except ModuleNotFoundError:
    print("rdkit not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", override_dir, "--quiet", "rdkit"])
    sys.path.insert(0, override_dir)
    from rdkit import Chem

try:
    import captum
    from captum.attr import IntegratedGradients
except ModuleNotFoundError:
    print("captum not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", override_dir, "--quiet", "captum"])
    sys.path.insert(0, override_dir)
    from captum.attr import IntegratedGradients

try:
    import statsmodels
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
    print("statsmodels not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", override_dir, "--quiet", "statsmodels"])
    sys.path.insert(0, override_dir)
    from statsmodels.stats.multitest import multipletests

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

class GNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, adj, mask):
        return self.model(x, adj, mask)

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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading trained GNN model...")
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    
    wrapper = GNNWrapper(model)
    ig = IntegratedGradients(wrapper)
    
    print("Computing Integrated Gradients...")
    node_attrs = []
    edge_attrs = []
    
    for x, adj, mask in test_loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        
        x_base = torch.zeros_like(x)
        adj_base = torch.zeros_like(adj)
        
        attr_x, attr_adj = ig.attribute(
            inputs=(x, adj),
            baselines=(x_base, adj_base),
            additional_forward_args=(mask,),
            n_steps=20,
            internal_batch_size=128
        )
        
        node_attrs.append(attr_x.cpu().detach().numpy())
        edge_attrs.append(attr_adj.cpu().detach().numpy())
        
    node_attrs = np.concatenate(node_attrs, axis=0)
    edge_attrs = np.concatenate(edge_attrs, axis=0)
    
    num_heavy_atoms = Mask_test.sum(dim=1).numpy()
    num_heavy_atoms = np.maximum(num_heavy_atoms, 1)
    
    node_attr_sum = np.sum(np.abs(node_attrs), axis=(1, 2))
    edge_attr_sum = np.sum(np.abs(edge_attrs), axis=(1, 2, 3))
    
    total_attr = node_attr_sum + edge_attr_sum
    norm_attr = total_attr / num_heavy_atoms
    
    test_df['ig_score'] = norm_attr
    
    has_aromatic = X_test[:, :, 10].sum(dim=1).numpy() > 0
    has_oxygen = X_test[:, :, 2].sum(dim=1).numpy() > 0
    has_nitrogen = X_test[:, :, 1].sum(dim=1).numpy() > 0
    has_fluorine = X_test[:, :, 3].sum(dim=1).numpy() > 0
    
    test_df['has_aromatic'] = has_aromatic
    test_df['has_oxygen'] = has_oxygen
    test_df['has_nitrogen'] = has_nitrogen
    test_df['has_fluorine'] = has_fluorine
    
    print("\nMerging intensive properties for statistical validation...")
    qm9_df = pd.read_csv('data/cleaned_qm9.csv')
    test_df = test_df.merge(qm9_df[['smiles', 'gap', 'mu']], on='smiles', how='left')
    
    valid_idx = test_df[['gap', 'mu']].notna().all(axis=1)
    test_df = test_df[valid_idx].reset_index(drop=True)
    
    motifs = ['has_aromatic', 'has_oxygen', 'has_nitrogen', 'has_fluorine']
    properties = ['gap', 'mu']
    
    results_list = []
    
    print("\nPerforming statistical validation...")
    for prop in properties:
        for motif in motifs:
            group_with = test_df[test_df[motif] == True][prop].values
            group_without = test_df[test_df[motif] == False][prop].values
            
            if len(group_with) < 5 or len(group_without) < 5:
                continue
                
            stat_lev, p_lev = stats.levene(group_with, group_without)
            
            if p_lev > 0.05:
                stat_test, p_val = stats.f_oneway(group_with, group_without)
                test_used = 'ANOVA'
            else:
                stat_test, p_val = stats.kruskal(group_with, group_without)
                test_used = 'Kruskal-Wallis'
                
            mean_diff = np.mean(group_with) - np.mean(group_without)
            pooled_std = np.sqrt((np.std(group_with, ddof=1)**2 + np.std(group_without, ddof=1)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            results_list.append({
                'Property': prop,
                'Motif': motif,
                'Test_Used': test_used,
                'Statistic': stat_test,
                'p_value': p_val,
                'Mean_With': np.mean(group_with),
                'Mean_Without': np.mean(group_without),
                'Cohens_d': cohens_d,
                'N_With': len(group_with),
                'N_Without': len(group_without)
            })
            
    results_df = pd.DataFrame(results_list)
    
    if len(results_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')
        results_df['p_value_adj'] = pvals_corrected
        results_df['Significant'] = reject
        
        print("\nStatistical Validation Results:")
        print(results_df.to_string(index=False, float_format="%.4f"))
        
        results_df.to_csv('data/motif_statistical_validation.csv', index=False)
        
        print("\nGenerating boxplots...")
        plt.rcParams['text.usetex'] = False
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for i, prop in enumerate(properties):
            for j, motif in enumerate(motifs):
                ax = axes[i, j]
                group_with = test_df[test_df[motif] == True][prop].values
                group_without = test_df[test_df[motif] == False][prop].values
                
                if len(group_with) > 0 and len(group_without) > 0:
                    ax.boxplot([group_without, group_with])
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(['Without Motif', 'With Motif'])
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                    
                ax.set_title(f"{motif}")
                ax.set_xlabel('Contains Motif')
                if prop == 'gap':
                    ax.set_ylabel('gap (Hartree)')
                else:
                    ax.set_ylabel('mu (Debye)')
                    
        plt.tight_layout()
        plt.savefig('data/motif_property_boxplots.png', dpi=300)
