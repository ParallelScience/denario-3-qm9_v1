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
        
    motifs = {
        'Aromatic_Ring': mol.HasSubstructMatch(Chem.MolFromSmarts('a1aaaaa1')) or mol.HasSubstructMatch(Chem.MolFromSmarts('a1aaaa1')),
        'Aliphatic_Ring': mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCCCC1')) or mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCCC1')) or mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCC1')),
        'Primary_Amine': mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3H2]')),
        'Secondary_Amine': mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3H1](C)C')),
        'Tertiary_Amine': mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3](C)(C)C')),
        'Alcohol': mol.HasSubstructMatch(Chem.MolFromSmarts('[OX2H]')),
        'Ether': mol.HasSubstructMatch(Chem.MolFromSmarts('[OX2](C)C')),
        'Carbonyl': mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3]=[OX1]')),
        'Nitrile': mol.HasSubstructMatch(Chem.MolFromSmarts('[NX1]#[CX2]'))
    }
    
    return x, adj, mask, target, motifs

if __name__ == '__main__':
    print("Loading test set data...")
    test_df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
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
    
    motif_list = [r[4] for r in results]
    motif_df = pd.DataFrame(motif_list)
    for col in motif_df.columns:
        test_df[col] = motif_df[col].astype(int)
        
    test_dataset = TensorDataset(X_test, Adj_test, Mask_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading trained GNN model...")
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(model)
    
    all_attr_x = []
    all_attr_adj = []
    
    for x, adj, mask in test_loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        x.requires_grad_()
        adj.requires_grad_()
        
        baseline_x = torch.zeros_like(x)
        baseline_adj = torch.zeros_like(adj)
        baseline_mask = mask.clone()
        
        attr = ig.attribute(
            inputs=(x, adj, mask),
            baselines=(baseline_x, baseline_adj, baseline_mask),
            n_steps=15,
            internal_batch_size=256
        )
        attr_x, attr_adj, attr_mask = attr
        
        all_attr_x.append(attr_x.detach().cpu().numpy())
        all_attr_adj.append(attr_adj.detach().cpu().numpy())
        
    all_attr_x = np.concatenate(all_attr_x, axis=0)
    all_attr_adj = np.concatenate(all_attr_adj, axis=0)
    
    print("Aggregating attributions by molecular motifs...")
    node_attr_sum = np.sum(np.abs(all_attr_x), axis=(1, 2))
    edge_attr_sum = np.sum(np.abs(all_attr_adj), axis=(1, 2, 3))
    total_attr = node_attr_sum + edge_attr_sum
    
    heavy_atom_counts = Mask_test.sum(dim=1).numpy()
    normalized_attr = total_attr / np.maximum(heavy_atom_counts, 1)
    test_df['normalized_attribution'] = normalized_attr
    
    print(f"\nProcessed {len(test_df)} molecules for attribution.")
    
    motif_cols = list(motif_df.columns)
    motif_attr_means = {}
    for motif in motif_cols:
        if test_df[motif].sum() > 0:
            motif_attr_means[motif] = test_df[test_df[motif] == 1]['normalized_attribution'].mean()
            
    top_motifs = sorted(motif_attr_means, key=motif_attr_means.get, reverse=True)[:5]
    print("Top Motifs by Average Normalized Attribution:")
    for m in top_motifs:
        print(f"  {m}: {motif_attr_means[m]:.4f}")
        
    print("\nPerforming statistical validation...")
    results_stats = []
    for motif in top_motifs:
        for prop in ['gap', 'mu']:
            group0 = test_df[test_df[motif] == 0][prop].dropna().values
            group1 = test_df[test_df[motif] == 1][prop].dropna().values
            
            if len(group0) < 5 or len(group1) < 5:
                continue
                
            stat_levene, p_levene = stats.levene(group0, group1)
            
            if p_levene > 0.05:
                stat_test, p_test = stats.f_oneway(group0, group1)
                test_name = 'ANOVA'
            else:
                stat_test, p_test = stats.kruskal(group0, group1)
                test_name = 'Kruskal-Wallis'
                
            results_stats.append({
                'Motif': motif,
                'Property': prop,
                'Group0_Size': len(group0),
                'Group1_Size': len(group1),
                'Group0_Mean': np.mean(group0),
                'Group1_Mean': np.mean(group1),
                'Levene_p': p_levene,
                'Test_Used': test_name,
                'Statistic': stat_test,
                'p_value': p_test
            })
            
    stats_df = pd.DataFrame(results_stats)
    if not stats_df.empty:
        _, pvals_adj, _, _ = multipletests(stats_df['p_value'], method='fdr_bh')
        stats_df['p_value_adj'] = pvals_adj
        
        print("\nStatistical Validation Results:")
        print(stats_df.to_string(index=False, float_format="%.4f"))
        
        stats_df.to_csv('data/motif_statistics.csv', index=False)
        print("\nsaved data/motif_statistics.csv columns:", list(stats_df.columns))
        
        print("\nGenerating boxplots...")
        plt.rcParams['text.usetex'] = False
        fig, axes = plt.subplots(len(top_motifs), 2, figsize=(10, 4 * len(top_motifs)))
        if len(top_motifs) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, motif in enumerate(top_motifs):
            for j, prop in enumerate(['gap', 'mu']):
                ax = axes[i, j]
                group0 = test_df[test_df[motif] == 0][prop].dropna().values
                group1 = test_df[test_df[motif] == 1][prop].dropna().values
                
                if len(group0) > 0 and len(group1) > 0:
                    ax.boxplot([group0, group1])
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(['False', 'True'])
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
        print("saved data/motif_property_boxplots.png")