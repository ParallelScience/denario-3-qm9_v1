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
import matplotlib.pyplot as plt
from scipy import stats

try:
    import rdkit
    from rdkit import Chem
except ModuleNotFoundError:
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", override_dir, "--quiet", "rdkit"])
    sys.path.insert(0, override_dir)
    from rdkit import Chem

try:
    import captum
    from captum.attr import IntegratedGradients
except ModuleNotFoundError:
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", override_dir, "--quiet", "captum"])
    sys.path.insert(0, override_dir)
    from captum.attr import IntegratedGradients

try:
    import statsmodels
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
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
    return x, adj, mask, target

if __name__ == '__main__':
    test_df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
    smiles_list = test_df['smiles'].tolist()
    targets = test_df['std_residual'].tolist()
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
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    ig = IntegratedGradients(model)
    attributions = []
    for x, adj, mask in test_loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        attr = ig.attribute(inputs=x, baselines=torch.zeros_like(x).to(device), additional_forward_args=(adj, mask), n_steps=20)
        sizes = mask.sum(dim=1, keepdim=True).unsqueeze(-1).float()
        sizes = torch.clamp(sizes, min=1.0)
        norm_attr = attr / sizes
        attributions.append(norm_attr.cpu().detach().numpy())
    attributions = np.concatenate(attributions, axis=0)
    node_importance = np.abs(attributions).sum(axis=-1)
    peak_motifs = []
    for i, row in test_df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            peak_motifs.append("Unknown")
            continue
        mol = Chem.RemoveHs(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            peak_motifs.append("Unknown")
            continue
        valid_importance = node_importance[i, :num_atoms]
        peak_idx = np.argmax(valid_importance)
        atom = mol.GetAtomWithIdx(int(peak_idx))
        bond_indices = [bond.GetIdx() for bond in atom.GetBonds()]
        if not bond_indices:
            motif_smiles = atom.GetSymbol()
        else:
            env = Chem.PathToSubmol(mol, bond_indices)
            motif_smiles = Chem.MolToSmiles(env)
        peak_motifs.append(motif_smiles)
    test_df['peak_motif'] = peak_motifs
    top_motifs = test_df['peak_motif'].value_counts().head(10).index.tolist()
    properties = ['gap', 'mu']
    stats_results = []
    for prop in properties:
        for motif in top_motifs:
            has_motif = test_df['peak_motif'] == motif
            group1 = test_df.loc[has_motif, prop].dropna().values
            group2 = test_df.loc[~has_motif, prop].dropna().values
            if len(group1) < 5 or len(group2) < 5:
                continue
            stat_l, p_l = stats.levene(group1, group2)
            if p_l > 0.05:
                stat_t, p_t = stats.f_oneway(group1, group2)
                test_used = 'ANOVA'
            else:
                stat_t, p_t = stats.kruskal(group1, group2)
                test_used = 'Kruskal-Wallis'
            stats_results.append({'Property': prop, 'Motif': motif, 'Group1_Size': len(group1), 'Group2_Size': len(group2), 'Group1_Mean': np.mean(group1), 'Group2_Mean': np.mean(group2), 'Levene_p': p_l, 'Test_Used': test_used, 'Test_Stat': stat_t, 'Raw_p': p_t})
    stats_df = pd.DataFrame(stats_results)
    for prop in properties:
        mask = stats_df['Property'] == prop
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(stats_df.loc[mask, 'Raw_p'], method='fdr_bh')
            stats_df.loc[mask, 'Adj_p'] = p_adj
    stats_df.to_csv('data/motif_statistics.csv', index=False)
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, prop in enumerate(properties):
        for j, motif in enumerate(top_motifs[:5]):
            ax = axes[i, j]
            has_motif = test_df['peak_motif'] == motif
            group1 = test_df.loc[~has_motif, prop].dropna().values
            group2 = test_df.loc[has_motif, prop].dropna().values
            ax.boxplot([group1, group2])
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['False', 'True'])
            ax.set_title(f'{prop} by {motif}')
            if j == 0: ax.set_ylabel(prop)
    plt.tight_layout()
    plt.savefig('data/motif_property_boxplots.png', dpi=300)
    motif_summary = test_df['peak_motif'].value_counts().reset_index()
    motif_summary.columns = ['Motif', 'Count']
    motif_summary.to_csv('data/shap_motif_summary.csv', index=False)
    np.savez_compressed('data/shap_values.npz', attributions=attributions, smiles=test_df['smiles'].values)