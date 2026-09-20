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
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.LayerNorm(2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim), nn.LayerNorm(emb_dim))
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
        self.gate_nn = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
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
        self.node_emb = nn.Sequential(nn.Linear(node_dim, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU())
        self.convs = nn.ModuleList([DenseGINEConv(emb_dim) for _ in range(num_layers)])
        self.pool = DenseGlobalAttentionPooling(emb_dim)
        self.out_nn = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2), nn.LayerNorm(emb_dim // 2), nn.ReLU(), nn.Linear(emb_dim // 2, 1))
    def forward(self, x, adj, mask):
        h = self.node_emb(x)
        h = h * mask.unsqueeze(-1).float()
        for conv in self.convs: h = conv(h, adj, mask)
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
    if mol is None: return None
    mol = Chem.RemoveHs(mol)
    max_N = 9
    x = np.zeros((max_N, 16), dtype=np.float32)
    adj = np.zeros((max_N, max_N, 4), dtype=np.float32)
    mask = np.zeros(max_N, dtype=bool)
    atom_mapping = {6: 0, 7: 1, 8: 2, 9: 3}
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_N: break
        mask[i] = True
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_mapping: x[i, atom_mapping[atomic_num]] = 1.0
        deg = atom.GetDegree()
        if deg <= 4: x[i, 4 + deg] = 1.0
        x[i, 9] = atom.GetFormalCharge()
        x[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
        h_count = atom.GetTotalNumHs()
        if h_count <= 4: x[i, 11 + h_count] = 1.0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i >= max_N or j >= max_N: continue
        btype = bond.GetBondType()
        idx = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}.get(btype, -1)
        if idx != -1:
            adj[i, j, idx] = 1.0
            adj[j, i, idx] = 1.0
    return x, adj, mask, target

def get_subgraph_smiles(mol, atom_indices):
    if not atom_indices: return ""
    env = Chem.EditableMol(Chem.Mol())
    idx_map = {}
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        idx_map[idx] = env.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_indices and a2 in atom_indices: env.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())
    submol = env.GetMol()
    try:
        Chem.SanitizeMol(submol)
        return Chem.MolToSmiles(submol)
    except: return Chem.MolToSmiles(submol, sanitize=False)

if __name__ == '__main__':
    test_df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
    smiles_list, targets = test_df['smiles'].tolist(), test_df['std_residual'].tolist()
    with mp.Pool(min(16, mp.cpu_count())) as pool: results = pool.map(process_row, zip(smiles_list, targets))
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    test_df = test_df.iloc[valid_indices].reset_index(drop=True)
    results = [results[i] for i in valid_indices]
    X_test = torch.tensor(np.stack([r[0] for r in results]), dtype=torch.float32)
    Adj_test = torch.tensor(np.stack([r[1] for r in results]), dtype=torch.float32)
    Mask_test = torch.tensor(np.stack([r[2] for r in results]), dtype=torch.bool)
    test_loader = DataLoader(TensorDataset(X_test, Adj_test, Mask_test), batch_size=128, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    ig = IntegratedGradients(GNNWrapper(model))
    all_node_attrs = []
    for x, adj, mask in test_loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        x.requires_grad_()
        attr, _ = ig.attribute(inputs=x, baselines=torch.zeros_like(x).to(device), additional_forward_args=(adj, mask), n_steps=20)
        all_node_attrs.append(attr.sum(dim=-1).detach().cpu().numpy())
    all_node_attrs = np.concatenate(all_node_attrs, axis=0)
    num_heavy_atoms = np.maximum(Mask_test.sum(dim=1).numpy(), 1)
    norm_node_attrs = all_node_attrs / num_heavy_atoms[:, None]
    peak_motifs = []
    for idx, smiles in enumerate(test_df['smiles']):
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        n_atoms = mol.GetNumAtoms()
        attrs = norm_node_attrs[idx, :n_atoms]
        peak_atom_idx = np.argmax(np.abs(attrs))
        env_atoms = set([peak_atom_idx])
        for neighbor in mol.GetAtomWithIdx(int(peak_atom_idx)).GetNeighbors():
            env_atoms.add(neighbor.GetIdx())
            if len(env_atoms) >= 3: break
        peak_motifs.append(get_subgraph_smiles(mol, list(env_atoms)) or "UNKNOWN")
    test_df['peak_motif'] = peak_motifs
    motif_counts = test_df['peak_motif'].value_counts()
    top_motifs = motif_counts.head(10).index.tolist()
    pd.DataFrame({'Motif': motif_counts.index, 'Peak_Frequency': motif_counts.values}).to_csv('data/motif_attribution_summary.csv', index=False)
    stat_results = []
    for motif in top_motifs:
        has_motif = test_df['peak_motif'] == motif
        for prop in ['gap', 'mu']:
            group_has, group_not = test_df.loc[has_motif, prop].dropna().values, test_df.loc[~has_motif, prop].dropna().values
            if len(group_has) < 10 or len(group_not) < 10: continue
            p_lev = stats.levene(group_has, group_not)[1]
            if p_lev > 0.05:
                stat_test, p_val = stats.f_oneway(group_has, group_not)
                test_name = "ANOVA"
            else:
                stat_test, p_val = stats.kruskal(group_has, group_not)
                test_name = "Kruskal-Wallis"
            stat_results.append({'Motif': motif, 'Property': prop, 'N_has': len(group_has), 'N_not': len(group_not), 'Mean_has': np.mean(group_has), 'Mean_not': np.mean(group_not), 'Levene_p': p_lev, 'Test_Used': test_name, 'Statistic': stat_test, 'p_value': p_val})
    stat_df = pd.DataFrame(stat_results)
    if not stat_df.empty:
        stat_df['p_value_adj'] = multipletests(stat_df['p_value'], method='fdr_bh')[1]
        stat_df.to_csv('data/motif_statistical_validation.csv', index=False)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, prop in enumerate(['gap', 'mu']):
        for j, motif in enumerate(top_motifs[:5]):
            ax = axes[i, j]
            has_motif = test_df['peak_motif'] == motif
            plot_data = [test_df.loc[~has_motif, prop].dropna().values, test_df.loc[has_motif, prop].dropna().values]
            ax.boxplot(plot_data)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['False', 'True'])
            ax.set_title(f'{prop} by {motif}')
    plt.tight_layout()
    plt.savefig('data/motif_property_boxplots.png', dpi=300)