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
        if btype == Chem.BondType.SINGLE: idx = 0
        elif btype == Chem.BondType.DOUBLE: idx = 1
        elif btype == Chem.BondType.TRIPLE: idx = 2
        elif btype == Chem.BondType.AROMATIC: idx = 3
        else: continue
        adj[i, j, idx] = 1.0
        adj[j, i, idx] = 1.0
    return x, adj, mask, target

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading trained GNN model...")
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    wrapper = GNNWrapper(model)
    ig = IntegratedGradients(wrapper)
    print("Computing Integrated Gradients...")
    batch_size = 256
    n_batches = int(np.ceil(len(X_test) / batch_size))
    all_node_attrs = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        x_batch = X_test[start_idx:end_idx].to(device)
        adj_batch = Adj_test[start_idx:end_idx].to(device)
        mask_batch = Mask_test[start_idx:end_idx].to(device)
        baseline_x = torch.zeros_like(x_batch).to(device)
        attributions, delta = ig.attribute(inputs=x_batch, baselines=baseline_x, additional_forward_args=(adj_batch, mask_batch), n_steps=20, return_convergence_delta=True)
        all_node_attrs.append(attributions.cpu().detach().numpy())
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  Processed batch {i+1}/{n_batches}")
    all_node_attrs = np.concatenate(all_node_attrs, axis=0)
    print("Normalizing attributions by molecule size...")
    node_attr_sum = np.sum(np.abs(all_node_attrs), axis=-1)
    n_heavy_atoms = Mask_test.sum(dim=1).numpy()
    peak_attrs = np.max(node_attr_sum, axis=1) / n_heavy_atoms
    peak_node_indices = np.argmax(node_attr_sum, axis=1)
    motifs = []
    for i in range(len(X_test)):
        peak_idx = peak_node_indices[i]
        node_feat = X_test[i, peak_idx].numpy()
        atom_type = "Unknown"
        if node_feat[0] == 1: atom_type = "C"
        elif node_feat[1] == 1: atom_type = "N"
        elif node_feat[2] == 1: atom_type = "O"
        elif node_feat[3] == 1: atom_type = "F"
        degree = -1
        for d in range(5):
            if node_feat[4 + d] == 1:
                degree = d
                break
        aromatic = "Aromatic" if node_feat[10] == 1 else "Aliphatic"
        motif = f"{aromatic}_{atom_type}_deg{degree}"
        motifs.append(motif)
    test_df['peak_motif'] = motifs
    test_df['peak_attribution'] = peak_attrs
    print("\nTop 5 Motifs by Frequency:")
    motif_counts = test_df['peak_motif'].value_counts()
    print(motif_counts.head(5).to_string())
    top_motifs = motif_counts.head(4).index.tolist()
    print("\nPerforming Statistical Validation...")
    properties = ['gap', 'mu']
    stat_results = []
    for prop in properties:
        for motif in top_motifs:
            group_with = test_df[test_df['peak_motif'] == motif][prop].dropna()
            group_without = test_df[test_df['peak_motif'] != motif][prop].dropna()
            if len(group_with) < 10 or len(group_without) < 10:
                continue
            stat_lev, p_lev = stats.levene(group_with, group_without)
            if p_lev > 0.05:
                stat_test, p_val = stats.f_oneway(group_with, group_without)
                test_name = "ANOVA"
            else:
                stat_test, p_val = stats.kruskal(group_with, group_without)
                test_name = "Kruskal-Wallis"
            mean_with = group_with.mean()
            mean_without = group_without.mean()
            effect_size = mean_with - mean_without
            stat_results.append({'Property': prop, 'Motif': motif, 'Test': test_name, 'Statistic': stat_test, 'p_value': p_val, 'Mean_With': mean_with, 'Mean_Without': mean_without, 'Effect_Size': effect_size, 'N_With': len(group_with), 'N_Without': len(group_without)})
    stat_df = pd.DataFrame(stat_results)
    if len(stat_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(stat_df['p_value'], alpha=0.05, method='fdr_bh')
        stat_df['p_value_adj'] = pvals_corrected
        stat_df['Significant'] = reject
    print("\nStatistical Test Results:")
    print(stat_df.to_string(index=False, float_format="%.4f"))
    stat_df.to_csv('data/motif_statistics.csv', index=False)
    print("\nsaved data/motif_statistics.csv columns:", list(stat_df.columns))
    print("\nGenerating Visualizations...")
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, len(top_motifs), figsize=(4 * len(top_motifs), 8))
    for i, prop in enumerate(properties):
        for j, motif in enumerate(top_motifs):
            ax = axes[i, j]
            data_with = test_df[test_df['peak_motif'] == motif][prop].dropna()
            data_without = test_df[test_df['peak_motif'] != motif][prop].dropna()
            ax.boxplot([data_without, data_with], labels=['False', 'True'])
            ax.set_title(f'{prop} by {motif}')
            if j == 0:
                ax.set_ylabel(prop)
    plt.tight_layout()
    plt.savefig('data/motif_property_boxplots.png', dpi=300)
    print("saved data/motif_property_boxplots.png")
    motif_summary = test_df['peak_motif'].value_counts().reset_index()
    motif_summary.columns = ['Motif', 'Count']
    motif_summary.to_csv('data/shap_motif_summary.csv', index=False)
    print("saved data/shap_motif_summary.csv columns:", list(motif_summary.columns))
    np.savez_compressed('data/shap_values.npz', attributions=all_node_attrs, smiles=test_df['smiles'].values)
    print("saved data/shap_values.npz keys:", list(np.load('data/shap_values.npz').files))