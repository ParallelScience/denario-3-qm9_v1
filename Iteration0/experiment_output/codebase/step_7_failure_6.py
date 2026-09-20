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
        return self.model(x, adj, mask).unsqueeze(-1)

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
    plt.rcParams['text.usetex'] = False
    print("Loading test set data...")
    test_df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
    if 'gap' not in test_df.columns or 'mu' not in test_df.columns:
        print("Merging intensive properties...")
        qm9_df = pd.read_csv('data/cleaned_qm9.csv')
        test_df = test_df.merge(qm9_df[['smiles', 'gap', 'mu']], on='smiles', how='left')
    smiles_list = test_df['smiles'].tolist()
    print(f"Processing {len(smiles_list)} SMILES to dense graphs...")
    num_workers = min(16, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        results_mp = pool.map(process_row, zip(smiles_list, [0]*len(smiles_list)))
    valid_indices = [i for i, r in enumerate(results_mp) if r is not None]
    test_df = test_df.iloc[valid_indices].reset_index(drop=True)
    results_mp = [results_mp[i] for i in valid_indices]
    X_test = torch.tensor(np.stack([r[0] for r in results_mp]), dtype=torch.float32)
    Adj_test = torch.tensor(np.stack([r[1] for r in results_mp]), dtype=torch.float32)
    Mask_test = torch.tensor(np.stack([r[2] for r in results_mp]), dtype=torch.bool)
    test_dataset = TensorDataset(X_test, Adj_test, Mask_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading trained GNN model...")
    model = GNNModel(emb_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    ig = IntegratedGradients(GNNWrapper(model))
    node_attrs = []
    edge_attrs = []
    print("Computing Integrated Gradients...")
    for x, adj, mask in test_loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        attr = ig.attribute(inputs=(x, adj), target=0, additional_forward_args=(mask,), baselines=(torch.zeros_like(x), torch.zeros_like(adj)), n_steps=10, internal_batch_size=256)
        node_attrs.append(attr[0].detach().cpu().numpy())
        edge_attrs.append(attr[1].detach().cpu().numpy())
    node_attrs = np.concatenate(node_attrs, axis=0)
    edge_attrs = np.concatenate(edge_attrs, axis=0)
    print("Aggregating attributions by molecular motifs...")
    node_scores = np.abs(node_attrs).sum(axis=-1)
    edge_scores = np.abs(edge_attrs).sum(axis=-1)
    num_atoms = Mask_test.sum(dim=1).numpy()
    node_scores = node_scores / num_atoms[:, None]
    edge_scores = edge_scores / num_atoms[:, None, None]
    motifs = {'Carbonyl': '[CX3]=[OX1]', 'Ether': '[OD2]([#6])[#6]', 'Hydroxyl': '[OX2H]', 'Primary_Amine': '[NX3H2]', 'Secondary_Amine': '[NX3H1]([#6])[#6]', 'Tertiary_Amine': '[NX3]([#6])([#6])[#6]', 'Nitrile': '[NX1]#[CX2]', 'Amide': '[NX3][CX3](=[OX1])', 'Aromatic_Ring': 'a1aaaaa1', 'Aliphatic_Ring': 'C1CCCC1', 'Fluorine': 'F', 'Alkyne': 'C#C', 'Alkene': 'C=C'}
    motif_attributions = {k: [] for k in motifs.keys()}
    molecule_motifs = {k: np.zeros(len(test_df), dtype=bool) for k in motifs.keys()}
    smiles_list = test_df['smiles'].tolist()
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        for motif_name, smarts in motifs.items():
            pattern = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                molecule_motifs[motif_name][i] = True
                attr_sum = 0.0
                for match in matches:
                    for idx in match:
                        if idx < 9:
                            attr_sum += node_scores[i, idx]
                    for idx1 in match:
                        for idx2 in match:
                            if idx1 < 9 and idx2 < 9 and idx1 != idx2:
                                attr_sum += edge_scores[i, idx1, idx2] / 2.0
                motif_attributions[motif_name].append(attr_sum)
    valid_motifs = [k for k, v in molecule_motifs.items() if v.sum() >= 10 and (~v).sum() >= 10]
    avg_motif_attr = {k: np.mean(motif_attributions[k]) for k in valid_motifs}
    top_motifs = sorted(avg_motif_attr.keys(), key=lambda k: avg_motif_attr[k], reverse=True)[:min(5, len(valid_motifs))]
    print(f"\nProcessed {len(test_df)} molecules for attribution.")
    print("Top Motifs by Average Normalized Attribution:")
    for m in top_motifs:
        print(f"  {m}: {avg_motif_attr[m]:.4f}")
    print("\nPerforming statistical validation...")
    results = []
    for motif in top_motifs:
        has_motif = molecule_motifs[motif]
        for prop in ['gap', 'mu']:
            group1 = test_df[has_motif][prop].dropna().values
            group0 = test_df[~has_motif][prop].dropna().values
            if len(group1) < 5 or len(group0) < 5:
                continue
            stat_l, p_l = stats.levene(group0, group1)
            if p_l > 0.05:
                stat, p_val = stats.f_oneway(group0, group1)
                test_used = 'ANOVA'
            else:
                stat, p_val = stats.kruskal(group0, group1)
                test_used = 'Kruskal-Wallis'
            results.append({'Motif': motif, 'Property': prop, 'Group0_Size': len(group0), 'Group1_Size': len(group1), 'Group0_Mean': np.mean(group0), 'Group1_Mean': np.mean(group1), 'Levene_p': p_l, 'Test_Used': test_used, 'Statistic': stat, 'p_value': p_val})
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        _, pvals_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_value_adj'] = pvals_corrected
        print("\nStatistical Validation Results:")
        print(results_df.to_string(index=False, float_format="%.4f"))
        results_df.to_csv('data/motif_statistics.csv', index=False)
        print("\nsaved data/motif_statistics.csv columns:", list(results_df.columns))
    else:
        print("\nNo valid statistical results generated.")
    print("\nGenerating boxplots...")
    if top_motifs:
        fig, axes = plt.subplots(2, len(top_motifs), figsize=(5 * len(top_motifs), 10))
        if len(top_motifs) == 1:
            axes = np.expand_dims(axes, axis=1)
        for i, prop in enumerate(['gap', 'mu']):
            for j, motif in enumerate(top_motifs):
                ax = axes[i, j]
                has_motif = molecule_motifs[motif]
                group1 = test_df[has_motif][prop].dropna()
                group0 = test_df[~has_motif][prop].dropna()
                if len(group0) > 0 and len(group1) > 0:
                    ax.boxplot([group0, group1], labels=['False', 'True'])
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