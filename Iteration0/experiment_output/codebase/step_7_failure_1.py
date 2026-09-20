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

# Ensure Captum is available
try:
    import captum
    from captum.attr import IntegratedGradients
except ModuleNotFoundError:
    print("captum not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", override_dir, "--quiet",
        "captum"
    ])
    sys.path.insert(0, override_dir)
    from captum.attr import IntegratedGradients

# Ensure statsmodels is available
try:
    import statsmodels
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
    print("statsmodels not found, installing locally to .pip_overrides...")
    override_dir = os.path.abspath("./.pip_overrides")
    os.makedirs(override_dir, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", override_dir, "--quiet",
        "statsmodels"
    ])
    sys.path.insert(0, override_dir)
    from statsmodels.stats.multitest import multipletests

from scipy.stats import levene, f_oneway, kruskal

# ---------------------------------------------------------
# GNN Architecture Definition (matching Step 4 & 5)
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

class IGModel(nn.Module):
    """Wrapper to ensure output is 2D for Captum."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, adj, mask):
        return self.model(x, adj, mask).unsqueeze(1)

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

def get_radius_1_motif(mol, atom_idx):
    """Extracts the canonical SMILES of the radius-1 environment around an atom."""
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        atoms_to_use = [atom_idx]
        bonds_to_use = []
        for nbr in atom.GetNeighbors():
            atoms_to_use.append(nbr.GetIdx())
            bonds_to_use.append(mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetIdx())
        return Chem.MolFragmentToSmiles(mol, atomsToUse=atoms_to_use, bondsToUse=bonds_to_use, canonical=True)
    except Exception:
        return "ERROR"

if __name__ == '__main__':
    print("Loading test set data...")
    df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
    smiles_list = df['smiles'].tolist()
    targets = df['std_residual'].tolist()
    
    print(f"Processing {len(smiles_list)} SMILES to dense graphs...")
    num_workers = min(16, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_row, zip(smiles_list, targets))
        
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    df = df.iloc[valid_indices].reset_index(drop=True)
    results = [results[i] for i in valid_indices]
    smiles_list = df['smiles'].tolist()
    
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
    
    ig_model = IGModel(model)
    ig = IntegratedGradients(ig_model)
    
    print("Computing Integrated Gradients attributions...")
    attr_x_list = []
    attr_adj_list = []
    
    for step, (x_batch, adj_batch, mask_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        adj_batch = adj_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        x_base = torch.zeros_like(x_batch)
        adj_base = torch.zeros_like(adj_batch)
        
        # Compute attributions
        attributions = ig.attribute(
            inputs=(x_batch, adj_batch),
            baselines=(x_base, adj_base),
            additional_forward_args=(mask_batch,),
            target=0,
            n_steps=20,
            internal_batch_size=256
        )
        
        attr_x_list.append(attributions[0].cpu().detach().numpy())
        attr_adj_list.append(attributions[1].cpu().detach().numpy())
        
        if (step + 1) % 20 == 0:
            print(f"  Processed {step + 1}/{len(test_loader)} batches...")
            
    attr_x = np.concatenate(attr_x_list, axis=0)
    attr_adj = np.concatenate(attr_adj_list, axis=0)
    
    print("Normalizing attributions by molecule size...")
    mask_all = Mask_test.numpy()
    num_atoms = mask_all.sum(axis=1).reshape(-1, 1, 1)
    attr_x_norm = attr_x / np.maximum(num_atoms, 1)
    
    print("Identifying peak motifs...")
    # Sum absolute attributions over feature dimensions
    node_scores = np.sum(np.abs(attr_x_norm), axis=-1)
    # Mask out non-existent atoms
    node_scores[~mask_all] = -1.0
    
    peak_indices = np.argmax(node_scores, axis=1)
    
    peak_motifs = []
    all_motifs_per_mol = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        
        motifs_in_mol = set()
        peak_motif = None
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            motif = get_radius_1_motif(mol, idx)
            motifs_in_mol.add(motif)
            if idx == peak_indices[i]:
                peak_motif = motif
                
        peak_motifs.append(peak_motif)
        all_motifs_per_mol.append(motifs_in_mol)
        
    from collections import Counter
    motif_counts = Counter(peak_motifs)
    
    motif_summary = pd.DataFrame(motif_counts.most_common(20), columns=['Motif', 'Peak_Frequency'])
    motif_summary.to_csv('data/motif_attribution_summary.csv', index=False)
    print("saved data/motif_attribution_summary.csv columns:", list(motif_summary.columns))
    print("\nTop 10 Peak Motifs:")
    print(motif_summary.head(10).to_string(index=False))
    
    # Select top 5 motifs for statistical validation
    top_motifs = [motif for motif, count in motif_counts.most_common(5) if motif != "ERROR"]
    
    for motif in top_motifs:
        df[f'has_{motif}'] = [motif in motifs for motifs in all_motifs_per_mol]
        
    print("\nPerforming statistical validation...")
    results = []
    properties = ['gap', 'mu']
    
    for motif in top_motifs:
        group_has = df[df[f'has_{motif}']]
        group_not = df[~df[f'has_{motif}']]
        
        for prop in properties:
            vals_has = group_has[prop].dropna().values
            vals_not = group_not[prop].dropna().values
            
            if len(vals_has) < 5 or len(vals_not) < 5:
                continue
                
            # Levene's test for homoscedasticity
            stat_l, p_l = levene(vals_has, vals_not)
            
            if p_l > 0.05:
                stat, p_val = f_oneway(vals_has, vals_not)
                test_used = 'ANOVA'
            else:
                stat, p_val = kruskal(vals_has, vals_not)
                test_used = 'Kruskal-Wallis'
                
            mean_has, std_has = np.mean(vals_has), np.std(vals_has, ddof=1)
            mean_not, std_not = np.mean(vals_not), np.std(vals_not, ddof=1)
            n_has, n_not = len(vals_has), len(vals_not)
            
            pooled_std = np.sqrt(((n_has - 1) * std_has**2 + (n_not - 1) * std_not**2) / (n_has + n_not - 2))
            cohens_d = (mean_has - mean_not) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'Motif': motif,
                'Property': prop,
                'N_has': n_has,
                'N_not': n_not,
                'Mean_has': mean_has,
                'Mean_not': mean_not,
                'Levene_p': p_l,
                'Test_Used': test_used,
                'Statistic': stat,
                'p_value': p_val,
                'Cohens_d': cohens_d
            })
            
    res_df = pd.DataFrame(results)
    
    if len(res_df) > 0:
        _, p_adj, _, _ = multipletests(res_df['p_value'], method='fdr_bh')
        res_df['p_value_adj'] = p_adj
        
    res_df.to_csv('data/motif_statistical_validation.csv', index=False)
    print("saved data/motif_statistical_validation.csv columns:", list(res_df.columns))
    print("\nStatistical Validation Results:")
    print(res_df.to_string(index=False, float_format="%.4f"))
    
    print("\nGenerating boxplots...")
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(len(top_motifs), 2, figsize=(10, 4 * len(top_motifs)))
    if len(top_motifs) == 1:
        axes = [axes]
        
    for i, motif in enumerate(top_motifs):
        for j, prop in enumerate(properties):
            ax = axes[i][j]
            
            group_has = df[df[f'has_{motif}']][prop].dropna().values
            group_not = df[~df[f'has_{motif}']][prop].dropna().values
            
            plot_data = []
            labels = []
            if len(group_not) > 0:
                plot_data.append(group_not)
                labels.append('False')
            if len(group_has) > 0:
                plot_data.append(group_has)
                labels.append('True')
                
            if plot_data:
                ax.boxplot(plot_data, labels=labels)
            ax.set_title(f'{prop} by presence of {motif}')
            ax.set_ylabel(prop)
            
    plt.tight_layout()
    plt.savefig('data/motif_property_boxplots.png', dpi=300)
    print("saved data/motif_property_boxplots.png")