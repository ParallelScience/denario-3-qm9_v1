import sys
import subprocess
import os

try:
    import rdkit
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

import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Fragments
from sklearn.model_selection import train_test_split

frag_funcs = {name: func for name, func in Fragments.__dict__.items() if name.startswith("fr_")}

def process_single_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    h_count = sum(1 for atom in mol_h.GetAtoms() if atom.GetSymbol() == 'H')
    heavy_atom_count = mol.GetNumHeavyAtoms()
    single_bonds = 0
    double_bonds = 0
    triple_bonds = 0
    aromatic_bonds = 0
    for bond in mol.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.BondType.SINGLE:
            single_bonds += 1
        elif btype == Chem.BondType.DOUBLE:
            double_bonds += 1
        elif btype == Chem.BondType.TRIPLE:
            triple_bonds += 1
        elif btype == Chem.BondType.AROMATIC:
            aromatic_bonds += 1
    res = {
        'smiles': smi,
        'heavy_atom_count': heavy_atom_count,
        'h_count': h_count,
        'single_bonds': single_bonds,
        'double_bonds': double_bonds,
        'triple_bonds': triple_bonds,
        'aromatic_bonds': aromatic_bonds
    }
    for name, func in frag_funcs.items():
        res[name] = func(mol)
    return res

if __name__ == "__main__":
    print("Loading QM9 dataset...")
    df = pd.read_csv('/home/node/work/data/qm9/qm9.csv')
    cols_to_drop = ['u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    agg_funcs = {col: 'mean' for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    if 'mol_id' in df.columns:
        agg_funcs['mol_id'] = 'first'
    df_grouped = df.groupby('smiles', as_index=False).agg(agg_funcs)
    smiles_list = df_grouped['smiles'].tolist()
    num_workers = min(16, mp.cpu_count())
    with mp.Pool(processes=num_workers) as pool:
        features_list = pool.map(process_single_smiles, smiles_list)
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    df_grouped = df_grouped.iloc[valid_indices].reset_index(drop=True)
    features_list = [f for f in features_list if f is not None]
    df_features = pd.DataFrame(features_list)
    df_grouped['u0_per_heavy_atom'] = df_grouped['u0'] / df_features['heavy_atom_count'].replace(0, np.nan)
    indices = np.arange(len(df_grouped))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    df_grouped.to_csv('data/cleaned_qm9.csv', index=False)
    df_features.to_csv('data/additive_features.csv', index=False)
    np.savez('data/split_indices.npz', train=train_idx, val=val_idx, test=test_idx)
    print(df_grouped[['u0', 'u0_per_heavy_atom']].describe().to_string())