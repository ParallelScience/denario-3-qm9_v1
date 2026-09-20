import pandas as pd
import numpy as np
from rdkit import Chem
import multiprocessing as mp

def get_rdkit_features(smiles):
    """
    Parses a SMILES string and extracts molecular features:
    heavy atom count, implicit hydrogen count, element counts (C, N, O, F),
    and bond type counts (single, double, triple, aromatic).
    Explicit hydrogens are added before counting bonds to ensure X-H bonds
    are included in the single bond count.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (np.nan,) * 10
    
    heavy_atom_count = mol.GetNumHeavyAtoms()
    # Implicit H count from the original molecule
    h_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    
    c_count = 0
    n_count = 0
    o_count = 0
    f_count = 0
    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        if num == 6: c_count += 1
        elif num == 7: n_count += 1
        elif num == 8: o_count += 1
        elif num == 9: f_count += 1
        
    # Add explicit Hs to count all bonds including X-H
    mol_with_hs = Chem.AddHs(mol)
    
    single_bonds = 0
    double_bonds = 0
    triple_bonds = 0
    aromatic_bonds = 0
    for bond in mol_with_hs.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE: single_bonds += 1
        elif btype == Chem.rdchem.BondType.DOUBLE: double_bonds += 1
        elif btype == Chem.rdchem.BondType.TRIPLE: triple_bonds += 1
        elif btype == Chem.rdchem.BondType.AROMATIC: aromatic_bonds += 1
        
    return (heavy_atom_count, h_count, c_count, n_count, o_count, f_count, 
            single_bonds, double_bonds, triple_bonds, aromatic_bonds)

if __name__ == '__main__':
    input_path = '/home/node/work/data/qm9/qm9.csv'
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Identify duplicates
    dup_counts = df['smiles'].value_counts()
    duplicated_smiles = dup_counts[dup_counts > 1]
    print(f"Found {len(duplicated_smiles)} duplicated SMILES.")
    
    # Group by SMILES and average numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_dict = {col: 'mean' for col in numeric_cols}
    
    # Keep the first mol_id for reference
    if 'mol_id' in df.columns:
        agg_dict['mol_id'] = 'first'
        
    print("Averaging target properties for duplicated SMILES...")
    df_grouped = df.groupby('smiles', as_index=False).agg(agg_dict)
    print(f"Dataset shape after deduplication: {df_grouped.shape}")
    
    print("Extracting RDKit features (heavy atom count, H count, element counts, bond types)...")
    with mp.Pool(processes=8) as pool:
        results = pool.map(get_rdkit_features, df_grouped['smiles'])
        
    feature_cols = [
        'heavy_atom_count', 'h_count', 'c_count', 'n_count', 'o_count', 'f_count',
        'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds'
    ]
    features_df = pd.DataFrame(results, columns=feature_cols)
    
    df_cleaned = pd.concat([df_grouped, features_df], axis=1)
    
    # Drop any rows where RDKit failed to parse SMILES
    initial_len = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['heavy_atom_count'])
    if len(df_cleaned) < initial_len:
        print(f"Dropped {initial_len - len(df_cleaned)} rows due to RDKit parsing failures.")
        
    # Ensure heavy_atom_count > 0 to avoid division by zero
    df_cleaned = df_cleaned[df_cleaned['heavy_atom_count'] > 0]
        
    # Compute per-heavy-atom u0
    print("Computing per-heavy-atom u0...")
    df_cleaned['u0_per_heavy_atom'] = df_cleaned['u0'] / df_cleaned['heavy_atom_count']
    
    # Save the cleaned dataset
    out_path = 'data/cleaned_qm9.csv'
    df_cleaned.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    print(f"saved {out_path} columns:", list(df_cleaned.columns))
    
    print("\nSummary statistics of derived features:")
    stats_cols = feature_cols + ['u0_per_heavy_atom']
    print(df_cleaned[stats_cols].describe().to_string())