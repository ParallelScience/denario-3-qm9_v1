import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Fragments
from sklearn.model_selection import train_test_split

# Extract all functional group descriptor functions from RDKit
frag_funcs = {name: func for name, func in Fragments.__dict__.items() if name.startswith("fr_")}

def process_single_smiles(smi):
    """
    Parses a SMILES string using RDKit and computes explicit size-normalization 
    features (heavy atom count, hydrogen count, specific atom counts), bond-type 
    counts, and functional-group counts.
    
    Args:
        smi (str): Canonical SMILES string.
        
    Returns:
        dict or None: Dictionary containing the computed features, or None if 
        the SMILES string could not be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    
    mol_h = Chem.AddHs(mol)
    
    h_count = 0
    c_count = 0
    n_count = 0
    o_count = 0
    f_count = 0
    
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "H":
            h_count += 1
        elif sym == "C":
            c_count += 1
        elif sym == "N":
            n_count += 1
        elif sym == "O":
            o_count += 1
        elif sym == "F":
            f_count += 1
            
    heavy_atom_count = mol.GetNumHeavyAtoms()
    
    single_bonds = 0
    double_bonds = 0
    triple_bonds = 0
    aromatic_bonds = 0
    
    for bond in mol.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            single_bonds += 1
        elif btype == Chem.rdchem.BondType.DOUBLE:
            double_bonds += 1
        elif btype == Chem.rdchem.BondType.TRIPLE:
            triple_bonds += 1
        elif btype == Chem.rdchem.BondType.AROMATIC:
            aromatic_bonds += 1
            
    feats = {
        "smiles": smi,
        "heavy_atom_count": heavy_atom_count,
        "h_count": h_count,
        "c_count": c_count,
        "n_count": n_count,
        "o_count": o_count,
        "f_count": f_count,
        "single_bonds": single_bonds,
        "double_bonds": double_bonds,
        "triple_bonds": triple_bonds,
        "aromatic_bonds": aromatic_bonds
    }
    
    for name, func in frag_funcs.items():
        feats[name] = func(mol)
        
    return feats

if __name__ == "__main__":
    print("Loading QM9 dataset...")
    df = pd.read_csv("/home/node/work/data/qm9/qm9.csv")
    print(f"Original dataset shape: {df.shape}")
    
    cols_to_drop = ["u0_atom", "u298_atom", "h298_atom", "g298_atom"]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    
    numeric_cols = ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]
    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict["mol_id"] = "first"
    
    df_unique = df.groupby("smiles", as_index=False).agg(agg_dict)
    num_duplicates = len(df) - len(df_unique)
    print(f"Found and averaged {num_duplicates} duplicated SMILES.")
    print(f"Unique SMILES count: {len(df_unique)}")
    
    print("Extracting RDKit features (atoms, bonds, functional groups)...")
    smiles_list = df_unique["smiles"].tolist()
    
    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        results = pool.map(process_single_smiles, smiles_list)
        
    results = [r for r in results if r is not None]
    df_feats = pd.DataFrame(results)
    print(f"Successfully processed {len(df_feats)} valid SMILES.")
    
    df_final = pd.merge(df_unique, df_feats, on="smiles")
    
    df_final["u0_per_heavy_atom"] = df_final["u0"] / df_final["heavy_atom_count"]
    
    print("\nSummary statistics for u0_per_heavy_atom:")
    print(df_final["u0_per_heavy_atom"].describe().to_string())
    
    train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_idx = train_df.index.values
    val_idx = val_df.index.values
    test_idx = test_df.index.values
    
    print(f"\nSplit sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    basic_cols = ["heavy_atom_count", "h_count", "c_count", "n_count", "o_count", "f_count", 
                  "single_bonds", "double_bonds", "triple_bonds", "aromatic_bonds"]
    frag_cols = list(frag_funcs.keys())
    
    cols_to_save = ["smiles", "mol_id"] + numeric_cols + basic_cols + ["u0_per_heavy_atom"]
    
    cleaned_csv_path = "data/cleaned_qm9.csv"
    df_final[cols_to_save].to_csv(cleaned_csv_path, index=False)
    print(f"\nSaved {cleaned_csv_path}")
    print(f"Columns: {list(df_final[cols_to_save].columns)}")
    
    features_npz_path = "data/additive_features.npz"
    X_features = df_final[basic_cols + frag_cols].values
    feature_names = np.array(basic_cols + frag_cols)
    np.savez(features_npz_path, X=X_features, features=feature_names)
    print(f"Saved {features_npz_path}")
    print(f"Keys: {list(np.load(features_npz_path).files)}")
    print(f"Feature matrix X shape: {X_features.shape}")
    
    splits_npz_path = "data/split_indices.npz"
    np.savez(splits_npz_path, train=train_idx, val=val_idx, test=test_idx)
    print(f"Saved {splits_npz_path}")
    print(f"Keys: {list(np.load(splits_npz_path).files)}")