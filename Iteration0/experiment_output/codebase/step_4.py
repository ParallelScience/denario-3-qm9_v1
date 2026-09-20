import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors

def worker_init():
    """Initialize the descriptor calculator once per worker process."""
    global calc
    desc_names = [x[0] for x in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def process_smiles(smiles):
    """
    Computes Morgan fingerprints, 2D descriptors, and the Murcko scaffold for a given SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 1. Morgan Fingerprint (Radius 2, 2048 bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    
    # 2. RDKit 2D Descriptors
    global calc
    try:
        descs = calc.CalcDescriptors(mol)
        descs_arr = np.array(descs, dtype=np.float32)
    except Exception:
        descs_arr = np.full(len(calc.GetDescriptorNames()), np.nan, dtype=np.float32)
        
    # 3. Murcko Scaffold
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
        
    return fp_arr, descs_arr, scaffold

def generate_scaffold_split(scaffolds, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=42):
    """
    Splits the dataset based on Murcko scaffolds to test out-of-distribution generalization.
    Largest scaffolds are placed in the training set first.
    """
    np.random.seed(random_state)
    scaffold_to_indices = {}
    for i, scaffold in enumerate(scaffolds):
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(i)
        
    # Sort scaffolds by size descending, then by scaffold string to ensure reproducibility
    scaffold_sets = list(scaffold_to_indices.items())
    scaffold_sets.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
    
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = int(frac_train * len(scaffolds))
    val_cutoff = int((frac_train + frac_val) * len(scaffolds))
    
    for scaffold, indices in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(indices)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
            
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

if __name__ == '__main__':
    print("Loading datasets...")
    df = pd.read_csv('data/cleaned_qm9.csv')
    res_df = pd.read_csv('data/residuals.csv')
    
    smiles_list = df['smiles'].tolist()
    print(f"Total SMILES to process: {len(smiles_list)}")
    
    print("Generating Morgan fingerprints and RDKit 2D descriptors (using multiprocessing)...")
    with mp.Pool(processes=16, initializer=worker_init) as pool:
        results = pool.map(process_smiles, smiles_list)
        
    valid_indices = []
    fps = []
    descs = []
    scaffolds = []
    
    for i, res in enumerate(results):
        if res is not None:
            valid_indices.append(i)
            fps.append(res[0])
            descs.append(res[1])
            scaffolds.append(res[2])
            
    fps = np.array(fps)
    descs = np.array(descs)
    scaffolds = np.array(scaffolds)
    
    print(f"Successfully processed {len(valid_indices)} molecules.")
    print(f"Feature shapes - Morgan FP: {fps.shape}, 2D Descriptors: {descs.shape}")
    
    # Handle NaNs in descriptors (replace with column mean, or 0 if entire column is NaN)
    print("Handling NaNs in descriptors...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(descs, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    
    inds = np.where(np.isnan(descs))
    descs[inds] = np.take(col_means, inds[1])
    
    # Perform scaffold split
    print("\nPerforming scaffold split...")
    train_idx, val_idx, test_idx = generate_scaffold_split(scaffolds)
    print(f"Split sizes - Train: {len(train_idx)} ({len(train_idx)/len(scaffolds):.1%}), "
          f"Val: {len(val_idx)} ({len(val_idx)/len(scaffolds):.1%}), "
          f"Test: {len(test_idx)} ({len(test_idx)/len(scaffolds):.1%})")
    
    # Filter residuals dataframe to match valid indices
    res_df = res_df.iloc[valid_indices].reset_index(drop=True)
    
    # Scale residuals
    print("\nScaling residuals using StandardScaler (fitted on training set only)...")
    targets = ['res_u0', 'res_u0_norm', 'res_gap', 'res_mu']
    scalers = {}
    scaled_residuals = {}
    
    for target in targets:
        scaler = StandardScaler()
        train_vals = res_df[target].values[train_idx].reshape(-1, 1)
        scaler.fit(train_vals)
        
        all_vals = res_df[target].values.reshape(-1, 1)
        scaled_vals = scaler.transform(all_vals).flatten()
        
        scalers[target] = scaler
        scaled_residuals[target] = scaled_vals
        
        train_scaled = scaled_vals[train_idx]
        test_scaled = scaled_vals[test_idx]
        print(f"  {target} - Train Mean: {train_scaled.mean():.4f}, Train Std: {train_scaled.std():.4f}")
        print(f"  {target} - Test Mean: {test_scaled.mean():.4f}, Test Std: {test_scaled.std():.4f}")
        
    scaled_res_df = pd.DataFrame(scaled_residuals)
    scaled_res_df['smiles'] = res_df['smiles']
    
    # Save artifacts
    print("\nSaving artifacts...")
    
    desc_names = [x[0] for x in Descriptors.descList]
    
    features_path = 'data/engineered_features.npz'
    np.savez_compressed(features_path, fps=fps, descs=descs, desc_names=np.array(desc_names, dtype=str))
    print(f"saved {features_path}")
    print(f"saved {features_path} keys:", np.load(features_path).files)
    
    split_path = 'data/split_indices.npz'
    np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"saved {split_path}")
    print(f"saved {split_path} keys:", np.load(split_path).files)
    
    scaled_res_path = 'data/scaled_residuals.csv'
    scaled_res_df.to_csv(scaled_res_path, index=False)
    print(f"saved {scaled_res_path}")
    print(f"saved {scaled_res_path} columns:", list(scaled_res_df.columns))
    
    scalers_path = 'data/target_scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"saved {scalers_path}")
    print(f"saved {scalers_path} keys:", list(scalers.keys()))