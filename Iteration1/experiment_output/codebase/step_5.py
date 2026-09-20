import os
os.environ["OMP_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import json
import multiprocessing as mp
import matplotlib.pyplot as plt

def get_mol_info(args):
    """
    Computes structural descriptors for a given molecule to identify isomers 
    and serve as baseline features for variance decomposition.
    
    Args:
        args (tuple): A tuple containing (mol_id, smiles).
        
    Returns:
        tuple: (mol_id, formula, aromatic_rings, double_bonds, ring_count, 
                heteroatoms, spiro, bridgehead, fcsp3) or None if parsing fails.
    """
    mol_id, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    formula = rdMolDescriptors.CalcMolFormula(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    
    double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
            
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    fcsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    
    return (mol_id, formula, aromatic_rings, double_bonds, ring_count, heteroatoms, spiro, bridgehead, fcsp3)

def partial_spearman(df, x_col, y_col, covar_cols):
    """
    Calculates the partial Spearman correlation between two variables, 
    controlling for a set of covariates.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        x_col (str): The name of the first variable.
        y_col (str): The name of the second variable.
        covar_cols (list): List of covariate column names.
        
    Returns:
        tuple: (partial_spearman_rho, p_value)
    """
    x_rank = df[x_col].rank()
    y_rank = df[y_col].rank()
    covar_ranks = df[covar_cols].rank()
    
    reg_x = LinearRegression().fit(covar_ranks, x_rank)
    x_res = x_rank - reg_x.predict(covar_ranks)
    
    reg_y = LinearRegression().fit(covar_ranks, y_rank)
    y_res = y_rank - reg_y.predict(covar_ranks)
    
    r, p = pearsonr(x_res, y_res)
    return r, p

if __name__ == '__main__':
    print("Loading datasets...")
    qm9_df = pd.read_csv('/home/node/work/data/qm9/qm9.csv')
    gnn_df = pd.read_csv('data/gnn_predictions.csv')
    
    print("Merging datasets...")
    merged = pd.merge(qm9_df, gnn_df, on=['mol_id', 'smiles'])
    print(f"Merged dataset size: {len(merged)}")
    
    print("Computing molecular descriptors for all molecules...")
    with mp.Pool(16) as pool:
        results = pool.map(get_mol_info, zip(merged['mol_id'], merged['smiles']))
        
    info_df = pd.DataFrame([r for r in results if r is not None], 
                           columns=['mol_id', 'formula', 'aromatic_rings', 'double_bonds', 
                                    'ring_count', 'heteroatoms', 'spiro', 'bridgehead', 'fcsp3'])
    
    merged = pd.merge(merged, info_df, on='mol_id')
    
    print("Identifying conjugation isomers...")
    isomer_pairs = []
    np.random.seed(42)
    
    for formula, group in merged.groupby('formula'):
        if len(group) < 2:
            continue
        
        # Group by conjugation-related features
        subgroups = group.groupby(['aromatic_rings', 'double_bonds'])
        if len(subgroups) < 2:
            continue
            
        subgroup_indices = [g.index.tolist() for _, g in subgroups]
        
        pairs = []
        for i in range(len(subgroup_indices)):
            for j in range(i+1, len(subgroup_indices)):
                for idx1 in subgroup_indices[i]:
                    for idx2 in subgroup_indices[j]:
                        pairs.append((idx1, idx2))
                        if len(pairs) > 5000:  # Hard cap per formula to prevent explosion
                            break
                    if len(pairs) > 5000:
                        break
            if len(pairs) > 5000:
                break
                
        if len(pairs) > 1000:
            np.random.shuffle(pairs)
            pairs = pairs[:1000]
            
        for idx1, idx2 in pairs:
            row_i = merged.loc[idx1]
            row_j = merged.loc[idx2]
            isomer_pairs.append({
                'mol_id_1': row_i['mol_id'],
                'mol_id_2': row_j['mol_id'],
                'formula': formula,
                'actual_diff': row_i['u0_res'] - row_j['u0_res'],
                'pred_diff': row_i['gnn_pred_u0_res'] - row_j['gnn_pred_u0_res'],
                'split_1': row_i['split'],
                'split_2': row_j['split']
            })
            
    isomer_df = pd.DataFrame(isomer_pairs)
    isomer_df.to_csv('data/isomer_results.csv', index=False)
    print("saved data/isomer_results.csv columns:", list(isomer_df.columns))
    
    print("\n--- Isomer Validation ---")
    print(f"Identified {len(isomer_df)} conjugation isomer pairs (Total).")
    if len(isomer_df) > 0:
        r_iso, p_iso = pearsonr(isomer_df['actual_diff'], isomer_df['pred_diff'])
        print(f"Pearson correlation (All pairs): {r_iso:.4f} (p-value: {p_iso:.2e})")
        
    test_isomer_df = isomer_df[(isomer_df['split_1'] == 'test') & (isomer_df['split_2'] == 'test')]
    print(f"Identified {len(test_isomer_df)} conjugation isomer pairs (Both in Test Set).")
    if len(test_isomer_df) > 0:
        r_iso_test, p_iso_test = pearsonr(test_isomer_df['actual_diff'], test_isomer_df['pred_diff'])
        print(f"Pearson correlation (Test pairs): {r_iso_test:.4f} (p-value: {p_iso_test:.2e})")

    print("\n--- Variance Decomposition (Test Set) ---")
    test_df = merged[merged['split'] == 'test'].copy()
    
    X_var = test_df[['ring_count', 'heteroatoms', 'aromatic_rings', 'double_bonds', 'spiro', 'bridgehead', 'fcsp3']].values
    y_var = test_df['gnn_pred_u0_res'].values
    
    reg = LinearRegression().fit(X_var, y_var)
    r2 = reg.score(X_var, y_var)
    
    variance_decomp = {
        'R2_explained_by_basic_descriptors': float(r2),
        'Variance_attributed_to_nonlinear_physics': float(1 - r2),
        'sample_size': len(test_df)
    }
    
    with open('data/variance_decomp.json', 'w') as f:
        json.dump(variance_decomp, f, indent=2)
    print("saved data/variance_decomp.json keys:", list(variance_decomp.keys()))
    
    print(f"Sample size: {variance_decomp['sample_size']}")
    print(f"Variance explained by basic RDKit descriptors (R²): {variance_decomp['R2_explained_by_basic_descriptors']:.4f}")
    print(f"Variance attributed to non-linear physics: {variance_decomp['Variance_attributed_to_nonlinear_physics']:.4f}")

    print("\n--- Partial Correlations with Intensive Properties (Test Set) ---")
    correlations = []
    for prop in ['gap', 'mu', 'homo', 'lumo']:
        r, p = partial_spearman(test_df, 'gnn_pred_u0_res', prop, ['ring_count', 'heteroatoms'])
        correlations.append({
            'property': prop,
            'partial_spearman_rho': r,
            'p_value': p
        })
        
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv('data/correlations.csv', index=False)
    print("saved data/correlations.csv columns:", list(corr_df.columns))
    
    print("Controlling for: Ring Count, Heteroatom Count")
    print(corr_df.to_string(index=False))

    # --- Plotting ---
    plt.rcParams['text.usetex'] = False
    
    if len(isomer_df) > 0:
        plt.figure(figsize=(8, 6))
        
        # Plot all pairs in light gray
        plt.scatter(isomer_df['actual_diff'], isomer_df['pred_diff'], alpha=0.2, s=10, color='gray', label='Train/Val Pairs')
        
        # Plot test pairs in blue
        if len(test_isomer_df) > 0:
            plt.scatter(test_isomer_df['actual_diff'], test_isomer_df['pred_diff'], alpha=0.7, s=15, color='blue', label='Test Pairs')
            
        plt.xlabel(r"Actual $\Delta u_0$ Residual (Hartree)")
        plt.ylabel(r"GNN Predicted $\Delta u_0$ Residual (Hartree)")
        plt.title("Conjugation Isomers: Predicted vs Actual Energy Differences")
        
        min_val = min(isomer_df['actual_diff'].min(), isomer_df['pred_diff'].min())
        max_val = max(isomer_df['actual_diff'].max(), isomer_df['pred_diff'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
        
        plt.axvline(0, color='k', linestyle=':', alpha=0.5)
        plt.axhline(0, color='k', linestyle=':', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/isomer_diff_scatter.png', dpi=300)
        print("saved data/isomer_diff_scatter.png")
        
    plt.figure(figsize=(8, 5))
    plt.bar(corr_df['property'], corr_df['partial_spearman_rho'], color='skyblue', edgecolor='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel(r"Partial Spearman $\rho$")
    plt.title(r"Correlation of GNN-Predicted $u_0$ Residuals with Intensive Properties" + "\n(Controlling for Ring & Heteroatom Counts)")
    plt.tight_layout()
    plt.savefig('data/partial_correlations.png', dpi=300)
    print("saved data/partial_correlations.png")
    
    plt.close('all')