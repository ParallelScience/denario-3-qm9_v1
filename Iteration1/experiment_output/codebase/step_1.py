import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "8"

def process_smiles(smiles):
    """
    Parses a SMILES string using RDKit and computes basic molecular features.
    
    Args:
        smiles (str): Canonical SMILES string of the molecule.
        
    Returns:
        tuple: A tuple containing (heavy_atom_count, h_count, single_bonds, 
               double_bonds, triple_bonds, aromatic_bonds, scaffold_smiles).
               Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol_h = Chem.AddHs(mol)
    
    heavy_atom_count = mol.GetNumHeavyAtoms()
    h_count = sum(1 for atom in mol_h.GetAtoms() if atom.GetAtomicNum() == 1)
    
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
            
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    
    return (heavy_atom_count, h_count, single_bonds, double_bonds, triple_bonds, aromatic_bonds, scaffold)

if __name__ == '__main__':
    print("Loading QM9 dataset...")
    df = pd.read_csv('/home/node/work/data/qm9/qm9.csv')
    print(f"Loaded {len(df)} molecules.")
    
    print("Processing SMILES with RDKit...")
    with mp.Pool(16) as pool:
        results = pool.map(process_smiles, df['smiles'].tolist())
        
    features = pd.DataFrame(results, columns=[
        'heavy_atom_count', 'h_count', 'single_bonds', 
        'double_bonds', 'triple_bonds', 'aromatic_bonds', 'scaffold'
    ])
    
    df = pd.concat([df, features], axis=1)
    
    initial_len = len(df)
    df = df.dropna(subset=['heavy_atom_count'])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} molecules due to RDKit parsing errors.")
        
    print("Fitting HuberRegressor for additive baseline...")
    X = df[['heavy_atom_count', 'h_count', 'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds']].values
    y = df['u0'].values
    
    reg = HuberRegressor(max_iter=2000)
    reg.fit(X, y)
    
    u0_pred = reg.predict(X)
    df['u0_res'] = y - u0_pred
    
    r2 = r2_score(y, u0_pred)
    rmse = np.sqrt(mean_squared_error(y, u0_pred))
    
    print("\n--- Additive Baseline Fit Results ---")
    print(f"Sample size: {len(df)}")
    print(f"R2 Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f} Hartree")
    print("Fitted Coefficients:")
    feature_names = ['heavy_atom_count', 'h_count', 'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds']
    for name, coef in zip(feature_names, reg.coef_):
        print(f"  {name}: {coef:.6f} Hartree")
    print(f"  Intercept: {reg.intercept_:.6f} Hartree")
    
    print("\n--- Residual Analysis ---")
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['heavy_atom_count'], df['u0_res'])
    print(f"Initial u0_res vs N_heavy:")
    print(f"  Slope: {slope:.6e} +/- {std_err:.6e} Hartree/atom")
    print(f"  R-squared: {r_value**2:.6f}")
    print(f"  p-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print("Slope is statistically significant (p < 0.05). Applying linear correction to residuals...")
        df['u0_res'] = df['u0_res'] - (slope * df['heavy_atom_count'] + intercept)
        
        slope_corr, intercept_corr, r_value_corr, p_value_corr, std_err_corr = stats.linregress(df['heavy_atom_count'], df['u0_res'])
        print(f"Corrected u0_res vs N_heavy:")
        print(f"  Slope: {slope_corr:.6e} +/- {std_err_corr:.6e} Hartree/atom")
        print(f"  p-value: {p_value_corr:.2e}")
    else:
        print("Slope is not statistically significant. No correction applied.")
        
    print("\n--- Generating Scaffold Split ---")
    scaffolds = df['scaffold'].values
    unique_scaffolds = np.unique(scaffolds)
    np.random.seed(42)
    np.random.shuffle(unique_scaffolds)
    
    n_scaffolds = len(unique_scaffolds)
    train_end = int(0.8 * n_scaffolds)
    val_end = int(0.9 * n_scaffolds)
    
    train_scaffolds = set(unique_scaffolds[:train_end])
    val_scaffolds = set(unique_scaffolds[train_end:val_end])
    
    def assign_split(scaffold):
        """
        Assigns a split label based on the molecule's Murcko scaffold.
        
        Args:
            scaffold (str): Murcko scaffold SMILES.
            
        Returns:
            str: 'train', 'val', or 'test'.
        """
        if scaffold in train_scaffolds:
            return 'train'
        elif scaffold in val_scaffolds:
            return 'val'
        else:
            return 'test'
            
    df['split'] = df['scaffold'].apply(assign_split)
    
    train_count = (df['split'] == 'train').sum()
    val_count = (df['split'] == 'val').sum()
    test_count = (df['split'] == 'test').sum()
    
    print(f"Split sizes (molecules): Train={train_count}, Val={val_count}, Test={test_count}")
    
    out_path = 'data/u0_res_baseline.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved data/u0_res_baseline.csv columns:", list(df.columns))
    
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.scatter(df['heavy_atom_count'], df['u0_res'], alpha=0.1, s=1)
    plt.xlabel('Heavy Atom Count')
    plt.ylabel(r'Residual $u_0$ (Hartree)')
    plt.title('Intensive Residuals vs Heavy Atom Count')
    plt.tight_layout()
    plot_path = 'data/u0_res_vs_nheavy.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Saved {plot_path}")