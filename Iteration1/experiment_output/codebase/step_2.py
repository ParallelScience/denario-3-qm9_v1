import sys
import subprocess
import os

# Install xgboost locally to resolve ModuleNotFoundError without polluting the global env
override_dir = os.path.abspath("./.pip_overrides")
os.makedirs(override_dir, exist_ok=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--target", override_dir, "--quiet",
    "xgboost"
])
sys.path.insert(0, override_dir)

import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["OMP_NUM_THREADS"] = "8"

# 37 topological and compositional descriptors
DESC_NAMES = [
    "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", 
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", 
    "BalabanJ", "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3", 
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", 
    "NumHeterocycles", "NumAromaticHeterocycles", "NumAromaticCarbocycles", 
    "NumAliphaticHeterocycles", "NumAliphaticCarbocycles", "NumSpiroAtoms", 
    "NumBridgeheadAtoms", "NumRotatableBonds", "NumHeteroatoms", 
    "TPSA", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount"
]

def compute_descriptors(smiles):
    """
    Computes 37 topological RDKit descriptors for a given SMILES string.
    Returns a list of floats, with np.nan for any descriptor that fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(DESC_NAMES)
    
    desc_dict = dict(Descriptors.descList)
    res = []
    for name in DESC_NAMES:
        try:
            if name in desc_dict:
                val = desc_dict[name](mol)
                res.append(float(val))
            else:
                res.append(np.nan)
        except Exception:
            res.append(np.nan)
    return res

if __name__ == '__main__':
    print("Loading data/u0_res_baseline.csv...")
    df = pd.read_csv('data/u0_res_baseline.csv')
    print(f"Loaded {len(df)} molecules.")
    
    print("Computing RDKit descriptors...")
    with mp.Pool(16) as pool:
        results = pool.map(compute_descriptors, df['smiles'].tolist())
        
    desc_df = pd.DataFrame(results, columns=DESC_NAMES)
    
    print("Imputing NaNs in descriptors with median values...")
    desc_df = desc_df.fillna(desc_df.median())
    
    df = pd.concat([df, desc_df], axis=1)
    
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    
    X_train = df.loc[train_mask, DESC_NAMES].values
    y_train = df.loc[train_mask, 'u0_res'].values
    
    X_val = df.loc[val_mask, DESC_NAMES].values
    y_val = df.loc[val_mask, 'u0_res'].values
    
    X_test = df.loc[test_mask, DESC_NAMES].values
    y_test = df.loc[test_mask, 'u0_res'].values
    
    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("\nEvaluating model...")
    metrics = {}
    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"--- {name.capitalize()} Set ---")
        print(f"  RMSE: {rmse:.6f} Hartree")
        print(f"  MAE:  {mae:.6f} Hartree")
        print(f"  R2:   {r2:.6f}")
        
        df.loc[df['split'] == name, 'xgb_pred_u0_res'] = preds
        
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': DESC_NAMES,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(fi_df.head(10).to_string(index=False))
    
    pred_df = df[['mol_id', 'smiles', 'split', 'u0_res', 'xgb_pred_u0_res']]
    pred_df.to_csv('data/xgb_predictions.csv', index=False)
    print(f"\nSaved data/xgb_predictions.csv columns: {list(pred_df.columns)}")
    
    with open('data/xgb_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved data/xgb_metrics.json")
    
    fi_df.to_csv('data/xgb_feature_importances.csv', index=False)
    print(f"Saved data/xgb_feature_importances.csv columns: {list(fi_df.columns)}")