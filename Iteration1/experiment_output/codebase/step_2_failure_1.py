import os
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import GraphDescriptors, rdMolDescriptors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["OMP_NUM_THREADS"] = "8"

DESC_NAMES = [
    "BertzCT", "Chi0", "Chi1", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v",
    "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n", "BalabanJ", "HallKierAlpha",
    "Ipc", "Kappa1", "Kappa2", "Kappa3", "NumRings", "NumAromaticRings",
    "NumAliphaticRings", "NumSaturatedRings", "NumHeterocycles",
    "NumAromaticHeterocycles", "NumAromaticCarbocycles", "NumAliphaticHeterocycles",
    "NumAliphaticCarbocycles", "NumSpiroAtoms", "NumBridgeheadAtoms",
    "NumAmideBonds", "NumRotatableBonds", "NumHeteroatoms", "TPSA", "FractionCSP3"
]

def compute_descriptors(smiles):
    """
    Computes 35 topological RDKit descriptors for a given SMILES string.
    Returns a list of floats, with np.nan for any descriptor that fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(DESC_NAMES)
    
    funcs = [
        GraphDescriptors.BertzCT,
        GraphDescriptors.Chi0,
        GraphDescriptors.Chi1,
        GraphDescriptors.Chi0v,
        GraphDescriptors.Chi1v,
        GraphDescriptors.Chi2v,
        GraphDescriptors.Chi3v,
        GraphDescriptors.Chi4v,
        GraphDescriptors.Chi0n,
        GraphDescriptors.Chi1n,
        GraphDescriptors.Chi2n,
        GraphDescriptors.Chi3n,
        GraphDescriptors.Chi4n,
        GraphDescriptors.BalabanJ,
        GraphDescriptors.HallKierAlpha,
        GraphDescriptors.Ipc,
        GraphDescriptors.Kappa1,
        GraphDescriptors.Kappa2,
        GraphDescriptors.Kappa3,
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumAromaticRings,
        rdMolDescriptors.CalcNumAliphaticRings,
        rdMolDescriptors.CalcNumSaturatedRings,
        rdMolDescriptors.CalcNumHeterocycles,
        rdMolDescriptors.CalcNumAromaticHeterocycles,
        rdMolDescriptors.CalcNumAromaticCarbocycles,
        rdMolDescriptors.CalcNumAliphaticHeterocycles,
        rdMolDescriptors.CalcNumAliphaticCarbocycles,
        rdMolDescriptors.CalcNumSpiroAtoms,
        rdMolDescriptors.CalcNumBridgeheadAtoms,
        rdMolDescriptors.CalcNumAmideBonds,
        rdMolDescriptors.CalcNumRotatableBonds,
        rdMolDescriptors.CalcNumHeteroatoms,
        rdMolDescriptors.CalcTPSA,
        rdMolDescriptors.CalcFractionCSP3
    ]
    
    desc = []
    for f in funcs:
        try:
            val = f(mol)
            desc.append(float(val))
        except Exception:
            desc.append(np.nan)
    return desc

if __name__ == '__main__':
    print("Loading data/u0_res_baseline.csv...")
    df = pd.read_csv('data/u0_res_baseline.csv')
    print(f"Loaded {len(df)} molecules.")
    
    print("Computing RDKit descriptors...")
    with mp.Pool(16) as pool:
        desc_list = pool.map(compute_descriptors, df['smiles'].tolist())
        
    desc_df = pd.DataFrame(desc_list, columns=DESC_NAMES)
    
    # Handle infinities and NaNs
    desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    nan_counts = desc_df.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"Found {total_nans} missing/infinite descriptor values. Applying median imputation...")
        medians = desc_df.median()
        desc_df.fillna(medians, inplace=True)
    else:
        print("No missing descriptor values found.")
        
    # Combine with original dataframe
    df = pd.concat([df, desc_df], axis=1)
    
    # Prepare data for XGBoost
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    
    X_train = df.loc[train_mask, DESC_NAMES].values
    y_train = df.loc[train_mask, 'u0_res'].values
    
    X_val = df.loc[val_mask, DESC_NAMES].values
    y_val = df.loc[val_mask, 'u0_res'].values
    
    X_test = df.loc[test_mask, DESC_NAMES].values
    y_test = df.loc[test_mask, 'u0_res'].values
    
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("Training XGBoost model...")
    try:
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=8,
            early_stopping_rounds=50
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError:
        # Fallback for older xgboost versions where early_stopping_rounds is in fit()
        model = xgb.XGBRegressor(
            n_estimators=1000,
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
            early_stopping_rounds=50,
            verbose=False
        )
        
    print(f"Best iteration: {model.best_iteration}")
    
    # Predict
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    
    # Evaluate
    def evaluate(y_true, y_pred):
        return {
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'R2': float(r2_score(y_true, y_pred))
        }
        
    metrics = {
        'train': evaluate(y_train, pred_train),
        'val': evaluate(y_val, pred_val),
        'test': evaluate(y_test, pred_test)
    }
    
    print("\n--- XGBoost Model Performance ---")
    print(json.dumps(metrics, indent=2))
    
    # Feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': DESC_NAMES,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 10 Feature Importances ---")
    print(fi_df.head(10).to_string(index=False))
    
    # Save predictions
    df['xgb_pred_u0_res'] = np.nan
    df.loc[train_mask, 'xgb_pred_u0_res'] = pred_train
    df.loc[val_mask, 'xgb_pred_u0_res'] = pred_val
    df.loc[test_mask, 'xgb_pred_u0_res'] = pred_test
    
    pred_df = df[['mol_id', 'smiles', 'split', 'u0_res', 'xgb_pred_u0_res']]
    pred_df.to_csv('data/xgb_predictions.csv', index=False)
    print(f"\nSaved data/xgb_predictions.csv columns: {list(pred_df.columns)}")
    
    with open('data/xgb_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved data/xgb_metrics.json")
    
    fi_df.to_csv('data/xgb_feature_importances.csv', index=False)
    print(f"Saved data/xgb_feature_importances.csv columns: {list(fi_df.columns)}")