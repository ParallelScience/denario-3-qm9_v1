import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df_qm9 = pd.read_csv('data/cleaned_qm9.csv')
    df_features = pd.read_csv('data/additive_features.csv')
    splits = np.load('data/split_indices.npz')
    
    train_idx = splits['train']
    val_idx = splits['val']
    test_idx = splits['test']
    
    print(f"Sample sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Prepare X and y
    feature_cols = [c for c in df_features.columns if c != 'smiles']
    X = df_features[feature_cols].values
    y = df_qm9['u0'].values
    smiles = df_qm9['smiles'].values
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Train Ridge regression
    print("\nTraining Ridge regression baseline (Group Contribution Method)...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate residuals
    res_train = y_train - y_train_pred
    res_val = y_val - y_val_pred
    res_test = y_test - y_test_pred
    
    # Standardize residuals based on train set
    res_mean = np.mean(res_train)
    res_std = np.std(res_train)
    
    std_res_train = (res_train - res_mean) / res_std
    std_res_val = (res_val - res_mean) / res_std
    std_res_test = (res_test - res_mean) / res_std
    
    # Evaluate performance
    print("\nBaseline Performance (u0):")
    metrics = {}
    for name, y_true, y_pred in [('Train', y_train, y_train_pred), 
                                 ('Val', y_val, y_val_pred), 
                                 ('Test', y_test, y_test_pred)]:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {'MSE': mse, 'R2': r2}
        print(f"  {name} - MSE: {mse:.4f} Hartree^2, R2: {r2:.4f}")
        
    # Print top coefficients
    coefs = pd.Series(model.coef_, index=feature_cols)
    print("\nTop 5 positive group contributions (Hartree/count):")
    print(coefs.nlargest(5).to_string())
    print("\nTop 5 negative group contributions (Hartree/count):")
    print(coefs.nsmallest(5).to_string())
    print(f"Intercept: {model.intercept_:.4f} Hartree")
    
    # Save metrics
    with open('data/ridge_metrics.txt', 'w') as f:
        for name, m in metrics.items():
            f.write(f"{name} - MSE: {m['MSE']:.4f}, R2: {m['R2']:.4f}\n")
    print("\nsaved data/ridge_metrics.txt")
    
    # Save model
    with open('data/ridge_baseline.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'res_mean': res_mean, 
            'res_std': res_std, 
            'feature_cols': feature_cols
        }, f)
    print("saved data/ridge_baseline.pkl")
    
    # Create a dataframe for standardized residuals
    df_res = pd.DataFrame({
        'smiles': smiles,
        'u0': y,
        'u0_pred': model.predict(X),
        'residual': y - model.predict(X)
    })
    df_res['std_residual'] = (df_res['residual'] - res_mean) / res_std
    
    # Add split info
    split_col = np.array([''] * len(df_res), dtype=object)
    split_col[train_idx] = 'train'
    split_col[val_idx] = 'val'
    split_col[test_idx] = 'test'
    df_res['split'] = split_col
    
    df_res.to_csv('data/standardized_residuals.csv', index=False)
    print("saved data/standardized_residuals.csv columns:", list(df_res.columns))
    
    # Print some summary statistics of the residuals
    print("\nStandardized Residuals Summary (Train):")
    print(pd.Series(std_res_train).describe().to_string())