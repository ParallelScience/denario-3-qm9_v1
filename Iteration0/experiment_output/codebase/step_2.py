import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

def train_and_evaluate_ridge(X: np.ndarray, y: np.ndarray, feature_names: list, target_name: str) -> tuple:
    """
    Trains a Ridge regression model and evaluates its performance.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: List of feature names.
        target_name: Name of the target variable.
        
    Returns:
        A tuple containing the trained model, predictions, and residuals.
    """
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    res = y - preds
    
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    print(f"\nTarget: {target_name}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print("  Coefficients:")
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
    print(coef_df.to_string(index=False))
    print(f"  Intercept: {model.intercept_:.4f}")
    
    return model, preds, res

def main():
    """
    Main execution function for baseline additive modeling.
    """
    plt.rcParams['text.usetex'] = False
    np.random.seed(42)
    
    # Load the cleaned dataset
    df = pd.read_csv('data/cleaned_qm9.csv')
    print(f"Loaded data shape: {df.shape}")
    print(f"Loaded columns: {list(df.columns)}")
    
    # Define additive features
    feature_cols = [
        'c_count', 'n_count', 'o_count', 'f_count', 'h_count',
        'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds'
    ]
    X = df[feature_cols].values
    
    targets = ['u0', 'gap', 'mu']
    residuals = {}
    
    print("\n--- Ridge Regression Results ---")
    for target in targets:
        y = df[target].values
        _, _, res = train_and_evaluate_ridge(X, y, feature_cols, target)
        residuals[target] = res
        
    # Normalize u0 residuals by heavy atom count
    residuals['u0_norm'] = residuals['u0'] / df['heavy_atom_count'].values
    
    # Check correlation to verify size independence
    corr_u0 = np.corrcoef(df['heavy_atom_count'], residuals['u0'])[0, 1]
    corr_abs_u0 = np.corrcoef(df['heavy_atom_count'], np.abs(residuals['u0']))[0, 1]
    print(f"\nCorrelation between heavy_atom_count and res_u0: {corr_u0:.4f}")
    print(f"Correlation between heavy_atom_count and abs(res_u0): {corr_abs_u0:.4f}")
    if abs(corr_u0) < 0.1 and abs(corr_abs_u0) < 0.1:
        print("No significant scaling for u0 residuals. Normalization is computed but may not be strictly necessary.")
    else:
        print("Significant scaling detected for u0 residuals. Normalization is recommended.")
    
    # Plot residuals vs heavy atom count
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    plot_targets = [
        ('u0', residuals['u0']), 
        ('u0_norm', residuals['u0_norm']), 
        ('gap', residuals['gap']), 
        ('mu', residuals['mu'])
    ]
    
    for i, (name, res_vals) in enumerate(plot_targets):
        ax = axes[i]
        ax.scatter(df['heavy_atom_count'], res_vals, alpha=0.1, s=5)
        
        # Set y-limits based on 1st and 99th percentiles to exclude extreme outliers
        p01 = np.percentile(res_vals, 1)
        p99 = np.percentile(res_vals, 99)
        margin = (p99 - p01) * 0.5
        if margin == 0:
            margin = 1.0
            
        ax.set_ylim(p01 - margin, p99 + margin)
        
        ax.set_xlabel('Heavy Atom Count')
        ax.set_ylabel(f'Residuals ({name})')
        ax.set_title(f'Residuals of {name} vs Heavy Atom Count')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = 'data/residuals_vs_size.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"saved {plot_path}")
    
    # Save residuals
    res_df = pd.DataFrame({
        'smiles': df['smiles'],
        'res_u0': residuals['u0'],
        'res_u0_norm': residuals['u0_norm'],
        'res_gap': residuals['gap'],
        'res_mu': residuals['mu']
    })
    res_path = 'data/residuals.csv'
    res_df.to_csv(res_path, index=False)
    print(f"saved {res_path}")
    print(f"saved {res_path} columns:", list(res_df.columns))
    
    # Save additive features
    feat_path = 'data/additive_features.npz'
    np.savez(feat_path, X=X, features=feature_cols)
    print(f"saved {feat_path} keys:", list(np.load(feat_path).files))

if __name__ == '__main__':
    main()