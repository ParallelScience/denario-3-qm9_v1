import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    plt.rcParams['text.usetex'] = False
    
    # Load features
    features_data = np.load('data/additive_features.npz')
    X = features_data['X']
    feature_names = features_data['features']
    print(f"Loaded features X shape: {X.shape}")
    print(f"Feature names: {list(feature_names)}")
    
    # Load residuals
    residuals_df = pd.read_csv('data/residuals.csv')
    print(f"Loaded residuals shape: {residuals_df.shape}")
    
    targets = ['res_u0', 'res_gap', 'res_mu']
    
    # We will use a standard train/test split (80/20)
    indices = np.arange(len(residuals_df))
    X_train, X_test, idx_train, idx_test = train_test_split(X, indices, test_size=0.2, random_state=42)
    
    print(f"Train set size: {len(idx_train)}")
    print(f"Test set size: {len(idx_test)}")
    
    metrics = {}
    predictions_df = pd.DataFrame({'smiles': residuals_df['smiles'].iloc[idx_test].values})
    
    for target in targets:
        print(f"\n--- Training Random Forest for {target} ---")
        y = residuals_df[target].values
        y_train, y_test = y[idx_train], y[idx_test]
        
        # Initialize and train model
        rf = RandomForestRegressor(n_estimators=100, n_jobs=8, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        # Evaluate
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"Results for {target}:")
        print(f"  Train MSE: {mse_train:.6f} | Train R2: {r2_train:.4f}")
        print(f"  Test MSE:  {mse_test:.6f} | Test R2:  {r2_test:.4f}")
        
        # Feature importances
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        print("  Feature Importances:")
        print(imp_df.to_string(index=False))
        
        metrics[target] = {
            'train_mse': mse_train,
            'train_r2': r2_train,
            'test_mse': mse_test,
            'test_r2': r2_test
        }
        
        # Save predictions
        predictions_df[f'{target}_true'] = y_test
        predictions_df[f'{target}_pred'] = y_pred_test
        
    # Save metrics
    metrics_path = 'data/baseline_metrics.txt'
    with open(metrics_path, 'w') as f:
        for target, m in metrics.items():
            f.write(f"Target: {target}\n")
            f.write(f"  Train MSE: {m['train_mse']:.6f}, R2: {m['train_r2']:.4f}\n")
            f.write(f"  Test MSE:  {m['test_mse']:.6f}, R2: {m['test_r2']:.4f}\n\n")
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save predictions
    preds_path = 'data/rf_baseline_predictions.csv'
    predictions_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to {preds_path}")
    print(f"Saved {preds_path} columns:", list(predictions_df.columns))
    
    # Plot true vs predicted for test set
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, target in enumerate(targets):
        y_true = predictions_df[f'{target}_true']
        y_pred = predictions_df[f'{target}_pred']
        
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.1, s=5)
        
        # Plot y=x line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax.set_xlabel(f'True {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{target} (Test R2: {metrics[target]["test_r2"]:.3f})')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = 'data/rf_baseline_predictions.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    main()