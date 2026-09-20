import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

if __name__ == '__main__':
    print('Loading data...')
    df_res = pd.read_csv('data/standardized_residuals.csv')
    df_features = pd.read_csv('data/additive_features.csv')
    splits = np.load('data/split_indices.npz')
    
    train_idx = splits['train']
    val_idx = splits['val']
    test_idx = splits['test']
    
    # Use the same features as the Ridge baseline
    feature_cols = [c for c in df_features.columns if c != 'smiles']
    X = df_features[feature_cols].values
    y = df_res['std_residual'].values
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f'Sample sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    
    print('\nTraining Random Forest Regressor on standardized residuals...')
    # Limit n_jobs to 16 to avoid CPU oversubscription
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=16)
    rf.fit(X_train, y_train)
    
    print('\nEvaluating model...')
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)
    
    metrics = {}
    print('Random Forest Performance (Standardized Residuals):')
    for name, y_true, y_pred in [('Train', y_train, y_train_pred),
                                 ('Val', y_val, y_val_pred),
                                 ('Test', y_test, y_test_pred)]:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {'MSE': mse, 'R2': r2}
        print(f'  {name} - MSE: {mse:.4f}, R2: {r2:.4f}')
        
    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print('\nTop 10 important features for predicting non-additive residuals:')
    print(importances.nlargest(10).to_string())
        
    # Save model
    with open('data/rf_baseline.pkl', 'wb') as f:
        pickle.dump({'model': rf, 'feature_cols': feature_cols}, f)
    print('\nsaved data/rf_baseline.pkl')
    
    # Save predictions for the entire dataset
    df_preds = pd.DataFrame({
        'smiles': df_res['smiles'],
        'std_residual_true': y,
        'std_residual_pred': rf.predict(X),
        'split': df_res['split']
    })
    df_preds.to_csv('data/rf_baseline_predictions.csv', index=False)
    print('saved data/rf_baseline_predictions.csv columns:', list(df_preds.columns))
    
    # Save metrics
    with open('data/rf_baseline_metrics.txt', 'w') as f:
        for name, m in metrics.items():
            f.write(f"{name} - MSE: {m['MSE']:.4f}, R2: {m['R2']:.4f}\n")
    print('saved data/rf_baseline_metrics.txt')