import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron for predicting molecular properties from 
    concatenated Morgan fingerprints and 2D descriptors.
    """
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

class WrappedModel(nn.Module):
    """
    Wraps the MLP to return a 2D tensor (batch_size, 1) instead of a 1D tensor.
    This prevents shape mismatch issues in SHAP explainers.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model.net(x)

def main():
    print("Loading features and splits...")
    features = np.load('data/engineered_features.npz')
    fps = features['fps']
    descs = features['descs']
    desc_names = features['desc_names']
    
    fp_names = [f"Morgan_bit_{i}" for i in range(fps.shape[1])]
    feature_names = np.array(fp_names + list(desc_names))
    
    X = np.hstack([fps, descs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    splits = np.load('data/split_indices.npz')
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    
    print("Loading scaler...")
    with open('data/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    X_train = scaler.transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Background dataset for SHAP (using 200 random training samples)
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), 200, replace=False)
    background = torch.tensor(X_train[bg_idx]).to(device)
    
    test_tensor = torch.tensor(X_test).to(device)
    
    targets = ['res_u0', 'res_gap', 'res_mu']
    shap_results = {}
    summary_dfs = []
    
    for target in targets:
        print(f"\nComputing SHAP values for {target}...")
        model = MLP(input_dim=X.shape[1]).to(device)
        model_path = f'data/mlp_{target}.pt'
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found, skipping.")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        wrapped_model = WrappedModel(model).to(device)
        wrapped_model.eval()
        
        # DeepExplainer is optimized for PyTorch NNs, but can occasionally fail on specific ops.
        # We wrap it in a try-except and fallback to GradientExplainer if needed.
        try:
            explainer = shap.DeepExplainer(wrapped_model, background)
            # Test a small batch first to catch initialization errors
            _ = explainer.shap_values(test_tensor[:2])
        except Exception as e:
            print(f"DeepExplainer failed with {e}, falling back to GradientExplainer...")
            explainer = shap.GradientExplainer(wrapped_model, background)
        
        batch_size = 500
        shap_values_list = []
        
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            sv = explainer.shap_values(batch)
            
            # Handle different return types across SHAP versions
            if isinstance(sv, list):
                sv = sv[0]
            if torch.is_tensor(sv):
                sv = sv.cpu().numpy()
                
            shap_values_list.append(sv)
            
            end_idx = min(i + batch_size, len(test_tensor))
            if end_idx % 2000 == 0 or end_idx == len(test_tensor):
                print(f"  Processed {end_idx}/{len(test_tensor)} samples")
            
        shap_values = np.vstack(shap_values_list)
        shap_results[target] = shap_values
        
        # Calculate mean absolute SHAP values across the test set
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        df_summary = pd.DataFrame({
            'Target': target,
            'Feature': feature_names,
            'MeanAbsSHAP': mean_abs_shap
        })
        df_summary = df_summary.sort_values('MeanAbsSHAP', ascending=False)
        
        print(f"Top 10 features for {target}:")
        print(df_summary.head(10).to_string(index=False))
        
        summary_dfs.append(df_summary)
        
    if shap_results:
        # Save SHAP values matrix
        shap_values_path = 'data/shap_values.npz'
        np.savez_compressed(shap_values_path, **shap_results)
        print(f"\nSaved SHAP values to {shap_values_path}")
        print(f"saved {shap_values_path} keys:", list(np.load(shap_values_path).files))
        
        # Save summary table
        summary_df = pd.concat(summary_dfs, ignore_index=True)
        summary_path = 'data/shap_motif_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved SHAP summary to {summary_path}")
        print(f"saved {summary_path} columns:", list(summary_df.columns))
        
        # Print overall top features across all targets
        print("\nOverall Top 15 Features across all targets (by average MeanAbsSHAP):")
        avg_shap = summary_df.groupby('Feature')['MeanAbsSHAP'].mean().reset_index()
        avg_shap = avg_shap.sort_values('MeanAbsSHAP', ascending=False)
        print(avg_shap.head(15).to_string(index=False))

if __name__ == '__main__':
    main()