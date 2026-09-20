import sys
import subprocess
import os

# Install shap locally to avoid ModuleNotFoundError without polluting the global environment
override_dir = os.path.abspath("./.pip_overrides")
os.makedirs(override_dir, exist_ok=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--target", override_dir, "--quiet",
    "shap"
])
sys.path.insert(0, override_dir)

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
        return self.model(x).view(-1, 1)

def main():
    print("Loading features and splits...")
    features = np.load('data/engineered_features.npz')
    print("loaded keys:", features.files)
    
    fps = features['fps']
    descs = features['descs']
    X = np.hstack([fps, descs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    splits = np.load('data/split_indices.npz')
    print("loaded keys:", splits.files)
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    
    print("Loading scaler...")
    with open('data/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    X_train = scaler.transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    
    # Feature names
    fp_names = [f"MorganFP_{i}" for i in range(2048)]
    desc_names = features['desc_names'].tolist()
    # Ensure desc_names are strings
    if isinstance(desc_names[0], bytes):
        desc_names = [d.decode('utf-8') for d in desc_names]
    elif not isinstance(desc_names[0], str):
        desc_names = [str(d) for d in desc_names]
        
    feature_names = np.array(fp_names + desc_names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    targets = ['res_u0', 'res_gap', 'res_mu']
    
    # We will use a background dataset of 1000 samples from the training set
    np.random.seed(42)
    bg_indices = np.random.choice(len(X_train), 1000, replace=False)
    X_bg = X_train[bg_indices]
    X_bg_tensor = torch.tensor(X_bg).to(device)
    
    # We will compute SHAP values for a subset of the test set to save time (e.g., 500 samples)
    test_sample_indices = np.random.choice(len(X_test), 500, replace=False)
    X_test_sample = X_test[test_sample_indices]
    X_test_sample_tensor = torch.tensor(X_test_sample).to(device)
    
    all_shap_values = {}
    summary_dfs = []
    
    for target in targets:
        print(f"\n--- Computing SHAP values for {target} ---")
        model_path = f'data/mlp_{target}.pt'
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping.")
            continue
            
        model = MLP(input_dim=X.shape[1]).to(device)
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        wrapped_model = WrappedModel(model)
        
        # Initialize DeepExplainer
        explainer = shap.DeepExplainer(wrapped_model, X_bg_tensor)
        
        print(f"Calculating SHAP values for {len(X_test_sample)} test samples...")
        # DeepExplainer returns a list of arrays for each output. Since we have 1 output, it's a list of length 1.
        shap_values = explainer.shap_values(X_test_sample_tensor)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Ensure shap_values is 2D: (num_samples, num_features)
        if shap_values.ndim > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
        all_shap_values[target] = shap_values
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_abs_shap is strictly 1D
        mean_abs_shap = np.ravel(mean_abs_shap)
        
        df = pd.DataFrame({
            'Target': target,
            'Feature': feature_names,
            'MeanAbsSHAP': mean_abs_shap
        })
        
        # Sort by importance
        df = df.sort_values('MeanAbsSHAP', ascending=False)
        summary_dfs.append(df)
        
        print(f"Top 10 features for {target}:")
        print(df.head(10).to_string(index=False))
        
    if all_shap_values:
        # Save SHAP values
        shap_path = 'data/shap_values.npz'
        np.savez_compressed(shap_path, **all_shap_values)
        print(f"\nSaved SHAP values to {shap_path}")
        print(f"saved {shap_path} keys:", list(np.load(shap_path).files))
        
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