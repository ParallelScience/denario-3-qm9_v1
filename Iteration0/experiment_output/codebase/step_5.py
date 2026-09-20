import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

def train_model(model, train_loader, val_loader, device, epochs=150, patience=15):
    """
    Trains the MLP with early stopping based on validation loss.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        if patience_counter >= patience:
            print(f"    Early stopping triggered at epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def main():
    plt.rcParams['text.usetex'] = False
    
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"CUDA memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
        
    # 2. Load data
    print("\nLoading datasets...")
    splits = np.load('data/split_indices.npz')
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    
    features = np.load('data/engineered_features.npz')
    fps = features['fps']
    descs = features['descs']
    
    # Concatenate and handle any potential NaNs
    X = np.hstack([fps, descs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    res_df = pd.read_csv('data/scaled_residuals.csv')
    
    # 3. Scale features
    print("Scaling features (fitting on training set only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    
    scaler_path = 'data/feature_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved feature scaler to {scaler_path}")
    
    targets = ['res_u0', 'res_gap', 'res_mu']
    metrics = {}
    predictions_df = pd.DataFrame({'smiles': res_df['smiles'].iloc[test_idx].values})
    
    batch_size = 1024
    
    # 4. Train and evaluate models
    for target in targets:
        print(f"\n--- Training PyTorch MLP for {target} ---")
        y = res_df[target].values.astype(np.float32)
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = MLP(input_dim=X.shape[1]).to(device)
        
        model = train_model(model, train_loader, val_loader, device, epochs=150, patience=15)
        
        # Evaluate on test set
        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                y_pred_test.append(preds.cpu().numpy())
                
        y_pred_test = np.concatenate(y_pred_test)
        
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print(f"  Final Test MSE: {mse:.4f}")
        print(f"  Final Test R2:  {r2:.4f}")
        
        metrics[target] = {'mse': mse, 'r2': r2}
        predictions_df[f'{target}_true'] = y_test
        predictions_df[f'{target}_pred'] = y_pred_test
        
        model_path = f'data/mlp_{target}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"  Saved model checkpoint to {model_path}")
        
    # 5. Save metrics
    metrics_path = 'data/topological_model_metrics.txt'
    with open(metrics_path, 'w') as f:
        for target, m in metrics.items():
            f.write(f"Target: {target}\n")
            f.write(f"  Test MSE: {m['mse']:.6f}\n")
            f.write(f"  Test R2:  {m['r2']:.6f}\n\n")
    print(f"\nSaved metrics to {metrics_path}")
    
    # 6. Save predictions
    preds_path = 'data/topological_model_predictions.csv'
    predictions_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to {preds_path}")
    print(f"Saved {preds_path} columns:", list(predictions_df.columns))
    
    # 7. Plot True vs Predicted
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, target in enumerate(targets):
        y_true = predictions_df[f'{target}_true']
        y_pred = predictions_df[f'{target}_pred']
        
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.1, s=5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax.set_xlabel(f'True {target} (scaled)')
        ax.set_ylabel(f'Predicted {target} (scaled)')
        ax.set_title(f'{target} (Test R2: {metrics[target]["r2"]:.3f})')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = 'data/topological_model_predictions.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    main()