import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import r2_score

def main():
    # 1. Parity Plots (XGBoost vs GNN)
    print("Loading predictions...")
    gnn_df = pd.read_csv('data/gnn_predictions.csv')
    xgb_df = pd.read_csv('data/xgb_predictions.csv')
    
    # Identify prediction columns
    gnn_pred_col = [c for c in gnn_df.columns if 'pred' in c.lower()][0]
    xgb_pred_col = [c for c in xgb_df.columns if 'pred' in c.lower()][0]
    
    # Merge on mol_id
    merged_df = pd.merge(
        gnn_df[['mol_id', 'split', 'u0_res', gnn_pred_col]], 
        xgb_df[['mol_id', xgb_pred_col]], 
        on='mol_id'
    )
    test_df = merged_df[merged_df['split'] == 'test'].dropna(subset=['u0_res', gnn_pred_col, xgb_pred_col])
    
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # XGBoost Parity Plot
    axes[0].scatter(test_df['u0_res'], test_df[xgb_pred_col], alpha=0.3, s=10, color='blue')
    min_val = min(test_df['u0_res'].min(), test_df[xgb_pred_col].min())
    max_val = max(test_df['u0_res'].max(), test_df[xgb_pred_col].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0].set_xlabel(r'Actual $u_0$ Residual (Hartree)')
    axes[0].set_ylabel('XGBoost Predicted (Hartree)')
    axes[0].set_title('XGBoost Parity Plot (Test Set)')
    
    # GNN Parity Plot
    axes[1].scatter(test_df['u0_res'], test_df[gnn_pred_col], alpha=0.3, s=10, color='green')
    min_val = min(test_df['u0_res'].min(), test_df[gnn_pred_col].min())
    max_val = max(test_df['u0_res'].max(), test_df[gnn_pred_col].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1].set_xlabel(r'Actual $u_0$ Residual (Hartree)')
    axes[1].set_ylabel('GNN Predicted (Hartree)')
    axes[1].set_title('GNN Parity Plot (Test Set)')
    
    plt.tight_layout()
    plt.savefig('data/parity_plot.png', dpi=300)
    print("saved data/parity_plot.png")
    
    # 2. Integrated Gradients Summary
    print("Loading IG attributions...")
    ig_data = torch.load('data/ig_attributions.pt', map_location='cpu', weights_only=False)
    
    ig_x = ig_data['ig_x']
    ig_edge = ig_data['ig_edge']
    
    # Calculate mean absolute attributions across all nodes/edges
    all_x_attr = np.concatenate(ig_x, axis=0)
    
    valid_edges = [e for e in ig_edge if e.shape[0] > 0]
    if len(valid_edges) > 0:
        all_edge_attr = np.concatenate(valid_edges, axis=0)
    else:
        all_edge_attr = np.zeros((1, 2))
    
    mean_abs_node_attr = np.mean(np.abs(all_x_attr), axis=0)
    mean_abs_edge_attr = np.mean(np.abs(all_edge_attr), axis=0)
    
    node_feat_names = ['Atomic Num', 'Degree', 'Hybridization', 'IsAromatic', 'FormalCharge']
    edge_feat_names = ['BondType', 'IsConjugated']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(node_feat_names, mean_abs_node_attr, color='purple')
    axes[0].set_ylabel('Mean Absolute Attribution')
    axes[0].set_title('Node Feature Attributions')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(edge_feat_names, mean_abs_edge_attr, color='orange')
    axes[1].set_ylabel('Mean Absolute Attribution')
    axes[1].set_title('Edge Feature Attributions')
    
    plt.tight_layout()
    plt.savefig('data/ig_summary_plot.png', dpi=300)
    print("saved data/ig_summary_plot.png")
    
    # 3. Mutation Shifts Distribution
    print("Loading mutation shifts...")
    mut_df = pd.read_csv('data/mutation_shifts.csv')
    
    xgb_shifts = mut_df['xgb_shift']
    gnn_shifts = mut_df['gnn_shift']
    
    plt.figure(figsize=(8, 5))
    plt.hist(xgb_shifts, bins=50, alpha=0.5, label='XGBoost Shift', density=True)
    plt.hist(gnn_shifts, bins=50, alpha=0.5, label='GNN Shift', density=True)
    plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('Predicted Energy Shift (Hartree)')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Energy Shifts upon Breaking Conjugation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/mutation_shifts_dist.png', dpi=300)
    print("saved data/mutation_shifts_dist.png")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Test Set R2 (XGBoost): {r2_score(test_df['u0_res'], test_df[xgb_pred_col]):.4f}")
    print(f"Test Set R2 (GNN): {r2_score(test_df['u0_res'], test_df[gnn_pred_col]):.4f}")
    
    print("\nMean Absolute Node Attributions:")
    for name, val in zip(node_feat_names, mean_abs_node_attr):
        print(f"  {name}: {val:.4f}")
        
    print("\nMean Absolute Edge Attributions:")
    for name, val in zip(edge_feat_names, mean_abs_edge_attr):
        print(f"  {name}: {val:.4f}")
        
    print("\nMutation Shifts Summary:")
    print(f"  XGBoost Shift Mean: {xgb_shifts.mean():.4f} ± {xgb_shifts.std():.4f}")
    print(f"  GNN Shift Mean: {gnn_shifts.mean():.4f} ± {gnn_shifts.std():.4f}")

if __name__ == '__main__':
    main()