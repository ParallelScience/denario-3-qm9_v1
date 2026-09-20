import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Loads precomputed motif statistics and correlations, and generates visualizations
    to illustrate the relationship between non-additive energy residuals and intensive
    electronic properties (gap, mu). Creates scatter plots for correlations and 
    bar/box plots for property distributions across high-influence motif groups.
    """
    plt.rcParams['text.usetex'] = False
    
    print("Loading precomputed statistics from Step 7...")
    corr_df = pd.read_csv('data/motif_correlations.csv')
    print("\nPrecomputed Correlations:")
    print(corr_df.to_string(index=False))
    
    stats_df = pd.read_csv('data/motif_statistics.csv')
    # Extract the top 5 unique motifs based on their order in the statistics file
    top_features = stats_df['Motif'].unique()[:5].tolist()
    print(f"\nTop 5 motifs selected for visualization: {top_features}")
    
    print("\nLoading predictions and QM9 dataset for plotting...")
    preds_df = pd.read_csv('data/topological_model_predictions.csv')
    qm9_df = pd.read_csv('data/cleaned_qm9.csv')
    
    print("Merging datasets on SMILES...")
    merged_df = pd.merge(preds_df, qm9_df[['smiles', 'gap', 'mu']], on='smiles', how='inner')
    
    print("\nGenerating scatter plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hexbin plot for res_u0_pred vs gap
    hb1 = axes[0].hexbin(merged_df['gap'], merged_df['res_u0_pred'], gridsize=50, cmap='Blues', mincnt=1)
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label('Count')
    axes[0].set_xlabel('HOMO-LUMO Gap (Hartree)')
    axes[0].set_ylabel('Predicted Non-Additive u0 Residual')
    axes[0].set_title('Energy Residual vs Gap')
    
    # Hexbin plot for res_u0_pred vs mu
    hb2 = axes[1].hexbin(merged_df['mu'], merged_df['res_u0_pred'], gridsize=50, cmap='Blues', mincnt=1)
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label('Count')
    axes[1].set_xlabel('Dipole Moment (Debye)')
    axes[1].set_ylabel('Predicted Non-Additive u0 Residual')
    axes[1].set_title('Energy Residual vs Dipole Moment')
    
    plt.tight_layout()
    scatter_path = 'data/residual_gap_corr.png'
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved {scatter_path}")
    
    print("\n--- Motif Distribution Analysis ---")
    print("Loading engineered features to reconstruct test set motif presence...")
    features = np.load('data/engineered_features.npz')
    fps = features['fps']
    descs = features['descs']
    
    # Decode descriptor names safely
    desc_names = features['desc_names'].tolist()
    if isinstance(desc_names[0], bytes):
        desc_names = [d.decode('utf-8') for d in desc_names]
    elif not isinstance(desc_names[0], str):
        desc_names = [str(d) for d in desc_names]
        
    fp_names = [f"MorganFP_{i}" for i in range(2048)]
    all_feature_names = fp_names + desc_names
    
    splits = np.load('data/split_indices.npz')
    test_idx = splits['test_idx']
    
    # Reconstruct the feature matrix for the test set
    X_test_fps = fps[test_idx]
    X_test_descs = descs[test_idx]
    X_test = np.hstack([X_test_fps, X_test_descs])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    top_feature_indices = [all_feature_names.index(f) for f in top_features]
    
    # Create a boolean dataframe indicating the presence (>0) of each top motif
    motif_df = pd.DataFrame({'smiles': preds_df['smiles'].values})
    for f, idx in zip(top_features, top_feature_indices):
        motif_df[f] = X_test[:, idx] > 0
        
    motif_merged = pd.merge(motif_df, merged_df[['smiles', 'gap', 'mu']], on='smiles', how='inner')
    
    print("\nGenerating bar and box plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    positions = np.arange(len(top_features))
    width = 0.35
    
    # Extract precomputed statistics for the top 5 motifs
    stats_gap = stats_df[stats_df['Property'] == 'gap'].set_index('Motif').loc[top_features]
    stats_mu = stats_df[stats_df['Property'] == 'mu'].set_index('Motif').loc[top_features]
    
    # Bar plot for gap
    ax = axes[0, 0]
    ax.bar(positions - width/2, stats_gap['Present_Mean'].values, width, yerr=stats_gap['Present_Std'].values, label='Present', color='lightblue', capsize=5)
    ax.bar(positions + width/2, stats_gap['Absent_Mean'].values, width, yerr=stats_gap['Absent_Std'].values, label='Absent', color='lightcoral', capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_ylabel('Mean HOMO-LUMO Gap (Hartree)')
    ax.set_title('Mean Gap by Motif Presence')
    ax.legend()
    
    # Bar plot for mu
    ax = axes[0, 1]
    ax.bar(positions - width/2, stats_mu['Present_Mean'].values, width, yerr=stats_mu['Present_Std'].values, label='Present', color='lightblue', capsize=5)
    ax.bar(positions + width/2, stats_mu['Absent_Mean'].values, width, yerr=stats_mu['Absent_Std'].values, label='Absent', color='lightcoral', capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_ylabel('Mean Dipole Moment (Debye)')
    ax.set_title('Mean Dipole Moment by Motif Presence')
    ax.legend()
    
    # Box plot for gap
    ax = axes[1, 0]
    bp_present = ax.boxplot([motif_merged[motif_merged[f] == True]['gap'].values for f in top_features],
                            positions=positions - width/2, widths=width, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'), showfliers=False)
    bp_absent = ax.boxplot([motif_merged[motif_merged[f] == False]['gap'].values for f in top_features],
                           positions=positions + width/2, widths=width, patch_artist=True,
                           boxprops=dict(facecolor='lightcoral'), showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_ylabel('HOMO-LUMO Gap (Hartree)')
    ax.set_title('Gap Distribution by Motif Presence')
    ax.legend([bp_present["boxes"][0], bp_absent["boxes"][0]], ['Present', 'Absent'])
    
    # Box plot for mu
    ax = axes[1, 1]
    bp_present_mu = ax.boxplot([motif_merged[motif_merged[f] == True]['mu'].values for f in top_features],
                               positions=positions - width/2, widths=width, patch_artist=True,
                               boxprops=dict(facecolor='lightblue'), showfliers=False)
    bp_absent_mu = ax.boxplot([motif_merged[motif_merged[f] == False]['mu'].values for f in top_features],
                              positions=positions + width/2, widths=width, patch_artist=True,
                              boxprops=dict(facecolor='lightcoral'), showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_ylabel('Dipole Moment (Debye)')
    ax.set_title('Dipole Moment Distribution by Motif Presence')
    ax.legend([bp_present_mu["boxes"][0], bp_absent_mu["boxes"][0]], ['Present', 'Absent'])
    
    plt.tight_layout()
    dist_path = 'data/motif_distributions.png'
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved {dist_path}")

if __name__ == '__main__':
    main()