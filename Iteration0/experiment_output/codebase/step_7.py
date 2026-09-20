import numpy as np
import pandas as pd
import scipy.stats as stats

def main():
    # Set pandas display options for cleaner output
    pd.options.display.float_format = '{:.4f}'.format
    
    print("Loading datasets...")
    preds_df = pd.read_csv('data/topological_model_predictions.csv')
    qm9_df = pd.read_csv('data/cleaned_qm9.csv')
    shap_df = pd.read_csv('data/shap_motif_summary.csv')
    
    print("Merging predictions with QM9 intensive properties...")
    # Merge on SMILES to align predictions with the true intensive properties
    merged_df = pd.merge(preds_df, qm9_df[['smiles', 'gap', 'mu', 'homo', 'lumo']], on='smiles', how='inner')
    
    print("\n--- Computing Correlations ---")
    intensive_props = ['gap', 'mu', 'homo', 'lumo']
    res_cols = ['res_u0_pred', 'res_gap_pred', 'res_mu_pred']
    
    corr_records = []
    for res_col in res_cols:
        if res_col not in merged_df.columns:
            continue
        for prop in intensive_props:
            pearson_r, pearson_p = stats.pearsonr(merged_df[res_col], merged_df[prop])
            spearman_r, spearman_p = stats.spearmanr(merged_df[res_col], merged_df[prop])
            corr_records.append({
                'Residual': res_col,
                'Property': prop,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p
            })
            
    corr_df = pd.DataFrame(corr_records)
    corr_path = 'data/motif_correlations.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved {corr_path}")
    print(corr_df.to_string(index=False))
    
    print("\n--- Motif Analysis ---")
    print("Loading engineered features to extract motif presence...")
    features = np.load('data/engineered_features.npz')
    fps = features['fps']
    descs = features['descs']
    
    # Decode descriptor names if they are stored as bytes
    desc_names = features['desc_names'].tolist()
    if isinstance(desc_names[0], bytes):
        desc_names = [d.decode('utf-8') for d in desc_names]
    elif not isinstance(desc_names[0], str):
        desc_names = [str(d) for d in desc_names]
        
    splits = np.load('data/split_indices.npz')
    test_idx = splits['test_idx']
    
    fp_names = [f"MorganFP_{i}" for i in range(2048)]
    all_feature_names = fp_names + desc_names
    
    # Filter for motif-like features (fragments, counts, fingerprints) to ensure interpretability
    motif_keywords = ['fr_', 'Num', 'Ring', 'MorganFP_']
    motif_features = set([f for f in all_feature_names if any(k in f for k in motif_keywords)])
    
    # Get top 15 motif features from SHAP summary (averaged across all targets)
    shap_motif_df = shap_df[shap_df['Feature'].isin(motif_features)]
    avg_shap = shap_motif_df.groupby('Feature')['MeanAbsSHAP'].mean().reset_index()
    top_features = avg_shap.sort_values('MeanAbsSHAP', ascending=False).head(15)['Feature'].tolist()
    
    print(f"Top 15 identified structural motifs: {top_features}")
    
    top_feature_indices = [all_feature_names.index(f) for f in top_features]
    
    # Reconstruct the feature matrix for the test set
    X_test_fps = fps[test_idx]
    X_test_descs = descs[test_idx]
    X_test = np.hstack([X_test_fps, X_test_descs])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create a boolean dataframe indicating the presence (>0) of each top motif
    motif_df = pd.DataFrame({'smiles': preds_df['smiles'].values})
    for f, idx in zip(top_features, top_feature_indices):
        motif_df[f] = X_test[:, idx] > 0
        
    motif_merged = pd.merge(motif_df, merged_df[['smiles', 'gap', 'mu']], on='smiles', how='inner')
    
    stat_records = []
    for f in top_features:
        for prop in ['gap', 'mu']:
            present = motif_merged[motif_merged[f] == True][prop]
            absent = motif_merged[motif_merged[f] == False][prop]
            
            stat_records.append({
                'Motif': f,
                'Property': prop,
                'Present_Count': len(present),
                'Present_Mean': present.mean() if len(present) > 0 else np.nan,
                'Present_Var': present.var() if len(present) > 1 else np.nan,
                'Present_Std': present.std() if len(present) > 1 else np.nan,
                'Absent_Count': len(absent),
                'Absent_Mean': absent.mean() if len(absent) > 0 else np.nan,
                'Absent_Var': absent.var() if len(absent) > 1 else np.nan,
                'Absent_Std': absent.std() if len(absent) > 1 else np.nan,
            })
            
    stats_df = pd.DataFrame(stat_records)
    stats_path = 'data/motif_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved {stats_path}")
    print(stats_df.to_string(index=False))

if __name__ == '__main__':
    main()