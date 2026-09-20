import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

if __name__ == '__main__':
    print("Loading latent embeddings and properties...")
    df = pd.read_csv('data/gnn_test_latent_embeddings.csv')
    latents_data = np.load('data/gnn_test_latents.npz')
    latents = latents_data['latents']
    
    properties = ['gap', 'mu', 'homo', 'lumo', 'residual']
    
    # Ensure no NaNs in properties
    valid_idx = df[properties].notna().all(axis=1)
    if not valid_idx.all():
        print(f"Dropping {(~valid_idx).sum()} rows with NaN properties.")
        df = df[valid_idx].reset_index(drop=True)
        latents = latents[valid_idx]
        
    print(f"Analyzing {len(df)} molecules...")
    
    # 1. Correlate 2D t-SNE projections with properties
    tsne_cols = ['tsne_1', 'tsne_2']
    tsne_corr = []
    for prop in properties:
        for tsne_col in tsne_cols:
            r, p = pearsonr(df[tsne_col], df[prop])
            rho, p_rho = spearmanr(df[tsne_col], df[prop])
            tsne_corr.append({
                'Property': prop,
                'Dimension': tsne_col,
                'Pearson_r': r,
                'Spearman_rho': rho
            })
    
    tsne_corr_df = pd.DataFrame(tsne_corr)
    tsne_corr_df.to_csv('data/tsne_property_correlations.csv', index=False)
    print("saved data/tsne_property_correlations.csv columns:", list(tsne_corr_df.columns))
    print("\nt-SNE vs Properties Correlations:")
    print(tsne_corr_df.to_string(index=False, float_format="%.4f"))
    
    # 2. Correlate 128D latents with properties
    latent_corrs = {prop: [] for prop in properties}
    for i in range(latents.shape[1]):
        dim_vals = latents[:, i]
        for prop in properties:
            r, _ = pearsonr(dim_vals, df[prop])
            latent_corrs[prop].append(r)
            
    latent_corr_df = pd.DataFrame(latent_corrs)
    latent_corr_df.index = [f'latent_{i}' for i in range(latents.shape[1])]
    latent_corr_df.to_csv('data/latent_128d_property_correlations.csv')
    print("\nsaved data/latent_128d_property_correlations.csv columns:", list(latent_corr_df.columns))
    
    print("\nTop 3 correlated latent dimensions (absolute Pearson r) for each property:")
    for prop in properties:
        top_dims = latent_corr_df[prop].abs().nlargest(3)
        print(f"  {prop}:")
        for dim, _ in top_dims.items():
            actual_r = latent_corr_df.loc[dim, prop]
            print(f"    {dim}: {actual_r:.4f}")
            
    # 3. Generate scatter plots of t-SNE colored by properties
    print("\nGenerating scatter plots...")
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        cmap = 'coolwarm' if prop == 'residual' else 'viridis'
        
        # Use percentiles for robust color limits to avoid outlier washout
        vmin = df[prop].quantile(0.01)
        vmax = df[prop].quantile(0.99)
        
        sc = ax.scatter(df['tsne_1'], df['tsne_2'], c=df[prop], cmap=cmap, 
                        s=2, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_title(f't-SNE colored by {prop}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(sc, ax=ax, label=prop)
        
    # Hide the empty subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/tsne_property_scatter.png', dpi=300)
    print("saved data/tsne_property_scatter.png")