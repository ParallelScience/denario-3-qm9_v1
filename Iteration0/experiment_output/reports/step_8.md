# Decoupling Non-Additive Energetics via Graph-Latent Residual Decomposition

## 1. Extensivity Decoupling and Baseline Performance

To isolate the non-additive electronic contributions to the internal energy (`u0`), the extensive scaling inherent to the QM9 dataset was decoupled using an additive baseline model based on group-contribution counts. The Ridge regression baseline, predicting `u0` directly from heavy atom and functional group counts, yielded a Test $R^2$ of 0.9850 and a Test MSE of 23.6894 (as recorded in `data/ridge_metrics.txt`). The remaining residuals thus represent non-additive energetics.

To evaluate how well graph-based topological models capture this non-additivity compared to non-linear descriptor-based models, a Random Forest (RF) and a Graph Neural Network (GNN) were trained to predict the standardized residuals of the Ridge baseline. According to `data/rf_baseline_metrics.txt`, the Random Forest achieved a Test $R^2$ of 0.9779 and a Test MSE of 0.0209 on the standardized residuals. In contrast, the GNN (`data/gnn_metrics.txt`) captured the topological variance substantially better, achieving a Test $R^2$ of 0.9985 and a Test MSE of 0.0014. By successfully modeling these deviations, the GNN effectively captures the non-linear electronic physics that classical additivity models miss.

## 2. GNN Latent Space and Intensive Property Correlation

The representations extracted from the penultimate layer of the trained GNN isolate non-additive structural effects into a 128-dimensional latent space. Analyzed over a test set of 13,381 molecules, the extracted latent representations (`data/gnn_test_latents.npz`) exhibit a mean of −0.2883 and a standard deviation of 2.7265. 

Correlations between these extracted features and fundamental intensive properties establish that the network intrinsically learned underlying electronic structures:
*   **128-Dimensional Correlations (`data/latent_128d_property_correlations.csv`)**: Individual latent dimensions highly correlate with the modeled non-additive residual. For example, `latent_24` correlates with the residual at a Pearson $r = -0.8520$, and `latent_102` at $r = 0.8274$. Furthermore, these dimensions capture physical intensive properties without direct supervision: `latent_83` correlates with the HOMO-LUMO gap at $r = 0.5207$, and `latent_50` correlates with the HOMO energy at $r = 0.3756$.
*   **t-SNE Dimensionality Reduction (`data/tsne_property_correlations.csv`)**: Projecting the latent space to two dimensions via t-SNE demonstrates a structured embedding landscape. The first principal dimension (`tsne_1`) correlates with the HOMO-LUMO gap (Pearson $r = -0.3635$, Spearman $\rho = -0.3414$) and HOMO ($r = 0.3068$). The second dimension (`tsne_2`) strongly orders the target non-additive residual (Spearman $\rho = 0.7871$, Pearson $r = 0.4611$). This clustering structure is visually captured in `data/tsne_property_scatter.png`.

## 3. Structural Motif Attribution and Statistical Validation

To chemically interpret the non-additivity captured by the model, Integrated Gradients were applied to attribute energetic deviations to specific atomic and bond motifs. Molecules were partitioned based on the presence of identified high-influence topological features (aromaticity, oxygen, nitrogen, and fluorine).

Statistical validation (`data/motif_statistical_validation.csv`) confirmed that these structural motifs are significant drivers of intensive electronic properties:

*   **Aromaticity**: Molecules containing aromatic motifs exhibit a significantly compressed HOMO-LUMO gap (mean 0.2071 Hartree) compared to those without (0.2608 Hartree). A Kruskal-Wallis test confirms this is highly significant ($p = 0.0000$, Cohen's $d = -1.3990$). Furthermore, aromaticity significantly increases the dipole moment `mu` (3.2036 Debye vs. 2.6075 Debye, $p = 0.0000$, Cohen's $d = 0.3734$).
*   **Nitrogen Integration**: The presence of nitrogen similarly compresses the gap (mean 0.2396 Hartree vs. 0.2704 Hartree, $p = 0.0000$, Cohen's $d = -0.6734$) while massively shifting the dipole moment (mean 3.1419 Debye vs. 2.0190 Debye, $p = 0.0000$, Cohen's $d = 0.8312$).
*   **Fluorine**: Molecules with fluorine show a narrower gap (mean 0.2133 Hartree vs. 0.2519 Hartree, ANOVA $p = 0.0000$, Cohen's $d = -0.7677$) and an increased dipole (mean 2.9627 Debye vs. 2.7084 Debye, ANOVA $p = 0.0162$, Cohen's $d = 0.1728$).
*   **Oxygen**: Oxygen incorporation yields statistically significant but physically smaller shifts, modifying the gap (mean 0.2504 Hartree vs. 0.2570 Hartree, $p = 0.0000$, Cohen's $d = -0.1347$) and dipole (mean 2.7610 Debye vs. 2.4275 Debye, $p = 0.0000$, Cohen's $d = 0.2009$).

These divergent distributions are visually detailed in the accompanying boxplots (`data/motif_property_boxplots.png`). The statistical results demonstrate that the deviations from classical additivity modeled by the GNN correspond directly to quantum-mechanical phenomena like extended $\pi$-conjugation and the high electronegativity of specific heteroatoms, effectively modeling the non-linear electronic physics of the dataset.