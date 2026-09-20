# Decoupling Non-Additive Energetics via Graph-Latent Residual Decomposition

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-09-20
**Best iteration:** 0

**[View Paper & Presentation](https://ParallelScience.github.io/denario-3-qm9_v1/)**

## Abstract

Classical group-contribution methods effectively capture the extensive scaling of molecular internal energy but inherently fail to account for non-additive electronic effects arising from complex topological features. To isolate and quantify these non-linear contributions, we propose a residual decomposition approach applied to the QM9 dataset. We first train a Ridge regression baseline on heavy atom and functional group counts to decouple the extensive energetic scaling. Subsequently, we employ a Graph Neural Network (GNN) to predict the standardized residuals of this additive baseline, utilizing Integrated Gradients and latent space dimensionality reduction to interpret the learned representations. The GNN successfully captures the topological variance missed by classical models, achieving a test $R^2$ of 0.9985 on the residuals and substantially outperforming a descriptor-based Random Forest baseline. Analysis of the GNN's 128-dimensional latent space reveals intrinsic correlations with fundamental intensive properties, such as the HOMO-LUMO gap and HOMO energy, despite the absence of direct supervision on these targets. Furthermore, structural motif attribution and statistical validation confirm that features such as aromaticity, nitrogen integration, and fluorine incorporation significantly drive these non-additive deviations, corresponding directly to compressed HOMO-LUMO gaps and shifted dipole moments. This framework rigorously separates extensive scaling from underlying non-linear electronic physics, demonstrating that graph-based message passing can effectively map specific structural motifs to their quantum-mechanical deviations from classical additivity.
