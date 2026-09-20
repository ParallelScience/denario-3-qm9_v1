1. **Data Preprocessing and Quality Control**:
   - Load the QM9 dataset and parse SMILES using RDKit.
   - Identify and handle the 83 duplicated SMILES by averaging their property values to ensure consistent labels.
   - Perform a molecule-aware train/validation/test split (80/10/10) to ensure all instances of the same SMILES reside in the same partition, preventing data leakage.
   - Calculate heavy atom counts and hydrogen counts to be used as explicit size-normalization features.

2. **Baseline Additive Model Construction**:
   - Construct a Group Contribution Method (GCM) baseline using Ridge regression. The model should predict `u0` using `N_heavy_atoms` and counts of bond types/functional groups as independent variables.
   - Train the baseline only on the training set.
   - Calculate residuals (`u0_actual - u0_predicted`) for all sets. Standardize these residuals (z-score normalization) to ensure stable training for the GNN.

3. **GNN Architecture Design**:
   - Implement a Graph Isomorphism Network (GIN) or Gated Graph Convolutional Network (GGCN) to capture topological nuances.
   - Initialize node features (atomic number, hybridization, degree) and edge features (bond type, aromaticity).
   - Incorporate a Global Attention Pooling layer to effectively aggregate global electronic effects like conjugation.
   - Ensure the output layer is linear (no activation) to accommodate the range of the standardized residuals.

4. **GNN Training and Benchmarking**:
   - Train the GNN on the standardized residuals using MSE loss.
   - Implement early stopping based on validation loss.
   - To validate the GNN's efficacy, train a non-linear baseline (e.g., Random Forest or MLP) on the same RDKit descriptors used in Step 1. Compare the GNN’s performance against this baseline to confirm that graph-based message passing captures unique topological dependencies.

5. **Attribution of Non-Additive Energetics**:
   - Apply Integrated Gradients (IG) to the trained GNN.
   - Define the baseline input as a "null" graph (zeroed-out node/edge features).
   - Normalize IG scores by molecule size to ensure that attribution is comparable across molecules of different scales.

6. **Latent Space Analysis**:
   - Extract latent representations from the penultimate layer of the GNN.
   - Use UMAP or t-SNE to visualize the latent space, checking for clusters corresponding to chemical families (e.g., aromatic vs. aliphatic).
   - Correlate these latent vectors with intensive properties (`gap`, `mu`, `homo`, `lumo`) to determine if the "non-additivity" aligns with known electronic descriptors.

7. **Statistical Validation of Structural Motifs**:
   - Group molecules based on high-influence subgraphs identified via IG scores.
   - Perform statistical tests (e.g., ANOVA) to compare `gap` and `mu` values between molecules containing these motifs versus those that do not, confirming the physical relevance of the GNN-identified features.

8. **Final Synthesis**:
   - Compile findings to demonstrate the decoupling of extensive scaling from non-linear electronic physics.
   - Map "non-additivity scores" to specific chemical motifs, providing a quantitative assessment of how these motifs deviate from classical group-contribution theory.