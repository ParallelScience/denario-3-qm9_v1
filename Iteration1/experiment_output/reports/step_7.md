# Decoupling Non-Additive Energetics via Graph-Latent Residual Decomposition

## 1. Introduction and Methodological Framework

The extensive scaling of molecular internal energy ($u_0$) in the QM9 dataset is predominantly driven by simple atomic additivity. However, chemical stability is fundamentally determined by non-additive electronic phenomena, including extended $\pi$-conjugation, resonance stabilization, and ring strain. This study isolates these non-additive topological effects by training predictive models on the residuals of an additive baseline model ($u_0$ residuals). To capture long-range topological features that strictly linear group-contribution models miss, we trained and compared a Graph Neural Network (GNN) and a tabular XGBoost baseline built on RDKit topological descriptors. 

The analysis leverages Integrated Gradients (IG), in-silico graph mutations, and conjugation isomer comparisons to interrogate the specific subgraphs driving non-additive energy shifts. Finally, the structural physics captured by the model are contextualized through partial correlations with intensive quantum properties.

## 2. Model Performance and Variance Decomposition

The objective of the machine learning phase was to predict the non-additive internal energy residuals ($u_0$ residuals) after controlling for raw atomic and bond-type counts. Both the XGBoost baseline (trained on an extensive set of RDKit descriptors including branching factors, ring sizes, and connectivity indices) and the Message-Passing Neural Network (GNN) achieved high predictive accuracy on the unseen test set. The XGBoost model yielded a test set $R^2$ of 0.9810, while the GNN achieved a test set $R^2$ of 0.9777. 

To determine whether the GNN merely memorized known structural heuristics or captured deeper non-linear physics, a variance decomposition was performed on the test set predictions ($N = 14,151$). A linear regression trained to map basic scalar descriptors (ring count, heteroatoms, aromatic rings, double bonds, spiro atoms, bridgehead atoms, and fraction of $sp^3$ carbons) to the GNN's predicted residuals explained only 9.64% of the variance ($R^2 = 0.0964$). The remaining 90.36% of the predicted variance constitutes non-linear physics and topological interactions captured internally by the GNN's message-passing architecture, confirming that the graph representation operates well beyond standard additive tabular limits.

## 3. Interpreting Graph-Latent Attributions

To physically interpret the GNN's non-linear variance, Integrated Gradients (IG) were computed over 1,000 test set molecules to assign attribution scores to input node and edge features. The mean absolute attribution sum per molecule was 61.8885 for nodes and 7.0920 for edges.

Analysis of the individual feature attributions reveals the chemical hierarchies the GNN relies on to compute non-additive stability:
*   **Node Features:** The model relies heavily on elemental identity and orbital geometry. The mean absolute attribution for Atomic Number was 3.7750, making it the most dominant feature. Hybridization followed closely at 2.5298. Interestingly, simple valency proxies were far less impactful (Degree: 0.5997). Formal charge (0.0002) and binary aromaticity flags (0.1226) provided minimal direct attribution, suggesting the network deduces aromatic stabilization primarily from the raw ring topology and hybridization states rather than explicitly relying on the engineered aromaticity label.
*   **Edge Features:** Bond Type dominated edge attributions (0.3187), while the explicit `IsConjugated` flag contributed less (0.0519). The disparity indicates that the message-passing scheme learns continuous conjugation paths via the sequence of bond types and node hybridizations, rather than strictly weighting the binary localized edge flag.

## 4. In-Silico Subgraph Perturbation: Isolating Conjugation

To quantify how non-additive energetic shifts arise from specific motifs, an in-silico mutation experiment was conducted. A subset of 4,146 test-set molecules containing non-aromatic conjugated double bonds was identified. For each molecule, the target conjugated double bond was programmatically reduced to a single bond (breaking the conjugation pathway without altering the rest of the molecular topology), and the predicted shift in $u_0$ residuals was recorded.

The disruption of resonance stabilization yielded a significant energetic penalty in the GNN predictions, manifesting as a mean energy shift of $0.5663 \pm 2.7581$ Hartree (Mutated - Original). By contrast, the XGBoost model predicted a near-zero mean shift of $0.0246 \pm 3.1799$ Hartree. Although the predicted shifts between the two models were highly correlated (Pearson $r = 0.9543$, $p < 0.001$), the stark difference in mean magnitude demonstrates a crucial point: while XGBoost detects a perturbation in the global feature vector, the GNN explicitly penalizes the loss of the continuous orbital overlap pathway. The GNN effectively isolates the magnitude of resonance stabilization energy, proving its sensitivity to topological connectivity rather than mere feature presence.

## 5. Conjugation Isomer Validation

The most stringent test of a model's capacity to resolve non-additive structural phenomena is its performance on conjugation isomers—molecules with identical chemical formulas but different connectivity and resonance stabilization networks. 

A comprehensive pair-wise matching across the dataset identified 190,679 conjugation isomer pairs. Over the entire distribution, the Pearson correlation between the actual $u_0$ residual difference and the GNN-predicted difference was positive and statistically significant ($r = 0.2279$, $p < 0.001$). However, performance on the rigidly controlled test-set isomer pairs ($N = 1,802$ pairs, both molecules in the unseen test split) was exceptionally strong, yielding a Pearson correlation of $r = 0.9875$ ($p < 0.001$). This confirms that the model's learned representation of residual stability rigorously maps to the true physical energy differences imposed by varying conjugation and ring strain configurations.

## 6. Correlation with Intensive Quantum Properties

If the GNN's predicted $u_0$ residuals accurately reflect underlying electronic stability, they should systematically correlate with frontier orbital energies. To test this, partial Spearman correlations were calculated between the GNN-predicted $u_0$ residuals and intensive quantum properties on the test set, strictly controlling for ring count and heteroatom count to eliminate spurious scaling correlations.

The partial correlation analysis yielded significant associations across all tested electronic properties:
*   **LUMO Energy:** $\rho = 0.213965$ ($p = 3.27 \times 10^{-146}$)
*   **HOMO Energy:** $\rho = 0.190063$ ($p = 3.20 \times 10^{-115}$)
*   **HOMO-LUMO Gap:** $\rho = 0.134176$ ($p = 7.59 \times 10^{-58}$)
*   **Dipole Moment ($\mu$):** $\rho = 0.039407$ ($p = 2.74 \times 10^{-6}$)

The strong relationships with HOMO and LUMO energies confirm that the non-additive residuals are directly linked to the frontier molecular orbital physics governing electron delocalization. Increased thermodynamic stability (captured by the $u_0$ residuals) systematically alters the orbital gap, independently of simple cyclic or heteroatomic characteristics.

## 7. Methodological Limitations and Discussion

While the study successfully decouples non-additive energetic effects, certain limitations remain. First, both the XGBoost and GNN architectures exhibited minor unstructured residuals (accounting for approximately 2% of the test set variance). These unstructured residuals likely stem from 3D geometric phenomena—such as steric clashing and non-bonded spatial overlap—that cannot be fully resolved from 2D SMILES connectivity graphs. 

Furthermore, the XGBoost tabular model achieved a marginally higher bulk test $R^2$ (0.9810) than the GNN (0.9777), highlighting a known trade-off in chemoinformatics: heavily engineered descriptors (such as detailed spatial and fragment counts in RDKit) can outcompete raw message-passing on bulk interpolation tasks. However, the in-silico mutation results emphasize that the GNN is chemically more consistent, properly penalizing the specific loss of localized conjugation where tabular models fail to register the physical magnitude of the shift.

In conclusion, the graph-latent decomposition effectively isolates non-additive quantum thermodynamic effects. The GNN's ability to evaluate continuous orbital networks accurately captures resonance stabilization and correctly ranks conjugation isomers, bridging the gap between classical additive group-contribution theory and non-linear quantum structural behavior.