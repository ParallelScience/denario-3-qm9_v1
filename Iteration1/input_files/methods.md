1. **Baseline Construction and Size Normalization**:
   - Perform a robust regression (e.g., Theil-Sen) of `u0` against heavy atom counts, hydrogen counts, and RDKit-derived bond-type counts (single, double, triple, aromatic).
   - Calculate residuals $u0_{res} = u0_{actual} - u0_{predicted}$.
   - Perform a t-test on the slope of $u0_{res}$ vs $N_{heavy}$. If statistically significant, apply a linear correction to ensure residuals are strictly intensive.

2. **Feature-Matched Baseline Comparison**:
   - Construct an expanded feature set for an XGBoost model including ring sizes, path lengths between heteroatoms, aromaticity indices, and branching factors.
   - Train the XGBoost model on these descriptors to predict $u0_{res}$ using a scaffold split.
   - Evaluate performance on the test set to establish a baseline for "topological" information capture.

3. **GNN Architecture and Training**:
   - Implement an MPNN using node features (atomic number, degree, hybridization, aromaticity, formal charge) and edge features (bond type, conjugation status).
   - Train the GNN to predict $u0_{res}$ using the same scaffold split as the XGBoost model to ensure a fair comparison.
   - Ensure the pipeline handles implicit hydrogen recalculation consistently during any structural modifications.

4. **Subgraph Perturbation and Integrated Gradients (IG)**:
   - Perform IG analysis using a "chemically neutral" baseline (e.g., a graph of isolated atoms with the same node count as the target) to identify motifs influencing $u0_{res}$.
   - Conduct "in-silico mutations" (e.g., converting conjugated double bonds to single bonds). Compare the GNN's predicted shift in $u0_{res}$ against the XGBoost baseline shift to isolate non-linear electronic effects from simple bond-count changes.

5. **Isomer-Specific Validation**:
   - Extract "conjugation isomers" (same formula, different conjugation patterns).
   - Calculate $\Delta u0_{res} = u0_{res, A} - u0_{res, B}$ for isomer pairs.
   - Evaluate if the GNN's predicted $\Delta u0_{res}$ correlates with actual energy differences, serving as a direct test of the model's ability to capture resonance stabilization.

6. **Residual Variance Decomposition**:
   - Quantify the variance in GNN-predicted residuals explained by known physical descriptors (e.g., RDKit-calculated ring strain energy) versus the variance attributed to "model-captured" electronic physics.
   - This distinguishes between the model learning known chemistry versus discovering new non-linear electronic effects.

7. **Correlation with Intensive Electronic Properties**:
   - Perform partial correlation analysis (using Spearman’s $\rho$ to account for non-linearity) between GNN-predicted $u0_{res}$ and intensive properties (`gap`, `mu`, `homo`, `lumo`), controlling for ring and heteroatom counts.
   - Determine if non-additive energy residuals are statistically linked to specific electronic phenomena like conjugation-induced gap narrowing.

8. **Final Evaluation of Non-Additivity**:
   - Synthesize results by categorizing residuals into "Chemical Non-Additivity" (resonance, strain) and "Model Residuals."
   - Map high-influence motifs identified by the GNN to their impacts on molecular stability, providing a clear distinction between classical group-contribution theory and the non-linear electronic physics captured by the GNN.