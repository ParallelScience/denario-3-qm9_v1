

Iteration 0:
### Summary: Decoupling Non-Additive Energetics via Graph-Latent Residual Decomposition

**1. Objective & Methodology**
The project aimed to isolate non-additive electronic contributions to internal energy ($U_0$) and intensive properties (`gap`, `mu`) in the QM9 dataset by removing size-extensive scaling. We compared a linear additive baseline (Ridge regression on fragment/bond counts) against a topological model (MLP on Morgan fingerprints/2D descriptors) to predict residuals.

**2. Key Findings**
*   **Model Performance:** Linear and shallow non-linear models (Random Forest) failed to capture non-additive residuals (Test $R^2 < 0$ for $U_0$). The topological model succeeded, achieving Test $R^2$ of 0.6493 for `res_u0`, 0.6205 for `res_gap`, and 0.5239 for `res_mu`.
*   **Structural Drivers (SHAP/IG):** Non-additivity is driven by structural rigidity and conjugation. 
    *   `res_u0` is influenced by spiro centers and rotatable bonds.
    *   `res_gap` is sensitive to polarizability (`BCUT2D_MRHI`) and oxygen-containing groups (ethers, ketones).
    *   `res_mu` is governed by topological electrotopological states.
*   **Physical Correlations:** Predicted non-additive energy residuals correlate inversely with the HOMO-LUMO gap ($r = -0.1981$), suggesting that structural motifs destabilizing the core geometry also promote tighter orbital spacing.

**3. Limitations & Uncertainties**
*   **Data:** 83 duplicated SMILES were averaged; 45 molecules have $\mu=0$.
*   **Methodology:** The topological model relies on 2D descriptors/fingerprints; while effective, it lacks explicit 3D geometry, which may limit the capture of subtle steric strain effects.
*   **Attribution:** SHAP/IG analysis provides feature importance but does not explicitly define the "non-additivity" mechanism beyond correlation with specific functional groups.

**4. Decisions for Future Work**
*   **Shift to GNNs:** The success of topological models justifies moving from fingerprint-based MLPs to full Message Passing Neural Networks (MPNNs) to better capture long-range graph topology.
*   **Refine Motif Analysis:** Future experiments should focus on the identified high-influence motifs (ethers, ketones, spiro centers) to determine if they can be used as "building blocks" for targeted property engineering.
*   **Geometry Integration:** Since 3D coordinates are absent, consider using RDKit (ETKDG + MMFF) to generate conformers if residual errors remain high for specific strained architectures.
        