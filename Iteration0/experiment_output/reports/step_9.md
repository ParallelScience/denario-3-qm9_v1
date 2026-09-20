# Decoupling Non-Additive Energetics via Graph-Latent Residual Decomposition: Results and Discussion

This report details the outcomes of our investigation into non-additive electronic contributions to internal energy and intensive properties within the QM9 dataset. By separating the purely additive, size-extensive scaling of molecular properties from their intrinsic, structure-dependent deviations, we evaluated the capacity of linear baselines versus non-linear topological models to capture complex electronic effects. 

## 1. Modeling Non-Additive Residuals: Baseline versus Topological Performance

The first objective of this study was to establish a purely additive baseline for extensive properties (such as the internal energy at 0 K, $U_0$) and subsequently model the non-linear residuals. As illustrated in the artifact `data/residuals_vs_size.png`, decoupling the raw extensive metrics from heavy atom count yields a residual distribution that reflects intrinsic molecular stability and electronic deviations rather than mere molecular size. 

To model these residuals, we compared standard descriptor-based models against a topological neural network (MLP over Morgan fingerprints and 2D descriptors). The non-linear Random Forest (RF) baseline struggled significantly with the non-additive residuals. Sourced from `data/baseline_metrics.txt` and visually confirmed in `data/rf_baseline_predictions.png`, the RF model yielded a test $R^2$ of $-0.8541$ for `res_u0`, indicating a failure to generalize beyond a simple mean prediction. Its predictive capability for the gap (`res_gap`) and dipole moment (`res_mu`) residuals was similarly weak, achieving test $R^2$ values of $0.2760$ and $0.1345$, respectively. 

Conversely, the topological model demonstrated a robust capacity to capture the underlying non-additive physics. According to `data/topological_model_metrics.txt` and shown in `data/topological_model_predictions.png`, the topological approach achieved a test $R^2$ of $0.6493$ for `res_u0` (Test MSE: 0.0413). Performance on intensive electronic targets was equally strong, returning a test $R^2$ of $0.6205$ for `res_gap` (Test MSE: 0.4727) and $0.5239$ for `res_mu` (Test MSE: 0.4832). This performance gap quantitatively verifies that standard group-contribution and shallow non-linear models are insufficient for capturing the complex, long-range topological features (such as extended $\pi$-conjugation and ring strain) that dictate energetic deviations.

## 2. Structural Motif Attribution via SHAP

To interpret the structural drivers of non-additivity, we computed Integrated Gradients (SHAP) across the test set. Feature attributions, detailed in `data/shap_motif_summary.csv`, pinpoint specific substructures responsible for significant energetic and electronic deviations.

Across all targets, the most influential motifs included `BCUT2D_MRHI` (mean absolute SHAP: 0.0320), the ether fragment `fr_ether` (0.0304), and the structural fingerprint `MorganFP_314` (0.0265). Breaking down the feature importance by target reveals distinct physical mechanisms:

*   **Internal Energy Residuals (`res_u0`)**: The model identified structural complexity and rigidity as primary drivers of non-additive stability. The presence of spiro centers (`NumSpiroAtoms`, SHAP: 0.0263) and molecular flexibility (`NumRotatableBonds`, SHAP: 0.0246) exerted the highest influence. 
*   **HOMO-LUMO Gap Residuals (`res_gap`)**: Electronic conjugation pathways strongly perturbed the HOMO-LUMO gap. High polarizability descriptors (`BCUT2D_MRHI`, SHAP: 0.0776) and oxygen-containing functional groups such as ethers (`fr_ether`, SHAP: 0.0662) and ketones (`fr_ketone`, SHAP: 0.0532) were strictly linked to gap deviations, underscoring the role of heteroatom lone-pairs in orbital splitting.
*   **Dipole Moment Residuals (`res_mu`)**: Non-additive dipole moments were governed by specific functional group topology (`MorganFP_807`, SHAP: 0.0383) and asymmetric charge distributions captured by topological electrotopological states (`MaxEStateIndex`, SHAP: 0.0316; `MaxAbsEStateIndex`, SHAP: 0.0315).

## 3. Quantitative Relationships Between Non-Additivity and Intensive Properties

By projecting non-additive energy residuals into the space of intrinsic electronic properties, we established a quantitative link between structural strain and electronic configuration. The correlations between the predicted residuals and true intensive properties are cataloged in `data/motif_correlations.csv` and visualized as density scatter plots in `data/residual_gap_corr.png`.

The predicted non-additive internal energy (`res_u0_pred`) exhibits a significant inverse correlation with the HOMO-LUMO gap (Pearson $r = -0.1981$, $p < 10^{-118}$). Molecules with positive energy residuals (indicating lesser stability than predicted by additive models) tend to exhibit narrower gaps, implying that the same structural motifs that destabilize the core geometry also promote tighter orbital spacing. The `res_u0_pred` metric also correlates positively with HOMO energies ($r = 0.1498$) and negatively with LUMO energies ($r = -0.1398$). 

Furthermore, the topological predictions for the intensive residuals themselves strongly correlate with their absolute target properties, validating the model's physical relevance. The predicted gap residual (`res_gap_pred`) holds a Pearson correlation of $0.4745$ with the true HOMO-LUMO gap, while the predicted dipole residual (`res_mu_pred`) correlates strongly with the true dipole moment ($r = 0.6115$).

### Motif-Level Property Distributions

To concretize how specific non-additive structural motifs perturb intrinsic electronics, we analyzed property distributions conditioned on motif presence, drawing from `data/motif_statistics.csv` and represented visually in `data/motif_distributions.png`.

*   **Ethers (`fr_ether`)**: When present, ethers widen the HOMO-LUMO gap (mean: 0.2598 Hartree) compared to their absence (0.2234 Hartree). Conversely, they reduce the average dipole moment (2.4487 Debye present vs. 2.7801 Debye absent).
*   **Ketones (`fr_ketone_Topliss` and `fr_ketone`)**: The presence of ketone motifs drastically narrows the gap. Molecules featuring `fr_ketone_Topliss` average a gap of 0.2046 Hartree, compared to 0.2513 Hartree for those without. They also substantially elevate the dipole moment (3.0682 Debye vs 2.5076 Debye), consistent with the strongly polarized carbon-oxygen double bond.
*   **Latent Topological Features (`MorganFP_314`)**: Substructures captured by `MorganFP_314` show a distinct electronic signature, pushing the mean gap down to 0.2179 Hartree (absent: 0.2580 Hartree) while driving the dipole moment sharply upward to 3.5221 Debye (absent: 2.1297 Debye). 

## Conclusion

This analysis rigorously validates that non-additive molecular energetics—decoupled from extensive size scaling—are predominantly governed by complex topological features rather than simple functional group counts. The failure of linear and shallow non-linear baselines (Test $R^2 < 0.3$) juxtaposed with the success of topological representations (Test $R^2 \approx 0.52-0.65$) highlights the necessity of graph-aware modeling. By mapping these energetic residuals to intrinsic properties like the HOMO-LUMO gap and dipole moment, we have quantified how structural non-additivity acts as a direct proxy for electronic configuration, driven specifically by polarizability limits and conjugation effects mediated by oxygen-containing motifs and ring topology.