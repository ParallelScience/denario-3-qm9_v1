1. **Data Cleaning and Preprocessing**:
   - Load the QM9 dataset and parse SMILES using RDKit.
   - Identify and handle the 83 duplicated SMILES by averaging their property values to ensure consistent targets.
   - Calculate heavy atom counts and implicit hydrogen counts for each molecule.

2. **Baseline Additive Modeling**:
   - Generate a comprehensive additive feature set including total heavy atom count, RDKit fragment descriptors (e.g., MACCS keys), and specific bond-type counts.
   - Train a Ridge regression model to predict `u0` using these features.
   - Calculate the residuals (actual `u0` minus predicted `u0`).
   - Verify that residuals are independent of molecule size by plotting them against the heavy atom count; if scaling remains, normalize residuals by the number of heavy atoms.

3. **Non-Linear Baseline Comparison**:
   - Train a Random Forest or XGBoost regressor on the same additive feature set used in Step 2 to predict the residuals.
   - Use this as a performance benchmark to quantify whether the GNN's graph-based representation provides superior predictive power over standard descriptor-based non-linear models.

4. **GNN Architecture Design**:
   - Construct a Message Passing Neural Network (MPNN).
   - Define node features to include atomic number, degree, hybridization state (sp, sp2, sp3), aromaticity flags, and formal charge.
   - Include the heavy atom count as a global feature to ensure the model focuses on intensive non-additive electronic effects.

5. **GNN Training and Validation**:
   - Perform a scaffold split (using RDKit) to divide the dataset into training, validation, and test sets to ensure generalization to unseen molecular architectures.
   - Scale the residuals using `StandardScaler` to improve training stability.
   - Train the GNN to predict the scaled residuals using MSE loss, monitoring validation performance to prevent overfitting.

6. **Attribution via Integrated Gradients**:
   - Apply Integrated Gradients (IG) to the trained GNN to attribute residual values to specific atoms and bonds.
   - Use a "null" graph (a graph with zeroed-out features and no edges) as the baseline reference for the IG calculation to ensure attribution scores are physically meaningful.

7. **Correlation with Intensive Properties**:
   - Extract the GNN-predicted residuals for each molecule.
   - Perform a correlation analysis between these residuals and intensive properties (`gap`, `mu`, `homo`, `lumo`) to determine if the non-additive energetic stability is physically linked to electronic properties.

8. **Structural Motif Analysis and Final Evaluation**:
   - Group molecules by high-influence subgraphs identified via IG scores.
   - Compare mean `gap` and `mu` values across these groups to quantify how specific motifs (e.g., conjugation, ring strain) deviate from classical group-contribution theory.
   - Summarize the findings by mapping the identified "non-additivity" motifs to their respective impacts on molecular electronic stability.