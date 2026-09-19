# QM9 Molecular Property Dataset — Data Description

## 1. Files

- `/home/node/work/data/qm9/qm9.csv` — the dataset. One CSV, 133,885 rows × 21 columns (one row per molecule), no missing values. Read with `pd.read_csv('/home/node/work/data/qm9/qm9.csv')`.

## 2. What the data is

QM9 (MoleculeNet release): quantum-mechanical properties of drug-like molecules drawn from the GDB-9 database. Molecules contain up to 9 heavy atoms, elements C, N, O, F only (plus implicit hydrogens). Properties were computed with DFT (B3LYP/6-31G(2df,p)) with atomization-energy bias correction. This is the *equilibrium-geometry* (relaxed) set: 133,885 molecules, identified as `gdb_1` … `gdb_133885`.

Structure is represented **only by SMILES** (canonical, stereochemistry stripped). No 3D coordinates are provided.

## 3. Columns, types, units

| Column | dtype | Meaning | Units | Typical range |
|---|---|---|---|---|
| `mol_id` | str | GDB-9 identifier (e.g. `gdb_4211`) | — | — |
| `smiles` | str | canonical SMILES, no stereo, C/N/O/F ≤ 9 heavy atoms | — | — |
| `A` | float | rotational constant A | GHz | 0 – 6.2e5 (huge for near-linear tops) |
| `B` | float | rotational constant B | GHz | 0.34 – 438 |
| `C` | float | rotational constant C | GHz | 0.33 – 283 |
| `mu` | float | dipole norm | Debye | 0 – 29.6 (mean 2.7) |
| `alpha` | float | isotropic polarizability | Bohr³ (a₀³) | 6.3 – 197 |
| `homo` | float | HOMO energy | Hartree | −0.429 – −0.102 |
| `lumo` | float | LUMO energy | Hartree | −0.175 – 0.194 |
| `gap` | float | HOMO–LUMO gap (= homo − lumo) | Hartree | 0.025 – 0.622 (mean 0.251) |
| `r2` | float | mean square radius ⟨r²⟩ | Bohr² | 19 – 3375 |
| `zpve` | float | zero-point vibrational energy | Hartree | 0.016 – 0.274 |
| `u0` | float | internal energy at 0 K | Hartree | −714.6 – −40.5 |
| `u298` | float | internal energy at 298.15 K | Hartree | ≈ u0 + 0.008 |
| `h298` | float | enthalpy at 298.15 K | Hartree | ≈ u298 + 0.001 |
| `g298` | float | free energy at 298.15 K | Hartree | ≈ u298 − 0.04 |
| `cv` | float | heat capacity at 298.15 K | cal/(mol·K) | 6.0 – 47.0 |
| `u0_atom`,`u298_atom`,`h298_atom`,`g298_atom` | float | **CAVEATED:** per-atom energies as shipped by MoleculeNet | inconsistent | see caveats |

## 4. Caveats and known structure

- **Size extensivity dominates raw correlations.** `u0, u298, h298, g298, zpve, alpha, r2, cv` are *extensive* — they scale roughly linearly with molecule size (heavy-atom / electron count). Raw pairwise correlations among them reach r > 0.99 almost entirely through molecule size. Any structure–property analysis must control for size (normalize per heavy atom, regress out atom counts, or predict intensive quantities like `gap`, `mu`, `homo`, `lumo`).
- **The four `_atom` columns are suspect.** Their scaling is inconsistent with u0 divided by atom count (e.g. methane u0 = −40.479 Eh gives `u0_atom` = −396.0). Do not use them without re-deriving; prefer computing per-heavy-atom energies directly from `u0` and the SMILES-derived atom count.
- **83 duplicated SMILES** (stereochemistry collapsed; different 3D isomers share a canonical SMILES). Expect small property scatter for duplicates.
- **45 rows with `mu` = 0 exactly**: symmetric molecules, physically real.
- **`A` spans 6 orders of magnitude** (near-linear tops → use log scale or drop for rotational analyses).
- Thermodynamic quantities satisfy near-exact identities (`gap = homo − lumo`; `h298 ≈ u298 + RT`; etc.) — good consistency checks, not new information.
- Hydrogens are implicit in SMILES; H counts must be derived (rdkit installed, see below).

## 5. Environment

- Python: `/opt/denario-venv/bin/python` (pandas, numpy, scipy, sklearn, torch available; **rdkit 2026.03.6 installed** for SMILES parsing, descriptors, fingerprints).
- No 3D geometries; if geometry-dependent features are wanted they must be optimized with a force field (rdkit ETKDG + MMFF/UFF).

## 6. Suggested analyses

1. **Extensivity decomposition**: separate extensive vs intensive properties; recover per-bond / per-group additivity (group-contribution structure) of u0, zpve, alpha by linear regression on bond/functional-group counts from SMILES; quantify residuals.
2. **Structure→property models** from SMILES (RDKit 2D descriptors or Morgan fingerprints) for intensive targets (`gap`, `mu`, `homo`, `lumo`) and size-normalized extensive targets; compare linear group-contribution vs GBM/NN; analyze error vs molecule size/composition.
3. **Scaling relations**: α(⟨r²⟩, N), Cv(N, ring counts), rotational constants vs inertia proxies — test physical scaling exponents.
4. **Property–property geometry** after size control: PCA of intensive residuals — what orthogonal "chemistry axes" remain (conjugation, electronegativity patterns, ring strain proxies)?
5. **Gap engineering**: which substructures minimize/maximize HOMO–LUMO gap; identify conjugation/aromaticity effects via group coefficients.
