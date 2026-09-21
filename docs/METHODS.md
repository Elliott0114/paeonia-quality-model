# Analysis methods

## Public commands

Run `python scripts/run_example.py` for feature/spectral processing and `python scripts/plot_results.py` for aggregate numerical panels. Both commands support `--data-dir` and `--output-dir`; `--help` describes the defaults. These commands do not retrain the cohort models or recompute confidence intervals from rounded summaries.

## Core functions

| Module | Functions and purpose |
|---|---|
| `paeonia.imaging` | `process_image`: the 12 selected image descriptors |
| `paeonia.spectra` | `exposure_factor`, `fixed_wavelength_indices`, `preprocessing_branches`: P0–P3 |
| `paeonia.visible` | `grouped_splits`, `select_model`, `estimators`, `cluster_bootstrap`: nested appearance classification |
| `paeonia.permutation` | `build_split_batches`, `evaluate_labels`, `permutation_task`: fixed-classifier batch-label sensitivity |
| `paeonia.chemical` | Calibration eligibility, `build_repeated_splits`, `run_endpoint_family`, OOF metrics, paired increments and CAT within-batch contrasts |
| `paeonia.cross_modal` | `fit_predict`, `select_alpha`: fold-local visible/HSI/joint models |
| `paeonia.multianalyte` | `run_split_endpoint`, `paired_errors`, `panel_summary`: seven-analyte cross-modal effects |
| `paeonia.panels` | `pair_predictions`, `panel_scores`, `summarize_panel`: mean/domain panel inference and leave-one-analyte sensitivity |
| `paeonia.loso` | `build_loso_splits`, `run_split_endpoint`, `endpoint_increment_results`, `panel_results`: cross-session analysis |
| `r/secondary_methods.R` | `secondary_analysis`: batch heterogeneity and appearance–chemistry association |

These are computational methods with explicit array/table arguments. Data acquisition, private-file adapters and a full-cohort command-line workflow are outside this release. See the [input schemas](DATA_DICTIONARY.md).

## Visible classification

The fixed visible15 panel combines 12 image descriptors and three stored machine-vision measures. Recorded appearance grade is the classification target. Outer validation uses 10 repeated five-fold stratified batch-grouped splits; four grouped inner folds select among five classifiers by mean macro-F1. Preprocessing is fitted within training folds. Evaluation uses 207 portions from 69 batches. Batch-cluster CIs use 5,000 draws; fixed-classifier batch-label sensitivity uses 199 permutations, seed 20260711. Per-grade F1 and agreement-stratum CIs use 10,000 draws. Higher recorded grades denote poorer appearance.

## Chemical models

The primary spectrum P0 contains 262 bands (430.42–969.81 nm). P1–P3 are sensitivity representations. Ridge alpha candidates are 0.01, 0.1, 1, 10, 100 and 1000. Ten repeated five-fold batch-grouped outer splits use seeds 20260711–20260720; five grouped inner folds choose alpha by pooled validation RMSE. Learned scaling, domain encoding, nuisance fitting and model fitting stay within the relevant training fold.

Baselines include the training mean, acquisition domain and the 15 visible features. Domain predictors are within-batch position, HSI acquisition session and HPLC calibration series. HPLC values are targets; expert grade is not a chemical predictor. Joint models fit a ridge nuisance baseline, regress spectra on nuisance predictors within the training fold, fit residual targets with ridge, and add the nuisance prediction.

Metrics use predictions averaged across 10 outer repeats. RMSE/MAE are in mg/g. HSI metric CIs use 2,000 batch-cluster bootstrap draws. Paired squared errors are computed before averaging over repeats and within batches. Endpoint inference uses 100,000 batch sign flips, with BH adjustment separately within each seven-analyte comparison family; S5A–D retain all four families.

## Panel inference

The seven-analyte panel is primary; excluding CAT is post hoc sensitivity. A batch is included when each panel analyte has at least one eligible portion. Sample sets can differ across analytes: 67 batches/200 unique portions enter the seven-analyte panel, and 68/203 enter the six-analyte panel.

Paired baseline-minus-augmented squared errors are averaged over repeats and eligible portions within endpoint/batch. Each endpoint's differences are divided by its baseline MSE pooled over the same panel batches, then averaged equally over endpoints and batches. Denominators remain fixed during resampling. CIs use 10,000 batch bootstrap draws and tests use 100,000 sign flips, base seed 20260801. Holm adjustment covers four mean/domain comparisons separately from four appearance/joint comparisons. Target-variance normalization and leave-one-analyte effects are sensitivity analyses. The S6 CAT-excluded baseline-MSE row shares the Table 4 comparison: 13.9%, CI 2.9–26.8%, unadjusted p = 0.008.

## Cross-session validation

Each of seven HSI sessions is held out once; inner leave-one-session-out validation tunes alpha using the remaining sessions. The baseline predicts the mean eligible content of the six training sessions. This is internal cross-session validation.

- S7B/Figure 5D: `100 * (baseline_mse - hsi_mse) / baseline_mse`, using each held-session baseline MSE.
- S7A: average absolute MSE differences equally across sessions, divide by the endpoint's pooled baseline MSE, then multiply by 100.
- LOSO panels: normalize endpoint/session differences by endpoint-pooled baseline MSE, then average equally across endpoints and sessions.

Negative relative reductions, including values below −100%, are retained. Pooled metric CIs use 5,000 batch-cluster draws; session-equal intervals use 10,000 session-bootstrap draws. Exact tests enumerate all 2⁷ sign assignments; endpoint tests have seven-analyte BH adjustment. Cross-session support requires positive pooled R² with its CI above zero and a positive session-equal MSE reduction with BH q < 0.05. No endpoint met all criteria.

## Secondary analyses and reported exploratory results

Within-batch heterogeneity uses the eligible within-batch range divided by the dataset-wide analyte IQR, held fixed during bootstrap. Appearance–chemistry Spearman CIs resample batches, and two-sided permutation tests retain the batch structure. BH adjustment is separate for each appearance measure across seven analytes. The R wrapper `secondary_analysis` preserves the manuscript's order of the heterogeneity and association calculations and uses seed 20260801 with 10,000 bootstrap/permutation draws.

CAT within-batch contrasts use eligible G1–G2, G1–G3 and G2–G3 pairs, keeping all contrasts from a batch together. The reported exploratory analysis has 92 contrasts from 58 batches; the paired increment was not supported (p = 0.613). Full estimates, including negative LOSO panel effects, remain in the [result notes](../paper_results/tables/README.md#supplementary-results-reported-in-text).

Optional R use: restore `r/renv.lock` with `renv::restore(project = "r")`, use that project library, source `r/secondary_methods.R`, and call `secondary_analysis(detail, visible_predictions)`. R is not required for the public Python commands.
