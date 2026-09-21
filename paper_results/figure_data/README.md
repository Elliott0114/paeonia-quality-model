# Aggregate figure data

These CSVs contain numerical summaries for selected panels of Figures 1–5. They contain no individual-level cohort predictions. Figure 3A–C use representative images; only the single example image is distributed. Figure 3D–F correspond to the confusion matrix, F1 metrics and prediction-agreement summaries.

For Figure 5D, each row is one analyte and held-out acquisition session. All 7 × 7 combinations are retained. `baseline_mse` and `hsi_mse` are in (mg/g)². `mean_squared_error_reduction` is the absolute baseline-minus-HSI difference. `relative_mse_reduction` divides that difference by the baseline MSE from the same held session; `percentage_mse_reduction` is 100 times the ratio. `label` rounds that percentage for display.

A relative reduction can be less than −100% when HSI error exceeds twice the baseline error. The values have not been clipped. GA in session S1 has a relative reduction of −119.4058517%, displayed as −119.4% in S7B and −119 in the heatmap.

S7A and LOSO panel summaries instead use endpoint baseline MSE pooled across sessions as their normalization denominator. Averaging the Figure 5D percentages would produce a different estimand.

Figure-source tables preserve numerical source precision. The final manuscript table CSVs preserve publication rounding. [Table index and definitions](../tables/README.md).

Run `python scripts/plot_results.py` from the repository root to redraw the supplied numerical panels. Figure 4D uses the final evidence-profile names from main Table 3. No intervals or tests are recomputed from rounded manuscript tables.
