# Manuscript-to-code map

| Manuscript content | Core methods / public results |
|---|---|
| Table 1; S1–S2 | HPLC eligibility in `paeonia.chemical`; final standard/eligibility CSVs |
| Table 2; Figure 3D–F | `paeonia.visible`, `paeonia.permutation`; aggregate classification CSVs |
| Table 3; S4–S5; Figure 4 | `paeonia.chemical`, `paeonia.cross_modal`, `paeonia.multianalyte`; endpoint metrics/increments |
| Table 4; S6; Figure 5A–B | `paeonia.panels`, `paeonia.multianalyte`; equal-weight panel effects |
| S3; Figure 2C–D | `r/secondary_methods.R`; batch heterogeneity and appearance–chemistry associations |
| S7; Figure 5C–D | `paeonia.loso`; pooled and session-specific results |
| S8 | `paeonia.spectra`, chemical/visible fitting methods; final sensitivity/variation CSVs |
| Supplementary CAT exploratory text | `paeonia.chemical.build_catechin_contrasts` and contrast inference; [reported values](../paper_results/tables/README.md#supplementary-results-reported-in-text) |

`scripts/run_example.py` processes the supplied sample. `scripts/plot_results.py` reads only the public aggregate CSVs and redraws Figure 2, Figure 3D–F, Figure 4 and Figure 5. Whole Figure 1 and representative image panels Figure 3A–C need additional original inputs.

Final [table CSVs](../paper_results/tables/README.md) preserve manuscript precision and S1–S8 numbering; [figure data](../paper_results/figure_data/README.md) preserve numerical precision used in plotting. The two representations need not have identical rounding.
