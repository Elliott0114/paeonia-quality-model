# Paeonia quality model

## Current Chinese Medicine manuscript — 16 November 2026

The current code entry point is [CM V4 CNN and regional models](releases/cm-v4-cnn-20261116/). It accompanies **Complementary information from appearance grading and hyperspectral chemical profiling of Paeoniae Radix Alba** and contains the actual EfficientNet1D, regional mean/attention trainers, frozen configurations, environment records and explicit run commands. Matching processed inputs and source records are supplied in the manuscript's Additional file 2. No model was retrained for this code publication.

The older examples and results below remain as a historical release. Their figures, seven-constituent and LOSO analyses must not be substituted for the current manuscript's V4 results.

## Historical manuscript release

Code and selected data accompanying **Disentangling appearance and multianalyte chemistry in Paeoniae Radix Alba decoction pieces using visible and hyperspectral imaging**.

This repository provides a **single real sample, final manuscript summary results, and the core analysis methods**. The study comprised 207 samples from 69 production batches. The full image/spectral collection, individual-level cohort tables and trained models are not distributed; full-cohort reproduction is therefore outside this release's scope.

## Run the public examples

```bash
conda env create -f environment.yml
conda activate paeonia-quality-model
python scripts/run_example.py
python scripts/plot_results.py
```

The first command calculates 12 image descriptors, combines them with three stored machine-vision measurements, and computes P0–P3 on 262 retained wavelengths from a measured 292-band ROI mean spectrum. It checks the outputs against supplied reference values. It does not train a model or use HPLC values or expert grade for feature extraction.

The second command redraws Figure 2, Figure 3D–F, Figure 4 and Figure 5 from public aggregate CSVs, producing PNG and PDF files. These preserve the numerical results; the layout is adapted for this standalone release. Whole-figure reproduction of Figure 1 and Figure 3A–C requires additional image/cube/chromatogram inputs. Outputs go to `outputs/`, which is excluded from Git.

## Results and methods

- [Final tables and notes](paper_results/tables/README.md): four main tables, eight supplementary table numbers, 17 supplementary CSV panels and 200 supplementary data rows.
- [Figure data](paper_results/figure_data/README.md): aggregate numerical panels, including all 49 analyte–session combinations in Figure 5D.
- [Methods and function interfaces](docs/METHODS.md), [data dictionary](docs/DATA_DICTIONARY.md), and [manuscript map](docs/MANUSCRIPT_MAP.md).
- [中文说明](README_zh.md).

`paeonia/` contains the image/spectral transformations, grouped model fitting, cross-modal comparisons, batch-label permutation, multianalyte panels and leave-one-session-out methods. These functions accept aligned arrays and tables; model fitting needs the original cohort. Optional R secondary methods are in `r/secondary_methods.R`, with dependency versions in `r/renv.lock`. Python is sufficient for both public commands.

The results distinguish appearance classification from chemical prediction. Cross-session results, unsupported effects and the post hoc status of the CAT-excluded sensitivity panel remain part of the report. The repository does not define a total-quality score.

No license has been added to this release.
