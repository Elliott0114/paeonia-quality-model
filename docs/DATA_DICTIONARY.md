# Data dictionary

## Public example

`examples/SBS-01-G1/` contains one physical portion from one production batch.

| File | Content |
|---|---|
| `visible.png` | Original RGB pixels stored as a lossless 5120 × 3840 PNG |
| `metadata.json` | Portion/batch/position identifiers, reference grade/session, exposure times and processing settings |
| `vendor_mean_spectrum_292.csv` | Measured ROI mean: `wavelength_nm`, `vendor_roi_mean` |
| `archived_machine_features.csv` | Three stored machine-vision measurements |
| `expected_visible15.csv` | Expected image and stored-machine feature values |
| `expected_preprocessing_262.csv` | Expected P0–P3 outputs; decimal values are rounded for CSV storage |
| `hplc_reference.csv` | Eight analyte reference measurements and calibration-range eligibility for this portion |
| `processing_preview.png` | Illustration of single-sample spectral processing |

The original HSI cube and ROI masks are not included. The demonstration starts from the measured ROI mean. The image is resized to a longest side of 2048 pixels for analysis; the minimum foreground component area is 64 analysis pixels. HPLC content and grade are reference metadata, never feature-extraction inputs.

Image comparisons allow eight float32 epsilons of relative difference and one epsilon of absolute difference; stored machine measurements must match exactly. Spectral comparisons use rtol 1e-8 and atol 1e-9; wavelength headings allow 1e-5 nm because the reference headings are rounded.

## Visible15 fields

| Feature | Interpretation | Source |
|---|---|---|
| `machine_average_area_cm2` | Mean piece area, cm² | Archived machine vision |
| `machine_short_axis_lt_0p8_pct` | Pieces with short axis below 0.8 cm, percent (0–100) | Archived machine vision |
| `machine_abnormal_piece_pct` | Abnormal pieces, percent (0–100) | Archived machine vision |
| `visible_mask_component_area_cv` | Component-area coefficient of variation | Image computation |
| `visible_lab_L_mean` | Mean CIE Lab lightness | Image computation |
| `visible_lab_a_mean` | Mean CIE Lab a coordinate | Image computation |
| `visible_lab_b_mean` | Mean CIE Lab b coordinate | Image computation |
| `visible_color_delta_e_fraction_gt_12` | Foreground fraction with color difference above 12 | Image computation |
| `visible_object_L_between_std` | Between-piece lightness variation | Image computation |
| `visible_relative_dark_spot_area_fraction` | Relative dark-region area fraction | Image computation |
| `visible_edge_minus_inner_L_mean` | Edge minus interior mean lightness | Image computation |
| `visible_texture_local_L_std_w5_mean` | Local lightness variation at window 5 | Image computation |
| `visible_texture_local_L_std_w33_mean` | Local lightness variation at window 33 | Image computation |
| `visible_texture_coarse_to_fine_ratio` | Coarse/fine texture variation ratio | Image computation |
| `visible_glcm_entropy` | Gray-level co-occurrence entropy | Image computation |

## Spectra

The signal is vendor-exported data, not absolute reflectance. Rescaling uses `white_exposure_us / sample_exposure_us`. No maximum normalization or clipping is introduced.

| Branch | Processing |
|---|---|
| P0 | Exposure-adjusted ROI mean; retain 430.422943–969.807861 nm |
| P1 | Row-wise SNV of P0, population standard deviation |
| P2 | Savitzky–Golay smoothing on all 292 bands (window 11, polynomial order 2), select the 262-band interval, then SNV |
| P3 | Wavelength-aware first derivative of the full-band SG-smoothed signal (`np.gradient`, edge order 2), select the interval, then SNV |

Retaining all 292 input bands matters for reproducing P2/P3 near the interval edges. The expected CSV has rounded wavelength headings, hence its comparison tolerance of 0.00001 nm; the calculation uses the original numeric wavelength array. No analyte chooses its preferred preprocessing branch.

## Cohort interfaces for core methods

Full-cohort data are not distributed. Functions in `paeonia/` use aligned in-memory tables and arrays, with one row per physical portion and a consecutive zero-based DataFrame index.

| Field | Meaning |
|---|---|
| `canonical_id` | Unique portion identifier |
| `sample_number` | Production batch; the grouping/resampling unit |
| `group_index` | Within-batch position G1–G3, not a quality grade |
| `session_code` | HSI acquisition session S1–S7 |
| `hplc_calibration_group` | Assigned HPLC calibration series |
| `expert_appearance_grade` | Recorded batch appearance anchor, inherited by its portions |
| `visible` array | Columns in `paeonia.visible.VISIBLE15`, in that exact order |
| `spectra["P0"]` | Exposure-rescaled mean signal on the retained 262 wavelengths |
| Target arrays | Analyte contents in mg/g; missing/out-of-range targets remain NaN |

GA, OPF, CAT, ALB, PF, GPF and BPF enter eligible endpoint models. PGG has no retained-range-eligible results in this cohort. [Analyte mapping](compound_to_metric_mapping.csv) supplies the source HPLC names. Eligibility is evaluated in extract-concentration units against the assigned retained calibration range; the study's content-to-concentration conversion is mg/g divided by 50 to obtain mg/mL.

`chemical.CohortData(table, spectra, compounds)` supports reference eligibility and CAT within-batch contrasts. `chemical.calibration_ranges(points_path, models_path)` reads calibration CSVs: points require `group_id`, `compound`, `level`, `concentration_mg_mL`; model records require `group_id`, `成分`, `保留level`, `R²`. Its output supplies retained and all-level concentration limits to `build_reference_qualification`. Targets are not imputed.

R secondary methods accept the eligibility detail joined to `compound_cn`, `compound`, `short`, `plot_order`. Visible predictions require `canonical_id`, `sample_number`, `group_index`, `expert_appearance_grade`, `cross_fitted_visible_score`; the latter is the mean grade prediction across outer repeats.

## Published results

Tables in `paper_results/tables/` use manuscript precision. Figure CSVs retain numerical source precision. Neither directory contains individual-level cohort predictions. Units, denominators and resampling definitions are given in the [table notes](../paper_results/tables/README.md) and [methods](METHODS.md).
