# Manuscript tables

Four main tables and eight supplementary table numbers are provided. Tables with different panel schemas use separate CSV files: 17 supplementary files, 200 data rows. CSV files use UTF-8 with BOM. Values retain manuscript reporting precision; confidence intervals and inequality signs are preserved as text.

## Main tables

| Table | Subject | Data rows |
|---|---|---|
| [Table 1](Table_1.csv) | Cohort and HPLC eligibility | 8 |
| [Table 2](Table_2.csv) | Visible appearance performance | 7 |
| [Table 3](Table_3.csv) | Analyte-specific HSI evidence | 7 |
| [Table 4](Table_4.csv) | Multianalyte panel increments | 8 |

## Supplementary tables

| Table / panel | Subject | Data rows |
|---|---|---|
| [Table S1](Table_S1.csv) | Reference standards and chromatographic peak identities | 8 |
| [Table S2A](Table_S2A.csv) | Measurement availability and calibration-range eligibility | 8 |
| [Table S2B](Table_S2B.csv) | CAT measurement availability by HPLC calibration series | 3 |
| [Table S2C](Table_S2C.csv) | PGG concentrations relative to calibration-range upper limits | 2 |
| [Table S3A](Table_S3A.csv) | Within-batch chemical variation | 7 |
| [Table S3B](Table_S3B.csv) | Appearance–chemistry associations | 14 |
| [Table S4](Table_S4.csv) | Endpoint-specific model performance | 28 |
| [Table S5A](Table_S5A.csv) | Acquisition domain → acquisition domain + residual HSI | 7 |
| [Table S5B](Table_S5B.csv) | Appearance → appearance + HSI | 7 |
| [Table S5C](Table_S5C.csv) | Domain + appearance → domain + appearance + residual HSI | 7 |
| [Table S5D](Table_S5D.csv) | HSI → HSI + appearance | 7 |
| [Table S6](Table_S6.csv) | Leave-one-analyte sensitivity of the domain-conditioned multianalyte panel | 7 |
| [Table S7A](Table_S7A.csv) | Pooled endpoint performance and cross-session criterion | 7 |
| [Table S7B](Table_S7B.csv) | Endpoint- and session-specific prediction errors | 49 |
| [Table S8A](Table_S8A.csv) | P0–P3 preprocessing sensitivity under identical grouped validation | 28 |
| [Table S8B](Table_S8B.csv) | Variation across 10 repeated batch-grouped validations | 8 |
| [Table S8C](Table_S8C.csv) | Visible prediction agreement and observed ensemble accuracy | 3 |

## Reading the tables

- GA: gallic acid; OPF: oxypaeoniflorin; CAT: catechin; ALB: albiflorin; PF: paeoniflorin; GPF: galloylpaeoniflorin; BPF: benzoylpaeoniflorin; PGG: pentagalloylglucose.
- In S4, all four metrics refer to the same analyte/model row. In S8A, both metrics refer to the same analyte/preprocessing row.
- S6 uses distinct column names for baseline-MSE and target-variance normalization. Its p values are unadjusted; main Table 4 reports Holm-adjusted panel tests.
- S5A–D are four separate seven-analyte BH families. Absolute MSE differences and their confidence intervals are in (mg/g)².
- S7A normalizes the equally weighted session differences by each endpoint's pooled baseline MSE. S7B normalizes each session difference by that session's baseline MSE. These are different summaries.
- S7B MSE values are displayed to three significant figures. Percentages are derived from unrounded scores, so recomputing them from displayed cells can differ.
- S8B gives medians and ranges across repeats, not confidence intervals for aggregated predictions.

## Table notes

### Table S1

All standards were obtained from Shanghai Yuanye Bio-Technology Co., Ltd. (Shanghai, China) and stored at 2–8°C. Purities are supplier-reported HPLC purities. Peak numbers and retention times refer to the first-series mixed-standard chromatogram in Fig. 1D.

### Table S2A

The study comprised 207 samples. The retained range spans the calibration levels used for fitting; the all-level range spans all matched calibration levels. Only measurements within the retained range were eligible for quantitative modeling. Eligible batch counts and complete triplets are reported in main-text Table 1. PGG had no eligible measurements.

### Table S2B

Eligible measurements were within the retained calibration range. Missing percentages use the sample count of the corresponding series as the denominator.

### Table S2C

Ratios use back-calculated extract concentrations and the upper calibration limit from the assigned series, both in mg/mL. All 206 measured PGG results exceeded the retained upper limits; 18 fell within the range spanning all matched calibration levels.

### Table S3A

Values summarize median within-batch ranges. Normalized ranges divide each within-batch range by the dataset-wide analyte IQR. A complete triplet contains three eligible samples from one batch. CIs use batch-cluster bootstrap resampling with the IQR denominator held fixed.

### Table S3B

Higher numerical appearance grades indicate poorer appearance. Correlations use analyte-specific eligible samples; CIs use batch-cluster resampling. Permutation p values are two-sided. BH correction is applied separately across seven analytes for each appearance measure.

### Table S4

RMSE and MAE are expressed in mg/g of weighed material. Metrics are calculated from aggregated OOF predictions, with batch-cluster bootstrap CIs. The four configurations are HSI P0, the training-fold mean, acquisition domain, and domain plus residual HSI. Conditional comparisons involving appearance predictors are reported in Table S5.

### Table S6

Each row compares acquisition domain with acquisition domain plus HSI for the six remaining analytes, using equal endpoint and batch weights. The CAT-excluded baseline-MSE result is the same comparison reported in main-text Table 4. The displayed sign-flip p values are unadjusted; main-text Table 4 reports Holm-adjusted panel tests.

### Table S7A

All endpoints include seven held-out sessions. R² CIs use batch-cluster resampling. Session-equal MSE reduction is normalized by the endpoint-wide baseline MSE pooled across held-out sessions. The criterion requires positive pooled R² with its CI above zero and positive session-equal error reduction with BH q < 0.05. Exact tests enumerate all 2⁷ session-level sign assignments; BH correction covers seven endpoints.

### Table S7B

MSE is expressed in (mg/g)² and shown to three significant figures. The baseline predicts the mean eligible target content of the training sessions. Relative MSE reduction is 100 × (MSEbaseline − MSEHSI)/MSEbaseline, using the baseline MSE of each held-out session. S1–S7 in this panel denote acquisition sessions.

### Table S8A

P0, primary exposure-rescaled mean spectrum; P1, SNV; P2, smoothing + SNV; P3, first derivative of the smoothed spectrum + SNV (Section 3). Estimates use aggregated OOF predictions and batch-cluster bootstrap CIs.

### Table S8B

Values are the median and minimum–maximum range of 10 repeat-specific performance estimates. They are distinct from metrics and CIs calculated from aggregated OOF predictions.

### Table S8C

Agreement is the modal-grade frequency across 10 OOF predictions. Accuracy is summarized within each agreement stratum, with CIs from 10,000 batch-cluster bootstrap resamples (Fig. 3F).

## Supplementary results reported in text

The exploratory CAT within-batch comparison uses 92 contrasts from 58 batches. The domain and domain-plus-HSI contrast models have R² = 0.355 (95% CI, 0.180–0.494) and 0.444 (0.238–0.597), respectively. The paired mean batch MSE reduction is −0.0011 (95% CI, −0.0093 to 0.0066) (mg/g)², p = 0.613; an incremental within-batch benefit was not supported.

The seven-analyte LOSO panel effect is −11.8% (95% CI, −41.7% to 12.8%; exact p = 0.750). The six-analyte panel excluding CAT has an effect of −7.0% (−32.1% to 14.4%; p = 0.688). Panel denominators pool baseline MSE within each endpoint across held-out sessions; confidence intervals resample sessions.
