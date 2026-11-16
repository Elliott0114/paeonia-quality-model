# V4 multiview structure and ROI-instance learning

Authorized 2026-11-06: implement the accepted research-upgrade plan using existing
data. No manuscript editing, new laboratory data, reference replacement, new
exclusions, session predictors or LOSO. Original V4/V3 and every old result remain
read-only. This protocol is fixed before any new predictive fits.

## Questions and fixed inputs

1. Which observed variation is shared or individual across appearance, P0 spectra
   and six-component HPLC profiles?
2. Does information within the existing ROI improve chemistry prediction beyond
   its mean spectrum, and does nonlinear instance aggregation help?
3. Are chemical differences within appearance grades recovered at the production
   batch level, and what prediction uncertainty remains?

207 grabs, 69 production batches, original 10 x 5 outer x 5 inner grouped splits;
V4 GA206 CAT173 ALB207 PF205 GPF199 BPF206. Use the same 262 wavelengths and
exposure scaling as P0. All eligibility masks remain unchanged. No dry-mass
correction or actual-net-weight assertion. All missing references remain missing.

## Input representation and computational bounds

Reuse frozen P0 and pixel IQR spectra; Ridge-D uses their concatenation (524
features). For the new bags, partition each frozen foreground ROI into exactly
eight spatially compact, nearly equal-pixel regions by recursive bisection along
the coordinate with the largest span (y breaks ties), with stable coordinate
ordering. This rule sees geometry only, not spectra or labels. Every foreground
pixel belongs to exactly one region. Retain the eight region mean spectra and
pixel-fraction weights. Their weighted mean must reproduce P0 within its stored
float32 rounding. Do not change masks, include labels/background or substitute
raw DN for the existing exposure-scaled vendor signals. Region weights describe
image area, not material mass. Region-level chemical truth is unavailable.

Read original cubes via read-only memory maps; do not duplicate large cubes.
Estimated new data/checkpoints under 2 GB; require 4 GB free before each new fit.
Preserve incomplete fits; do not delete or reseed them. Existing completed fits
are reused after checksums and source identity checks, never silently replaced.

## Five pipelines

1. Ridge-P0: exact existing V4 hsi_P0 predictions.
2. Ridge-D: P0 plus pixel IQR, original alpha grid .01,.1,1,10,100,1000, train-only
   StandardScaler, per-endpoint pooled inner grab RMSE (largest alpha on ties).
3. EfficientNet-P0: exact original six-output EfficientNet1D, original recipe,
   common six-endpoint inner-Huber epoch selection; reuse completed fits/predictions.
4. MIL-mean: same EfficientNet convolutional backbone and 64-unit prediction head;
   encode the eight regions, pixel-fraction-weight their 48-dimensional embeddings,
   then apply the original head. Six outputs jointly trained.
5. MIL-attention: same model with a 48->16->1 tanh attention score, added to log
   pixel weights before softmax. The last attention layer starts at zero, so the
   initial forward pass matches MIL-mean. No per-pixel HPLC supervision.

For both new CNNs use the ORIGINAL fixed recipe, not another search: Huber delta1,
AdamW lr .002/decay .0001, batch16 grabs,160-epoch OneCycle horizon,dropout .2,
existing fit seeds and shuffle, original +5 fixed augmentation copies in training
only. Band normalization uses training original P0 means/population SD so all
instances share the same wavelength scale; target scaling and missingness weights
use training-observed V4 only. Apply each old scale/drift/noise perturbation
coherently across all regions of its grab, retaining original clipping. Augmented
bags are not independent samples. Maximum optimizer updates match the old CNN;
actual chosen duration is reported. Attention initialization uses an isolated RNG
so shared model initialization is identical.

Select one epoch per pipeline/outer fold using the mean of the six pooled
standardized Huber losses over the five original inner validation sets; earliest
tie. Refit outer training from fresh initialization to the selected epoch with
the full160-epoch schedule prefix. No outer labels enter fitting or selection.
The two new pipelines require500 inner fits and100 outer refits, with a two-arm
R01F01 technical gate before the remaining49 folds. Runtime alone may change job
concurrency; model settings, seeds and update budgets cannot change after scores.

The previously tuned/per-endpoint CNN results are context, not identical controls.
This experiment does not simultaneously alter training target or loss unit.

## Joint/individual structure (descriptive, not a deployable predictor)

Use official r.jive2.4 in a separate local R library. Features standardized over
the described cohort, then each block centered and Frobenius-scaled by r.jive.
Primary: batch-mean appearance15, P0, and six HPLC values, on the intersection
with all six observed batch means (expected66). Use method='perm', package default
permutation rank selection, fixed seed20261106, maxiter1000. Report selected
ranks, convergence, joint/individual/residual fractions and reconstruction checks.
These fractions are observed statistical structure, not pure biological signal.

Also fit the same-batch noCAT blocks and the three pairwise block combinations;
JIVE joint structure in a multi-block fit is not automatically evidence that every
block participates equally. Check CAT complete-three-grab batches separately.
Bootstrap200 production batches, fixed selected ranks, for descriptive stability;
do not label these intervals as including rank-selection uncertainty. Associate
latent scores with grade/manufacturer descriptively and noncausally. No clustering
or data-derived quality ranking. All full-cohort transformations stay in this
descriptive branch and never enter predictive training.

## Evaluation and integration with appearance

Primary grain: predicted and observed batch means computed using identical
eligible grabs. Report per-repeat R2,RMSE,MAE,bias,Spearman and predicted/observed
SD; also all grab metrics. Complete50 folds constitute ten repeats of69 batches,
not independent external datasets. Average paired repeated losses within batch
before10000 batch-bootstrap draws. Four new/old contrasts are prespecified:
Ridge-D vs Ridge-P0; MIL-mean vs EfficientNet-P0; MIL-attention vs MIL-mean;
MIL-attention vs Ridge-P0. Intervals are exploratory and conditional on existing
fits, with no unadjusted significance claims after this long exploratory history.

Six/noCAT panels use the SAME six-observable batch intersection; normalize by
each outer-training endpoint SD and give endpoints/batches equal weight. This
is error aggregation, not a quality score. Report CAT complete3 vs incomplete1-2,
grades, manufacturer groups (small groups explicitly descriptive), and low/middle/
high concentration defined by outer-training batch tertiles. Report all six
components including harms. Within-grade recovery includes centered ranks,
variation amplitude and comparison with training-batch grade-typical chemistry.
Pairwise chemical-distance analyses resample batches, not dependent sample pairs.

Reuse photo OOF grades only after matching outer test-batch membership. Keep
known-expert-grade diagnostics separate from photo-predicted-grade diagnostics;
the latter includes classification error. HPLC is reference/supervision, never
an input silently supplied at deployment. Produce a joint descriptive table,
not a new weighted quality grade or a trained grade-fusion network.

Pre-fit alignment inspection found0/50 matching outer memberships between old
appearance and chemical partitions. Therefore retain historical appearance
results as context and refit only the existing fixed hist_gradient_boosting
configuration on each chemical outer training set, using the existing15 features
and no tuning. Save50 aligned photo predictions and parameters. This necessary
alignment is not a new classification family or a reported improvement over the
old independently partitioned Mac result.

## Prediction intervals: separate calibrated pipeline

To avoid using calibration labels in epoch/alpha selection, reuse the fifth inner
split as proper-training/calibration in each original outer fold. For this
SEPARATE uncertainty arm fix CNN epoch160 and Ridge alpha1000 a priori for all
five pipelines. Reuse inner5 checkpoints; no outer-trained selected checkpoint is
given a spurious conformal guarantee. Calibrate absolute batch-mean residuals on
the held-out inner5 calibration batches; predict original outer-test batches.
For each endpoint use the split-conformal ceil((m+1)*.90) order statistic, including
infinity if m is insufficient. Report actual calibration counts, coverage,width,
point error and empirical selective-risk curves. Intervals are marginal per
endpoint, not simultaneous six-component coverage or subgroup guarantees.
The interval width is constant within an endpoint/fold and is not an individual
confidence score. For descriptive risk-versus-retention curves only, use distance
to the nearest proper-training batch spectrum in proper-training standardized
P0 space. That target-free distance is fixed before fitting; it is not a coverage
guarantee or a validated prospective rejection threshold.
Exchangeability across production batches is an assumption, not established by
this observational cohort. Clearly distinguish this fixed calibrated pipeline
from the primary nested-selected pipelines and do not mix their metrics.

## Gates, preservation and delivery

Before prediction: validate all300 group boundaries; held-out perturbation leaves
training normalization unchanged; missing-target gradient zero; coherent training-
only augmentation; regional-weight reconstruction; synthetic attention initial
equivalence, permutation invariance and one-instance equivalence to old backbone;
checkpoint replay and one-full-outer-fold inner-selection independent recomputation.
R01F01 gate is technical correctness, not improved outer performance.

Independent statistical verification recomputes key metrics and checks counts,
denominators, CI sign/scale, conformal quantiles and all input/source protection.
Write separate code, inputs, results, source hashes, Chinese Data report/preview,
completion receipts and Git records. No manuscripts, V4/V3 or old outputs change.
Background continuation is authorized by the accepted plan and standing user
preference; notify completion or actual failure, not ordinary training progress.
