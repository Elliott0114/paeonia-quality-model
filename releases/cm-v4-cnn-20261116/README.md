# Chinese Medicine V4 CNN and regional model code

This version accompanies **Complementary information from appearance grading and hyperspectral chemical profiling of Paeoniae Radix Alba**. It exposes the actual six-output EfficientNet1D and eight-region mean/attention models used as absolute-content controls in the manuscript. It does not replace the V4 reference values or historical results.

The three CNNs share 207 grabs from 69 production batches, V4 targets GA/CAT/ALB/PF/GPF/BPF, 262 P0 bands and the original 10 ×5 outer ×5 inner grouped partitions. Original plus five fixed augmented copies are used in training only. Six targets train jointly, and one common epoch is selected from pooled inner validation losses. These are not the 197-sample historical CNN or the later independently selected endpoint experiments. No ratio CNN was trained for the manuscript.

## Files and provenance

- `network.py`, `backbones.py`, `roi_network.py`, and `roi_training.py`: byte-identical archived architecture and regional training files.
- `base.py`: original P0 normalization, augmentation, loss, seed, fit and outer-selection functions, extracted without changing their function text.
- `fit_prepermuted.py`: the actual archived full-run P0 trainer. It gathers the seeded row permutation once per epoch; the earlier execution-equivalence record is preserved under `provenance/`.
- `region_runner.py`, `region_partition.py`, `ridge.py`: unchanged scientific functions extracted from the original run. Ridge comparators use the retained alpha grid and grouped inner RMSE rule.
- `common.py`, `storage.py`, `run.py`: directory/I/O bindings, frozen-input checks and explicit command-line entry points. They avoid importing the author's old experiment directory tree.
- `config.json`, `environment.json`, `requirements.lock`: frozen design and actual recorded environment, not newly optimized settings.
- `provenance/source_map.json`: originating file and function hashes. `archived_outer_selections.csv` contains all 150 saved model/fold epoch selections; no selections were re-estimated for this release.
- `archive/`: original orchestration, region extraction, scoring and separate interval scripts, retained as source records. Their historical filesystem bindings are documented and not the portable entry point.

The adapters were checked by source/function hashes, input and grouped-split validation, imports and existing selection metadata. No training, model inference, bootstrap or scientific-result update was performed while preparing this code release. Full numerical rerun equivalence of the packaging adapters has not been newly measured.

## Environment

The recorded training environment is Python 3.11.17 on Apple Silicon macOS 27.0.1, PyTorch 2.14.1 with MPS, NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.1 and scikit-learn 1.8.0. The complete recorded Python package list is `requirements.lock`. Install in a separate environment; do not overwrite a frozen research environment:

```bash
python3.11 -m venv .venv-cm-v4
source .venv-cm-v4/bin/activate
python -m pip install -r requirements.lock
```

MPS, fused AdamW and deterministic algorithms are intentionally retained. A CPU/CUDA substitution is a different execution environment and has not been established to reproduce the archived numbers. Installing these pinned packages was not repeated during packaging. Raw cubes and model checkpoints are not uploaded to this GitHub release.

## Inputs

Extract `model_inputs/` from the manuscript's **Additional file 2**. It contains the six byte-preserved files listed in `input_manifest.json`:

| File | Contents |
|---|---|
| cohort.csv | Ordered canonical grab IDs, production batch, grab index, consensus grade and manufacturer |
| V4_targets.csv | Six adopted mg/g targets, missing values retained |
| features.npz | Ordered IDs and 207 ×262 original P0 matrix |
| regions.npz | Ordered IDs,207 ×8 ×262 region spectra,207 ×8 pixel-fraction weights and wavelengths |
| dispersion.npz | Ordered IDs and 207 ×524 concatenated P0/IQR features |
| frozen_grouped_splits.json | Exact50 outer partitions and all 250 inner partitions in original ID order |

These processed scientific inputs are supplied with the manuscript, not committed to the public code tree. The command stops if a hash differs, a batch crosses a train/test boundary, a target count changes, or regional spectra do not recompose the archived P0. No new splitting, filtering, exposure rescaling, dry-weight conversion or target adjustment occurs here.

## Commands

From this directory, check inputs and inspect a run plan without training:

```bash
python run.py check --inputs /path/to/Additional_file_2/model_inputs
python run.py plan --inputs /path/to/Additional_file_2/model_inputs --model mil_attention --split R01F01
```

The following commands **would train models** and were not executed for this release. Outputs must be in a new directory separate from inputs and the code snapshot:

```bash
python run.py train --inputs /path/to/model_inputs --output /path/to/new/eff --model efficientnet_p0 --split R01F01
python run.py train --inputs /path/to/model_inputs --output /path/to/new/mean --model mil_mean --split R01F01
python run.py train --inputs /path/to/model_inputs --output /path/to/new/attention --model mil_attention --split R01F01
```

Use `--split all` for the original 50 folds, and `--model ridge_p0` or `--model ridge_dispersion` for the two Ridge comparators. The primary manuscript reused completed P0 Ridge/EfficientNet predictions; this entry point allows explicit future reproduction. It does not silently replace those archived predictions. Completed CNN fits are hash-checked before reuse and incomplete fits are preserved. No command starts a background monitor.

## Relationship to the manuscript

Figure 5 and the CNN part of Table S4 use these absolute-content comparisons. Performance is summarized across complete repeats; batch-mean errors and per-grab errors have different units. Six/noCAT panels share the same 66 eligible batches and use training-fold target variance for standardization. The original scoring implementation is `archive/multiview_roi/analyze.py`; its existing outputs remain in Additional file 2.

Figure 6's intervals use a **separate fixed-model arm**, not the selected outer refits: CNN 160 epochs, Ridge alpha 1000, fifth inner split as proper-training/calibration. `archive/multiview_roi/intervals.py` preserves this code and its original source bindings. The present portable `train` command implements the main nested-selected models; it does not claim to regenerate intervals, JIVE, photo grades or manuscript figures automatically. Those archived result files have not changed.

Ratio/G-adjusted prediction summaries remain exploratory and are not generated by this CNN entry point. HPLC preliminary-experiment numerical records remain pending; this code packaging supplies no additional laboratory evidence.

## Citation

Cite the immutable GitHub commit specified in the revised manuscript and `CODE_VERSION.json` in Additional file 2. The repository root's older examples, tables and LOSO analysis describe the historical manuscript. This directory is the entry point for the current CNN controls.
