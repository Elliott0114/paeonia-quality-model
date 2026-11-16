#!/usr/bin/env python3
"""V4 six-output EfficientNet1D, strict production-batch nested evaluation.

Commands: prepare, pilot, full. Full execution requires a passed pilot record.
Each completed fit is immutable and hash checked before reuse. Existing model
results, raw inputs, manuscripts and environments are never updated in place.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from network import build_model
from paeonia import analysis_io as io, publication_pilot as pp, v3_workflow as w
from project import load_runtime

NAME = 'efficientnet_v4_grouped_20261026'
CORE = ['GA', 'CAT', 'ALB', 'PF', 'GPF', 'BPF']
COUNTS = dict(zip(CORE, [206, 173, 207, 205, 199, 206]))
MODEL = 'efficientnet1d_P0'
RIDGE = 'hsi_P0'
CONFIG = dict(
    name=NAME, reference='V4', targets=CORE, expected_counts=COUNTS,
    model='archived custom EfficientNet1D with six-output head',
    old_weights_reused=False, input='frozen P0, 262 bands',
    input_transform='bandwise training-original mean and population SD; no SNV',
    input_dtype='float32 network, float64 train-only normalization statistics',
    target_transform='per-target training-observed mean and population SD; no log',
    loss='Huber delta=1, missing target mask, equal endpoint expected weight',
    batch_size=16, max_epochs=160, learning_rate=0.002, weight_decay=0.0001,
    optimizer='AdamW, fused=True on MPS', scheduler='OneCycleLR, full 160-epoch schedule',
    augmentation=dict(copies_including_original=6, noise_std=0.0001,
        scale_jitter=0.02, baseline_drift=0.0005,
        application='raw spectrum before normalization; training only; fixed per fit/copy'),
    epoch_selection='minimum pooled inner validation per-endpoint standardized Huber loss; earliest tie',
    outer_refit='fresh initialization on all outer training rows; chosen epoch; same 160-epoch LR schedule',
    split_policy='exact frozen 10 repeats x 5 outer folds x 5 inner folds; production batch grouped',
    outer_test_used_for_selection=False, session_predictors=False, loso_enabled=False,
    training_device='mps', deterministic_algorithms=True, base_seed=42,
    negative_predictions='retained without clipping',
    ridge_reuse='exact existing V4 pure P0 predictions for all 50 identical outer folds',
    evaluation='six endpoints independently scored on their V4 eligibility; no complete-case deletion',
    metrics='mean of ten OOF predictions plus single-repeat metrics; batch bootstrap',
    comparison='paired batch-average repeated squared error; BH across six endpoints',
    manuscript_modified=False, original_model_results_modified=False,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (42,) + parts)).encode()).digest()[:4], 'little') % (2**31 - 1)


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.backends.mps.is_available(), 'This sealed run requires the tested MPS environment.'


def paths():
    runtime = load_runtime()
    project = Path(runtime['project_root']) / '01_白芍质量评价'
    return runtime, project, project / '04_核查记录' / NAME


def environment():
    import scipy, sklearn, matplotlib, joblib
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        machine=platform.machine(), torch=torch.__version__, numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        matplotlib=matplotlib.__version__, joblib=joblib.__version__,
        mps_available=torch.backends.mps.is_available(), torch_threads=1,
        baseline_environment_modified=False, system_site_packages=False)


def prepare():
    runtime, project, out = paths()
    audit = project / '04_核查记录'
    old = audit / 'model_v4_no_session_20261025'
    frozen = Path(runtime['frozen_run'])
    table, _, spectra, _, splits, _, _ = w.load_prepared(frozen)
    for split in splits:
        pp.validate_split(table, split)
    assert len(splits) == 50 and all(len(s.inner_splits) == 5 for s in splits)
    targets = w.read_csv(old / 'inputs/V4_targets.csv').set_index('canonical_id').loc[table.canonical_id, CORE]
    assert targets.notna().sum().to_dict() == COUNTS
    assert np.isfinite(spectra['P0']).all() and spectra['P0'].shape == (207, 262)
    legacy = project / '02_数据与分析/液相-高光谱分析/白芍程序20260513/白芍程序20260513'
    protected = [legacy, frozen, audit / 'model_v4_main_20261024', old,
        audit / 'hplc_reference_v4_final_20261024', audit / 'analysis_configured_baseline_20261022',
        audit / 'v4_separate_ridge_no_session_20261025',
        project / '01_论文与版本/外观与化学质量评价/00_主要稿件']
    sources = {'network': HERE / 'network.py', 'runner': HERE / 'run.py'}
    # Legacy raw HSI cubes live elsewhere; only the legacy model package is read.
    for i, folder in enumerate(protected):
        assert folder.is_dir(), folder
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['.DS_Store', 'Thumbs.db']:
                sources[f'protected_{i}/{p.relative_to(folder)}'] = p
    out, identity = io.create_run(out, runtime['project_root'], CONFIG, sources,
        dict(environment=environment(), preservation_scope=[str(p) for p in protected]))
    inp = out / 'inputs'
    inp.mkdir()
    cols = ['canonical_id', 'sample_number', 'group_index']
    w.write_csv(table[cols], inp / 'cohort.csv')
    w.write_csv(targets.reset_index(), inp / 'V4_targets.csv')
    np.savez_compressed(inp / 'features.npz', canonical_id=table.canonical_id.to_numpy(str), P0=spectra['P0'])
    shutil.copy2(old / 'inputs/frozen_grouped_splits.json', inp / 'frozen_grouped_splits.json')
    original_splits = json.loads((frozen / 'inputs/frozen_splits.json').read_text())['grouped']
    assert json.loads((inp / 'frozen_grouped_splits.json').read_text())['grouped'] == original_splits
    predictions = w.read_csv(old / 'models/predictions.csv.gz')
    ridge = predictions[predictions.model_id.eq(RIDGE) & predictions['mode'].eq('grouped')].copy()
    assert len(ridge) == 12420 and set(ridge.short) == set(CORE)
    w.write_csv(ridge, inp / 'ridge_predictions.csv.gz')
    old_metrics = w.read_csv(old / 'statistics/model_metrics.csv')
    w.write_csv(old_metrics[old_metrics.model_id.eq(RIDGE)], inp / 'ridge_metric_reference.csv')
    for file in ['network.py', 'run.py']:
        (out / 'code').mkdir(exist_ok=True)
        shutil.copy2(HERE / file, out / 'code' / file)
    dump(inp / 'manifest.json', {p.name:w.digest(p) for p in inp.iterdir() if p.is_file()})
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out / 'environment.txt').write_text(freeze)
    dump(out / 'progress.json', dict(stage='prepared', completed_fits=0, total_fits=300, time=now()))
    print(f'Prepared {len(table)} samples, {table.sample_number.nunique()} batches; {len(sources)} protected files.', flush=True)


def check_run(out, full_sources=False):
    identity = json.loads((out / 'run.json').read_text())
    assert identity['config'] == CONFIG
    assert identity['environment'] == environment()
    for key in ['network', 'runner']:
        assert w.digest(HERE / (key + '.py' if key == 'network' else 'run.py')) == identity['source_hashes'][key]
    for key, value in json.loads((out / 'inputs/manifest.json').read_text()).items():
        assert w.digest(out / 'inputs' / key) == value, key
    if full_sources:
        io.check_inputs(identity)
    return identity


def load_inputs(out):
    table = w.read_csv(out / 'inputs/cohort.csv')
    targets = w.read_csv(out / 'inputs/V4_targets.csv').set_index('canonical_id').loc[table.canonical_id, CORE].to_numpy(float)
    with np.load(out / 'inputs/features.npz', allow_pickle=False) as f:
        assert f['canonical_id'].tolist() == table.canonical_id.tolist()
        x = f['P0'].copy()
    raw = json.loads((out / 'inputs/frozen_grouped_splits.json').read_text())['grouped']
    splits = w.restore_splits(raw, table)
    return table, x, targets, splits


def normalization(x, y):
    xmean, xstd = x.mean(0), x.std(0)
    ymean, ystd = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    xstd = np.where(xstd < 1e-6, 1., xstd)
    ystd = np.where(ystd < 1e-6, 1., ystd)
    assert np.isfinite([xmean, xstd]).all() and np.isfinite([ymean, ystd]).all()
    return dict(input_mean=xmean, input_std=xstd, target_mean=ymean, target_std=ystd)


def augment(x, ids, seed):
    """Only receives training originals; each source has one original + five copies."""
    aug = CONFIG['augmentation']
    copies, lineage = [], []
    drift = np.linspace(-1., 1., x.shape[1], dtype=np.float32)
    for i, (row, cid) in enumerate(zip(x, ids)):
        for v in range(aug['copies_including_original']):
            a = row.astype(np.float32).copy()
            s = seed_for(seed, cid, 'augmentation', v)
            if v:
                rng = np.random.default_rng(s)
                a *= np.float32(rng.normal(1., aug['scale_jitter']))
                a += np.float32(rng.normal(0., aug['baseline_drift'])) * drift
                a += rng.normal(0., aug['noise_std'], a.shape).astype(np.float32)
                a = np.clip(a, 0., None)
            copies.append(a)
            lineage.append(dict(canonical_id=cid, source_row=i, variant=v, augmented=bool(v), seed=s))
    return np.stack(copies), pd.DataFrame(lineage)


def masked_huber(pred, target, mask, observed_fraction):
    # A global train-only observation fraction keeps expected component weights
    # equal even when a mini-batch contains more missing CAT values.
    raw = F.huber_loss(pred, target, reduction='none', delta=1.)
    return (raw * mask / observed_fraction).mean()


def predict(model, x, norm):
    model.eval()
    with torch.no_grad():
        z = torch.tensor((x - norm['input_mean']) / norm['input_std'], dtype=torch.float32, device='mps')
        p = model(z).cpu().numpy()
    return p * norm['target_std'] + norm['target_mean']


def verify_completed(folder):
    marker = folder / 'complete.json'
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text())
    for filename, expected in meta['hashes'].items():
        assert w.digest(folder / filename) == expected, folder / filename
    return True


def fit(train_x, train_y, train_ids, valid_x, valid_y, folder, seed, epochs, role):
    """The outer fit never receives outer test features or target values."""
    if verify_completed(folder):
        return
    if folder.exists():
        raise RuntimeError(f'Incomplete fit preserved. Inspect before recovery: {folder}')
    folder.mkdir(parents=True)
    setup(seed)
    start = time.perf_counter()
    norm = normalization(train_x, train_y)
    expanded, lineage = augment(train_x, train_ids, seed)
    source_rows = lineage.source_row.to_numpy(int)
    x = torch.tensor((expanded - norm['input_mean']) / norm['input_std'], dtype=torch.float32, device='mps')
    zy = (train_y[source_rows] - norm['target_mean']) / norm['target_std']
    mask = torch.tensor(np.isfinite(zy), dtype=torch.float32, device='mps')
    y = torch.tensor(np.nan_to_num(zy, nan=0.), dtype=torch.float32, device='mps')
    fraction = mask.mean(0)
    assert (np.isfinite(train_y).sum(0) > 0).all()
    if valid_x is not None:
        vx = torch.tensor((valid_x - norm['input_mean']) / norm['input_std'], dtype=torch.float32, device='mps')
        vy0 = (valid_y - norm['target_mean']) / norm['target_std']
        vm = torch.tensor(np.isfinite(vy0), dtype=torch.float32, device='mps')
        vy = torch.tensor(np.nan_to_num(vy0, nan=0.), dtype=torch.float32, device='mps')
        valid_counts = np.isfinite(vy0).sum(0)
        assert (valid_counts > 0).all()
    model = build_model().to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'], fused=True)
    bs = CONFIG['batch_size']
    steps = math.ceil(len(x) / bs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['max_epochs'], steps_per_epoch=steps)
    shuffle = np.random.default_rng(seed_for(seed, 'shuffle'))
    history = []
    w.write_csv(lineage, folder / 'augmentation_lineage.csv.gz')
    np.savez_compressed(folder / 'normalization.npz', **norm, train_ids=np.asarray(train_ids, str),
        observed_counts=np.isfinite(train_y).sum(0))
    dump(folder / 'fit.json', dict(role=role, seed=seed, epochs=epochs,
        original_train_n=len(train_x), expanded_train_n=len(x), validation_n=0 if valid_x is None else len(valid_x),
        target_counts=np.isfinite(train_y).sum(0).tolist(), train_ids=list(train_ids),
        started_at=now(), outer_test_access_during_fit=False))
    for epoch in range(1, epochs + 1):
        model.train()
        order = shuffle.permutation(len(x))
        loss_sum = torch.zeros((), device='mps')
        for off in range(0, len(x), bs):
            indices = torch.tensor(order[off:off+bs], device='mps')
            optimizer.zero_grad(set_to_none=True)
            loss = masked_huber(model(x[indices]), y[indices], mask[indices], fraction)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(indices)
        item = dict(epoch=epoch, train_huber=float((loss_sum / len(x)).cpu()),
            learning_rate=optimizer.param_groups[0]['lr'])
        if valid_x is not None:
            model.eval()
            with torch.no_grad():
                vloss = (F.huber_loss(model(vx), vy, reduction='none') * vm).sum(0).cpu().numpy()
            item['validation_huber'] = float(np.mean(vloss / valid_counts))
            for j, short in enumerate(CORE):
                item['val_sum_' + short] = float(vloss[j])
                item['val_n_' + short] = int(valid_counts[j])
        assert np.isfinite(list(item.values())).all(), (folder, epoch, item)
        history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            dump(folder / 'progress.json', dict(epoch=epoch, epochs=epochs, seconds=time.perf_counter()-start, time=now()))
    torch.mps.synchronize()
    w.write_csv(pd.DataFrame(history), folder / 'history.csv')
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=state, norm={k:torch.tensor(v) for k,v in norm.items()},
        targets=CORE, epoch=epochs, seed=seed), folder / 'model.pt')
    train_pred = predict(model, train_x, norm)
    np.savez_compressed(folder / 'training_predictions.npz', prediction=train_pred,
        truth=train_y, ids=np.asarray(train_ids, str))
    del model, optimizer, scheduler, x, y, mask
    torch.mps.empty_cache()
    hashes = {p.name:w.digest(p) for p in folder.iterdir() if p.is_file() and p.name not in ['progress.json', 'complete.json']}
    dump(folder / 'complete.json', dict(complete=True, elapsed_seconds=time.perf_counter()-start,
        completed_at=now(), hashes=hashes))
    print(f'{folder.parent.name}/{folder.name}: {epochs} epochs, {time.perf_counter()-start:.1f}s', flush=True)


def run_outer(out, table, x, y, split):
    folder = out / 'models' / split.split_id
    if (folder / 'predictions.csv.gz').exists():
        meta = json.loads((folder / 'outer_complete.json').read_text())
        assert w.digest(folder / 'predictions.csv.gz') == meta['prediction_sha256']
        return
    folder.mkdir(parents=True, exist_ok=True)
    curves = []
    for k, (train, valid) in enumerate(split.inner_splits, 1):
        f = folder / f'inner{k}'
        fit(x[train], y[train], table.iloc[train].canonical_id.tolist(), x[valid], y[valid],
            f, seed_for(split.split_id, 'inner', k), CONFIG['max_epochs'], 'inner_validation')
        curves.append(w.read_csv(f / 'history.csv'))
    losses = np.zeros(CONFIG['max_epochs'])
    for short in CORE:
        sums = sum(c['val_sum_' + short].to_numpy(float) for c in curves)
        counts = sum(c['val_n_' + short].to_numpy(float) for c in curves)
        losses += sums / counts / len(CORE)
    epoch = int(np.argmin(losses)) + 1
    w.write_csv(pd.DataFrame(dict(epoch=np.arange(1, len(losses)+1), pooled_inner_huber=losses)),
        folder / 'epoch_selection.csv')
    dump(folder / 'selection.json', dict(selected_epoch=epoch, inner_folds=5,
        selected_score=float(losses[epoch-1]), rule=CONFIG['epoch_selection'],
        outer_test_used=False, train_ids=table.iloc[split.train_idx].canonical_id.tolist(),
        test_ids=table.iloc[split.test_idx].canonical_id.tolist()))
    tr, te = split.train_idx, split.test_idx
    final = folder / 'outer_refit'
    fit(x[tr], y[tr], table.iloc[tr].canonical_id.tolist(), None, None,
        final, seed_for(split.split_id, 'outer'), epoch, 'outer_refit_no_validation')
    checkpoint = torch.load(final / 'model.pt', map_location='cpu', weights_only=True)
    model = build_model().to('mps')
    model.load_state_dict(checkpoint['state_dict'])
    norm = {k:v.numpy() for k,v in checkpoint['norm'].items()}
    pred = predict(model, x[te], norm)
    del model
    torch.mps.empty_cache()
    assert np.isfinite(pred).all()
    records = []
    for j, short in enumerate(CORE):
        for i, index in enumerate(te):
            records.append(dict(short=short, compound=pp.CN[short], model_id=MODEL,
                split_id=split.split_id, repeat=split.repeat, outer_fold=split.fold,
                canonical_id=table.iloc[index].canonical_id,
                sample_number=int(table.iloc[index].sample_number), group_index=int(table.iloc[index].group_index),
                observed_target=bool(np.isfinite(y[index,j])), y_true=float(y[index,j]),
                y_pred=float(pred[i,j]), selected_epoch=epoch, mode='grouped', reference_version='V4'))
    w.write_csv(pd.DataFrame(records), folder / 'predictions.csv.gz')
    dump(folder / 'outer_complete.json', dict(complete=True, selected_epoch=epoch,
        prediction_rows=len(records), completed_at=now(),
        prediction_sha256=w.digest(folder / 'predictions.csv.gz')))
    print(f'{split.split_id}: outer complete; selected epoch {epoch}; {len(te)} unaugmented test samples.', flush=True)


def execute(stage):
    _, _, out = paths()
    check_run(out, full_sources=True)
    table, x, y, splits = load_inputs(out)
    if stage == 'pilot':
        chosen = [s for s in splits if s.split_id == 'R01F01']
    else:
        gate = json.loads((out / 'pilot/verification.json').read_text())
        assert gate['passed'] is True, 'Full execution requires passed pilot verification.'
        chosen = splits
    with io.run_lock(out):
        start = time.perf_counter()
        for split in chosen:
            pp.validate_split(table, split)
            run_outer(out, table, x, y, split)
            done = len(list((out / 'models').glob('R*/outer_complete.json')))
            dump(out / 'progress.json', dict(stage=stage, completed_outer_folds=done,
                total_outer_folds=50, seconds_this_stage=time.perf_counter()-start, time=now()))
        if stage == 'full':
            frames = [w.read_csv(out / 'models' / s.split_id / 'predictions.csv.gz') for s in splits]
            w.write_csv(pd.concat(frames, ignore_index=True), out / 'models/predictions.csv.gz')
            dump(out / 'models/complete.json', dict(complete=True, outer_folds=50, inner_fits=250,
                final_outer_fits=50, completed_at=now(), prior_predictions_reused=12420))
    check_run(out, full_sources=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'pilot', 'full'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        execute(args.stage)
