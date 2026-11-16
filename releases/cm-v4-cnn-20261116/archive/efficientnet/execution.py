#!/usr/bin/env python3
"""Equivalent MPS data staging, without changing the sealed scientific trainer.

Copy the same seeded permutation to the GPU once per epoch, then slice the
result. The reference trainer gathers the same rows separately per mini-batch.
Full execution is allowed only after all pilot outer weights reproduce exactly.
The reference runner and its completed pilot are preserved unchanged.
"""
import argparse
import inspect
import json
import time
import torch
import numpy as np
import run as r


def make_fit():
    source=inspect.getsource(r.fit)
    replacements=[
        ('        order = shuffle.permutation(len(x))\n',
         '        order = shuffle.permutation(len(x))\n'
         '        epoch_index = torch.tensor(order, device="mps")\n'
         '        epoch_x, epoch_y, epoch_mask = x[epoch_index], y[epoch_index], mask[epoch_index]\n'),
        ("            indices = torch.tensor(order[off:off+bs], device='mps')\n",
         '            indices = slice(off, off+bs)\n'),
        ('            loss = masked_huber(model(x[indices]), y[indices], mask[indices], fraction)\n',
         '            loss = masked_huber(model(epoch_x[indices]), epoch_y[indices], epoch_mask[indices], fraction)\n'),
        ('            loss_sum += loss.detach() * len(indices)\n',
         '            loss_sum += loss.detach() * len(epoch_x[indices])\n'),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    namespace=dict(vars(r))
    exec(compile(source,str(r.HERE/'execution.py')+'::equivalent_fit','exec'),namespace)
    return namespace['fit'],source


def equivalence():
    _,_,out=r.paths()
    r.check_run(out,full_sources=True)
    table,x,y,splits=r.load_inputs(out)
    split=next(s for s in splits if s.split_id=='R01F01')
    old=out/'models/R01F01/outer_refit'
    oldstate=torch.load(old/'model.pt',map_location='cpu',weights_only=True)
    fast,source=make_fit()
    new=out/'pilot/optimized_outer_replay'
    fast(x[split.train_idx],y[split.train_idx],table.iloc[split.train_idx].canonical_id.tolist(),
        None,None,new,r.seed_for(split.split_id,'outer'),oldstate['epoch'],'numerical_equivalence_only')
    newstate=torch.load(new/'model.pt',map_location='cpu',weights_only=True)
    diffs={k:float(torch.max(torch.abs(a-newstate['state_dict'][k])).item())
           for k,a in oldstate['state_dict'].items()}
    assert max(diffs.values())==0.,max(diffs.values())
    for key in oldstate['norm']:
        assert torch.equal(oldstate['norm'][key],newstate['norm'][key])
    a=r.w.read_csv(old/'history.csv');b=r.w.read_csv(new/'history.csv')
    np.testing.assert_array_equal(a.to_numpy(),b.to_numpy())
    old_seconds=json.loads((old/'complete.json').read_text())['elapsed_seconds']
    new_seconds=json.loads((new/'complete.json').read_text())['elapsed_seconds']
    (out/'code/fit_prepermuted.py').write_text(source)
    r.dump(out/'pilot/execution_equivalence.json',dict(passed=True,
        all_state_tensors_bitwise_identical=True,all_epoch_loss_and_lr_values_identical=True,
        real_pilot_outer_epochs=oldstate['epoch'],real_pilot_outer_train_n=len(split.train_idx),
        original_seconds=old_seconds,prepermuted_seconds=new_seconds,
        throughput_ratio=old_seconds/new_seconds,max_state_abs_difference=max(diffs.values()),
        generated_fit_sha256=r.w.digest(out/'code/fit_prepermuted.py'),
        execution_script_sha256=r.w.digest(r.HERE/'execution.py'),
        evaluation_predictions_not_used_for_choice=True))
    print((out/'pilot/execution_equivalence.json').read_text())


def full():
    _,_,out=r.paths()
    gate=json.loads((out/'pilot/execution_equivalence.json').read_text())
    assert gate['passed'] and gate['all_state_tensors_bitwise_identical']
    assert gate['execution_script_sha256']==r.w.digest(r.HERE/'execution.py')
    r.fit,source=make_fit()
    assert source==(out/'code/fit_prepermuted.py').read_text()
    identity=dict(reference_runner_sha256=r.w.digest(r.HERE/'run.py'),
        execution_script_sha256=r.w.digest(r.HERE/'execution.py'),
        generated_fit_sha256=r.w.digest(out/'code/fit_prepermuted.py'),
        difference='GPU row gathering once per epoch instead of once per mini-batch',
        changed_training_rules=False,parallel_jobs=1)
    record=out/'execution_patch.json'
    if record.exists():assert json.loads(record.read_text())==identity
    else:r.dump(record,identity)
    r.execute('full')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['equivalence','full'])
    args=ap.parse_args()
    if args.stage=='equivalence':equivalence()
    else:full()
