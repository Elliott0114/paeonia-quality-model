#!/usr/bin/env python3
"""Portable entry point for the frozen manuscript models; check/plan never fit."""
import argparse,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def check_inputs(folder):
    import numpy as np
    import pandas as pd
    expected=json.loads((ROOT/'input_manifest.json').read_text())
    for name,h in expected.items():
        p=folder/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:
            raise ValueError('Missing or changed frozen input: '+name)
    t=pd.read_csv(folder/'cohort.csv',float_precision='round_trip')
    ids=t.canonical_id.tolist();assert len(ids)==207 and len(set(ids))==207
    assert t.sample_number.nunique()==69 and (t.groupby('sample_number').size()==3).all()
    core=json.loads((ROOT/'config.json').read_text())['targets']
    y=pd.read_csv(folder/'V4_targets.csv',float_precision='round_trip').set_index('canonical_id').loc[ids,core].to_numpy(float)
    assert np.isfinite(y).sum(0).tolist()==[206,173,207,205,199,206]
    with np.load(folder/'features.npz',allow_pickle=False) as z:
        assert z['canonical_id'].tolist()==ids;x=z['P0'].copy()
    with np.load(folder/'regions.npz',allow_pickle=False) as z:
        assert z['canonical_id'].tolist()==ids;bags=z['spectra'].astype(float);weights=z['weights'].astype(float)
    with np.load(folder/'dispersion.npz',allow_pickle=False) as z:
        assert z['canonical_id'].tolist()==ids;d=z['features'].astype(float)
    assert x.shape==(207,262) and bags.shape==(207,8,262) and d.shape==(207,524)
    assert weights.shape==(207,8) and (weights>0).all()
    assert np.isfinite(x).all() and np.isfinite(bags).all() and np.isfinite(d).all()
    np.testing.assert_array_equal(d[:,:262],x)
    np.testing.assert_allclose(weights.sum(1),1,rtol=0,atol=1e-14)
    np.testing.assert_allclose((bags*weights[:,:,None]).sum(1),x,rtol=2e-7,atol=2e-8)
    raw=json.loads((folder/'frozen_grouped_splits.json').read_text())['grouped']
    assert len(raw)==50 and len({r['split_id'] for r in raw})==50
    index={v:i for i,v in enumerate(ids)}
    convert=lambda values:np.array([index[v] for v in values],dtype=int)
    splits=[];boundaries=0
    for r in raw:
        tr,te=convert(r['train_ids']),convert(r['test_ids'])
        assert len(tr)==len(set(tr)) and len(te)==len(set(te))
        assert not set(tr)&set(te) and set(tr)|set(te)==set(range(207))
        assert not set(t.iloc[tr].sample_number)&set(t.iloc[te].sample_number)
        boundaries+=1;pairs=[]
        assert len(r['inner'])==5
        validation=[]
        for k in r['inner']:
            a,v=convert(k['train_ids']),convert(k['valid_ids'])
            assert not set(a)&set(v) and set(a)|set(v)==set(tr)
            assert not set(t.iloc[a].sample_number)&set(t.iloc[v].sample_number)
            pairs.append((a,v));validation.extend(v.tolist());boundaries+=1
        assert sorted(validation)==sorted(tr.tolist())
        splits.append(SimpleNamespace(split_id=r['split_id'],repeat=r['repeat'],fold=r['fold'],train_idx=tr,test_idx=te,inner_splits=tuple(pairs)))
    for rep in range(1,11):
        assert sorted(i for s in splits if s.repeat==rep for i in s.test_idx)==list(range(207))
    return t,x,y,bags,weights,d,splits,dict(passed=True,grabs=207,batches=69,valid_targets=np.isfinite(y).sum(0).tolist(),outer_folds=50,group_boundaries=boundaries,training_calls=0,inference_calls=0)

def classical(mode,s,t,x,y,d,output):
    import numpy as np
    import pandas as pd
    import base
    from ridge import select_alpha,fit_predict_model
    from region_runner import prediction_rows
    fd=output/'models'/mode/s.split_id
    if fd.exists():raise RuntimeError('Existing output preserved: '+str(fd))
    fd.mkdir(parents=True);features=x if mode=='ridge_p0' else d
    predictions=[];selections=[]
    for j,ep in enumerate(base.CORE):
        alpha,score=select_alpha('hsi_P0',t,features,y[:,j],s.inner_splits,[],(.01,.1,1.,10.,100.,1000.))
        predictions.append(fit_predict_model('hsi_P0',t,features,y[:,j],s.train_idx,s.test_idx,[],alpha))
        score['short']=ep;selections.append(score)
    base.w.write_csv(pd.concat(selections),fd/'selection.csv')
    base.w.write_csv(pd.DataFrame(prediction_rows(t,y,s.test_idx,np.column_stack(predictions),s,mode)),fd/'predictions.csv.gz')
    base.dump(fd/'complete.json',dict(complete=True,prediction_sha256=base.w.digest(fd/'predictions.csv.gz')))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','plan','train'])
    p.add_argument('--inputs',required=True,type=Path,help='model_inputs directory from Additional file2')
    p.add_argument('--output',type=Path,help='New writable directory, outside all supplied inputs')
    p.add_argument('--model',choices=['efficientnet_p0','mil_mean','mil_attention','ridge_p0','ridge_dispersion'],default='efficientnet_p0')
    p.add_argument('--split',default='R01F01',help='One frozen split ID, or all for the50 original outer folds')
    a=p.parse_args();data=check_inputs(a.inputs.resolve());*_,ss,qa=data
    selected=ss if a.split=='all' else [s for s in ss if s.split_id==a.split]
    if not selected:p.error('Unknown frozen split: '+a.split)
    if a.action=='check':print(json.dumps(qa,indent=2));return
    if a.action=='plan':
        print(json.dumps(dict(model=a.model,splits=[s.split_id for s in selected],cnn_max_inner_fits=5*len(selected) if not a.model.startswith('ridge') else 0,cnn_outer_refits=len(selected) if not a.model.startswith('ridge') else 0,epochs_max=160,training_calls=0),indent=2));return
    if a.output is None:p.error('train requires --output')
    output=a.output.resolve();inputs=a.inputs.resolve()
    if output==inputs or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        p.error('Output must not overlap the input directory')
    if output==ROOT or output.is_relative_to(ROOT):p.error('Keep outputs outside the published code snapshot')
    output.mkdir(parents=True,exist_ok=True);os.environ['PAEONIA_OUTPUT']=str(output)
    import base
    # Use the archived fast staging function, whose earlier bitwise equivalence is retained.
    namespace=dict(vars(base));source=(ROOT/'fit_prepermuted.py').read_text()
    exec(compile(source,str(ROOT/'fit_prepermuted.py'),'exec'),namespace)
    base.fit=namespace['fit']
    t,x,y,bags,weights,d,_,_=data
    import region_runner as region
    for s in selected:
        if a.model=='efficientnet_p0':base.run_outer(output,t,x,y,s)
        elif a.model.startswith('ridge'):classical(a.model,s,t,x,y,d,output)
        else:
            for k in range(1,6):region.new_inner(a.model,s,k,t,x,y,bags,weights)
            region.new_outer(a.model,s,t,x,y,bags,weights)

if __name__=='__main__':main()
