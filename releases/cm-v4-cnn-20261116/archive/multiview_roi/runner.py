"""Bounded execution of the frozen five-pipeline contrast."""
import argparse,time,subprocess,concurrent.futures,multiprocessing
from common import *
import roi_training as train
from paeonia.chemical import fit_predict_model
from paeonia.analysis_config import load_config,configured_estimators

def folder(mode,sid):return OUT/'models'/mode/sid

def check_code():
    m=json.loads((OUT/'plan/code_frozen.json').read_text())
    for p,h in m.items():assert digest(Path(p))==h,p

def prediction_rows(t,y,indices,p,s,model,**extra):
    rows=[]
    for j,ep in enumerate(CORE):
        for k,i in enumerate(indices):
            rows.append(dict(model=model,short=ep,split_id=s.split_id,repeat=s.repeat,outer_fold=s.fold,canonical_id=t.iloc[i].canonical_id,sample_number=int(t.iloc[i].sample_number),group_index=int(t.iloc[i].group_index),grade=int(t.iloc[i].recorded_grade),manufacturer=t.iloc[i].manufacturer,y_true=float(y[i,j]),y_pred=float(p[k,j]),observed_target=bool(np.isfinite(y[i,j])),**extra))
    return rows

def ridge(s,t,x,y):
    fd=folder('ridge_dispersion',s.split_id);mark=fd/'complete.json'
    if mark.exists():
        assert digest(fd/'predictions.csv.gz')==json.loads(mark.read_text())['prediction_sha256'];return
    assert not fd.exists(),fd;fd.mkdir(parents=True)
    d=dispersion();rows=[];choices=[];pred=np.zeros((len(s.test_idx),6))
    for j,ep in enumerate(CORE):
        scores=[]
        for alpha in ALPHAS:
            err=[]
            for k,(tr,va) in enumerate(s.inner_splits,1):
                p=fit_predict_model('hsi_P0',t,d,y[:,j],tr,va,[],alpha)
                err.extend(((p-y[va,j])**2)[np.isfinite(y[va,j])])
                rows.extend(dict(short=ep,alpha=alpha,inner=k,canonical_id=t.iloc[i].canonical_id,y_true=float(y[i,j]),y_pred=float(q)) for i,q in zip(va,p))
            scores.append((float(np.mean(err)),alpha))
        score,alpha=min(scores,key=lambda z:(z[0],-z[1]));choices.append(dict(short=ep,selected_alpha=alpha,inner_mse=score))
        pred[:,j]=fit_predict_model('hsi_P0',t,d,y[:,j],s.train_idx,s.test_idx,[],alpha)
    write(pd.DataFrame(rows),fd/'inner_predictions.csv.gz');write(pd.DataFrame(choices),fd/'selection.csv')
    dump(fd/'selection_frozen.json',dict(hash=digest(fd/'selection.csv'),outer_labels_used=False))
    write(pd.DataFrame(prediction_rows(t,y,s.test_idx,pred,s,'ridge_dispersion')),fd/'predictions.csv.gz')
    dump(mark,dict(complete=True,prediction_sha256=digest(fd/'predictions.csv.gz'),at=now()))

def new_inner(mode,s,k,t,x,y,bags,weights):
    tr,va=s.inner_splits[k-1];train.fit(mode,bags[tr],weights[tr],x[tr],y[tr],t.iloc[tr].canonical_id.tolist(),bags[va],weights[va],y[va],folder(mode,s.split_id)/f'inner{k}',b.seed_for(s.split_id,'inner',k),160,'inner_validation')

def new_outer(mode,s,t,x,y,bags,weights):
    fd=folder(mode,s.split_id)
    if (fd/'outer_complete.json').exists():
        assert digest(fd/'predictions.csv.gz')==json.loads((fd/'outer_complete.json').read_text())['prediction_sha256'];return
    hs=[read(fd/f'inner{k}/history.csv') for k in range(1,6)]
    assert all(len(h)==160 for h in hs)
    score=sum(sum(h['val_sum_'+ep].to_numpy() for h in hs)/sum(h['val_n_'+ep].to_numpy() for h in hs)/6 for ep in CORE)
    ep=int(np.argmin(score))+1;write(pd.DataFrame(dict(epoch=np.arange(1,161),pooled_inner_huber=score)),fd/'selection.csv')
    dump(fd/'selection_frozen.json',dict(epoch=ep,hash=digest(fd/'selection.csv'),outer_labels_used=False,at=now()))
    tr,te=s.train_idx,s.test_idx;ff=fd/'outer_refit'
    train.fit(mode,bags[tr],weights[tr],x[tr],y[tr],t.iloc[tr].canonical_id.tolist(),None,None,None,ff,b.seed_for(s.split_id,'outer'),ep,'outer_no_validation')
    model,norm=train.load_model(ff);p,att=train.predict(model,bags[te],weights[te],norm,True)
    write(pd.DataFrame(prediction_rows(t,y,te,p,s,mode,selected_epoch=ep)),fd/'predictions.csv.gz')
    np.savez_compressed(fd/'outer_attention.npz',weights=att,canonical_id=t.iloc[te].canonical_id.to_numpy(str))
    del model;torch.mps.empty_cache()
    dump(fd/'outer_complete.json',dict(complete=True,prediction_sha256=digest(fd/'predictions.csv.gz'),at=now()))

def photo(s,t):
    fd=folder('photo_aligned',s.split_id);mark=fd/'complete.json'
    if mark.exists():assert digest(fd/'predictions.csv')==json.loads(mark.read_text())['hash'];return
    assert not fd.exists();fd.mkdir(parents=True)
    z=np.load(OUT/'inputs/visible.npz');xx=z['features'];cfg=load_config()
    assert z['feature_names'].tolist()==cfg['visible']['features']
    seed=b.seed_for(s.split_id,'photo_aligned');m=configured_estimators(cfg,seed)['hist_gradient_boosting']
    m.fit(xx[s.train_idx],t.iloc[s.train_idx].recorded_grade.to_numpy());p=m.predict(xx[s.test_idx])
    df=t.iloc[s.test_idx].copy();df['predicted_grade']=p;df['repeat']=s.repeat;df['outer_fold']=s.fold;write(df,fd/'predictions.csv')
    dump(fd/'parameters.json',dict(seed=seed,candidate_parameters=cfg['visible']['candidate_parameters']['hist_gradient_boosting'],train_ids=t.iloc[s.train_idx].canonical_id.tolist(),tuning=False))
    dump(mark,dict(complete=True,hash=digest(fd/'predictions.csv')))

def baseline(s,t,y):
    fd=folder('references',s.split_id);mark=fd/'complete.json'
    if mark.exists():assert digest(fd/'predictions.csv.gz')==json.loads(mark.read_text())['hash'];return
    fd.mkdir(parents=True,exist_ok=True);frames=[]
    for name,src in [('ridge_p0',OUT/'inputs/ridge_predictions.csv.gz'),('efficientnet_p0',EFFICIENT/'models'/s.split_id/'predictions.csv.gz')]:
        df=read(src);df=df[df.split_id.eq(s.split_id)].copy();df['model']=name
        assert set(df.canonical_id)==set(t.iloc[s.test_idx].canonical_id)
        df=df.merge(t[['canonical_id','recorded_grade','manufacturer']],on='canonical_id',validate='many_to_one').rename(columns={'recorded_grade':'grade'})
        frames.append(df[[c for c in prediction_rows(t,y,s.test_idx,np.zeros((len(s.test_idx),6)),s,name)[0]]])
    write(pd.concat(frames,ignore_index=True),fd/'predictions.csv.gz');dump(mark,dict(complete=True,hash=digest(fd/'predictions.csv.gz')))

def neural_task(task):
    stage,mode,sid=task;t,x,y,ss=load();bags,weights=spatial();s=next(z for z in ss if z.split_id==sid)
    for k in range(1,6):
        dump(OUT/f'worker_{mode}.json',dict(stage=stage,split_id=sid,mode=mode,phase=f'inner{k}',pid=os.getpid(),at=now()))
        new_inner(mode,s,k,t,x,y,bags,weights)
    new_outer(mode,s,t,x,y,bags,weights)
    return dict(mode=mode,split_id=sid)

def run(stage,jobs=2):
    check_sources();check_code();assert json.loads((OUT/'qa/preflight.json').read_text())['passed']
    if stage=='full':assert json.loads((OUT/'qa/pilot.json').read_text())['passed']
    t,x,y,ss=load();bags,weights=spatial();chosen=ss[:1] if stage=='pilot' else ss
    tasks=[]
    for s in chosen:
        dump(OUT/'progress.json',dict(stage=stage,split_id=s.split_id,phase='classical_and_reuse',at=now()))
        baseline(s,t,y);ridge(s,t,x,y);photo(s,t)
        tasks.extend((stage,mode,s.split_id) for mode in NEW)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as ex:
        futures=[ex.submit(neural_task,z) for z in tasks]
        try:
            for f in concurrent.futures.as_completed(futures):
                done=f.result();completed=sum(all((folder(m,v.split_id)/'outer_complete.json').exists() for m in NEW) for v in ss)
                dump(OUT/'progress.json',dict(stage=stage,completed_outer_folds=completed,total_outer_folds=50,last_completed=done,at=now()))
        except BaseException:
            for f in futures:f.cancel()
            ex.shutdown(wait=True,cancel_futures=True)
            raise
    check_sources();check_code()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','full']);p.add_argument('--jobs',type=int,default=2);a=p.parse_args();run(a.stage,a.jobs)
