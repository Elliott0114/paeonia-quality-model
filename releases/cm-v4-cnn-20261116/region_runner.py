from common import *
import roi_training as train

def folder(mode,sid):return OUT/'models'/mode/sid


def prediction_rows(t,y,indices,p,s,model,**extra):
    rows=[]
    for j,ep in enumerate(CORE):
        for k,i in enumerate(indices):
            rows.append(dict(model=model,short=ep,split_id=s.split_id,repeat=s.repeat,outer_fold=s.fold,canonical_id=t.iloc[i].canonical_id,sample_number=int(t.iloc[i].sample_number),group_index=int(t.iloc[i].group_index),grade=int(t.iloc[i].recorded_grade),manufacturer=t.iloc[i].manufacturer,y_true=float(y[i,j]),y_pred=float(p[k,j]),observed_target=bool(np.isfinite(y[i,j])),**extra))
    return rows


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
