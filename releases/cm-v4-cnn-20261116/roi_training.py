"""Train only on supplied groups. Selection and outer prediction live elsewhere."""
import math,time,shutil
from common import *
from roi_network import build
import torch.nn.functional as F

def augment_bags(x,weights,ids,seed):
    ans=[];ww=[];line=[];drift=np.linspace(-1.,1.,x.shape[-1],dtype=np.float32)
    for i,(row,cid) in enumerate(zip(x,ids)):
        for v in range(6):
            a=row.astype(np.float32).copy();s=b.seed_for(seed,cid,'augmentation',v)
            if v:
                rng=np.random.default_rng(s)
                a*=np.float32(rng.normal(1.,.02))
                a+=np.float32(rng.normal(0.,.0005))*drift
                a+=rng.normal(0.,.0001,(x.shape[-1],)).astype(np.float32)
                a=np.clip(a,0.,None)
            ans.append(a);ww.append(weights[i]);line.append(dict(canonical_id=cid,source_row=i,variant=v,augmented=bool(v),seed=s))
    return np.stack(ans),np.asarray(ww),pd.DataFrame(line)

def loss(pred,y,mask,fraction):return (F.huber_loss(pred,y,reduction='none',delta=1.)*mask/fraction).mean()

def predict(model,x,weights,norm,return_attention=False):
    model.eval();pred=[];att=[]
    with torch.no_grad():
        for st in range(0,len(x),16):
            q=torch.tensor((x[st:st+16]-norm['input_mean'])/norm['input_std'],dtype=torch.float32,device='mps')
            w=torch.tensor(weights[st:st+16],dtype=torch.float32,device='mps')
            p,a=model(q,w,True);pred.append(p.cpu().numpy());att.append(a.cpu().numpy())
    result=np.concatenate(pred)*norm['target_std']+norm['target_mean']
    return (result,np.concatenate(att)) if return_attention else result

def fit(mode,train_x,train_weights,train_p0,train_y,train_ids,valid_x,valid_weights,valid_y,folder,seed,epochs,role):
    if b.verify_completed(folder):return
    assert not folder.exists(),f'Incomplete fit preserved: {folder}'
    assert shutil.disk_usage(OUT).free>4*1024**3,'Less than4GB free; preserve fits and stop.'
    folder.mkdir(parents=True);b.setup(seed);start=time.perf_counter()
    norm=b.normalization(train_p0,train_y);expanded,ew,line=augment_bags(train_x,train_weights,train_ids,seed)
    source=line.source_row.to_numpy(int);zy=(train_y[source]-norm['target_mean'])/norm['target_std']
    x=torch.tensor((expanded-norm['input_mean'])/norm['input_std'],dtype=torch.float32,device='mps')
    weights=torch.tensor(ew,dtype=torch.float32,device='mps');mask=torch.tensor(np.isfinite(zy),dtype=torch.float32,device='mps')
    y=torch.tensor(np.nan_to_num(zy,nan=0.),dtype=torch.float32,device='mps');fraction=mask.mean(0)
    model=build(mode).to('mps');initial={k:array_hash(v.cpu().numpy()) for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001,fused=True)
    steps=math.ceil(len(x)/16);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,epochs=160,steps_per_epoch=steps)
    shuffle=np.random.default_rng(b.seed_for(seed,'shuffle'));history=[];vp=[];order_hash=hashlib.sha256()
    write(line,folder/'augmentation_lineage.csv.gz')
    np.savez_compressed(folder/'normalization.npz',**norm,train_ids=np.asarray(train_ids,str),observed_counts=np.isfinite(train_y).sum(0))
    dump(folder/'fit.json',dict(mode=mode,seed=seed,epochs=int(epochs),role=role,train_ids=list(train_ids),original_train_n=len(train_ids),validation_n=0 if valid_x is None else len(valid_x),target_counts=np.isfinite(train_y).sum(0).tolist(),expanded_train_n=len(x),outer_test_access_during_fit=False,steps_per_epoch=steps,actual_updates=steps*int(epochs),max_updates=steps*160,initial_state_hashes=initial,augmentation_hash=array_hash(expanded),started_at=now()))
    for epoch in range(1,int(epochs)+1):
        model.train();order=shuffle.permutation(len(x));order_hash.update(order.tobytes());ix=torch.tensor(order,device='mps')
        xx,yy,mm,ww=x[ix],y[ix],mask[ix],weights[ix];total=torch.zeros((),device='mps')
        for off in range(0,len(x),16):
            sl=slice(off,off+16);opt.zero_grad(set_to_none=True)
            l=loss(model(xx[sl],ww[sl]),yy[sl],mm[sl],fraction);l.backward();opt.step();sched.step();total+=l.detach()*len(xx[sl])
        item=dict(epoch=epoch,train_huber=float((total/len(x)).cpu()),learning_rate=opt.param_groups[0]['lr'])
        if valid_x is not None:
            pred=predict(model,valid_x,valid_weights,norm);vp.append(pred)
            zz=(valid_y-norm['target_mean'])/norm['target_std'];pp=(pred-norm['target_mean'])/norm['target_std']
            err=np.abs(pp-zz);h=np.where(err<1,.5*err**2,err-.5);sums=np.nansum(h,axis=0);counts=np.isfinite(zz).sum(0)
            item['validation_huber']=float(np.mean(sums/counts))
            for j,ep in enumerate(CORE):item['val_sum_'+ep]=float(sums[j]);item['val_n_'+ep]=int(counts[j])
        assert np.isfinite(list(item.values())).all(),(folder,epoch,item)
        history.append(item)
        if epoch%10==0 or epoch==epochs:dump(folder/'progress.json',dict(epoch=epoch,epochs=int(epochs),seconds=time.perf_counter()-start,at=now()))
    torch.mps.synchronize();write(pd.DataFrame(history),folder/'history.csv')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},norm={k:torch.tensor(v) for k,v in norm.items()},epoch=int(epochs),seed=seed,mode=mode),folder/'model.pt')
    if valid_x is not None:np.savez_compressed(folder/'validation_by_epoch.npz',prediction=np.stack(vp),truth=valid_y)
    p=predict(model,train_x,train_weights,norm)
    np.savez_compressed(folder/'training_predictions.npz',prediction=p,truth=train_y,ids=np.asarray(train_ids,str))
    dump(folder/'training_trace.json',dict(shuffle_hash=order_hash.hexdigest()))
    del model,opt,sched,x,y,weights,mask;torch.mps.empty_cache()
    hashes={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name not in ['progress.json','complete.json']}
    dump(folder/'complete.json',dict(complete=True,hashes=hashes,elapsed_seconds=time.perf_counter()-start,completed_at=now()))
    print(f'{folder}: {epochs} epochs in {time.perf_counter()-start:.1f}s',flush=True)

def load_model(folder):
    assert b.verify_completed(folder)
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=build(ck['mode']).to('mps');model.load_state_dict(ck['state_dict'])
    return model,{k:v.numpy() for k,v in ck['norm'].items()}
