"""Separate fixed-model split-conformal analysis; no calibration-label tuning."""
from common import *
from runner import prediction_rows,check_code
from paeonia.chemical import fit_predict_model
import roi_training as trn
from scipy.spatial.distance import cdist

def main():
    check_sources();check_code();assert json.loads((OUT/'qa/full.json').read_text())['passed']
    t,x,y,ss=load();bags,w=spatial();d=dispersion();root=OUT/'uncertainty';root.mkdir(exist_ok=True)
    for s in ss:
        fd=root/s.split_id;mark=fd/'complete.json'
        if mark.exists():
            for n,h in json.loads(mark.read_text())['hashes'].items():assert digest(fd/n)==h
            continue
        fd.mkdir(exist_ok=True);a,v=s.inner_splits[4];te=s.test_idx;ii=np.r_[v,te]
        assert set(t.iloc[a].sample_number).isdisjoint(t.iloc[ii].sample_number)
        n=b.normalization(x[a],y[a]);z=(x-n['input_mean'])/n['input_std']
        proto=np.stack([z[t.sample_number.eq(k)].mean(0) for k in sorted(t.iloc[a].sample_number.unique())])
        btest=sorted(map(int,t.iloc[te].sample_number.unique()));bcal=sorted(map(int,t.iloc[v].sample_number.unique()))
        dist={k:float(cdist(z[t.sample_number.eq(k)].mean(0)[None],proto).min()) for k in btest}
        tables=[];raw=[];sources={}
        for model in MODELS:
            if model.startswith('ridge'):
                feats=x if model=='ridge_p0' else d
                p=np.column_stack([fit_predict_model('hsi_P0',t,feats,y[:,j],a,ii,[],1000.) for j in range(6)])
            elif model=='efficientnet_p0':
                cp=EFFICIENT/'models'/s.split_id/'inner5';assert b.verify_completed(cp)
                ck=torch.load(cp/'model.pt',map_location='cpu',weights_only=True);assert ck['epoch']==160
                net=old.build_1d_backbone_model('efficientnet1d',6).to('mps');net.load_state_dict(ck['state_dict']);norm={k:v.numpy() for k,v in ck['norm'].items()}
                for k in n:np.testing.assert_array_equal(norm[k],n[k])
                p=b.predict(net,x[ii],norm);sources[str(cp/'model.pt')]=digest(cp/'model.pt');del net;torch.mps.empty_cache()
            else:
                cp=OUT/'models'/model/s.split_id/'inner5';net,norm=trn.load_model(cp)
                for k in n:np.testing.assert_array_equal(norm[k],n[k])
                p=trn.predict(net,bags[ii],w[ii],norm);sources[str(cp/'model.pt')]=digest(cp/'model.pt');del net;torch.mps.empty_cache()
            rows=pd.DataFrame(prediction_rows(t,y,ii,p,s,model));rows['role']=np.where(rows.canonical_id.isin(t.iloc[v].canonical_id),'calibration','test');raw.append(rows)
            for ep in CORE:
                g=rows[rows.short.eq(ep)&rows.observed_target].groupby(['role','sample_number'],as_index=False).agg(y_true=('y_true','mean'),y_pred=('y_pred','mean'),n_grabs=('canonical_id','size'),grade=('grade','first'),manufacturer=('manufacturer','first'))
                cal=g[g.role.eq('calibration')];err=np.abs(cal.y_true-cal.y_pred).to_numpy();m=len(err);rank=int(np.ceil((m+1)*.90));q=float(np.sort(err)[rank-1]) if rank<=m else float('inf')
                for rr in g[g.role.eq('test')].itertuples():
                    tables.append(dict(model=model,short=ep,split_id=s.split_id,repeat=s.repeat,sample_number=rr.sample_number,grade=rr.grade,manufacturer=rr.manufacturer,n_grabs=rr.n_grabs,y_true=rr.y_true,y_pred=rr.y_pred,calibration_batches=m,quantile_rank=rank,radius=q,lower=rr.y_pred-q,upper=rr.y_pred+q,covered=bool(abs(rr.y_true-rr.y_pred)<=q),distance=dist[rr.sample_number],fixed_epoch=160 if 'ridge' not in model else 0,fixed_alpha=1000 if 'ridge' in model else 0))
        write(pd.DataFrame(tables),fd/'batch_intervals.csv.gz');write(pd.concat(raw,ignore_index=True),fd/'grab_predictions.csv.gz')
        dump(fd/'design.json',dict(proper_training_batches=sorted(t.iloc[a].sample_number.unique().tolist()),calibration_batches=bcal,test_batches=btest,calibration_not_used_for_tuning=True,checkpoint_sources=sources))
        dump(mark,dict(complete=True,hashes={p.name:digest(p) for p in fd.iterdir() if p.is_file() and p.name!='complete.json'}))
    check_sources();print('Separate fixed-model intervals completed',flush=True)

if __name__=='__main__':main()
