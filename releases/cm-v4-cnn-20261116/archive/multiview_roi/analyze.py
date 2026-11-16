"""Summaries from sealed OOF predictions only; no fitting or candidate selection."""
from common import *
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

CONTRASTS=[('ridge_dispersion','ridge_p0'),('mil_mean','efficientnet_p0'),('mil_attention','mil_mean'),('mil_attention','ridge_p0')]

def metric(g):
    a=g.y_true.to_numpy(float);p=g.y_pred.to_numpy(float);e=p-a;var=np.mean((a-a.mean())**2)
    return dict(n=len(g),mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)),r2=float(1-np.mean(e*e)/var) if var>1e-14 else np.nan,spearman=float(spearmanr(a,p).statistic) if np.std(a)>1e-12 and np.std(p)>1e-12 else np.nan,sd_ratio=float(np.std(p)/np.std(a)) if np.std(a)>1e-12 else np.nan)

def ci_rows(loss,keys):
    ans=[];rng=np.random.default_rng(20261106)
    for key,g in loss.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        z=g.groupby(['sample_number','model']).loss.mean().unstack()
        for new,old in CONTRASTS:
            pair=z[[new,old]].dropna();a=pair[new].to_numpy();b_=pair[old].to_numpy();draw=rng.integers(0,len(pair),(10000,len(pair)));delta=(a[draw]-b_[draw]).mean(1);pct=100*(1-a[draw].mean(1)/b_[draw].mean(1))
            ans.append(dict(zip(keys,key))|dict(new=new,reference=old,n_batches=len(pair),new_mse=float(a.mean()),reference_mse=float(b_.mean()),mse_difference=float(np.mean(a-b_)),difference_low=float(np.quantile(delta,.025)),difference_high=float(np.quantile(delta,.975)),reduction_pct=float(100*(1-a.mean()/b_.mean())),reduction_low=float(np.quantile(pct,.025)),reduction_high=float(np.quantile(pct,.975)),bootstrap_unit='production batch; loss averaged across repeats first',inference='exploratory conditional interval; not multiplicity-adjusted or external validation'))
    return pd.DataFrame(ans)

def main():
    check_sources();assert json.loads((OUT/'qa/full.json').read_text())['passed'];t,x,y,ss=load();st=OUT/'statistics'
    df=read(OUT/'models/full_predictions.csv.gz');df=df[df.observed_target].copy()
    counts={ep:int(np.isfinite(y[:,j]).sum()) for j,ep in enumerate(CORE)}
    bm=pd.DataFrame(y,columns=CORE).assign(sample_number=t.sample_number).groupby('sample_number').mean()
    nn=pd.DataFrame(np.isfinite(y).astype(int),columns=CORE).assign(sample_number=t.sample_number).groupby('sample_number').sum();eligible=bm.dropna().index
    assert len(eligible)==66
    # Grade-typical reference uses training batches, with each batch weighted once.
    df['train_sd']=np.nan;df['grade_typical']=np.nan;df['concentration_stratum']=''
    for s in ss:
        a=s.train_idx;training=pd.DataFrame(y[a],columns=CORE).assign(sample_number=t.iloc[a].sample_number.to_numpy(),grade=t.iloc[a].recorded_grade.to_numpy()).groupby(['sample_number','grade'],as_index=False).mean()
        for j,ep in enumerate(CORE):
            z=(df.split_id==s.split_id)&(df.short==ep);typ=training.groupby('grade')[ep].mean();lo,hi=np.nanquantile(training[ep],[1/3,2/3]);sd=float(np.nanstd(y[a,j]))
            df.loc[z,'train_sd']=sd;df.loc[z,'grade_typical']=df.loc[z,'grade'].map(typ).fillna(training[ep].mean())
            df.loc[z,'concentration_stratum']=np.where(df.loc[z,'y_true']<lo,'low',np.where(df.loc[z,'y_true']>hi,'high','middle'))
    assert df[['train_sd','grade_typical']].notna().all().all()
    batch=df.groupby(['model','short','repeat','split_id','sample_number'],as_index=False).agg(y_true=('y_true','mean'),y_pred=('y_pred','mean'),n_grabs=('canonical_id','size'),grade=('grade','first'),manufacturer=('manufacturer','first'),train_sd=('train_sd','first'),grade_typical=('grade_typical','first'))
    # Batch concentration strata must use batch truth and training-batch cutpoints.
    batch['concentration_stratum']=''
    for s in ss:
        for ep in CORE:
            lo,hi=np.nanquantile(bm.loc[t.iloc[s.train_idx].sample_number.unique(),ep],[1/3,2/3]);z=(batch.split_id==s.split_id)&(batch.short==ep)
            batch.loc[z,'concentration_stratum']=np.where(batch.loc[z,'y_true']<lo,'low',np.where(batch.loc[z,'y_true']>hi,'high','middle'))
    frames=[];groups=[];within=[];losses=[]
    for grain,data in [('grab',df),('batch',batch)]:
        data['loss']=(data.y_pred-data.y_true)**2;data['grade_loss']=(data.grade_typical-data.y_true)**2;data['normalized_loss']=data.loss/data.train_sd**2
        for key,g in data.groupby(['model','short','repeat']):
            frames.append(dict(grain=grain,model=key[0],short=key[1],repeat=key[2],**metric(g),grade_typical_mse=float(g.grade_loss.mean()),mse_reduction_vs_grade=float(1-g.loss.mean()/g.grade_loss.mean())))
            centered=g[['y_true','y_pred']]-g.groupby('grade')[['y_true','y_pred']].transform('mean');within.append(dict(grain=grain,model=key[0],short=key[1],repeat=key[2],**metric(centered),reference='within-repeat observed grade centering; descriptive',grade_typical_mse=float(g.grade_loss.mean())))
        for kind in ['grade','manufacturer','concentration_stratum']:
            for key,g in data.groupby(['model','short','repeat',kind]):groups.append(dict(grain=grain,model=key[0],short=key[1],repeat=key[2],subgroup_type=kind,subgroup=str(key[3]),n_batches=g.sample_number.nunique(),**metric(g)))
        loss=data[['model','short','repeat','sample_number','loss']].copy();loss['grain']=grain;losses.append(loss)
    write(pd.DataFrame(frames),st/'repeat_metrics.csv');write(pd.DataFrame(groups),st/'subgroup_metrics.csv');write(pd.DataFrame(within),st/'within_grade_metrics.csv');write(ci_rows(pd.concat(losses),['grain','short']),st/'paired_batch_intervals.csv')
    write(df,st/'grab_predictions_enriched.csv.gz');write(batch,st/'batch_predictions.csv.gz')
    panels=[]
    for name,eps in [('six',CORE),('noCAT',[e for e in CORE if e!='CAT'])]:
        g=batch[batch.sample_number.isin(eligible)&batch.short.isin(eps)]
        z=g.groupby(['model','repeat','sample_number'],as_index=False).agg(loss=('normalized_loss','mean'),endpoints=('short','nunique'));assert (z.endpoints==len(eps)).all();z['panel']=name;panels.append(z)
    panel=pd.concat(panels);write(panel,st/'panel_batch_losses.csv.gz');write(panel.groupby(['panel','model','repeat'],as_index=False).loss.mean(),st/'panel_repeat_metrics.csv');write(ci_rows(panel,['panel']),st/'panel_paired_intervals.csv')
    cat=batch[batch.short.eq('CAT')].copy();cat['subset']=np.where(cat.n_grabs==3,'complete3','incomplete1to2')
    write(pd.DataFrame([dict(model=k[0],repeat=k[1],subset=k[2],**metric(g)) for k,g in cat.groupby(['model','repeat','subset'])]),st/'CAT_subset_metrics.csv');write(ci_rows(cat,['subset']),st/'CAT_subset_paired_intervals.csv')
    # Photo/chemical joint display on exactly aligned held-out batches.
    photos=[]
    for s in ss:
        pp=read(OUT/'models/photo_aligned'/s.split_id/'predictions.csv');pp['split_id']=s.split_id;photos.append(pp)
    photo=pd.concat(photos);pb=photo.groupby(['repeat','split_id','sample_number'],as_index=False).agg(grade=('recorded_grade','first'),predicted_grade=('predicted_grade',lambda v:int(v.mode().iloc[0])))
    write(pd.DataFrame([dict(repeat=k,macro_f1=f1_score(g.grade,g.predicted_grade,labels=[1,2,3],average='macro'),n_batches=len(g)) for k,g in pb.groupby('repeat')]),st/'photo_aligned_metrics.csv')
    merged=batch.merge(pb,on=['repeat','split_id','sample_number','grade'],validate='many_to_one');merged['photo_correct']=merged.grade==merged.predicted_grade
    # Apparent deviations are descriptions of composition, not good/bad grades.
    merged['measured_minus_grade_typical']=merged.y_true-merged.grade_typical;merged['predicted_minus_grade_typical']=merged.y_pred-merged.grade_typical
    write(merged,st/'appearance_chemistry_joint_display.csv.gz')
    write(pd.DataFrame([dict(model=k[0],short=k[1],repeat=k[2],photo_correct=k[3],**metric(g)) for k,g in merged.groupby(['model','short','repeat','photo_correct'])]),st/'photo_error_conditioned_metrics.csv')
    # Profile distances compare pairs inside the SAME outer test fold/grade,
    # so every pair shares one training-only scale. Pairs are not independent.
    pairs=[]
    for panel_name,eps in [('six',CORE),('noCAT',[e for e in CORE if e!='CAT'])]:
        z=batch[batch.sample_number.isin(eligible)&batch.short.isin(eps)]
        for (model,rep,sid,grade),g in z.groupby(['model','repeat','split_id','grade']):
            truth=g.pivot(index='sample_number',columns='short',values='y_true')[eps]
            pred=g.pivot(index='sample_number',columns='short',values='y_pred')[eps]
            sd=g.drop_duplicates('short').set_index('short').loc[eps,'train_sd'].to_numpy()
            for i in range(len(truth)):
                for j in range(i+1,len(truth)):
                    a=float(np.sqrt(np.mean(((truth.iloc[i]-truth.iloc[j])/sd)**2)))
                    p=float(np.sqrt(np.mean(((pred.iloc[i]-pred.iloc[j])/sd)**2)))
                    pairs.append(dict(panel=panel_name,model=model,repeat=rep,split_id=sid,grade=grade,batch_a=int(truth.index[i]),batch_b=int(truth.index[j]),observed_distance=a,predicted_distance=p,absolute_error=abs(p-a)))
    pairs=pd.DataFrame(pairs);write(pairs,st/'within_grade_profile_pairs.csv.gz')
    write(pd.DataFrame([dict(panel=k[0],model=k[1],repeat=k[2],n_dependent_pairs=len(g),distance_mae=float(g.absolute_error.mean()),distance_spearman=float(spearmanr(g.observed_distance,g.predicted_distance).statistic)) for k,g in pairs.groupby(['panel','model','repeat'])]),st/'profile_distance_metrics.csv')
    distance_ci=[];rng=np.random.default_rng(20261107)
    ids=list(sorted(eligible));lookup={n:i for i,n in enumerate(ids)};draw=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=10000)
    for pa,g in pairs.groupby('panel'):
        z=g.groupby(['batch_a','batch_b','model']).absolute_error.mean().unstack()
        ia=np.array([lookup[v] for v in z.index.get_level_values(0)]);ib=np.array([lookup[v] for v in z.index.get_level_values(1)]);weights=draw[:,ia]*draw[:,ib];den=weights.sum(1);ok=den>0
        for new,old in CONTRASTS:
            delta=(z[new]-z[old]).to_numpy();values=(weights[ok]@delta)/den[ok]
            distance_ci.append(dict(panel=pa,new=new,reference=old,distance_mae_difference=float(delta.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),n_unique_dependent_pairs=len(z),bootstrap_unit='batch multiplicities; pair weight is product of endpoint multiplicities',fits_fixed=True))
    write(pd.DataFrame(distance_ci),st/'profile_distance_paired_intervals.csv')
    # Full-cohort descriptive JIVE score associations, with no predictive claims.
    jr=[];assoc=[]
    for directory in (OUT/'structure').glob('*/'):
        if not (directory/'fractions.csv').exists():continue
        if not read(directory/'ranks.csv').converged.all():continue
        f=read(directory/'fractions.csv');boots=read(directory/'bootstrap_fixed_rank.csv');assert boots.bootstrap.nunique()==200
        for row in f.itertuples():
            for part in ['joint','individual','residual']:
                values=boots.loc[boots.block.eq(row.block),part]
                jr.append(dict(scenario=directory.name,block=row.block,part=part,fraction=getattr(row,part),low=float(values.quantile(.025)),high=float(values.quantile(.975)),bootstrap_n=200,rank_selection_uncertainty_included=False))
        meta=read(directory/'batches.csv');mats=[read(directory/f'{b}_joint.csv').to_numpy() for b in f.block];joint=np.concatenate(mats,axis=1);u,sv,_=np.linalg.svd(joint,full_matrices=False);rank=int(read(directory/'ranks.csv').joint_rank.iloc[0]);score=u[:,:rank]*sv[:rank]
        sd=pd.DataFrame(score,columns=[f'joint_score_{j+1}' for j in range(rank)]);sd.insert(0,'sample_number',meta.sample_number);write(sd,directory/'joint_scores.csv')
        for j in range(rank):
            for factor in ['grade','manufacturer']:
                zz=pd.DataFrame({'score':score[:,j],'factor':meta[factor]});total=np.sum((zz.score-zz.score.mean())**2);between=sum(len(g)*(g.score.mean()-zz.score.mean())**2 for _,g in zz.groupby('factor'))
                assoc.append(dict(scenario=directory.name,score=j+1,factor=factor,descriptive_eta2=between/total,n_batches=len(zz),causal=False))
    write(pd.DataFrame(jr),st/'JIVE_variance_summary.csv');write(pd.DataFrame(assoc),st/'JIVE_score_associations.csv')
    # Separate fixed-model conformal summaries.
    ints=pd.concat([read(OUT/'uncertainty'/s.split_id/'batch_intervals.csv.gz') for s in ss],ignore_index=True)
    us=[];risk=[]
    for key,g in ints.groupby(['model','short','repeat']):
        us.append(dict(model=key[0],short=key[1],repeat=key[2],**metric(g),coverage=float(g.covered.mean()),mean_width=float((2*g.radius).mean()),infinite_intervals=int(np.isinf(g.radius).sum()),min_calibration_batches=int(g.calibration_batches.min())))
        for retain in [.5,.75,1.]:
            kept=g.sort_values(['distance','sample_number']).iloc[:int(np.ceil(len(g)*retain))]
            risk.append(dict(model=key[0],short=key[1],repeat=key[2],retained_fraction=retain,**metric(kept),selection='nearest proper-training P0 distance; descriptive only'))
    write(ints,st/'fixed_model_intervals.csv.gz');write(pd.DataFrame(us),st/'fixed_model_interval_metrics.csv');write(pd.DataFrame(risk),st/'fixed_model_selective_risk.csv')
    findings=dict(completed_at=now(),counts=counts,grabs=207,batches=69,complete_panel_batches=66,folds=50,repeats=10,pipelines=MODELS,primary='mean per-repeat OOF metrics; batch means use identical observed grabs',limits=['Repeated69batches, no independent external validation','JIVE is full-cohort descriptive structure','Bootstrap conditions on existing fitted predictions and selected ranks','Separate fixed160epoch/alpha1000 uncertainty arm, calibration labels not tuned','No new quality grade; no HPLC replacement claim'])
    dump(st/'findings.json',findings);check_sources();print('Statistics completed',flush=True)

if __name__=='__main__':main()
