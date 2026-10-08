from common import *
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import sklearn
import time

ALPHAS=np.logspace(-3,3,7)
GAMMAS=(.25,1.,4.)

def candidates(x,y,z,kind):
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-12]=1
    x=(x-mean)/scale; z=(z-mean)/scale
    ym=y.mean(); yc=y-ym
    pred=[]; params=[]
    if kind=='ridge':
        e,v=eigh(x.T@x); e=np.maximum(e,0)
        rhs=v.T@(x.T@yc)
        for a in ALPHAS:
            pred.append(z@v@(rhs/(e+a))+ym); params.append((a,0.))
    else:
        dd=cdist(x,x,'sqeuclidean'); zd=cdist(z,x,'sqeuclidean')
        for mult in GAMMAS:
            gamma=mult/x.shape[1]
            e,v=eigh(np.exp(-gamma*dd)); e=np.maximum(e,0)
            rhs=v.T@yc; zv=np.exp(-gamma*zd)@v
            for a in ALPHAS:
                pred.append(zv@(rhs/(e+a))+ym); params.append((a,mult))
    return np.asarray(pred),params

def nested_oof(x,y,groups,kind):
    out=np.full(len(y),np.nan); tune=[]
    for group in np.unique(groups):
        test=groups==group; train=~test
        xi,yi,gi=x[train],y[train],groups[train]
        score=None; count=0
        for inner in np.unique(gi):
            val=gi==inner
            ps,params=candidates(xi[~val],yi[~val],xi[val],kind)
            s=np.sum((ps-yi[val])**2,axis=1)
            score=s if score is None else score+s; count+=val.sum()
        best=int(np.argmin(score/count))
        ps,params=candidates(xi,yi,x[test],kind)
        out[test]=ps[best]
        tune.append(dict(held_out_block=int(group),alpha=float(params[best][0]),
          gamma_multiplier=float(params[best][1]),inner_rmse=float(np.sqrt(score[best]/count))))
    return out,tune

def main():
    dest=HERE/'enhanced_baseline'; dest.mkdir(exist_ok=True)
    saved=np.load(SAVED/'strict_model_inputs.npz')
    rows=[]; predictions=[]; tunings=[]; legacy=[]; fine_audit=[]; arrays={}
    old=pd.read_csv(SAVED/'strict_upwind_all_tests.csv'); old=old[old.metric.eq('10_1000')]
    for d in DIRS:
        t=tag(d); x=saved['baseline_'+t]; c=saved['network_'+t]; y=saved['y_10_1000_'+t]; xy=saved['xy_'+t]
        df=mapped(cells(d)); df.to_csv(dest/f'fine_cells_{t}.csv',index=False)
        fine=[]
        for i in range(len(y)):
            group=df[df.active_id==i]; stats=[]
            for case in ('real','homo'):
                a=group['fai_'+case].to_numpy()
                if len(a)==0: raise ValueError('Missing fine cells')
                stats.extend([a.std(),np.quantile(a,.1),np.quantile(a,.9),np.mean(a<=.1*Q)])
            fine.append(stats)
            fine_audit.append(dict(direction=d,active_id=i,n_fine=len(group)))
        fine=np.asarray(fine); b=np.column_stack([x,fine])
        networks=df.groupby('active_id').apply(lambda z:float((z.current_real-z.current_homo).mean()),include_groups=False) if pd.__version__.startswith('2.2') else df.assign(dc=df.current_real-df.current_homo).groupby('active_id').dc.mean()
        assert np.allclose(networks.reindex(range(len(y))).to_numpy(),c.ravel(),atol=1e-12)
        arrays.update({f'baseline_{t}':x,f'enhanced_{t}':b,f'network_{t}':c,f'y_{t}':y,f'xy_{t}':xy})
        for seed,g in enumerate(blocks(xy)):
            # Exact reproduction, separate from the revised inner-spatial-CV experiment.
            ra=cross.rmse_cv(x,y,g); rc=cross.rmse_cv(np.column_stack([x,c]),y,g)
            ref=old[(old.direction==d)&(old.seed==seed)].iloc[0]
            legacy.append(dict(direction=d,seed=seed,rmse_baseline=ra,rmse_circuit=rc,
                baseline_abs_error=abs(ra-ref.rmse_baseline),circuit_abs_error=abs(rc-ref.rmse_circuit)))
            models=[('A','ridge',x),('A_C','ridge',np.column_stack([x,c])),
                    ('B','ridge',b),('B_C','ridge',np.column_stack([b,c])),
                    ('B','kernel_ridge',b),('B_C','kernel_ridge',np.column_stack([b,c]))]
            for name,kind,features in models:
                pred,tune=nested_oof(features,y,g,kind)
                rows.append(dict(direction=d,seed=seed,model=name,learner=kind,n=len(y),p=features.shape[1],
                    rmse=float(np.sqrt(mean_squared_error(y,pred))),mae=float(mean_absolute_error(y,pred)),r2=float(r2_score(y,pred))))
                predictions.extend(dict(direction=d,seed=seed,model=name,learner=kind,active_id=i,
                    block=int(g[i]),observed=float(y[i]),predicted=float(pred[i])) for i in range(len(y)))
                tunings.extend(dict(direction=d,seed=seed,model=name,learner=kind,**z) for z in tune)
            print('MODEL_DONE',d,'seed',seed,flush=True)
        pd.DataFrame(rows).to_csv(dest/'model_metrics.csv',index=False)
        pd.DataFrame(legacy).to_csv(dest/'legacy_reproduction.csv',index=False)
    pd.DataFrame(predictions).to_csv(dest/'oof_predictions.csv',index=False)
    pd.DataFrame(tunings).to_csv(dest/'selected_hyperparameters.csv',index=False)
    pd.DataFrame(fine_audit).to_csv(dest/'fine_cell_counts.csv',index=False)
    np.savez_compressed(dest/'model_inputs.npz',**arrays)
    result=pd.DataFrame(rows); comps=[]
    for learner,base,plus in [('ridge','A','A_C'),('ridge','B','B_C'),('kernel_ridge','B','B_C')]:
        a=result[(result.learner==learner)&(result.model==base)]
        b=result[(result.learner==learner)&(result.model==plus)]
        m=a.merge(b,on=['direction','seed'],suffixes=('_base','_network'))
        m['gain_pct']=100*(m.rmse_base-m.rmse_network)/m.rmse_base
        m['comparison']=learner+'_'+base+'_to_'+plus
        comps.append(m)
    comps=pd.concat(comps,ignore_index=True); comps.to_csv(dest/'paired_comparisons.csv',index=False)
    summaries=[]
    for (comparison,d),g in comps.groupby(['comparison','direction']):
        summaries.append(dict(comparison=comparison,direction=d,n=len(g),improves=int((g.gain_pct>0).sum()),
          median_gain_pct=float(g.gain_pct.median()),median_baseline_rmse=float(g.rmse_base.median()),median_network_rmse=float(g.rmse_network.median()),
          median_baseline_mae=float(g.mae_base.median()),median_network_mae=float(g.mae_network.median()),
          median_baseline_r2=float(g.r2_base.median()),median_network_r2=float(g.r2_network.median())))
    for comparison,g in comps.groupby('comparison'):
        summaries.append(dict(comparison=comparison,direction='all',n=len(g),improves=int((g.gain_pct>0).sum()),
          median_gain_pct=float(g.gain_pct.median()),median_baseline_rmse=float(g.rmse_base.median()),median_network_rmse=float(g.rmse_network.median()),
          median_baseline_mae=float(g.mae_base.median()),median_network_mae=float(g.mae_network.median()),
          median_baseline_r2=float(g.r2_base.median()),median_network_r2=float(g.r2_network.median())))
    pd.DataFrame(summaries).to_csv(dest/'summary.csv',index=False)
    save_json(dest/'metadata.json',dict(sklearn=sklearn.__version__,numpy=np.__version__,
       low_fai_threshold=.1*Q,alphas=ALPHAS.tolist(),gamma_multipliers=GAMMAS,
       fine_feature_names=[case+'_'+stat for case in ('real','homo') for stat in ('std','p10','p90','low_fraction')],
       inner_selection='pooled squared error, four remaining spatial blocks; scaler fitted in each fold',
       legacy_max_difference=max(max(z['baseline_abs_error'],z['circuit_abs_error']) for z in legacy)))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
if __name__=='__main__': main()
