
from release_paths import resolve_path as _release_path
from common import *
from run_models import candidates
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import hashlib

data=np.load(HERE/'enhanced_baseline/model_inputs.npz')
# Numerical check of the fast analytic implementation against sklearn.
errors={}
for kind in ('ridge','kernel_ridge'):
    x=data['enhanced_1125']; y=data['y_1125']; train=np.arange(len(y))<65
    pred,params=candidates(x[train],y[train],x[~train],kind)
    sc=StandardScaler().fit(x[train]); xx=sc.transform(x[train]); zz=sc.transform(x[~train])
    maximum=0.
    for k,(alpha,mult) in enumerate(params):
        if kind=='ridge':
            model=Ridge(alpha=alpha,solver='cholesky').fit(xx,y[train]); ref=model.predict(zz)
        else:
            ym=y[train].mean(); model=KernelRidge(alpha=alpha,kernel='rbf',gamma=mult/x.shape[1]).fit(xx,y[train]-ym);ref=model.predict(zz)+ym
        maximum=max(maximum,float(np.max(abs(pred[k]-ref))))
    errors[kind]=maximum
    assert maximum<1e-8

# Bridge the original CV/tuning implementation to the enhanced features.
# Diagnostic prompted by observed tuning sensitivity; not a replacement for preregistered comparisons.
rows=[]
for d in DIRS:
    t=tag(d); x=data['enhanced_'+t]; y=data['y_'+t]; c=data['network_'+t]
    for seed,g in enumerate(blocks(data['xy_'+t])):
        a=cross.rmse_cv(x,y,g); b=cross.rmse_cv(np.column_stack([x,c]),y,g)
        rows.append(dict(direction=d,seed=seed,rmse_baseline=a,rmse_circuit=b,gain_pct=100*(a-b)/a))
    print('BRIDGE_DONE',d,flush=True)
bridge=pd.DataFrame(rows); bridge.to_csv(HERE/'enhanced_baseline/legacy_tuning_enhanced_bridge.csv',index=False)
summ=[]
for d,g in [('all',bridge)]+list(bridge.groupby('direction')):
    summ.append(dict(direction=d,n=len(g),improves=int((g.gain_pct>0).sum()),median_gain_pct=float(g.gain_pct.median())))
pd.DataFrame(summ).to_csv(HERE/'enhanced_baseline/legacy_tuning_enhanced_bridge_summary.csv',index=False)

old=pd.read_csv(SAVED/'strict_upwind_all_tests.csv').query("metric == '10_1000'")
absrows=[]
for d,g in [('all',old)]+list(old.groupby('direction')):
    absrows.append(dict(direction=d,n=len(g),baseline_median_rmse=float(g.rmse_baseline.median()),
      circuit_median_rmse=float(g.rmse_circuit.median()),median_gain_pct=float((-100*g.relative_rmse_change).median())))
pd.DataFrame(absrows).to_csv(HERE/'original_absolute_rmse.csv',index=False)

metrics=pd.read_csv(HERE/'enhanced_baseline/model_metrics.csv')
predictions=pd.read_csv(HERE/'enhanced_baseline/oof_predictions.csv')
assert len(metrics)==240 and len(predictions)==240*85
assert np.isfinite(metrics[['rmse','mae','r2']].to_numpy()).all()
assert predictions.groupby(['learner','model','direction','seed']).active_id.nunique().eq(85).all()
assert np.isfinite(predictions[['observed','predicted']].to_numpy()).all()
rep=json.loads((HERE/'q_sensitivity/same_q_replication.json').read_text(encoding='utf8'))
assert max(x['max_absolute_difference'] for x in rep)<1e-10
assert len(pd.read_csv(HERE/'q_sensitivity/added_value_tests.csv'))==120
inputaudit=json.loads((HERE/'q_sensitivity/input_audit.json').read_text(encoding='utf8'))
assert all(x['mask_unchanged'] and x['focal_sha256']==x['copied_focal_sha256'] for x in inputaudit)
qa=dict(status='PASS',analytic_vs_sklearn_max_abs_errors=errors,n_model_comparisons=len(metrics),
  n_oof_rows=len(predictions),q_input_cases=len(inputaudit),q_added_value_tests=120,
  note='PASS verifies reproducibility and output integrity, not universal positive scientific results.')
save_json(HERE/'QA.json',qa)
paths=[cross.TEMPLATE,SAVED/'strict_model_inputs.npz',SAVED/'strict_upwind_all_tests.csv',
  ROOT/'manuscript_paris/manuscript_zh.md',HERE/'PROTOCOL.md']
paths += [original_map(d,c) for d in DIRS for c in ('real','homo')]
paths += [_release_path('E:/lunwen4/2/空气质量/Paris_2022_event/analysis')/s for s in ('D075_background_observation_wrf_pairs.csv','pollutant_transport_lag_diagnostics.csv')]
save_json(HERE/'source_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
print(qa,flush=True);print(pd.DataFrame(summ).to_string(index=False),flush=True)
