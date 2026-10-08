from common import *

dest=HERE/'q_sensitivity'; saved=np.load(SAVED/'strict_model_inputs.npz')
old=pd.read_csv(SAVED/'strict_upwind_all_tests.csv'); old=old[old.metric.eq('10_1000')]
qa=[]; patterns=[]; tests=[]; all_networks={}
for d in DIRS:
    t=tag(d); original=cells(d)
    for factor in (.5,1.,2.):
        df=original.copy()
        if factor!=1:
            for case in ('real','homo'):
                a,h=cross.read_ascii(dest/f'q{factor:g}'/t/case/'network_cum_curmap.asc')
                df['current_'+case]=cross.sample_grid(a,h,df)
        elif d==112.5:
            for case in ('real','homo'):
                a,h=cross.read_ascii(dest/'q1'/t/case/'network_cum_curmap.asc')
                replica=cross.sample_grid(a,h,df); base=df['current_'+case].to_numpy()
                qa.append(dict(case=case,max_absolute_difference=float(np.max(abs(replica-base))),
                  rmse=float(np.sqrt(np.mean((replica-base)**2))),relative_rmse=float(np.sqrt(np.mean((replica-base)**2))/np.mean(abs(base)))))
        p=summary_current(df)
        for case in ('real','homo'):
            p['vs_q1_'+case+'_rho']=float(spearmanr(df['current_'+case],original['current_'+case]).statistic)
        delta=df.current_real-df.current_homo; ref=original.current_real-original.current_homo
        p['delta_vs_q1_rho']=float(spearmanr(delta,ref).statistic)
        patterns.append(dict(direction=d,q_factor=factor,**p))
        eligible=mapped(df); eligible['delta_current']=eligible.current_real-eligible.current_homo
        network=eligible.groupby('active_id').delta_current.mean().reindex(range(85)).to_numpy()[:,None]
        assert np.isfinite(network).all()
        all_networks[f'q{factor:g}_{t}']=network
        x=saved['baseline_'+t]; y=saved['y_10_1000_'+t]; xy=saved['xy_'+t]
        if factor==1: assert np.allclose(network,saved['network_'+t],atol=1e-12)
        for seed,g in enumerate(blocks(xy)):
            baseline=old[(old.direction==d)&(old.seed==seed)].iloc[0]
            extra=float(baseline.rmse_circuit) if factor==1 else cross.rmse_cv(np.column_stack([x,network]),y,g)
            tests.append(dict(direction=d,q_factor=factor,seed=seed,rmse_baseline=float(baseline.rmse_baseline),rmse_circuit=extra,
              gain_pct=100*(baseline.rmse_baseline-extra)/baseline.rmse_baseline))
        print('Q_ANALYSIS_DONE',d,factor,flush=True)
pd.DataFrame(patterns).to_csv(dest/'network_patterns.csv',index=False)
pd.DataFrame(tests).to_csv(dest/'added_value_tests.csv',index=False)
np.savez_compressed(dest/'aggregated_networks.npz',**all_networks)
save_json(dest/'same_q_replication.json',qa)
tests=pd.DataFrame(tests); rows=[]
for factor,g0 in tests.groupby('q_factor'):
    for d,g in [('all',g0)]+list(g0.groupby('direction')):
        rows.append(dict(q_factor=factor,direction=d,n=len(g),improves=int((g.gain_pct>0).sum()),median_gain_pct=float(g.gain_pct.median()),
          min_gain_pct=float(g.gain_pct.min()),max_gain_pct=float(g.gain_pct.max())))
pd.DataFrame(rows).to_csv(dest/'added_value_summary.csv',index=False)
pd.DataFrame(patterns).query('q_factor==1').to_csv(HERE/'four_direction_network_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
