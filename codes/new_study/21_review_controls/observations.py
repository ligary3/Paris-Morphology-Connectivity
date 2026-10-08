
from release_paths import resolve_path as _release_path
from common import *
import shutil

dest=HERE/'observations'; dest.mkdir(exist_ok=True)
AIR=_release_path('E:/lunwen4/2/空气质量/Paris_2022_event/analysis')
WEATHER=_release_path('E:/lunwen4/2/气象观测/Paris_WRF_validation')
raw=pd.read_csv(AIR/'D075_background_observation_wrf_pairs.csv')
raw['time_utc']=pd.to_datetime(raw.time_utc,utc=True)
raw=raw[raw.background_bin_deg.isin(DIRS)].dropna(subset=['O3','delta_transport']).copy()

def eligible(df):
    return df[df.groupby('time_utc').station_code.transform('nunique')>=2].copy()

def residual(a):
    return a-np.nanmean(a,axis=0,keepdims=True)-np.nanmean(a,axis=1,keepdims=True)+np.nanmean(a)

def stats(df,seed=20260910,boot=1000):
    x=df.pivot(index='time_utc',columns='station_code',values='delta_transport').to_numpy()
    y=df.pivot(index='time_utc',columns='station_code',values='O3').to_numpy()
    assert np.array_equal(np.isnan(x),np.isnan(y))
    def rho(a,b):
        aa,bb=residual(a),residual(b); ok=np.isfinite(aa)&np.isfinite(bb)
        return float(spearmanr(aa[ok],bb[ok]).statistic)
    observed=rho(x,y); rng=np.random.default_rng(seed); values=[]; n=len(x)
    for _ in range(boot):
        starts=rng.integers(0,max(1,n-6+1),size=int(np.ceil(n/6)))
        ix=np.concatenate([np.arange(s,min(s+6,n)) for s in starts])[:n]
        r=rho(x[ix],y[ix])
        if np.isfinite(r): values.append(r)
    lo,hi=np.quantile(values,[.025,.975])
    return dict(n_pairs=len(df),n_stations=df.station_code.nunique(),n_hours=df.time_utc.nunique(),rho=observed,
       ci_low=float(lo),ci_high=float(hi),n_bootstrap_valid=len(values),first_time=str(df.time_utc.min()),last_time=str(df.time_utc.max()))

stations=sorted(raw.station_code.unique()); sets={}
sets['none']=eligible(raw)
for station in stations: sets[station]=eligible(raw[raw.station_code.ne(station)])
common=set.intersection(*[set(df.time_utc) for df in sets.values()])
rows=[]
for support in ('available_hours','common_hours'):
    for station,frame in sets.items():
        frame=frame if support=='available_hours' else frame[frame.time_utc.isin(common)]
        result=stats(frame)
        rows.append(dict(support=support,omitted_station=station,**result))
        print('O3_DONE',support,station,result['rho'],flush=True)
pd.DataFrame(rows).to_csv(dest/'o3_leave_one_station_out.csv',index=False)
old=pd.read_csv(AIR/'pollutant_transport_lag_diagnostics.csv')
old0=old[(old.regime=='stable_network_hours')&(old.pollutant=='O3')&(old.lag_hours_pollutant_after_transport==0)].iloc[0]
assert abs(rows[0]['rho']-old0.spearman_rho_double_demeaned)<1e-12
save_json(dest/'o3_metadata.json',dict(bootstrap_replicates=1000,bootstrap_seed=20260910,
    bootstrap_block='6 retained hourly indices, not necessarily consecutive clock hours',
    stations=stations,common_hours=len(common),original_rho=float(old0.spearman_rho_double_demeaned),
    reproduction_abs_error=abs(rows[0]['rho']-old0.spearman_rho_double_demeaned),
    demeaning='original one-pass station mean and common-hour mean subtraction, recomputed per subset and bootstrap'))
for p in ['surface_validation_metrics.json','surface_validation_by_station.csv','trappes_profile_validation_metrics.json','trappes_profile_bias_by_pressure.csv']:
    shutil.copy2(WEATHER/'processed'/p,dest/p)
shutil.copy2(AIR/'pollutant_transport_lag_diagnostics.csv',dest/'pollutant_transport_lag_diagnostics.csv')
shutil.copy2(WEATHER/'WRF_METEOROLOGICAL_VALIDATION_REPORT.md',dest/'EXISTING_WRF_METEOROLOGICAL_VALIDATION_REPORT.md')
print(pd.DataFrame(rows).to_string(index=False),flush=True)
