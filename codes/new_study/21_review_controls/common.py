from __future__ import annotations

from release_paths import resolve_path as _release_path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'_vendor'))
import importlib.util
import json
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXT = _release_path('E:/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/extended_domain_v1')
SAVED = ROOT / 'new_study/17_transport_metric_candidates/analysis_strict_upwind_pbl'
DIRS = (45.,67.5,90.,112.5)
Q = 0.976565954884937

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

cross=module(ROOT/'new_study/16_multidirection_circuit/scripts/test_cross_direction_added_value.py','original_cross')

def tag(d): return cross.tag(d)
def folder(d):
    return EXT/'circuit_real_homo_1125_v1' if d==112.5 else EXT/'circuit_multidirection_pairs_v1'/tag(d)

def original_map(d,case):
    return folder(d)/case/f'{case}_{tag(d)}_cum_curmap.asc'

def cells(d):
    if d==112.5:
        df=pd.read_csv(cross.PAIR1125).rename(columns={'lambda_f_real':'fai_real','lambda_f_homo':'fai_homo'})
    else:
        df=pd.read_csv(folder(d)/'cells.csv')
        for case in ('real','homo'):
            a,h=cross.read_ascii(original_map(d,case))
            df['current_'+case]=cross.sample_grid(a,h,df)
    return df.loc[df.analysis_core.astype(bool)].copy()

def mapping():
    export=np.load(ROOT/'new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz')
    template=np.load(cross.TEMPLATE)
    active=template['PRIMARY_ACTIVATION'].astype(bool)&template['TREATMENT_CORE'].astype(bool)
    x,y=Transformer.from_crs(4326,2154,always_xy=True).transform(export['lon'],export['lat'])
    full=np.column_stack([x.ravel(),y.ravel()]); xy=np.column_stack([x[active],y[active]])
    lut={int(f):i for i,f in enumerate(np.flatnonzero(active))}
    return cKDTree(full),lut,xy

def mapped(df):
    tree,lut,xy=mapping()
    dist,near=tree.query(df[['x_center','y_center']].to_numpy())
    df=df.copy(); df['active_id']=[lut.get(int(i),-1) for i in near]
    return df.loc[(df.active_id>=0)&(dist<=850)].copy()

def blocks(xy):
    return [KMeans(n_clusters=5,random_state=s,n_init=20).fit_predict(xy) for s in range(10)]

def summary_current(df):
    a=df.current_real.to_numpy(); b=df.current_homo.to_numpy(); n=len(a); k=int(np.ceil(n*.1))
    # Deterministic top-K selection; ties broken by input row order.
    topa=np.argsort(-a,kind='stable')[:k]; topb=np.argsort(-b,kind='stable')[:k]
    return dict(n_cells=n,top10_n=k,rank_rho=float(spearmanr(a,b).statistic),
      top10_overlap=float(len(np.intersect1d(topa,topb))/k),mean_abs_delta=float(np.mean(np.abs(a-b))),
      mean_homo=float(np.mean(b)),mean_abs_delta_pct_homo=float(100*np.mean(np.abs(a-b))/np.mean(b)))

def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
