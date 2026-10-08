"""Reproduce the primary spatial-CV test from the distributed frozen inputs.

No WRF run, raw building data, or Circuitscape solve is needed for this test.
Outputs are written to outputs/primary_reproduction, not over archived results.
"""
from pathlib import Path
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

def rmse_cv(x, y, groups):
    predicted = np.full(len(y), np.nan)
    for group in np.unique(groups):
        test = groups == group
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
        model.fit(x[~test], y[~test])
        predicted[test] = model.predict(x[test])
    return float(np.sqrt(np.mean((predicted-y)**2)))

def main():
    inputs = np.load(ROOT/'datas/analysis_inputs/strict_model_inputs.npz')
    reference = pd.read_csv(ROOT/'datas/results/primary/strict_upwind_all_tests.csv')
    rows = []
    for direction in (45., 67.5, 90., 112.5):
        tag = f'{int(round(direction*10)):04d}'
        x, c = inputs['baseline_'+tag], inputs['network_'+tag]
        y, xy = inputs['y_10_1000_'+tag], inputs['xy_'+tag]
        for seed in range(10):
            blocks = KMeans(n_clusters=5, random_state=seed, n_init=20).fit_predict(xy)
            a, b = rmse_cv(x,y,blocks), rmse_cv(np.column_stack([x,c]),y,blocks)
            ref = reference[(reference.metric=='10_1000')&(reference.direction==direction)&(reference.seed==seed)].iloc[0]
            rows.append(dict(direction=direction,seed=seed,rmse_baseline=a,rmse_circuit=b,
                             reduction_pct=100*(a-b)/a,
                             reference_error=max(abs(a-ref.rmse_baseline),abs(b-ref.rmse_circuit))))
    frame = pd.DataFrame(rows)
    output = ROOT/'outputs/primary_reproduction'
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'paired_comparisons.csv',index=False)
    result = dict(comparisons=len(frame),improved=int((frame.reduction_pct>0).sum()),
                  median_rmse_reduction_pct=float(frame.reduction_pct.median()),
                  max_difference_from_archived_rmse=float(frame.reference_error.max()))
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['max_difference_from_archived_rmse']>1e-7:
        raise RuntimeError('Numerical reproduction differs from the archived reference; check package versions.')

if __name__=='__main__':
    main()
