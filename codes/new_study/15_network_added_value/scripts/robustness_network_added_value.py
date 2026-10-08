"""Sensitivity audit for the Phase-I Circuit network-added-value result."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = _release_path('D:\\研究生\\代码\\论文4\\lunwen4\\new_study\\15_network_added_value\\phase1_1125')
TEMPLATE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_activation_pilot_v1\\injection_templates\\paris_d03_real_homo_injection_templates.npz')
RADII_M = (1500.0, 2500.0, 3500.0, 5500.0)
SEEDS = tuple(range(10))


def cv_rmse(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pred = np.full_like(y, np.nan, dtype=float)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5),
        )
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> None:
    wrf = np.load(ROOT / "wrf_phase1_export.npz")
    template = np.load(TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    table = pd.read_csv(ROOT / "network_added_value_cells.csv")
    xy = table[["x_l93", "y_l93"]].to_numpy()
    local = table[["delta_mh_m", "delta_lp", "delta_scalar_fai"]].to_numpy()
    circuit = table[["delta_circuit_current"]].to_numpy()
    tree = cKDTree(xy)

    times = np.asarray([str(v) for v in wrf["all_times"]])
    all_y = wrf["delta_along_all_hourly"][:, active]
    speed = wrf["background_speed_ms"]
    bins = wrf["background_bin_deg"]
    persistence = wrf["background_persistence6h"]
    stable1125 = (bins == 112.5) & (speed >= 1.0) & (persistence >= 0.90)
    july18 = np.asarray([t.startswith("2022-07-18") for t in times])
    hours = np.asarray([int(t[11:13]) for t in times])
    selectors = {
        "full_event_stable1125": stable1125,
        "july18_all_stable1125": stable1125 & july18,
        "july18_day_06_18": stable1125 & july18 & (hours >= 6) & (hours <= 18),
        "july18_night": stable1125 & july18 & ~((hours >= 6) & (hours <= 18)),
    }
    for date in ("2022-07-16", "2022-07-17", "2022-07-19"):
        selector = stable1125 & np.asarray([t.startswith(date) for t in times])
        if int(selector.sum()) >= 4:
            selectors[f"{date}_stable1125"] = selector

    rows = []
    for radius in RADII_M:
        context = np.zeros_like(local)
        for i, point in enumerate(xy):
            neighbors = [n for n in tree.query_ball_point(point, radius) if n != i]
            context[i] = np.mean(local[neighbors], axis=0) if neighbors else 0.0
        base_context = np.column_stack([local, context])
        for period, selector in selectors.items():
            y = np.mean(all_y[selector], axis=0)
            for seed in SEEDS:
                groups = KMeans(n_clusters=5, random_state=seed, n_init=20).fit_predict(xy)
                for baseline_name, baseline in (("local", local), ("local_context", base_context)):
                    rmse_base = cv_rmse(baseline, y, groups)
                    rmse_circuit = cv_rmse(np.column_stack([baseline, circuit]), y, groups)
                    rows.append({
                        "period": period,
                        "n_hours": int(selector.sum()),
                        "radius_m": radius,
                        "seed": seed,
                        "baseline": baseline_name,
                        "rmse_baseline": rmse_base,
                        "rmse_plus_circuit": rmse_circuit,
                        "relative_rmse_change": (rmse_circuit - rmse_base) / rmse_base,
                        "circuit_improves": rmse_circuit < rmse_base,
                    })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "network_added_value_robustness_all_runs.csv", index=False)
    summary = (
        result.groupby(["period", "baseline"], as_index=False)
        .agg(
            n_tests=("circuit_improves", "size"),
            improvement_fraction=("circuit_improves", "mean"),
            median_relative_rmse_change=("relative_rmse_change", "median"),
            p10_relative_rmse_change=("relative_rmse_change", lambda x: np.percentile(x, 10)),
            p90_relative_rmse_change=("relative_rmse_change", lambda x: np.percentile(x, 90)),
        )
    )
    summary.to_csv(ROOT / "network_added_value_robustness_summary.csv", index=False)
    overall = {
        "status": "ROBUSTNESS_COMPLETE",
        "n_total_tests": int(len(result)),
        "overall_improvement_fraction": float(result["circuit_improves"].mean()),
        "local_improvement_fraction": float(result.loc[result["baseline"] == "local", "circuit_improves"].mean()),
        "local_context_improvement_fraction": float(result.loc[result["baseline"] == "local_context", "circuit_improves"].mean()),
        "median_relative_rmse_change": float(result["relative_rmse_change"].median()),
        "periods": {name: int(selector.sum()) for name, selector in selectors.items()},
        "radii_m": list(RADII_M),
        "spatial_block_seeds": list(SEEDS),
    }
    (ROOT / "network_added_value_robustness.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
