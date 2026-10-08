"""Phase-I test: does Circuit add predictive value beyond local/context morphology?"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
OUT = WORKSPACE / "new_study/15_network_added_value/phase1_1125"
TEMPLATE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_activation_pilot_v1\\injection_templates\\paris_d03_real_homo_injection_templates.npz')
CIRCUIT_CSV = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\circuit_real_homo_1125_v1\\analysis\\circuit_real_homo_cellwise.csv')
WRF_EXPORT = WORKSPACE / "new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz"


def metric_bundle(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "pearson": float(pearsonr(y, pred).statistic),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def spatial_cv_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[float]]:
    pred = np.full_like(y, np.nan, dtype=float)
    alphas: list[float] = []
    candidates = np.logspace(-3, 3, 13)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=candidates, cv=min(5, int(train.sum()))))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
        alphas.append(float(model[-1].alpha_))
    return pred, alphas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wrf = np.load(WRF_EXPORT)
    lat, lon = wrf["lat"], wrf["lon"]
    response_map = wrf["delta_along_matched_mean"]
    matched_times = [str(value) for value in wrf["matched_times"]]

    template = np.load(TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    jj, ii = np.where(active)
    y = response_map[active]

    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    wrf_x, wrf_y = transformer.transform(lon, lat)
    active_xy = np.column_stack([wrf_x[active], wrf_y[active]])
    all_xy = np.column_stack([wrf_x.ravel(), wrf_y.ravel()])
    active_flat = np.flatnonzero(active)
    flat_to_active = {int(flat): index for index, flat in enumerate(active_flat)}

    cells = pd.read_csv(CIRCUIT_CSV)
    cells = cells[cells["analysis_core"].astype(bool)].copy()
    # Assign each 100 m cell to its genuinely nearest d03 center first. Querying only
    # active centers would incorrectly pull neighboring park/water cells into edge
    # urban cells and inflate their sample counts.
    tree = cKDTree(all_xy)
    distance, nearest_flat = tree.query(cells[["x_center", "y_center"]].to_numpy(), k=1)
    cells["active_id"] = [flat_to_active.get(int(value), -1) for value in nearest_flat]
    cells["nearest_wrf_distance_m"] = distance
    cells = cells[(cells["active_id"] >= 0) & (distance <= 850.0)].copy()
    grouped = cells.groupby("active_id").agg(
        circuit_delta_mean=("delta_current", "mean"),
        circuit_delta_abs_mean=("delta_current", lambda x: float(np.mean(np.abs(x)))),
        circuit_real_mean=("current_real", "mean"),
        circuit_homo_mean=("current_homo", "mean"),
        n_circuit_100m=("grid_id", "size"),
        assignment_max_distance_m=("nearest_wrf_distance_m", "max"),
    )
    if len(grouped) != len(y):
        missing = sorted(set(range(len(y))) - set(grouped.index))
        raise ValueError(f"Circuit did not cover every active WRF cell: {missing}")
    grouped = grouped.reindex(range(len(y)))

    dmh = (template["REAL_MH_URB2D"] - template["HOMO_MH_URB2D"])[active].astype(float)
    dlp = (template["REAL_BUILD_AREA_FRACTION"] - template["HOMO_BUILD_AREA_FRACTION"])[active].astype(float)
    dlf = (template["REAL_LF_URB2D_S"] - template["HOMO_LF_URB2D_S"])[active].astype(float)
    local = np.column_stack([dmh, dlp, dlf])

    near = cKDTree(active_xy)
    context = np.zeros_like(local)
    for index, point in enumerate(active_xy):
        neighbors = near.query_ball_point(point, 3500.0)
        neighbors = [n for n in neighbors if n != index]
        context[index] = np.mean(local[neighbors], axis=0) if neighbors else 0.0
    network = grouped[["circuit_delta_mean"]].to_numpy()

    groups = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(active_xy)
    designs = {
        "M1_local": local,
        "M2_local_context3p5km": np.column_stack([local, context]),
        "M3_local_context_circuit": np.column_stack([local, context, network]),
    }
    results: dict[str, object] = {}
    predictions = {}
    for name, x in designs.items():
        pred, selected = spatial_cv_predictions(x, y, groups)
        predictions[name] = pred
        results[name] = {**metric_bundle(y, pred), "n_features": int(x.shape[1]), "selected_alphas": selected}

    m2, m3 = results["M2_local_context3p5km"], results["M3_local_context_circuit"]
    results["network_added_value"] = {
        "delta_rmse_m3_minus_m2": float(m3["rmse"] - m2["rmse"]),
        "relative_rmse_change": float((m3["rmse"] - m2["rmse"]) / m2["rmse"]),
        "delta_r2_m3_minus_m2": float(m3["r2"] - m2["r2"]),
        "delta_pearson_m3_minus_m2": float(m3["pearson"] - m2["pearson"]),
        "improves_spatial_cv_rmse": bool(m3["rmse"] < m2["rmse"]),
    }
    results["direct_associations"] = {
        "spearman_y_vs_circuit_delta": float(spearmanr(y, network[:, 0]).statistic),
        "pearson_y_vs_circuit_delta": float(pearsonr(y, network[:, 0]).statistic),
    }
    results["audit"] = {
        "n_wrf_active_cells": int(len(y)),
        "matched_hour_count": int(len(matched_times)),
        "matched_times": sorted(matched_times),
        "circuit_bin_inflow_deg": 112.5,
        "context_radius_m": 3500,
        "spatial_cv_blocks": 5,
        "circuit_100m_count_min": int(grouped["n_circuit_100m"].min()),
        "circuit_100m_count_median": float(grouped["n_circuit_100m"].median()),
        "circuit_100m_count_max": int(grouped["n_circuit_100m"].max()),
        "max_assignment_distance_m": float(grouped["assignment_max_distance_m"].max()),
        "scope": "Exploratory Phase-I spatial-CV test on one event direction; not yet cross-event validation.",
    }

    table = pd.DataFrame({
        "wrf_j": jj, "wrf_i": ii, "x_l93": active_xy[:, 0], "y_l93": active_xy[:, 1],
        "delta_along_bg_ms": y, "delta_mh_m": dmh, "delta_lp": dlp, "delta_scalar_fai": dlf,
        "context_delta_mh": context[:, 0], "context_delta_lp": context[:, 1],
        "context_delta_scalar_fai": context[:, 2], "delta_circuit_current": network[:, 0],
        "spatial_block": groups,
    })
    for name, pred in predictions.items():
        table[f"pred_{name}"] = pred
    table.to_csv(OUT / "network_added_value_cells.csv", index=False)
    (OUT / "network_added_value_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    nav = results["network_added_value"]
    verdict = "SUPPORTED" if nav["improves_spatial_cv_rmse"] else "NOT SUPPORTED in this Phase-I test"
    report = f"""# Phase-I network-added-value test

## Question

After local and 3.5 km contextual morphology are known, does the 112.5° REAL–HOMO Circuit
contrast improve spatial prediction of the July-18 direction-matched WRF transport response?

## Result

**{verdict}.** M2 spatial-CV RMSE = {m2['rmse']:.4f} m s-1; M3 RMSE = {m3['rmse']:.4f} m s-1.
The relative RMSE change after adding Circuit is {nav['relative_rmse_change'] * 100:.2f}%
(negative means improvement). Cross-validated R2 changes by {nav['delta_r2_m3_minus_m2']:.3f}.

Direct Spearman association between delta-current and delta-transport is
{results['direct_associations']['spearman_y_vs_circuit_delta']:.3f}.

## Scope

This is the first, deliberately small Phase-I diagnostic: 85 active d03 cells, one dominant
direction and one event day. It determines whether the core mechanism is promising; it cannot
establish cross-event generality by itself.
"""
    (OUT / "NETWORK_ADDED_VALUE_PHASE1_REPORT.md").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    sc = axes[0].scatter(network[:, 0], y, c=groups, cmap="tab10", s=48, edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color="0.6", lw=0.8); axes[0].axvline(0, color="0.6", lw=0.8)
    axes[0].set_xlabel("Mean delta Circuit current"); axes[0].set_ylabel("Mean delta along-flow wind (m/s)")
    axes[0].set_title("Direct association")
    names = list(designs)
    axes[1].bar(range(3), [results[n]["rmse"] for n in names], color=["#9ecae1", "#4292c6", "#08519c"])
    axes[1].set_xticks(range(3), ["Local", "+ context", "+ Circuit"]); axes[1].set_ylabel("Spatial-CV RMSE (m/s)")
    axes[1].set_title("Out-of-block prediction")
    axes[2].scatter(y, predictions["M2_local_context3p5km"], label="Morphology", alpha=0.8)
    axes[2].scatter(y, predictions["M3_local_context_circuit"], label="+ Circuit", alpha=0.8)
    lim = [float(min(y.min(), *(p.min() for p in predictions.values()))), float(max(y.max(), *(p.max() for p in predictions.values())))]
    axes[2].plot(lim, lim, color="0.4", ls="--"); axes[2].set_xlim(lim); axes[2].set_ylim(lim)
    axes[2].set_xlabel("Observed delta transport"); axes[2].set_ylabel("Cross-validated prediction")
    axes[2].legend(); axes[2].set_title("Predicted versus WRF")
    fig.suptitle("Does citywide Circuit connectivity add information?")
    fig.savefig(OUT / "network_added_value_phase1.png", dpi=240)
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
