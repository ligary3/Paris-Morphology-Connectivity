"""Estimate hour-specific Circuit added value and relate it to shared atmospheric state."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = _release_path('D:\\研究生\\代码\\论文4\\lunwen4\\new_study\\15_network_added_value\\phase1_1125')
SEEDS = tuple(range(10))


def rmse_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    prediction = np.full_like(y, np.nan, dtype=float)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def main() -> None:
    wrf = np.load(ROOT / "wrf_phase1_export.npz")
    cells = pd.read_csv(ROOT / "network_added_value_cells.csv")
    xy = cells[["x_l93", "y_l93"]].to_numpy()
    local = cells[["delta_mh_m", "delta_lp", "delta_scalar_fai"]].to_numpy()
    context = cells[["context_delta_mh", "context_delta_lp", "context_delta_scalar_fai"]].to_numpy()
    circuit = cells[["delta_circuit_current"]].to_numpy()
    baseline = np.column_stack([local, context])
    extended = np.column_stack([baseline, circuit])

    times = np.asarray([str(value) for value in wrf["all_times"]])
    stable = (
        (wrf["background_bin_deg"] == 112.5)
        & (wrf["background_speed_ms"] >= 1.0)
        & (wrf["background_persistence6h"] >= 0.90)
    )
    indices = np.where(stable)[0]
    rows = []
    group_sets = [KMeans(n_clusters=5, random_state=seed, n_init=20).fit_predict(xy) for seed in SEEDS]
    for index in indices:
        y = wrf["delta_along_all_hourly"][index][cells["wrf_j"].to_numpy(int), cells["wrf_i"].to_numpy(int)]
        changes = []
        for seed, groups in zip(SEEDS, group_sets):
            base_rmse = rmse_cv(baseline, y, groups)
            circuit_rmse = rmse_cv(extended, y, groups)
            changes.append((circuit_rmse - base_rmse) / base_rmse)
        rows.append({
            "time_utc": times[index],
            "hour_utc": int(times[index][11:13]),
            "background_speed_ms": float(wrf["background_speed_ms"][index]),
            "persistence6h": float(wrf["background_persistence6h"][index]),
            "active_pblh_m": float(wrf["common_active_pblh_m"][index]),
            "active_t2_k": float(wrf["common_active_t2_k"][index]),
            "active_ust_ms": float(wrf["common_active_ust_ms"][index]),
            "wrf_delta_spatial_rms_ms": float(np.sqrt(np.mean(y ** 2))),
            "median_relative_rmse_change": float(np.median(changes)),
            "p10_relative_rmse_change": float(np.percentile(changes, 10)),
            "p90_relative_rmse_change": float(np.percentile(changes, 90)),
            "improvement_seed_fraction": float(np.mean(np.asarray(changes) < 0)),
        })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "hourly_network_added_value.csv", index=False)

    outcome = result["median_relative_rmse_change"].to_numpy()
    relationships = {}
    for column in ("background_speed_ms", "active_pblh_m", "active_t2_k", "active_ust_ms", "wrf_delta_spatial_rms_ms"):
        relationships[column] = float(spearmanr(outcome, result[column]).statistic)
    summary = {
        "status": "HOURLY_REGIME_DIAGNOSTIC_COMPLETE",
        "n_stable_1125_hours": int(len(result)),
        "hour_improvement_fraction": float((outcome < 0).mean()),
        "median_relative_rmse_change": float(np.median(outcome)),
        "spearman_rmse_change_vs_state": relationships,
        "note": "Negative RMSE change means Circuit improved spatially blocked prediction.",
    }
    (ROOT / "hourly_network_regime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = np.where(outcome < 0, "#238b45", "#cb181d")
    axes[0].bar(np.arange(len(result)), outcome * 100, color=colors)
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].set_xticks(np.arange(len(result))[::3], result["time_utc"].str[5:16].iloc[::3], rotation=60, ha="right")
    axes[0].set_ylabel("RMSE change after Circuit (%)")
    axes[0].set_title("Hour-specific added value")
    for ax, column, label in (
        (axes[1], "active_pblh_m", "Shared Paris PBLH (m)"),
        (axes[2], "background_speed_ms", "Shared background wind (m/s)"),
    ):
        ax.scatter(result[column], outcome * 100, c=colors, s=46, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel(label); ax.set_ylabel("RMSE change after Circuit (%)")
    axes[1].set_title(f"PBLH: Spearman {relationships['active_pblh_m']:.2f}")
    axes[2].set_title(f"Wind: Spearman {relationships['background_speed_ms']:.2f}")
    fig.suptitle("When does Circuit add information beyond local and contextual morphology?")
    fig.savefig(ROOT / "hourly_network_regime_diagnostic.png", dpi=240)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
