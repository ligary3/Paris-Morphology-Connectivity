"""Compare LCZ-network and detailed 3-D-network added value using the frozen strict baseline."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
LCZ_ROOT = _release_path('E:\\lunwen4\\2\\LCZ\\Paris\\Cerema_2022\\circuit_benchmark_v1')
OUT = LCZ_ROOT / "analysis"
STRICT = WORKSPACE / "new_study" / "17_transport_metric_candidates" / "analysis_strict_upwind_pbl" / "strict_model_inputs.npz"
DIRECTIONS = (45.0, 67.5, 90.0, 112.5)
SEEDS = tuple(range(10))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def top_overlap(a: np.ndarray, b: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(a) * fraction)))
    ia = set(np.argpartition(a, -n)[-n:])
    ib = set(np.argpartition(b, -n)[-n:])
    return len(ia & ib) / n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cross = load_module(WORKSPACE / "new_study" / "16_multidirection_circuit" / "scripts" / "test_cross_direction_added_value.py", "cross_lcz")
    data = np.load(STRICT)
    template = np.load(cross.TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    export = np.load(WORKSPACE / "new_study" / "15_network_added_value" / "phase1_1125" / "wrf_phase1_export.npz")
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    wx, wy = transformer.transform(export["lon"], export["lat"])
    all_xy = np.column_stack([wx.ravel(), wy.ravel()])
    xy = np.column_stack([wx[active], wy[active]])
    active_flat = np.flatnonzero(active)
    flat_to_active = {int(value): index for index, value in enumerate(active_flat)}
    wrf_tree = cKDTree(all_xy)
    groups = [KMeans(n_clusters=5, random_state=seed, n_init=20).fit_predict(xy) for seed in SEEDS]

    rows, network_audit = [], []
    for direction in DIRECTIONS:
        dtag = cross.tag(direction)
        cells = pd.read_csv(LCZ_ROOT / dtag / "cells.csv")
        cells = cells[cells.analysis_core.astype(bool)].copy()
        lcz_current, lcz_header = cross.read_ascii(LCZ_ROOT / dtag / f"lcz_{dtag}_cum_curmap.asc")
        cells["current_lcz"] = cross.sample_grid(lcz_current, lcz_header, cells)

        if direction == 112.5:
            old = pd.read_csv(cross.PAIR1125)[["grid_id", "current_real", "current_homo"]]
            cells = cells.merge(old, on="grid_id", how="left")
        else:
            old_folder = cross.MULTI / dtag
            real, real_header = cross.read_ascii(old_folder / "real" / f"real_{dtag}_cum_curmap.asc")
            homo, homo_header = cross.read_ascii(old_folder / "homo" / f"homo_{dtag}_cum_curmap.asc")
            cells["current_real"] = cross.sample_grid(real, real_header, cells)
            cells["current_homo"] = cross.sample_grid(homo, homo_header, cells)
        cells["delta_current_lcz"] = cells.current_lcz - cells.current_homo
        cells["delta_current_3d"] = cells.current_real - cells.current_homo

        distance, nearest = wrf_tree.query(cells[["x_center", "y_center"]].to_numpy())
        cells["active_id"] = [flat_to_active.get(int(value), -1) for value in nearest]
        cells = cells[(cells.active_id >= 0) & (distance <= 850)].copy()
        aggregate = cells.groupby("active_id")[["delta_current_lcz", "delta_current_3d"]].mean().reindex(range(len(xy)))
        if aggregate.isna().any().any():
            raise ValueError(f"Missing WRF-cell network aggregation for {direction}")
        lcz_network = aggregate.delta_current_lcz.to_numpy()[:, None]
        detailed_network = aggregate.delta_current_3d.to_numpy()[:, None]
        frozen_detailed = data[f"network_{dtag}"]
        if not np.allclose(detailed_network, frozen_detailed, rtol=1e-6, atol=1e-9):
            raise AssertionError(f"Detailed 3-D network differs from frozen strict input for {direction}")

        baseline = data[f"baseline_{dtag}"]
        y = data[f"y_10_1000_{dtag}"]
        for seed, blocks in zip(SEEDS, groups):
            rmse_base = cross.rmse_cv(baseline, y, blocks)
            rmse_lcz = cross.rmse_cv(np.column_stack([baseline, lcz_network]), y, blocks)
            rmse_3d = cross.rmse_cv(np.column_stack([baseline, detailed_network]), y, blocks)
            rows.append({
                "direction": direction,
                "seed": seed,
                "rmse_baseline": rmse_base,
                "rmse_lcz": rmse_lcz,
                "rmse_3d": rmse_3d,
                "relative_change_lcz": (rmse_lcz - rmse_base) / rmse_base,
                "relative_change_3d": (rmse_3d - rmse_base) / rmse_base,
                "lcz_improves": rmse_lcz < rmse_base,
                "detailed_3d_improves": rmse_3d < rmse_base,
                "detailed_3d_beats_lcz": rmse_3d < rmse_lcz,
            })
        network_audit.append({
            "direction": direction,
            "network_delta_spearman_lcz_vs_3d": float(spearmanr(lcz_network[:, 0], detailed_network[:, 0]).statistic),
            "top10_overlap": top_overlap(lcz_network[:, 0], detailed_network[:, 0], .10),
            "top05_overlap": top_overlap(lcz_network[:, 0], detailed_network[:, 0], .05),
            "direct_y_spearman_lcz": float(spearmanr(y, lcz_network[:, 0]).statistic),
            "direct_y_spearman_3d": float(spearmanr(y, detailed_network[:, 0]).statistic),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "lcz_vs_3d_nav_all_folds.csv", index=False)
    summary = result.groupby("direction", as_index=False).agg(
        median_relative_change_lcz=("relative_change_lcz", "median"),
        median_relative_change_3d=("relative_change_3d", "median"),
        lcz_improvement_fraction=("lcz_improves", "mean"),
        detailed_3d_improvement_fraction=("detailed_3d_improves", "mean"),
        detailed_3d_beats_lcz_fraction=("detailed_3d_beats_lcz", "mean"),
    )
    summary.to_csv(OUT / "lcz_vs_3d_nav_summary.csv", index=False)
    final = {
        "status": "LCZ_VS_3D_NAV_COMPLETE",
        "target": "Post-spinup stable-hour, direction-matched WRF REAL-HOMO 10-1000 m mean along-flow transport response",
        "baseline": "Frozen strict baseline: local H/LP/scalar and directional FAI, 3.5 km radial context, and 5 km +/-45 degree upwind context",
        "network_audit": network_audit,
        "overall": {
            "median_relative_change_lcz": float(result.relative_change_lcz.median()),
            "median_relative_change_3d": float(result.relative_change_3d.median()),
            "lcz_improvement_fraction": float(result.lcz_improves.mean()),
            "detailed_3d_improvement_fraction": float(result.detailed_3d_improves.mean()),
            "detailed_3d_beats_lcz_fraction": float(result.detailed_3d_beats_lcz.mean()),
        },
    }
    (OUT / "lcz_vs_3d_nav_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    x = np.arange(len(summary)); width = .34
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.bar(x - width/2, summary.median_relative_change_lcz * 100, width, label="LCZ categorical network", color="#9ecae1")
    ax.bar(x + width/2, summary.median_relative_change_3d * 100, width, label="Detailed 3-D network", color="#08519c")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, [f"{value:g}°" for value in summary.direction])
    ax.set_xlabel("Inflow direction")
    ax.set_ylabel("Median cross-validated RMSE change (%)")
    ax.set_title("Network added value: LCZ benchmark versus detailed 3-D morphology")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "lcz_vs_3d_nav.png", dpi=240)
    plt.close(fig)
    print(json.dumps(final, indent=2)); print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
