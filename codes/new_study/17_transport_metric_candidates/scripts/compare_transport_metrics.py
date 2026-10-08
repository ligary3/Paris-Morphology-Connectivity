"""Compare Circuit added value for near-surface and low-layer WRF transport targets."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
ROOT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')
OUT = WORKSPACE / "new_study/17_transport_metric_candidates/analysis_directional_baseline"
DIRECTIONS = (22.5, 45.0, 67.5, 90.0, 112.5)
METRICS = ("10", "100", "300", "10_300", "10_500", "10_800", "10_1000")
SEEDS = tuple(range(5))


def load_cross_module():
    path = WORKSPACE / "new_study/16_multidirection_circuit/scripts/test_cross_direction_added_value.py"
    spec = importlib.util.spec_from_file_location("cross", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = load_cross_module()
    transport = np.load(WORKSPACE / "new_study/17_transport_metric_candidates/paired_transport_metrics.npz")
    export = np.load(WORKSPACE / "new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz")
    template = np.load(cross.TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    wx, wy = transformer.transform(export["lon"], export["lat"])
    all_xy = np.column_stack([wx.ravel(), wy.ravel()])
    active_xy = np.column_stack([wx[active], wy[active]])
    active_flat = np.flatnonzero(active)
    flat_to_active = {int(flat): i for i, flat in enumerate(active_flat)}
    wrf_tree = cKDTree(all_xy)

    local = np.column_stack([
        (template["REAL_MH_URB2D"] - template["HOMO_MH_URB2D"])[active],
        (template["REAL_BUILD_AREA_FRACTION"] - template["HOMO_BUILD_AREA_FRACTION"])[active],
        (template["REAL_LF_URB2D_S"] - template["HOMO_LF_URB2D_S"])[active],
    ]).astype(float)
    near = cKDTree(active_xy)
    groups_by_seed = [KMeans(n_clusters=5, random_state=s, n_init=20).fit_predict(active_xy) for s in SEEDS]

    networks = {}
    directional_fai = {}
    for direction in DIRECTIONS:
        dtag = cross.tag(direction)
        if direction == 112.5:
            cells = pd.read_csv(cross.PAIR1125)
            cells = cells[cells.analysis_core.astype(bool)].copy()
            cells["delta_current"] = cells.current_real - cells.current_homo
            cells["delta_directional_fai"] = cells.lambda_f_real - cells.lambda_f_homo
        else:
            folder = cross.MULTI / dtag
            cells = pd.read_csv(folder / "cells.csv")
            cells = cells[cells.analysis_core.astype(bool)].copy()
            real, rh = cross.read_ascii(folder / "real" / f"real_{dtag}_cum_curmap.asc")
            homo, hh = cross.read_ascii(folder / "homo" / f"homo_{dtag}_cum_curmap.asc")
            cells["delta_current"] = cross.sample_grid(real, rh, cells) - cross.sample_grid(homo, hh, cells)
            cells["delta_directional_fai"] = cells.fai_real - cells.fai_homo
        distance, nearest = wrf_tree.query(cells[["x_center", "y_center"]].to_numpy())
        cells["active_id"] = [flat_to_active.get(int(v), -1) for v in nearest]
        cells = cells[(cells.active_id >= 0) & (distance <= 850)]
        networks[direction] = cells.groupby("active_id").delta_current.mean().reindex(range(int(active.sum()))).to_numpy()
        directional_fai[direction] = cells.groupby("active_id").delta_directional_fai.mean().reindex(range(int(active.sum()))).to_numpy()

    stable = (transport["background_speed_ms"] >= 1) & (transport["background_persistence6h"] >= .9)
    rows = []
    for metric in METRICS:
        response = transport[f"delta_along_{metric}"]
        for direction in DIRECTIONS:
            selector = stable & (transport["background_bin_deg"] == direction)
            y = np.mean(response[selector][:, active], axis=0)
            network = networks[direction][:, None]
            local_directional = np.column_stack([local, directional_fai[direction]])
            context = np.zeros_like(local_directional)
            for i, point in enumerate(active_xy):
                neighbors = [n for n in near.query_ball_point(point, 3500) if n != i]
                context[i] = np.mean(local_directional[neighbors], axis=0) if neighbors else 0
            baseline = np.column_stack([local_directional, context])
            extended = np.column_stack([baseline, network])
            direct = float(spearmanr(y, network[:, 0]).statistic)
            for seed, groups in zip(SEEDS, groups_by_seed):
                base = cross.rmse_cv(baseline, y, groups)
                extra = cross.rmse_cv(extended, y, groups)
                rows.append({"metric": metric, "direction": direction, "n_hours": int(selector.sum()), "seed": seed, "direct_spearman": direct, "rmse_baseline": base, "rmse_circuit": extra, "relative_rmse_change": (extra-base)/base, "improves": extra < base})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "transport_metric_all_tests.csv", index=False)
    summary_metric = result.groupby("metric", as_index=False).agg(improvement_fraction=("improves", "mean"), median_relative_rmse_change=("relative_rmse_change", "median"), median_direct_spearman=("direct_spearman", "median"))
    summary_detail = result.groupby(["metric", "direction"], as_index=False).agg(improvement_fraction=("improves", "mean"), median_relative_rmse_change=("relative_rmse_change", "median"), direct_spearman=("direct_spearman", "first"))
    summary_metric.to_csv(OUT / "transport_metric_summary.csv", index=False)
    summary_detail.to_csv(OUT / "transport_metric_direction_summary.csv", index=False)
    final = {"status": "TRANSPORT_METRIC_COMPARISON_COMPLETE", "best_metric_by_median_rmse_change": str(summary_metric.sort_values("median_relative_rmse_change").iloc[0].metric), "summary": summary_metric.to_dict(orient="records")}
    (OUT / "transport_metric_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2)); print(summary_detail.to_string(index=False))


if __name__ == "__main__":
    main()
