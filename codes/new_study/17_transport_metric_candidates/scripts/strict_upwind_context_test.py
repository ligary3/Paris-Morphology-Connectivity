"""Strict NAV test with directional local, radial-context, and upwind-sector morphology."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
OUT = WORKSPACE / "new_study/17_transport_metric_candidates/analysis_strict_upwind_pbl"
DIRECTIONS = (45.0, 67.5, 90.0, 112.5)
METRICS = ("10_500", "10_800", "10_1000", "pbl_mean", "pbl_int")
SEEDS = tuple(range(10))


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = module_from(WORKSPACE / "new_study/16_multidirection_circuit/scripts/test_cross_direction_added_value.py", "cross")
    transport = np.load(WORKSPACE / "new_study/17_transport_metric_candidates/paired_transport_metrics.npz")
    export = np.load(WORKSPACE / "new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz")
    template = np.load(cross.TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    wx, wy = transformer.transform(export["lon"], export["lat"])
    all_xy = np.column_stack([wx.ravel(), wy.ravel()])
    xy = np.column_stack([wx[active], wy[active]])
    active_flat = np.flatnonzero(active)
    flat_to_active = {int(flat): i for i, flat in enumerate(active_flat)}
    wrf_tree = cKDTree(all_xy)
    local_scalar = np.column_stack([
        (template["REAL_MH_URB2D"] - template["HOMO_MH_URB2D"])[active],
        (template["REAL_BUILD_AREA_FRACTION"] - template["HOMO_BUILD_AREA_FRACTION"])[active],
        (template["REAL_LF_URB2D_S"] - template["HOMO_LF_URB2D_S"])[active],
    ]).astype(float)
    times = np.asarray([str(value) for value in transport["times"]])
    stable = (transport["background_speed_ms"] >= 1) & (transport["background_persistence6h"] >= .9)
    post_spinup = times >= "2022-07-17_00:00:00"
    groups = [KMeans(n_clusters=5, random_state=s, n_init=20).fit_predict(xy) for s in SEEDS]
    rows, audit, saved_inputs = [], [], {}

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
        cells["active_id"] = [flat_to_active.get(int(value), -1) for value in nearest]
        cells = cells[(cells.active_id >= 0) & (distance <= 850)]
        aggregate = cells.groupby("active_id")[["delta_current", "delta_directional_fai"]].mean().reindex(range(len(xy)))
        network = aggregate.delta_current.to_numpy()[:, None]
        local = np.column_stack([local_scalar, aggregate.delta_directional_fai.to_numpy()])

        radial = np.zeros_like(local)
        upwind = np.zeros_like(local)
        for i in range(len(xy)):
            dx = xy[:, 0] - xy[i, 0]
            dy = xy[:, 1] - xy[i, 1]
            distance_m = np.hypot(dx, dy)
            bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
            angle_difference = np.abs((bearing - direction + 180) % 360 - 180)
            radial_neighbors = (distance_m > 0) & (distance_m <= 3500)
            upwind_neighbors = (distance_m > 0) & (distance_m <= 5000) & (angle_difference <= 45)
            radial[i] = np.mean(local[radial_neighbors], axis=0) if radial_neighbors.any() else 0
            upwind[i] = np.mean(local[upwind_neighbors], axis=0) if upwind_neighbors.any() else 0
        baseline = np.column_stack([local, radial, upwind])
        extended = np.column_stack([baseline, network])
        selector = stable & post_spinup & (transport["background_bin_deg"] == direction)
        saved_inputs[f"baseline_{dtag}"] = baseline
        saved_inputs[f"network_{dtag}"] = network
        saved_inputs[f"xy_{dtag}"] = xy
        audit.append({"direction": direction, "post_spinup_hours": int(selector.sum()), "baseline_features": int(baseline.shape[1])})
        for metric in METRICS:
            y = np.mean(transport[f"delta_along_{metric}"][selector][:, active], axis=0)
            saved_inputs[f"y_{metric}_{dtag}"] = y
            for seed, block in zip(SEEDS, groups):
                base = cross.rmse_cv(baseline, y, block)
                extra = cross.rmse_cv(extended, y, block)
                rows.append({"metric": metric, "direction": direction, "n_hours": int(selector.sum()), "seed": seed, "rmse_baseline": base, "rmse_circuit": extra, "relative_rmse_change": (extra-base)/base, "improves": extra < base})
    result = pd.DataFrame(rows)
    np.savez_compressed(OUT / "strict_model_inputs.npz", **saved_inputs)
    result.to_csv(OUT / "strict_upwind_all_tests.csv", index=False)
    detail = result.groupby(["metric", "direction"], as_index=False).agg(improvement_fraction=("improves", "mean"), median_relative_rmse_change=("relative_rmse_change", "median"), p10=("relative_rmse_change", lambda x: np.percentile(x, 10)), p90=("relative_rmse_change", lambda x: np.percentile(x, 90)))
    summary = result.groupby("metric", as_index=False).agg(improvement_fraction=("improves", "mean"), median_relative_rmse_change=("relative_rmse_change", "median"))
    detail.to_csv(OUT / "strict_upwind_detail.csv", index=False)
    summary.to_csv(OUT / "strict_upwind_summary.csv", index=False)
    final = {"status": "STRICT_UPWIND_COMPLETE", "spinup_excluded_before": "2022-07-17_00:00:00", "audit": audit, "summary": summary.to_dict(orient="records")}
    (OUT / "strict_upwind_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2)); print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
