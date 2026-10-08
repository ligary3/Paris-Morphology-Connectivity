"""Test Circuit added value across the five dominant stable inflow-direction stages."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
ROOT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')
MULTI = ROOT / "circuit_multidirection_pairs_v1"
PAIR1125 = ROOT / "circuit_real_homo_1125_v1" / "analysis" / "circuit_real_homo_cellwise.csv"
OUT = WORKSPACE / "new_study/16_multidirection_circuit/cross_direction_analysis"
TEMPLATE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_activation_pilot_v1\\injection_templates\\paris_d03_real_homo_injection_templates.npz')
DIRECTIONS = (22.5, 45.0, 67.5, 90.0, 112.5)
RADII = (1500.0, 2500.0, 3500.0, 5500.0)
SEEDS = tuple(range(5))


def tag(direction):
    return f"{int(round(direction * 10)):04d}"


def read_ascii(path):
    header = {}
    with path.open("r", encoding="ascii") as stream:
        for _ in range(6):
            key, value = stream.readline().split()
            header[key.lower()] = float(value)
        values = np.loadtxt(stream)
    return values, header


def sample_grid(array, header, cells):
    size = header["cellsize"]
    col = np.rint((cells.x_center.to_numpy() - header["xllcorner"] - size / 2) / size).astype(int)
    south = np.rint((cells.y_center.to_numpy() - header["yllcorner"] - size / 2) / size).astype(int)
    row = int(header["nrows"]) - 1 - south
    return array[row, col]


def rmse_cv(x, y, groups):
    pred = np.full_like(y, np.nan, dtype=float)
    for group in np.unique(groups):
        test, train = groups == group, groups != group
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wrf = np.load(WORKSPACE / "new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz")
    template = np.load(TEMPLATE)
    active = template["PRIMARY_ACTIVATION"].astype(bool) & template["TREATMENT_CORE"].astype(bool)
    jj, ii = np.where(active)
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    wx, wy = transformer.transform(wrf["lon"], wrf["lat"])
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
    times = np.asarray([str(v) for v in wrf["all_times"]])
    stable = (wrf["background_speed_ms"] >= 1) & (wrf["background_persistence6h"] >= .9)
    rows, qa = [], []
    direction_vectors = {}
    for direction in DIRECTIONS:
        dtag = tag(direction)
        if direction == 112.5:
            cells = pd.read_csv(PAIR1125)
            cells = cells[cells.analysis_core.astype(bool)].copy()
            cells["delta_current"] = cells.current_real - cells.current_homo
            rho_network = float(spearmanr(cells.current_real, cells.current_homo).statistic)
        else:
            folder = MULTI / dtag
            cells = pd.read_csv(folder / "cells.csv")
            cells = cells[cells.analysis_core.astype(bool)].copy()
            real, rh = read_ascii(folder / "real" / f"real_{dtag}_cum_curmap.asc")
            homo, hh = read_ascii(folder / "homo" / f"homo_{dtag}_cum_curmap.asc")
            cells["current_real"] = sample_grid(real, rh, cells)
            cells["current_homo"] = sample_grid(homo, hh, cells)
            cells["delta_current"] = cells.current_real - cells.current_homo
            rho_network = float(spearmanr(cells.current_real, cells.current_homo).statistic)
        distance, nearest = wrf_tree.query(cells[["x_center", "y_center"]].to_numpy())
        cells["active_id"] = [flat_to_active.get(int(v), -1) for v in nearest]
        cells = cells[(cells.active_id >= 0) & (distance <= 850)].copy()
        network = cells.groupby("active_id").delta_current.mean().reindex(range(len(jj)))
        if network.isna().any():
            raise ValueError(f"Missing WRF aggregation for direction {direction}")
        direction_vectors[direction] = network.to_numpy()[:, None]
        selector = stable & (wrf["background_bin_deg"] == direction)
        y = np.mean(wrf["delta_along_all_hourly"][selector][:, active], axis=0)
        qa.append({
            "direction": direction, "n_hours": int(selector.sum()), "network_real_homo_spearman": rho_network,
            "direct_y_network_spearman": float(spearmanr(y, network).statistic),
        })
        near = cKDTree(active_xy)
        for radius in RADII:
            context = np.zeros_like(local)
            for i, point in enumerate(active_xy):
                neighbors = [n for n in near.query_ball_point(point, radius) if n != i]
                context[i] = np.mean(local[neighbors], axis=0) if neighbors else 0
            baseline = np.column_stack([local, context])
            extended = np.column_stack([baseline, network.to_numpy()])
            for seed in SEEDS:
                groups = KMeans(n_clusters=5, random_state=seed, n_init=20).fit_predict(active_xy)
                base = rmse_cv(baseline, y, groups)
                extra = rmse_cv(extended, y, groups)
                rows.append({"direction": direction, "n_hours": int(selector.sum()), "radius_m": radius, "seed": seed, "rmse_baseline": base, "rmse_circuit": extra, "relative_rmse_change": (extra-base)/base, "improves": extra < base})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cross_direction_all_tests.csv", index=False)
    summary = result.groupby("direction", as_index=False).agg(n_tests=("improves", "size"), improvement_fraction=("improves", "mean"), median_relative_rmse_change=("relative_rmse_change", "median"), p10=("relative_rmse_change", lambda x: np.percentile(x, 10)), p90=("relative_rmse_change", lambda x: np.percentile(x, 90)))
    summary.to_csv(OUT / "cross_direction_summary.csv", index=False)
    final = {"status": "CROSS_DIRECTION_COMPLETE", "direction_qa": qa, "overall_improvement_fraction": float(result.improves.mean()), "overall_median_relative_rmse_change": float(result.relative_rmse_change.median()), "directions_improving_at_median": [float(row.direction) for row in summary.itertuples() if row.median_relative_rmse_change < 0]}
    (OUT / "cross_direction_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2)); print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
