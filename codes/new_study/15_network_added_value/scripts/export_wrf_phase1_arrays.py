"""Run inside WSL to export the matched July-18 WRF response for Windows-side analysis."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


WORKSPACE = _release_path('/mnt/d/研究生/代码/论文4/lunwen4')
OUT = WORKSPACE / "new_study/15_network_added_value/phase1_1125"
DIAGNOSTICS = WORKSPACE / "new_study/13_wrf_homo_pair/full96h_analysis/full96h_diagnostics.json"
BACKGROUND = WORKSPACE / "new_study/13_wrf_homo_pair/full96h_analysis/d02_background_wind_hourly.csv"
WRF_SAMPLE = _release_path('/home/gary/wrf-work/cases/paris_real_pair24h_v1/wrfout_d03_2022-07-16_00:00:00')
TEMPLATE = _release_path('/mnt/e/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')


def helper_module():
    source = WORKSPACE / "new_study/13_wrf_homo_pair/scripts/analyze_full96h_regimes.py"
    spec = importlib.util.spec_from_file_location("full96h", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    helper = helper_module()
    real_t, real, lat, lon, _ = helper.load_segments(helper.REAL_SEGMENTS, 3, helper.D03_FIELDS)
    homo_t, homo, _, _, _ = helper.load_segments(helper.HOMO_SEGMENTS, 3, helper.D03_FIELDS)
    if real_t != homo_t:
        raise ValueError("Unpaired WRF timeline")
    matched_times = set(json.loads(DIAGNOSTICS.read_text())["july18_regime"]["dominant_matched_hours"])
    matched = np.asarray([i for i, time in enumerate(real_t) if time in matched_times], dtype=int)
    if len(matched) != 18:
        raise ValueError(f"Expected 18 matched hours, got {len(matched)}")

    background_by_time = {}
    with BACKGROUND.open() as stream:
        header = stream.readline().strip().split(",")
        for line in stream:
            row = dict(zip(header, line.strip().split(",")))
            background_by_time[row["time_utc"]] = row
    to_rad = np.deg2rad(np.asarray([float(background_by_time[t]["transport_to_deg"]) for t in real_t]))
    ru, rv = helper.earth_wind(real)
    hu, hv = helper.earth_wind(homo)
    along = (ru - hu) * np.sin(to_rad)[:, None, None] + (rv - hv) * np.cos(to_rad)[:, None, None]
    masks = np.load(TEMPLATE)
    active = masks["PRIMARY_ACTIVATION"].astype(bool) & masks["TREATMENT_CORE"].astype(bool)

    with nc.Dataset(WRF_SAMPLE) as ds:
        sample_lat = np.asarray(ds.variables["XLAT"][0], dtype=np.float64)
        sample_lon = np.asarray(ds.variables["XLONG"][0], dtype=np.float64)
    if not np.allclose(sample_lat, lat) or not np.allclose(sample_lon, lon):
        raise ValueError("WRF sample coordinates differ from assembled output")
    np.savez_compressed(
        OUT / "wrf_phase1_export.npz",
        lat=lat,
        lon=lon,
        delta_along_matched_mean=np.mean(along[matched], axis=0),
        delta_along_matched_hourly=along[matched],
        matched_times=np.asarray(sorted(matched_times)),
        delta_along_all_hourly=along,
        all_times=np.asarray(real_t),
        background_speed_ms=np.asarray([float(background_by_time[t]["speed_ms"]) for t in real_t]),
        background_bin_deg=np.asarray([float(background_by_time[t]["circuit_inflow_bin_deg"]) for t in real_t]),
        background_persistence6h=np.asarray([float(background_by_time[t]["persistence6h"]) for t in real_t]),
        common_active_pblh_m=np.mean(0.5 * (real["PBLH"] + homo["PBLH"])[:, active], axis=1),
        common_active_t2_k=np.mean(0.5 * (real["T2"] + homo["T2"])[:, active], axis=1),
        common_active_ust_ms=np.mean(0.5 * (real["UST"] + homo["UST"])[:, active], axis=1),
    )
    print("exported", OUT / "wrf_phase1_export.npz", "matched_hours", len(matched))


if __name__ == "__main__":
    main()
