"""Derive paired 10/100/300 m and 10–500 m WRF transport metrics for all 97 hours."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import csv
from pathlib import Path

import netCDF4 as nc
import numpy as np


ROOT = _release_path('/home/gary/wrf-work/cases')
OUT = _release_path('/mnt/d/研究生/代码/论文4/lunwen4/new_study/17_transport_metric_candidates')
BACKGROUND = _release_path('/mnt/d/研究生/代码/论文4/lunwen4/new_study/13_wrf_homo_pair/full96h_analysis/d02_background_wind_hourly.csv')
REAL = [ROOT / "paris_real_pair24h_v1", ROOT / "paris_real_core48h_v1", ROOT / "paris_real_final24h_v1"]
HOMO = [ROOT / "paris_homo_pair24h_v1", ROOT / "paris_homo_core48h_v1", ROOT / "paris_homo_final24h_v1", ROOT / "paris_homo_resume03_v1"]
G = 9.81


def interpolate(profile: np.ndarray, height: np.ndarray, target: float) -> np.ndarray:
    below = np.sum(height <= target, axis=0) - 1
    below = np.clip(below, 0, height.shape[0] - 2)
    above = below + 1
    z0 = np.take_along_axis(height, below[None], axis=0)[0]
    z1 = np.take_along_axis(height, above[None], axis=0)[0]
    v0 = np.take_along_axis(profile, below[None], axis=0)[0]
    v1 = np.take_along_axis(profile, above[None], axis=0)[0]
    weight = np.divide(target - z0, z1 - z0, out=np.zeros_like(z0), where=np.abs(z1-z0) > 1e-6)
    return v0 + np.clip(weight, 0, 1) * (v1 - v0)


def integrate_to_top(samples: np.ndarray, heights: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Piecewise-linear vertical integral from 10 m to a spatially varying top."""
    clipped_top = np.clip(top, heights[0] + 1.0, heights[-1])
    total = np.zeros_like(clipped_top, dtype=float)
    for index in range(len(heights) - 1):
        h0, h1 = heights[index], heights[index + 1]
        extent = np.clip(clipped_top - h0, 0.0, h1 - h0)
        fraction = extent / (h1 - h0)
        value_end = samples[index] + fraction * (samples[index + 1] - samples[index])
        total += 0.5 * extent * (samples[index] + value_end)
    return total


def derive(path: Path) -> dict[str, dict[str, np.ndarray]]:
    records = {}
    with nc.Dataset(path) as ds:
        times = [str(x) for x in nc.chartostring(ds.variables["Times"][:]).tolist()]
        for t, time in enumerate(times):
            u_grid = 0.5 * (np.asarray(ds.variables["U"][t, :, :, :-1], dtype=float) + np.asarray(ds.variables["U"][t, :, :, 1:], dtype=float))
            v_grid = 0.5 * (np.asarray(ds.variables["V"][t, :, :-1, :], dtype=float) + np.asarray(ds.variables["V"][t, :, 1:, :], dtype=float))
            cos = np.asarray(ds.variables["COSALPHA"][t], dtype=float)
            sin = np.asarray(ds.variables["SINALPHA"][t], dtype=float)
            u = u_grid * cos - v_grid * sin
            v = v_grid * cos + u_grid * sin
            u10g = np.asarray(ds.variables["U10"][t], dtype=float)
            v10g = np.asarray(ds.variables["V10"][t], dtype=float)
            u10 = u10g * cos - v10g * sin
            v10 = v10g * cos + u10g * sin
            zstag = (np.asarray(ds.variables["PH"][t], dtype=float) + np.asarray(ds.variables["PHB"][t], dtype=float)) / G
            terrain = np.asarray(ds.variables["HGT"][t], dtype=float)
            z = 0.5 * (zstag[:-1] + zstag[1:]) - terrain
            z_profile = np.concatenate([np.full((1, *terrain.shape), 10.0), z], axis=0)
            u_profile = np.concatenate([u10[None], u], axis=0)
            v_profile = np.concatenate([v10[None], v], axis=0)
            heights = np.asarray([10., 50., 100., 200., 300., 400., 500., 800., 1000., 1500., 2000., 3000.])
            us = np.stack([interpolate(u_profile, z_profile, value) for value in heights])
            vs = np.stack([interpolate(v_profile, z_profile, value) for value in heights])
            pblh = np.asarray(ds.variables["PBLH"][t], dtype=float)
            u_pbl_integral = integrate_to_top(us, heights, pblh)
            v_pbl_integral = integrate_to_top(vs, heights, pblh)
            pbl_depth = np.clip(pblh, heights[0] + 1.0, heights[-1]) - heights[0]
            records[time] = {
                "u10": u10, "v10": v10,
                "u100": us[2], "v100": vs[2],
                "u300": us[4], "v300": vs[4],
                "u10_300": np.trapz(us[:5], heights[:5], axis=0) / 290.0,
                "v10_300": np.trapz(vs[:5], heights[:5], axis=0) / 290.0,
                "u10_500": np.trapz(us[:7], heights[:7], axis=0) / 490.0,
                "v10_500": np.trapz(vs[:7], heights[:7], axis=0) / 490.0,
                "u10_800": np.trapz(us[:8], heights[:8], axis=0) / 790.0,
                "v10_800": np.trapz(vs[:8], heights[:8], axis=0) / 790.0,
                "u10_1000": np.trapz(us[:9], heights[:9], axis=0) / 990.0,
                "v10_1000": np.trapz(vs[:9], heights[:9], axis=0) / 990.0,
                "upbl_mean": u_pbl_integral / pbl_depth,
                "vpbl_mean": v_pbl_integral / pbl_depth,
                "upbl_int": u_pbl_integral,
                "vpbl_int": v_pbl_integral,
            }
    return records


def assemble(segments):
    output = {}
    for segment in segments:
        for path in sorted(segment.glob("wrfout_d03_*")):
            print("reading", path, flush=True)
            for time, record in derive(path).items():
                if time in output:
                    maximum = max(float(np.max(np.abs(record[k] - output[time][k]))) for k in record)
                    if maximum > 1e-4:
                        raise ValueError(f"Nonidentical duplicate {time}: {maximum}")
                else:
                    output[time] = record
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real, homo = assemble(REAL), assemble(HOMO)
    times = sorted(real)
    if times != sorted(homo) or len(times) != 97:
        raise ValueError(f"Timeline mismatch REAL={len(real)} HOMO={len(homo)}")
    with BACKGROUND.open() as stream:
        bg = {row["time_utc"]: row for row in csv.DictReader(stream)}
    result = {}
    for name in ("10", "100", "300", "10_300", "10_500", "10_800", "10_1000", "pbl_mean", "pbl_int"):
        ukey, vkey = f"u{name}", f"v{name}"
        delta = []
        for time in times:
            angle = np.deg2rad(float(bg[time]["transport_to_deg"]))
            delta.append((real[time][ukey] - homo[time][ukey]) * np.sin(angle) + (real[time][vkey] - homo[time][vkey]) * np.cos(angle))
        result[f"delta_along_{name}"] = np.stack(delta).astype(np.float32)
    np.savez_compressed(
        OUT / "paired_transport_metrics.npz",
        times=np.asarray(times),
        background_bin_deg=np.asarray([float(bg[t]["circuit_inflow_bin_deg"]) for t in times]),
        background_speed_ms=np.asarray([float(bg[t]["speed_ms"]) for t in times]),
        background_persistence6h=np.asarray([float(bg[t]["persistence6h"]) for t in times]),
        **result,
    )
    print("COMPLETE", OUT / "paired_transport_metrics.npz")


if __name__ == "__main__":
    main()
