from __future__ import annotations

from release_paths import resolve_path as _release_path

import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc
import numpy as np


ROOT = _release_path('/home/gary/wrf-work/cases')
OUT = _release_path('/mnt/d/研究生/代码/论文4/lunwen4/new_study/13_wrf_homo_pair/full96h_analysis')
FIG = OUT / "figures"
TEMPLATE = _release_path('/mnt/e/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')

REAL_SEGMENTS = [
    ROOT / "paris_real_pair24h_v1",
    ROOT / "paris_real_core48h_v1",
    ROOT / "paris_real_final24h_v1",
]
HOMO_SEGMENTS = [
    ROOT / "paris_homo_pair24h_v1",
    ROOT / "paris_homo_core48h_v1",
    ROOT / "paris_homo_final24h_v1",
    ROOT / "paris_homo_resume03_v1",
]

D03_FIELDS = ("U10", "V10", "COSALPHA", "SINALPHA", "T2", "PBLH", "UST", "HFX")
D02_FIELDS = ("U10", "V10", "COSALPHA", "SINALPHA")


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d_%H:%M:%S")


def load_segments(segments: list[Path], domain: int, fields: tuple[str, ...]):
    records: dict[str, dict[str, np.ndarray]] = {}
    sources: dict[str, list[str]] = {}
    duplicate_max_abs: dict[str, float] = {}
    lat = lon = None
    files: list[str] = []

    for segment in segments:
        for path in sorted(segment.glob(f"wrfout_d{domain:02d}_*")):
            files.append(str(path))
            with nc.Dataset(path) as ds:
                times = [str(x) for x in nc.chartostring(ds.variables["Times"][:]).tolist()]
                if lat is None:
                    lat = np.asarray(ds.variables["XLAT"][0], dtype=np.float64)
                    lon = np.asarray(ds.variables["XLONG"][0], dtype=np.float64)
                loaded = {field: np.asarray(ds.variables[field][:], dtype=np.float64) for field in fields}
                for index, time in enumerate(times):
                    current = {field: loaded[field][index] for field in fields}
                    sources.setdefault(time, []).append(str(path))
                    if time in records:
                        max_abs = max(
                            float(np.max(np.abs(current[field] - records[time][field])))
                            for field in fields
                        )
                        duplicate_max_abs[time] = max(duplicate_max_abs.get(time, 0.0), max_abs)
                    else:
                        records[time] = current

    ordered = sorted(records, key=parse_time)
    arrays = {field: np.stack([records[time][field] for time in ordered]) for field in fields}
    return ordered, arrays, lat, lon, {
        "files": files,
        "duplicate_times": {time: sources[time] for time in sources if len(sources[time]) > 1},
        "duplicate_max_abs": duplicate_max_abs,
    }


def earth_wind(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    u = data["U10"] * data["COSALPHA"] - data["V10"] * data["SINALPHA"]
    v = data["V10"] * data["COSALPHA"] + data["U10"] * data["SINALPHA"]
    return u, v


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    radius = 6371.0
    p1 = np.deg2rad(lat)
    p0 = np.deg2rad(lat0)
    dp = p1 - p0
    dl = np.deg2rad(lon - lon0)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p0) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def bearing_to(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def nearest_direction(direction: np.ndarray) -> np.ndarray:
    return (np.round(direction / 22.5) * 22.5) % 360.0


def trailing_persistence(u: np.ndarray, v: np.ndarray, width: int = 6) -> np.ndarray:
    speed = np.hypot(u, v)
    unit_u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 1.0e-9)
    unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 1.0e-9)
    result = np.empty_like(speed)
    for i in range(len(speed)):
        start = max(0, i - width + 1)
        result[i] = np.hypot(np.mean(unit_u[start:i + 1]), np.mean(unit_v[start:i + 1]))
    return result


def expected_times() -> list[str]:
    start = datetime(2022, 7, 16, 0)
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%d_%H:%M:%S") for i in range(97)]


def sample_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sample = values[mask]
    return {
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "rms": float(np.sqrt(np.mean(sample ** 2))),
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    real3_t, real3, lat3, lon3, real3_meta = load_segments(REAL_SEGMENTS, 3, D03_FIELDS)
    homo3_t, homo3, _, _, homo3_meta = load_segments(HOMO_SEGMENTS, 3, D03_FIELDS)
    real2_t, real2, lat2, lon2, real2_meta = load_segments(REAL_SEGMENTS, 2, D02_FIELDS)
    homo2_t, homo2, _, _, homo2_meta = load_segments(HOMO_SEGMENTS, 2, D02_FIELDS)

    expected = expected_times()
    coverage = {
        "expected_hourly_records": 97,
        "real_d03_records": len(real3_t),
        "homo_d03_records": len(homo3_t),
        "real_d02_records": len(real2_t),
        "homo_d02_records": len(homo2_t),
        "real_d03_exact_timeline": real3_t == expected,
        "homo_d03_exact_timeline": homo3_t == expected,
        "real_d02_exact_timeline": real2_t == expected,
        "homo_d02_exact_timeline": homo2_t == expected,
        "paired_d03_times_equal": real3_t == homo3_t,
        "paired_d02_times_equal": real2_t == homo2_t,
    }
    if not all(coverage[key] for key in coverage if key.endswith("timeline") or key.startswith("paired_")):
        raise AssertionError(json.dumps(coverage, indent=2))

    masks = np.load(TEMPLATE)
    treatment = masks["TREATMENT_CORE"].astype(bool)
    active = masks["PRIMARY_ACTIVATION"].astype(bool) & treatment
    lat0 = float(np.mean(lat3[active]))
    lon0 = float(np.mean(lon3[active]))

    r2u, r2v = earth_wind(real2)
    h2u, h2v = earth_wind(homo2)
    shared_max_abs = float(max(np.max(np.abs(r2u - h2u)), np.max(np.abs(r2v - h2v))))
    distance2 = haversine_km(lat2, lon2, lat0, lon0)
    background_mask = (distance2 >= 15.0) & (distance2 <= 35.0)
    bg_u = np.mean(0.5 * (r2u + h2u)[:, background_mask], axis=1)
    bg_v = np.mean(0.5 * (r2v + h2v)[:, background_mask], axis=1)
    bg_speed = np.hypot(bg_u, bg_v)
    bg_to = bearing_to(bg_u, bg_v)
    bg_from = (bg_to + 180.0) % 360.0
    bg_bin = nearest_direction(bg_from)
    persistence = trailing_persistence(bg_u, bg_v, 6)

    with (OUT / "d02_background_wind_hourly.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_utc", "speed_ms", "transport_to_deg", "inflow_from_deg", "circuit_inflow_bin_deg", "persistence6h"])
        for row in zip(real2_t, bg_speed, bg_to, bg_from, bg_bin, persistence):
            writer.writerow([row[0], *[f"{float(x):.6f}" for x in row[1:]]])

    dates = np.asarray([parse_time(time).date().isoformat() for time in real3_t])
    july18 = np.where(dates == "2022-07-18")[0]
    if len(july18) != 24:
        raise AssertionError(f"Expected 24 July-18 records, found {len(july18)}")

    r3u, r3v = earth_wind(real3)
    h3u, h3v = earth_wind(homo3)
    du = r3u - h3u
    dv = r3v - h3v
    unit_u = np.sin(np.deg2rad(bg_to))[:, None, None]
    unit_v = np.cos(np.deg2rad(bg_to))[:, None, None]
    delta = {
        "ALONG_BG": du * unit_u + dv * unit_v,
        "WSPD10": np.hypot(r3u, r3v) - np.hypot(h3u, h3v),
        "T2": real3["T2"] - homo3["T2"],
        "PBLH": real3["PBLH"] - homo3["PBLH"],
        "UST": real3["UST"] - homo3["UST"],
        "HFX": real3["HFX"] - homo3["HFX"],
    }

    stable18 = july18[(bg_speed[july18] >= 1.0) & (persistence[july18] >= 0.90)]
    direction_counts = Counter(float(value) for value in bg_bin[stable18])
    dominant_bin = direction_counts.most_common(1)[0][0] if direction_counts else None
    matched18 = stable18[bg_bin[stable18] == dominant_bin] if dominant_bin is not None else np.asarray([], dtype=int)

    vector_u18 = float(np.mean(bg_u[july18]))
    vector_v18 = float(np.mean(bg_v[july18]))
    vector_to18 = float(bearing_to(np.asarray([vector_u18]), np.asarray([vector_v18]))[0])
    vector_from18 = (vector_to18 + 180.0) % 360.0

    diagnostic: dict[str, object] = {
        "status": "FULL96H_ANALYSIS_PASS",
        "coverage": coverage,
        "source_metadata": {
            "real_d03": real3_meta,
            "homo_d03": homo3_meta,
            "real_d02": real2_meta,
            "homo_d02": homo2_meta,
        },
        "shared_d02_background": {
            "definition": "Vector mean geographic 10 m wind in a 15-35 km annulus around the D075 active-cell centroid.",
            "cell_count": int(background_mask.sum()),
            "real_homo_wind_max_abs_difference_ms": shared_max_abs,
        },
        "july18_regime": {
            "hour_count": int(len(july18)),
            "stable_hour_count": int(len(stable18)),
            "stable_definition": "speed >= 1.0 m/s and trailing 6 h directional persistence >= 0.90",
            "vector_mean_speed_ms": float(np.hypot(vector_u18, vector_v18)),
            "vector_mean_transport_to_deg": vector_to18,
            "vector_mean_inflow_from_deg": vector_from18,
            "nearest_circuit_inflow_bin_deg": float(nearest_direction(np.asarray([vector_from18]))[0]),
            "stable_hour_direction_counts": {str(key): value for key, value in sorted(direction_counts.items())},
            "dominant_circuit_inflow_bin_deg": dominant_bin,
            "dominant_matched_hours": [real3_t[i] for i in matched18],
        },
        "july18_real_minus_homo": {},
        "scope": "WRF REAL-HOMO physical-response and direction-matching diagnostic; not yet a Circuit network-added-value test.",
    }

    for field, values in delta.items():
        hourly_mean = np.asarray([float(np.mean(values[i][active])) for i in july18])
        hourly_rms = np.asarray([float(np.sqrt(np.mean(values[i][active] ** 2))) for i in july18])
        mean_map = np.mean(values[july18], axis=0)
        matched_map = np.mean(values[matched18], axis=0) if len(matched18) else np.full_like(mean_map, np.nan)
        diagnostic["july18_real_minus_homo"][field] = {
            "all_24h_time_mean_active": float(np.mean(hourly_mean)),
            "all_24h_mean_abs_active": float(np.mean(np.abs(hourly_mean))),
            "all_24h_mean_rms_active": float(np.mean(hourly_rms)),
            "all_24h_time_mean_map_active": sample_stats(mean_map, active),
            "dominant_direction_matched_hour_count": int(len(matched18)),
            "dominant_direction_matched_time_mean_map_active": sample_stats(matched_map, active) if len(matched18) else None,
            "hourly_active_mean": hourly_mean.tolist(),
            "hourly_active_rms": hourly_rms.tolist(),
        }

    (OUT / "full96h_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")

    x = np.arange(24)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True, sharex=True)
    axes[0].plot(x, bg_speed[july18], color="#205493", marker="o", label="Shared d02 speed")
    axes[0].set_ylabel("m s$^{-1}$")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, bg_from[july18], color="#7a3e9d", marker="o", label="Inflow from")
    axes[1].step(x, bg_bin[july18], color="#d95f02", where="mid", label="Nearest Circuit bin")
    axes[1].set_ylabel("degrees")
    axes[1].set_ylim(0, 360)
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].grid(alpha=0.25)
    axes[2].plot(x, persistence[july18], color="#238b45", marker="o")
    axes[2].axhline(0.90, color="0.3", linestyle="--")
    axes[2].set_ylabel("6 h persistence")
    axes[2].set_xlabel("UTC hour on 2022-07-18")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Shared d02 background wind and Circuit direction matching")
    fig.savefig(FIG / "july18_d02_wind_regime.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharex=True)
    for ax, field in zip(axes.flat, delta):
        values = diagnostic["july18_real_minus_homo"][field]
        ax.plot(x, values["hourly_active_mean"], marker="o", markersize=3, label="Active-cell mean")
        ax.plot(x, values["hourly_active_rms"], label="Active-cell RMS")
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_title(field)
        ax.grid(alpha=0.25)
        ax.set_xlabel("UTC hour")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Paris REAL − HOMO response on 2022-07-18")
    fig.savefig(FIG / "july18_real_minus_homo_timeseries.png", dpi=220)
    plt.close(fig)

    if len(matched18):
        jj, ii = np.where(treatment)
        pad = 12
        j0, j1 = max(0, jj.min() - pad), min(treatment.shape[0], jj.max() + pad + 1)
        i0, i1 = max(0, ii.min() - pad), min(treatment.shape[1], ii.max() + pad + 1)
        sl = np.s_[j0:j1, i0:i1]
        map_fields = (("ALONG_BG", "m s$^{-1}$"), ("WSPD10", "m s$^{-1}$"), ("T2", "K"), ("PBLH", "m"))
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
        for ax, (field, units) in zip(axes, map_fields):
            panel = np.mean(delta[field][matched18], axis=0)[sl]
            limit = max(float(np.percentile(np.abs(panel), 99)), 1.0e-9)
            mesh = ax.pcolormesh(lon3[sl], lat3[sl], panel, shading="auto", cmap="RdBu_r",
                                 norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
            ax.contour(lon3[sl], lat3[sl], treatment[sl].astype(float), levels=[0.5], colors="black", linewidths=0.9)
            ax.set_title(field)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cb = fig.colorbar(mesh, ax=ax, shrink=0.82)
            cb.set_label(units)
        fig.suptitle(f"REAL − HOMO during dominant matched inflow bin {dominant_bin:.1f}°")
        fig.savefig(FIG / "july18_dominant_direction_matched_maps.png", dpi=220)
        plt.close(fig)

    report = f"""# Paris WRF full-96-hour audit and first regime diagnostic

## Completion

- REAL d03 hourly records: **{len(real3_t)} / 97**
- HOMO d03 hourly records: **{len(homo3_t)} / 97**
- Exact hourly timeline in both experiments: **{coverage['real_d03_exact_timeline'] and coverage['homo_d03_exact_timeline']}**
- Shared d02 REAL/HOMO maximum wind difference: **{shared_max_abs:.3e} m s-1**

## 2022-07-18 shared d02 wind regime

- Vector-mean background speed: **{np.hypot(vector_u18, vector_v18):.3f} m s-1**
- Vector-mean inflow-from direction: **{vector_from18:.1f}°**
- Nearest 16-direction Circuit inflow bin: **{nearest_direction(np.asarray([vector_from18]))[0]:.1f}°**
- Stable hours: **{len(stable18)} / 24**
- Dominant stable Circuit bin: **{dominant_bin}°**
- Dominant-bin matched hours: **{len(matched18)}**

## Interpretation boundary

This is the WRF REAL-HOMO physical-response and direction-matching diagnostic. It does not yet demonstrate Circuit network added value. The next gate is to construct the direction-matched REAL/HOMO Circuit pair and test whether its network difference explains WRF transport differences beyond local and neighborhood morphology.
"""
    (OUT / "FULL96H_FIRST_DIAGNOSTIC_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": diagnostic["status"],
        "coverage": coverage,
        "july18_regime": diagnostic["july18_regime"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
