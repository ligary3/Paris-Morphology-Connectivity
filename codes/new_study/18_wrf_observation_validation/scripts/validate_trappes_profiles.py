"""Validate WRF REAL lower-tropospheric profiles against NOAA IGRA Trappes soundings."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


DATA = _release_path('E:\\lunwen4\\2\\气象观测\\Paris_WRF_validation')
IGRA = DATA / "raw" / "noaa_igra" / "FRM00007145" / "FRM00007145-data.txt"
WRF = DATA / "processed" / "wrf_real_trappes_profiles_97h.npz"
PROCESSED = DATA / "processed"
FIGURES = DATA / "figures"
EVENT_START = pd.Timestamp("2022-07-16 00:00:00")
EVENT_END = pd.Timestamp("2022-07-20 00:00:00")


def integer_field(line: str, start: int, end: int) -> float:
    text = line[start:end].strip()
    if not text:
        return np.nan
    value = int(text)
    return np.nan if value in {-9999, -8888} else float(value)


def release_offset_minutes(nominal: datetime, release_hhmm: int) -> int:
    hour, minute = divmod(release_hhmm, 100)
    candidates = [nominal.replace(hour=hour, minute=minute) + timedelta(days=shift) for shift in (-1, 0, 1)]
    actual = min(candidates, key=lambda value: abs((value - nominal).total_seconds()))
    return int((actual - nominal).total_seconds() / 60)


def read_event_soundings() -> list[dict]:
    soundings = []
    with IGRA.open("r", encoding="ascii", errors="replace") as stream:
        iterator = iter(stream)
        for line in iterator:
            if not line.startswith("#"):
                continue
            year, month, day, hour = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
            count = int(line[32:36])
            levels = [next(iterator).rstrip("\n") for _ in range(count)]
            nominal = datetime(year, month, day, hour)
            if not (EVENT_START.to_pydatetime() <= nominal <= EVENT_END.to_pydatetime()) or hour not in {0, 12}:
                continue
            release = int(line[27:31])
            rows = []
            for level in levels:
                pressure = integer_field(level, 9, 15)
                height = integer_field(level, 16, 21)
                temperature = integer_field(level, 22, 27)
                direction = integer_field(level, 40, 45)
                speed = integer_field(level, 46, 51)
                rows.append({
                    "pressure_pa": pressure,
                    "height_msl_m": height,
                    "temperature_c": temperature / 10 if np.isfinite(temperature) else np.nan,
                    "wind_from_deg": direction,
                    "wind_speed_ms": speed / 10 if np.isfinite(speed) else np.nan,
                })
            soundings.append({
                "nominal_time": nominal,
                "release_hhmm": release,
                "release_offset_min": release_offset_minutes(nominal, release),
                "levels": pd.DataFrame(rows),
            })
    return soundings


def interpolate_log_pressure(model_pressure, model_value, target_pressure):
    good = np.isfinite(model_pressure) & np.isfinite(model_value)
    x = np.log(model_pressure[good][::-1])
    y = model_value[good][::-1]
    target = np.log(target_pressure)
    result = np.interp(target, x, y)
    outside = (target < x.min()) | (target > x.max())
    result[outside] = np.nan
    return result


def circular_error(model, observed):
    return np.abs((model - observed + 180) % 360 - 180)


def metric_block(frame: pd.DataFrame) -> dict:
    result = {"n_level_pairs": int(len(frame)), "n_soundings": int(frame.nominal_time.nunique())}
    for variable in ("temperature_c", "wind_speed_ms"):
        obs = frame[f"obs_{variable}"].to_numpy()
        mod = frame[f"wrf_{variable}"].to_numpy()
        good = np.isfinite(obs) & np.isfinite(mod)
        error = mod[good] - obs[good]
        result[f"{variable}_n"] = int(good.sum())
        result[f"{variable}_bias"] = float(np.mean(error))
        result[f"{variable}_mae"] = float(np.mean(np.abs(error)))
        result[f"{variable}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{variable}_pearson"] = float(pearsonr(obs[good], mod[good]).statistic) if good.sum() >= 3 else np.nan
    good = np.isfinite(frame.obs_wind_from_deg) & np.isfinite(frame.wrf_wind_from_deg) & (frame.obs_wind_speed_ms >= 1)
    result["wind_direction_n"] = int(good.sum())
    result["wind_direction_mae_deg"] = float(np.mean(circular_error(frame.loc[good, "wrf_wind_from_deg"], frame.loc[good, "obs_wind_from_deg"])))
    du = frame.wrf_u_ms - frame.obs_u_ms
    dv = frame.wrf_v_ms - frame.obs_v_ms
    good = np.isfinite(du) & np.isfinite(dv)
    result["wind_vector_rmse_ms"] = float(np.sqrt(np.mean(du[good] ** 2 + dv[good] ** 2)))
    return result


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    wrf = np.load(WRF)
    wrf_times = pd.to_datetime([str(value).replace("_", " ") for value in wrf["times"]])
    soundings = read_event_soundings()
    rows = []
    for sounding in soundings:
        nominal = pd.Timestamp(sounding["nominal_time"])
        ti = int(np.where(wrf_times == nominal)[0][0])
        obs = sounding["levels"].copy()
        obs = obs[(obs.pressure_pa >= 70000) & (obs.pressure_pa <= 100000)].copy()
        pressure = obs.pressure_pa.to_numpy()
        wrf_temp = interpolate_log_pressure(wrf["pressure_pa"][ti], wrf["temperature_k"][ti] - 273.15, pressure)
        wrf_u = interpolate_log_pressure(wrf["pressure_pa"][ti], wrf["u_earth_ms"][ti], pressure)
        wrf_v = interpolate_log_pressure(wrf["pressure_pa"][ti], wrf["v_earth_ms"][ti], pressure)
        wrf_speed = np.hypot(wrf_u, wrf_v)
        wrf_direction = (np.degrees(np.arctan2(-wrf_u, -wrf_v)) + 360) % 360
        radians = np.deg2rad(obs.wind_from_deg.to_numpy())
        obs_u = -obs.wind_speed_ms.to_numpy() * np.sin(radians)
        obs_v = -obs.wind_speed_ms.to_numpy() * np.cos(radians)
        for index, item in enumerate(obs.itertuples()):
            rows.append({
                "nominal_time": nominal,
                "release_offset_min": sounding["release_offset_min"],
                "pressure_hpa": item.pressure_pa / 100,
                "height_msl_m": item.height_msl_m,
                "obs_temperature_c": item.temperature_c,
                "wrf_temperature_c": wrf_temp[index],
                "obs_wind_speed_ms": item.wind_speed_ms,
                "wrf_wind_speed_ms": wrf_speed[index],
                "obs_wind_from_deg": item.wind_from_deg,
                "wrf_wind_from_deg": wrf_direction[index],
                "obs_u_ms": obs_u[index], "obs_v_ms": obs_v[index],
                "wrf_u_ms": wrf_u[index], "wrf_v_ms": wrf_v[index],
            })
    pairs = pd.DataFrame(rows)
    pairs.to_csv(PROCESSED / "wrf_igra_trappes_profile_pairs.csv", index=False)

    post = pairs[pairs.nominal_time >= "2022-07-17"].copy()
    metrics = {
        "all_event": metric_block(pairs),
        "post_spinup": metric_block(post),
        "profile_times": [str(value) for value in sorted(post.nominal_time.unique())],
        "pressure_layer": "1000-700 hPa; WRF log-pressure interpolation to observed IGRA levels",
        "station_grid_distance_km": float(wrf["station_grid_distance_km"]),
        "release_offsets_minutes": {str(time): int(frame.release_offset_min.iloc[0]) for time, frame in post.groupby("nominal_time")},
        "source": "NOAA NCEI IGRA v2.2, station FRM00007145 (Trappes)",
    }
    (PROCESSED / "trappes_profile_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    colors = plt.cm.plasma(np.linspace(0.05, 0.95, post.nominal_time.nunique()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True, sharey=True)
    for color, (time, frame) in zip(colors, post.groupby("nominal_time")):
        label = pd.Timestamp(time).strftime("%d %HZ")
        axes[0].plot(frame.obs_temperature_c, frame.pressure_hpa, color=color, linewidth=2, label=f"Obs {label}")
        axes[0].plot(frame.wrf_temperature_c, frame.pressure_hpa, color=color, linestyle="--", linewidth=1.4)
        axes[1].plot(frame.obs_wind_speed_ms, frame.pressure_hpa, color=color, linewidth=2, label=f"Obs {label}")
        axes[1].plot(frame.wrf_wind_speed_ms, frame.pressure_hpa, color=color, linestyle="--", linewidth=1.4)
    for ax in axes:
        ax.set_ylim(1000, 700); ax.grid(alpha=.25); ax.set_ylabel("Pressure (hPa)")
    axes[0].set_xlabel("Temperature (°C)"); axes[1].set_xlabel("Wind speed (m/s)")
    axes[0].set_title("Temperature profiles"); axes[1].set_title("Wind-speed profiles")
    axes[1].plot([], [], color="black", label="IGRA: solid")
    axes[1].plot([], [], color="black", linestyle="--", label="WRF: dashed")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Trappes lower-tropospheric profiles, 17–20 July 2022")
    fig.savefig(FIGURES / "wrf_igra_trappes_profiles.png", dpi=240)
    plt.close(fig)

    pressure_bins = np.arange(700, 1001, 25)
    post["pressure_bin"] = pd.cut(post.pressure_hpa, bins=np.append(pressure_bins, 1025), labels=pressure_bins, right=False)
    profile_error = post.assign(
        temp_error=post.wrf_temperature_c - post.obs_temperature_c,
        speed_error=post.wrf_wind_speed_ms - post.obs_wind_speed_ms,
    ).groupby("pressure_bin", observed=True).agg(temp_bias=("temp_error", "mean"), speed_bias=("speed_error", "mean"), n=("temp_error", "count")).reset_index()
    profile_error.to_csv(PROCESSED / "trappes_profile_bias_by_pressure.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
