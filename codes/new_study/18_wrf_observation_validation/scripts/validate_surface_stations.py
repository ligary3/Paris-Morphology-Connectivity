"""Clean NOAA ISD observations and validate REAL WRF d03 at Paris-area stations."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


DATA = _release_path('E:\\lunwen4\\2\\气象观测\\Paris_WRF_validation')
RAW = DATA / "raw" / "noaa_isd" / "2022"
PROCESSED = DATA / "processed"
FIGURES = DATA / "figures"
WRF = PROCESSED / "wrf_real_surface_97h.npz"
ACCEPTED_QC = {"0", "1", "4", "5"}


def decode_tmp(value: str) -> float:
    parts = str(value).split(",")
    if len(parts) < 2 or parts[0] in {"+9999", "-9999", "9999"} or parts[1] not in ACCEPTED_QC:
        return np.nan
    return float(parts[0]) / 10.0


def decode_wnd(value: str) -> tuple[float, float]:
    parts = str(value).split(",")
    if len(parts) < 5:
        return np.nan, np.nan
    direction, dq, speed, sq = parts[0], parts[1], parts[3], parts[4]
    wd = np.nan if direction == "999" or dq not in ACCEPTED_QC else float(direction)
    ws = np.nan if speed == "9999" or sq not in ACCEPTED_QC else float(speed) / 10.0
    return wd, ws


def circular_mean(direction, speed):
    good = np.isfinite(direction) & np.isfinite(speed)
    if not good.any():
        return np.nan
    radians = np.deg2rad(direction[good])
    # Meteorological FROM direction to vector TO components.
    u = np.mean(-speed[good] * np.sin(radians))
    v = np.mean(-speed[good] * np.cos(radians))
    return float((np.degrees(np.arctan2(-u, -v)) + 360) % 360)


def circular_abs_error(model_from, observed_from):
    return np.abs((model_from - observed_from + 180) % 360 - 180)


def metrics(frame, prefix=""):
    out = {"n_pairs": int(len(frame))}
    for variable in ("temp_c", "wind_speed_ms"):
        obs, mod = frame[f"obs_{variable}"].to_numpy(), frame[f"wrf_{variable}"].to_numpy()
        good = np.isfinite(obs) & np.isfinite(mod)
        error = mod[good] - obs[good]
        out[f"{variable}_n"] = int(good.sum())
        out[f"{variable}_bias"] = float(np.mean(error)) if good.any() else np.nan
        out[f"{variable}_mae"] = float(np.mean(np.abs(error))) if good.any() else np.nan
        out[f"{variable}_rmse"] = float(np.sqrt(np.mean(error**2))) if good.any() else np.nan
        out[f"{variable}_pearson"] = float(pearsonr(obs[good], mod[good]).statistic) if good.sum() >= 3 else np.nan
    good = np.isfinite(frame.obs_wind_from_deg) & np.isfinite(frame.wrf_wind_from_deg) & (frame.obs_wind_speed_ms >= 1)
    out["wind_direction_n"] = int(good.sum())
    out["wind_direction_mae_deg"] = float(np.mean(circular_abs_error(frame.loc[good, "wrf_wind_from_deg"], frame.loc[good, "obs_wind_from_deg"]))) if good.any() else np.nan
    du = frame.wrf_u_ms - frame.obs_u_ms
    dv = frame.wrf_v_ms - frame.obs_v_ms
    goodv = np.isfinite(du) & np.isfinite(dv)
    out["wind_vector_rmse_ms"] = float(np.sqrt(np.mean((du[goodv]**2 + dv[goodv]**2)))) if goodv.any() else np.nan
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    observations = []
    station_rows = []
    for path in sorted(RAW.glob("*.csv")):
        raw = pd.read_csv(path, low_memory=False)
        raw["time_utc"] = pd.to_datetime(raw.DATE, utc=True).dt.tz_localize(None)
        raw = raw[(raw.time_utc >= "2022-07-16") & (raw.time_utc <= "2022-07-20")].copy()
        decoded = raw.WND.apply(decode_wnd)
        raw["wind_from_deg"] = [value[0] for value in decoded]
        raw["wind_speed_ms"] = [value[1] for value in decoded]
        raw["temp_c"] = raw.TMP.apply(decode_tmp)
        raw = raw[raw.time_utc.dt.minute == 0].copy()
        rows = []
        for time, group in raw.groupby("time_utc"):
            direction = circular_mean(group.wind_from_deg.to_numpy(), group.wind_speed_ms.to_numpy())
            rows.append({"station": path.stem, "station_name": group.NAME.iloc[0], "lat": float(group.LATITUDE.iloc[0]), "lon": float(group.LONGITUDE.iloc[0]), "elevation_m": float(group.ELEVATION.iloc[0]), "time_utc": time, "obs_temp_c": float(group.temp_c.median()), "obs_wind_speed_ms": float(group.wind_speed_ms.median()), "obs_wind_from_deg": direction, "raw_reports_at_hour": int(len(group))})
        station = pd.DataFrame(rows)
        observations.append(station)
        station_rows.append({"station": station.station.iloc[0], "station_name": station.station_name.iloc[0], "lat": station.lat.iloc[0], "lon": station.lon.iloc[0], "elevation_m": station.elevation_m.iloc[0], "event_hour_records": int(len(station))})
    obs = pd.concat(observations, ignore_index=True)

    wrf = np.load(WRF)
    times = pd.to_datetime([str(value).replace("_", " ") for value in wrf["times"]], format="%Y-%m-%d %H:%M:%S")
    lat, lon = wrf["lat"], wrf["lon"]
    paired = []
    station_meta = pd.DataFrame(station_rows)
    for station in station_meta.itertuples():
        distance2 = ((lat - station.lat) * 111.0)**2 + ((lon - station.lon) * 111.0 * np.cos(np.deg2rad(station.lat)))**2
        j, i = np.unravel_index(np.argmin(distance2), distance2.shape)
        u, v = wrf["u10_earth"][:, j, i], wrf["v10_earth"][:, j, i]
        speed = np.hypot(u, v)
        direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        model = pd.DataFrame({"time_utc": times, "wrf_temp_c": wrf["t2_k"][:, j, i] - 273.15, "wrf_wind_speed_ms": speed, "wrf_wind_from_deg": direction, "wrf_u_ms": u, "wrf_v_ms": v})
        one = obs[obs.station == station.station].merge(model, on="time_utc", how="inner")
        radians = np.deg2rad(one.obs_wind_from_deg)
        one["obs_u_ms"] = -one.obs_wind_speed_ms * np.sin(radians)
        one["obs_v_ms"] = -one.obs_wind_speed_ms * np.cos(radians)
        one["wrf_j"], one["wrf_i"] = j, i
        one["station_grid_distance_km"] = float(np.sqrt(distance2[j, i]))
        paired.append(one)
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(PROCESSED / "wrf_isd_hourly_pairs.csv", index=False)
    station_meta.to_csv(PROCESSED / "station_inventory.csv", index=False)

    results = {"all_96h": metrics(paired), "post_spinup": metrics(paired[paired.time_utc >= "2022-07-17"]), "stations_post_spinup": {}}
    station_metric_rows = []
    for station, frame in paired[paired.time_utc >= "2022-07-17"].groupby("station"):
        station_result = {"name": frame["station_name"].iloc[0], **metrics(frame)}
        results["stations_post_spinup"][station] = station_result
        station_metric_rows.append({"station": station, "station_name": station_result["name"], **{key: value for key, value in station_result.items() if key != "name"}})
    station_metrics = pd.DataFrame(station_metric_rows)
    station_metrics.to_csv(PROCESSED / "surface_validation_by_station.csv", index=False)
    results["source"] = "NOAA NCEI Global Hourly / Integrated Surface Database; QC codes 0,1,4,5 retained; exact UTC-hour reports."
    (PROCESSED / "surface_validation_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    post = paired[paired.time_utc >= "2022-07-17"].copy()
    hourly = post.groupby("time_utc").agg(obs_temp=("obs_temp_c", "mean"), wrf_temp=("wrf_temp_c", "mean"), obs_ws=("obs_wind_speed_ms", "mean"), wrf_ws=("wrf_wind_speed_ms", "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)
    axes[0].plot(hourly.index, hourly.obs_temp, label="ISD station mean", color="#252525")
    axes[0].plot(hourly.index, hourly.wrf_temp, label="WRF REAL", color="#d7301f")
    axes[0].set_ylabel("2 m temperature (°C)"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(hourly.index, hourly.obs_ws, label="ISD station mean", color="#252525")
    axes[1].plot(hourly.index, hourly.wrf_ws, label="WRF REAL", color="#2b8cbe")
    axes[1].set_ylabel("10 m wind speed (m/s)"); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[1].set_xlabel("UTC")
    fig.suptitle("Paris-area surface meteorology: observations versus WRF REAL")
    fig.savefig(FIGURES / "wrf_isd_surface_timeseries.png", dpi=240)
    plt.close(fig)

    station_plot = station_meta.merge(station_metrics, on=["station", "station_name"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    plot_specs = [
        ("temp_c_bias", "Temperature bias (°C)", "coolwarm", -5, 5),
        ("wind_speed_ms_bias", "Wind-speed bias (m/s)", "coolwarm", -2, 2),
        ("wind_direction_mae_deg", "Wind-direction MAE (°)", "viridis", 0, 40),
    ]
    for ax, (column, title, cmap, vmin, vmax) in zip(axes, plot_specs):
        scatter = ax.scatter(station_plot.lon, station_plot.lat, c=station_plot[column], s=100, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black")
        for row in station_plot.itertuples():
            ax.annotate(row.station_name.split(",")[0].title(), (row.lon, row.lat), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.scatter(2.3522, 48.8566, marker="*", s=140, color="gold", edgecolor="black", label="Paris center")
        ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=.2)
        fig.colorbar(scatter, ax=ax, shrink=.78)
    fig.suptitle("Station-level WRF REAL validation, 17–20 July 2022 UTC")
    fig.savefig(FIGURES / "wrf_isd_station_metrics_map.png", dpi=240)
    plt.close(fig)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
