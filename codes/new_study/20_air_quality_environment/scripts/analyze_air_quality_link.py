"""Link Airparif observations to the paired WRF morphology response.

This is an environmental-meaning diagnostic, not a WRF-Chem validation.  The
primary spatial sample is the six D075 urban-background stations.  Pollutant
and WRF-response values are double demeaned by station and hour, removing each
station's persistent offset and the event-wide hourly signal before testing
whether local departures covary.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATA = _release_path('E:\\lunwen4\\2\\空气质量\\Paris_2022_event')
OBS = DATA / "processed" / "airparif_event_wide.csv"
TRANSPORT = Path("new_study/17_transport_metric_candidates/paired_transport_metrics.npz")
GRID = Path("new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz")
OUT = DATA / "analysis"
FIG = OUT / "figures"
POST_SPINUP = pd.Timestamp("2022-07-17 00:00:00", tz="UTC")

STATIONS = {
    "PA01H": (48.862128, 2.344623),
    "PA07": (48.8571944, 2.2932778),
    "PA12": (48.8371944, 2.3938056),
    "PA13": (48.8284722, 2.3595583),
    "PA15L": (48.8303889, 2.2698861),
    "PA18": (48.8917278, 2.3455750),
}


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    value = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(value))


def double_demean(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame[column]
        - frame.groupby("station_code")[column].transform("mean")
        - frame.groupby("time_utc")[column].transform("mean")
        + frame[column].mean()
    )


def moving_block_ci(frame: pd.DataFrame, pollutant: str, seed: int = 20260904) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    times = np.asarray(sorted(frame.time_utc.unique()))
    if len(times) < 12:
        return np.nan, np.nan
    values = []
    block = 6
    for _ in range(150):
        chosen = []
        while len(chosen) < len(times):
            start = int(rng.integers(0, max(1, len(times) - block + 1)))
            chosen.extend(times[start : start + block])
        parts = []
        for replicate, time in enumerate(chosen[: len(times)]):
            sample = frame.loc[frame.time_utc.eq(time)].copy()
            sample["bootstrap_time"] = replicate
            parts.append(sample)
        sample = pd.concat(parts, ignore_index=True)
        sample["time_utc"] = sample.pop("bootstrap_time")
        x = double_demean(sample, "delta_transport")
        y = double_demean(sample, pollutant)
        rho = spearmanr(x, y, nan_policy="omit").statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def station_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    transport = np.load(TRANSPORT)
    grid = np.load(GRID)
    lat, lon = grid["lat"], grid["lon"]
    times = pd.to_datetime(
        [str(value).replace("_", " ") for value in transport["times"]], utc=True
    )
    metric = transport["delta_along_10_1000"]
    records, station_meta = [], []
    for code, (station_lat, station_lon) in STATIONS.items():
        distance = haversine_km(station_lat, station_lon, lat, lon)
        row, col = np.unravel_index(np.argmin(distance), distance.shape)
        station_meta.append(
            {
                "station_code": code,
                "latitude": station_lat,
                "longitude": station_lon,
                "wrf_row": row,
                "wrf_col": col,
                "nearest_grid_distance_km": float(distance[row, col]),
            }
        )
        records.extend(
            {
                "station_code": code,
                "time_utc": time,
                "delta_transport": float(metric[index, row, col]),
                "background_speed_ms": float(transport["background_speed_ms"][index]),
                "background_bin_deg": float(transport["background_bin_deg"][index]),
            }
            for index, time in enumerate(times)
        )
    return pd.DataFrame(records), pd.DataFrame(station_meta)


def correlation_table(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regimes = {
        "stable_network_hours": matched.background_bin_deg.isin([45.0, 67.5, 90.0, 112.5]),
        "all_postspinup_hours": pd.Series(True, index=matched.index),
        "dominant_1125_hours": matched.background_bin_deg.eq(112.5),
    }
    for regime, mask in regimes.items():
      source = matched.loc[mask].copy()
      predictors = source[
          ["station_code", "time_utc", "delta_transport", "background_speed_ms", "background_bin_deg"]
      ].drop_duplicates()
      for pollutant in ("NO2", "O3", "Ox_O3eq_ugm3"):
        observations = source[["station_code", "time_utc", pollutant]].dropna()
        for lag in range(-3, 4):
            shifted_observations = observations.copy()
            # Positive lag means that the pollutant observation follows the WRF
            # transport response by `lag` hours.
            shifted_observations["time_utc"] -= pd.Timedelta(hours=lag)
            shifted = predictors.merge(
                shifted_observations, on=["station_code", "time_utc"], how="inner"
            )
            counts = shifted.groupby("time_utc").station_code.transform("nunique")
            shifted = shifted.loc[counts >= 2]
            shifted["transport_residual"] = double_demean(shifted, "delta_transport")
            shifted["pollutant_residual"] = double_demean(shifted, pollutant)
            rho = spearmanr(
                shifted["transport_residual"], shifted["pollutant_residual"], nan_policy="omit"
            ).statistic
            if lag == 0 and regime != "dominant_1125_hours":
                low, high = moving_block_ci(
                    shifted.rename(columns={"pollutant_residual": "_unused"}), pollutant
                )
            else:
                low, high = np.nan, np.nan
            rows.append(
                {
                    "regime": regime,
                    "pollutant": pollutant,
                    "lag_hours_pollutant_after_transport": lag,
                    "n_pairs": len(shifted),
                    "n_stations": shifted.station_code.nunique(),
                    "n_hours": shifted.time_utc.nunique(),
                    "spearman_rho_double_demeaned": float(rho),
                    "block_bootstrap_ci_low": low,
                    "block_bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def event_summary(obs: pd.DataFrame) -> dict:
    core = obs.loc[obs.station_code.isin(STATIONS)].copy()
    o3 = core.dropna(subset=["O3"])
    no2 = core.dropna(subset=["NO2"])
    ox = core.dropna(subset=["Ox_O3eq_ugm3"])
    return {
        "time_window_utc": [str(core.time_utc.min()), str(core.time_utc.max())],
        "D075_background_NO2_stations": int(no2.station_code.nunique()),
        "D075_background_O3_stations": int(o3.station_code.nunique()),
        "D075_background_Ox_stations": int(ox.station_code.nunique()),
        "maximum_hourly_O3_ugm3": float(o3.O3.max()),
        "maximum_hourly_O3_station": str(o3.loc[o3.O3.idxmax(), "station_code"]),
        "maximum_hourly_O3_time_utc": str(o3.loc[o3.O3.idxmax(), "time_utc"]),
        "O3_station_hours_above_120": int((o3.O3 > 120).sum()),
        "O3_station_hours_above_180": int((o3.O3 > 180).sum()),
        "maximum_NO2_ugm3": float(no2.NO2.max()),
        "maximum_Ox_O3eq_ugm3": float(ox.Ox_O3eq_ugm3.max()),
    }


def make_figure(obs: pd.DataFrame, correlations: pd.DataFrame) -> None:
    core = obs.loc[obs.station_code.isin(STATIONS)].copy()
    hourly = core.groupby("time_utc")[["NO2", "O3", "Ox_O3eq_ugm3"]].median()
    colors = {"NO2": "#d95f02", "O3": "#1b9e77", "Ox_O3eq_ugm3": "#5e3c99"}
    labels = {"NO2": "NO$_2$", "O3": "O$_3$", "Ox_O3eq_ugm3": "O$_x$ (O$_3$-equiv.)"}
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), constrained_layout=True)
    for column in colors:
        axes[0].plot(hourly.index, hourly[column], lw=2, color=colors[column], label=labels[column])
    axes[0].axhline(180, color="0.4", lw=1, ls="--", label="O$_3$ hourly information threshold")
    axes[0].set_ylabel("Concentration (µg m$^{-3}$ or O$_3$-equiv.)")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].set_title("D075 urban-background observations during the July 2022 event")
    axes[0].grid(alpha=0.2)

    for pollutant in ("NO2", "O3", "Ox_O3eq_ugm3"):
        data = correlations.loc[
            correlations.pollutant.eq(pollutant)
            & correlations.regime.eq("stable_network_hours")
        ]
        axes[1].plot(
            data.lag_hours_pollutant_after_transport,
            data.spearman_rho_double_demeaned,
            marker="o", lw=2, color=colors[pollutant], label=labels[pollutant],
        )
    axes[1].axhline(0, color="0.3", lw=1)
    axes[1].axvline(0, color="0.6", lw=1, ls=":")
    axes[1].set_xlabel("Pollutant response lag after WRF morphology-induced transport (h)")
    axes[1].set_ylabel("Two-way-demeaned Spearman ρ")
    axes[1].set_xticks(range(-3, 4))
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=3)
    fig.savefig(FIG / "air_quality_event_and_transport_link.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(OBS)
    obs["time_utc"] = pd.to_datetime(obs["time_utc"], utc=True)
    obs = obs.loc[obs.time_utc >= POST_SPINUP].copy()
    wrf, station_meta = station_samples()
    matched = obs.merge(wrf, on=["station_code", "time_utc"], how="inner")
    matched.to_csv(OUT / "D075_background_observation_wrf_pairs.csv", index=False, encoding="utf-8-sig")
    station_meta.to_csv(OUT / "D075_station_wrf_grid_match.csv", index=False, encoding="utf-8-sig")
    correlations = correlation_table(matched)
    correlations.to_csv(OUT / "pollutant_transport_lag_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary = event_summary(obs)
    primary = correlations.loc[
        correlations.lag_hours_pollutant_after_transport.eq(0)
        & correlations.regime.eq("stable_network_hours")
    ]
    summary["contemporaneous_double_demeaned_results"] = primary.to_dict(orient="records")
    (OUT / "air_quality_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(obs, correlations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLag diagnostics")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
