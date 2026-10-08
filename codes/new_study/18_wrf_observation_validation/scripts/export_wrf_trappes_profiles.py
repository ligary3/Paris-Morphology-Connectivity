"""Export hourly WRF REAL vertical profiles at the nearest Trappes d03 grid cell."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np


ROOT = _release_path('/home/gary/wrf-work/cases')
SEGMENTS = [
    ROOT / "paris_real_pair24h_v1",
    ROOT / "paris_real_core48h_v1",
    ROOT / "paris_real_final24h_v1",
]
OUTPUT = _release_path('/mnt/e/lunwen4/2/气象观测/Paris_WRF_validation/processed/wrf_real_trappes_profiles_97h.npz')
STATION_LAT = 48.7666666
STATION_LON = 2.0
G = 9.81
P0 = 100000.0
RD_OVER_CP = 287.05 / 1004.0


def expected_times() -> list[str]:
    start = datetime(2022, 7, 16)
    return [(start + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S") for hour in range(97)]


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6371.0
    p1 = np.deg2rad(lat)
    p0 = np.deg2rad(STATION_LAT)
    dp = p1 - p0
    dl = np.deg2rad(lon - STATION_LON)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p0) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.minimum(1, np.sqrt(a)))


def main() -> None:
    paths = [path for segment in SEGMENTS for path in sorted(segment.glob("wrfout_d03_*"))]
    if not paths:
        raise FileNotFoundError("No REAL d03 wrfout files found")

    with nc.Dataset(paths[0]) as ds:
        lat = np.asarray(ds.variables["XLAT"][0], dtype=float)
        lon = np.asarray(ds.variables["XLONG"][0], dtype=float)
        distance = haversine_km(lat, lon)
        j, i = np.unravel_index(np.argmin(distance), distance.shape)

    records: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        with nc.Dataset(path) as ds:
            times = [str(value) for value in nc.chartostring(ds.variables["Times"][:]).tolist()]
            for ti, time in enumerate(times):
                if time in records:
                    continue
                pressure = np.asarray(ds.variables["P"][ti, :, j, i] + ds.variables["PB"][ti, :, j, i], dtype=float)
                theta = np.asarray(ds.variables["T"][ti, :, j, i], dtype=float) + 300.0
                temperature = theta * (pressure / P0) ** RD_OVER_CP
                geopotential = np.asarray(ds.variables["PH"][ti, :, j, i] + ds.variables["PHB"][ti, :, j, i], dtype=float)
                height_stag = geopotential / G
                height = 0.5 * (height_stag[:-1] + height_stag[1:])
                u_grid = 0.5 * np.asarray(ds.variables["U"][ti, :, j, i:i + 2], dtype=float).sum(axis=1)
                v_grid = 0.5 * np.asarray(ds.variables["V"][ti, :, j:j + 2, i], dtype=float).sum(axis=1)
                cosalpha = float(ds.variables["COSALPHA"][ti, j, i])
                sinalpha = float(ds.variables["SINALPHA"][ti, j, i])
                u_earth = u_grid * cosalpha - v_grid * sinalpha
                v_earth = v_grid * cosalpha + u_grid * sinalpha
                records[time] = {
                    "pressure_pa": pressure,
                    "height_msl_m": height,
                    "temperature_k": temperature,
                    "u_earth_ms": u_earth,
                    "v_earth_ms": v_earth,
                }

    ordered = expected_times()
    missing = [time for time in ordered if time not in records]
    if missing:
        raise RuntimeError(f"Missing WRF hours: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.stack([records[time][key] for time in ordered]) for key in next(iter(records.values()))}
    np.savez_compressed(
        OUTPUT,
        times=np.asarray(ordered),
        wrf_lat=float(lat[j, i]),
        wrf_lon=float(lon[j, i]),
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        station_grid_distance_km=float(distance[j, i]),
        grid_j=j,
        grid_i=i,
        **payload,
    )
    print(f"records={len(ordered)} levels={payload['pressure_pa'].shape[1]} distance_km={distance[j, i]:.3f} output={OUTPUT}")


if __name__ == "__main__":
    main()
