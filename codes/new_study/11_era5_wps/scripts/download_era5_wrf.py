"""Download ERA5 GRIB forcing for the Paris WRF Phase-I event.

The full run is deliberately split into daily pressure-level and single-level
files. Existing non-empty final files are skipped; downloads first use a .part
suffix and are atomically renamed only after cdsapi returns successfully.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import cdsapi


AREA = [56.25, -15.75, 37.00, 14.00]  # north, west, south, east
DATES = [date(2022, 7, 16) + timedelta(days=i) for i in range(5)]
TIMES = [f"{hour:02d}:00" for hour in range(24)]

PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100",
    "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800",
    "825", "850", "875", "900", "925", "950", "975", "1000",
]

PRESSURE_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

SINGLE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "geopotential",
    "land_sea_mask",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "skin_temperature",
    "snow_depth",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def retrieve_atomic(client: cdsapi.Client, dataset: str, request: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"SKIP existing {target.name} ({target.stat().st_size:,} bytes)")
        return
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    request_sidecar = target.with_suffix(target.suffix + ".request.json")
    request_sidecar.write_text(json.dumps({"dataset": dataset, "request": request}, indent=2), encoding="utf-8")
    print(f"REQUEST {dataset} -> {target.name}", flush=True)
    client.retrieve(dataset, request, str(part))
    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"CDS returned no data for {target.name}")
    part.replace(target)
    print(f"DONE {target.name} ({target.stat().st_size:,} bytes)", flush=True)


def pressure_request(day: date, times: list[str], levels: list[str], variables: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{day.year:04d}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": times,
        "pressure_level": levels,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }


def single_request(day: date, times: list[str], variables: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{day.year:04d}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": times,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-kind", choices=("pressure", "single"), default="pressure")
    parser.add_argument("--output", type=Path, default=_release_path('/home/gary/wrf-work/data/ERA5/paris_20220716_20'))
    args = parser.parse_args()

    # Fail fast here: experiment monitoring should see the first CDS error
    # instead of the library silently retrying hundreds of times.
    client = cdsapi.Client(progress=True, quiet=False, retry_max=0)
    if args.mode == "smoke":
        day = DATES[0]
        if args.smoke_kind == "pressure":
            retrieve_atomic(
                client,
                "reanalysis-era5-pressure-levels",
                pressure_request(day, ["00:00"], ["1000"], ["geopotential"]),
                args.output / "smoke_era5_pl_20220716_00.grib",
            )
        else:
            retrieve_atomic(
                client,
                "reanalysis-era5-single-levels",
                single_request(day, ["00:00"], ["2m_temperature"]),
                args.output / "smoke_era5_sl_20220716_00.grib",
            )
        return

    for day in DATES:
        stamp = day.strftime("%Y%m%d")
        retrieve_atomic(
            client,
            "reanalysis-era5-pressure-levels",
            pressure_request(day, TIMES, PRESSURE_LEVELS, PRESSURE_VARIABLES),
            args.output / f"era5_pressure_{stamp}.grib",
        )
        retrieve_atomic(
            client,
            "reanalysis-era5-single-levels",
            single_request(day, TIMES, SINGLE_VARIABLES),
            args.output / f"era5_single_{stamp}.grib",
        )


if __name__ == "__main__":
    main()
