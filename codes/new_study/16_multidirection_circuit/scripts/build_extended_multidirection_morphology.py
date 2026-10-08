"""Build extended-domain REAL FAI for the four event-stage directions preceding 112.5°."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np


HERE = Path(__file__).resolve()
MORPH = HERE.parents[2] / "01_phase1a_morphology" / "scripts"
sys.path.insert(0, str(MORPH))
from build_observed_morphology import build_resolution, make_grid, projected_width  # noqa: E402


INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_buildings_targeted_imputed.gpkg')
OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_morphology_multidirection_v1.gpkg')
REPORT = OUTPUT.with_suffix(".json")
DIRECTIONS = (22.5, 45.0, 67.5, 90.0)
RESOLUTION = 100
ORIGIN_X = 643000
ORIGIN_Y = 6857000


def suffix(direction: float) -> str:
    return f"{int(round(direction * 10)):04d}"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    buildings = gpd.read_file(INPUT, layer="extended_buildings_imputed", engine="pyogrio")
    paris = gpd.read_file(INPUT, layer="paris_boundary", engine="pyogrio")
    support = gpd.read_file(INPUT, layer="support_footprint_500m", engine="pyogrio")
    support_geom = support.geometry.union_all()
    paris_geom = paris.geometry.union_all()
    buildings = buildings[buildings.geometry.intersects(support_geom)].copy()
    buildings["height_m"] = buildings["height_analysis_m"].astype(float)
    buildings["height_valid"] = True

    frontal = []
    for direction in DIRECTIONS:
        tag = suffix(direction)
        print(f"Projected widths for {direction:.1f} degrees...", flush=True)
        width = buildings.geometry.apply(lambda geom, d=direction: projected_width(geom, d))
        column = f"frontal_area_{tag}"
        buildings[column] = width * buildings["height_m"]
        frontal.append(column)

    grid = make_grid(support_geom, RESOLUTION, ORIGIN_X, ORIGIN_Y)
    print(f"Single overlay: {len(buildings):,} buildings x {len(grid):,} cells", flush=True)
    result, qa = build_resolution(buildings, grid, frontal)
    result = result.rename(columns={"paris_fraction": "support_fraction", "core_cell": "support_core_cell"})
    result["paris_fraction"] = result.geometry.intersection(paris_geom).area / RESOLUTION**2
    result["analysis_core"] = result["paris_fraction"] >= 0.5

    keep = [
        "grid_id", "ix", "iy", "x_center", "y_center", "support_fraction",
        "support_core_cell", "paris_fraction", "analysis_core", "lambda_p",
        "h_area_weighted", *[f"lambda_f_{suffix(d)}" for d in DIRECTIONS], "geometry",
    ]
    result[keep].to_file(OUTPUT, layer="extended_multidirection", driver="GPKG", engine="pyogrio")
    paris.to_file(OUTPUT, layer="paris_boundary", driver="GPKG", engine="pyogrio", append=True)
    support.to_file(OUTPUT, layer="support_footprint_500m", driver="GPKG", engine="pyogrio", append=True)
    report = {
        "status": "COMPLETE",
        "directions_deg": list(DIRECTIONS),
        "n_buildings": int(len(buildings)),
        "n_grid_cells": int(len(result)),
        "analysis_core_cells": int(result["analysis_core"].sum()),
        "q95_core_by_direction": {
            str(d): float(result.loc[result["analysis_core"], f"lambda_f_{suffix(d)}"].quantile(0.95))
            for d in DIRECTIONS
        },
        "overlay_qa": qa,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
