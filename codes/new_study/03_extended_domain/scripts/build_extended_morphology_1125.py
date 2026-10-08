"""Build the 100 m extended-domain morphology and frozen FAI resistance surface.

Only the variables needed for the first REAL Circuit pilot are produced:
plan-area fraction, area-weighted height, 112.5-degree frontal-area index, and
the pre-registered R1 resistance mapping. Paris remains the analysis core.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
MORPHOLOGY_SCRIPTS = HERE.parents[2] / "01_phase1a_morphology" / "scripts"
sys.path.insert(0, str(MORPHOLOGY_SCRIPTS))
from build_observed_morphology import build_resolution, make_grid, projected_width  # noqa: E402


DEFAULT_INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_buildings_targeted_imputed.gpkg')
DEFAULT_OUTPUT_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')
FROZEN_D075_Q95_FAI = 0.976565954884937
INFLOW_DEGREES = 112.5
TRANSPORT_DEGREES = 292.5
RESOLUTION = 100
ORIGIN_X = 643000
ORIGIN_Y = 6857000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gpkg", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def projection_bounds(geometry, bearing_degrees: float) -> tuple[float, float]:
    theta = math.radians(bearing_degrees)
    dx, dy = math.sin(theta), math.cos(theta)
    coordinates = np.asarray(geometry.boundary.coords) if geometry.geom_type == "Polygon" else None
    if coordinates is None:
        arrays = [np.asarray(part.boundary.coords) for part in geometry.geoms]
        coordinates = np.vstack(arrays)
    projected = coordinates[:, 0] * dx + coordinates[:, 1] * dy
    return float(projected.min()), float(projected.max())


def make_map(grid: gpd.GeoDataFrame, paris: gpd.GeoDataFrame, output: Path) -> None:
    display = grid[grid["support_fraction"] >= 0.5]
    items = [
        ("h_area_weighted", "Area-weighted building height (m)", "viridis"),
        ("lambda_p", "Plan-area fraction", "YlOrBr"),
        ("lambda_f_1125", "FAI, inflow 112.5°", "magma"),
        ("r1_c1", "Frozen directional resistance R1", "inferno"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=180)
    for ax, (column, title, cmap) in zip(axes.flat, items):
        display.plot(column=column, ax=ax, cmap=cmap, linewidth=0, legend=True)
        paris.boundary.plot(ax=ax, color="#2b8cbe", linewidth=0.8)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Paris extended 100 m REAL morphology and resistance pilot", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_gpkg = output_dir / "extended_morphology_resistance_1125.gpkg"
    report_json = output_dir / "extended_morphology_resistance_1125_qa.json"
    figure_png = output_dir / "extended_morphology_resistance_1125_QA.png"
    if any(path.exists() for path in [output_gpkg, report_json, figure_png]):
        raise FileExistsError("Refusing to overwrite an existing extended morphology run")

    buildings = gpd.read_file(
        args.input_gpkg, layer="extended_buildings_imputed", engine="pyogrio"
    )
    paris = gpd.read_file(args.input_gpkg, layer="paris_boundary", engine="pyogrio")
    support = gpd.read_file(
        args.input_gpkg, layer="support_footprint_500m", engine="pyogrio"
    )
    support_geom = support.geometry.union_all()
    paris_geom = paris.geometry.union_all()
    buildings = buildings[buildings.geometry.intersects(support_geom)].copy()
    buildings["height_m"] = buildings["height_analysis_m"].astype(float)
    buildings["height_valid"] = True

    print(f"Computing projected widths for {len(buildings):,} buildings...", flush=True)
    buildings["projected_width_1125_m"] = buildings.geometry.apply(
        lambda geom: projected_width(geom, INFLOW_DEGREES)
    )
    buildings["frontal_area_1125"] = (
        buildings["projected_width_1125_m"] * buildings["height_m"]
    )
    grid = make_grid(support_geom, RESOLUTION, ORIGIN_X, ORIGIN_Y)
    print(f"Overlaying buildings with {len(grid):,} 100 m cells...", flush=True)
    morphology, build_qa = build_resolution(buildings, grid, ["frontal_area_1125"])
    # build_resolution inherited the original name; all heights are complete here.
    morphology = morphology.rename(
        columns={"paris_fraction": "support_fraction", "core_cell": "support_core_cell"}
    )
    morphology["paris_fraction"] = (
        morphology.geometry.intersection(paris_geom).area / RESOLUTION**2
    )
    morphology["analysis_core"] = morphology["paris_fraction"] >= 0.5
    morphology["lambda_f_1125"] = morphology["lambda_f_1125"].fillna(0).clip(lower=0)
    morphology["lambda_p"] = morphology["lambda_p"].fillna(0).clip(0, 1)
    morphology["r1_c05"] = 1 + 0.5 * morphology["lambda_f_1125"] / FROZEN_D075_Q95_FAI
    morphology["r1_c1"] = 1 + morphology["lambda_f_1125"] / FROZEN_D075_Q95_FAI
    morphology["r1_c2"] = 1 + 2 * morphology["lambda_f_1125"] / FROZEN_D075_Q95_FAI

    theta = math.radians(TRANSPORT_DEGREES)
    morphology["flow_s_m"] = (
        morphology["x_center"] * math.sin(theta)
        + morphology["y_center"] * math.cos(theta)
    )
    paris_s_min, paris_s_max = projection_bounds(paris_geom, TRANSPORT_DEGREES)
    morphology["domain_3km"] = morphology["flow_s_m"].between(
        paris_s_min - 3000, paris_s_max + 3000
    ) & morphology["support_core_cell"]
    morphology["domain_5km"] = morphology["flow_s_m"].between(
        paris_s_min - 5000, paris_s_max + 5000
    ) & morphology["support_core_cell"]

    keep = [
        "grid_id", "ix", "iy", "x_center", "y_center", "cell_area_full_m2",
        "support_fraction", "support_core_cell", "paris_fraction", "analysis_core",
        "building_count", "footprint_union_area_m2", "lambda_p", "h_area_weighted",
        "lambda_f_1125", "r1_c05", "r1_c1", "r1_c2", "flow_s_m",
        "domain_3km", "domain_5km", "geometry",
    ]
    morphology = morphology[keep]
    morphology.to_file(
        output_gpkg, layer="extended_morphology_r1_1125", driver="GPKG", engine="pyogrio"
    )
    paris.to_file(output_gpkg, layer="paris_boundary", driver="GPKG", engine="pyogrio", append=True)
    support.to_file(output_gpkg, layer="support_footprint_500m", driver="GPKG", engine="pyogrio", append=True)
    make_map(morphology, paris, figure_png)

    core = morphology[morphology["analysis_core"]]
    occupied = morphology[morphology["lambda_p"] > 0]
    report = {
        "input_buildings_intersecting_support": int(len(buildings)),
        "n_grid_cells": int(len(morphology)),
        "n_analysis_core_cells": int(len(core)),
        "n_domain_3km_cells": int(morphology["domain_3km"].sum()),
        "n_domain_5km_cells": int(morphology["domain_5km"].sum()),
        "frozen_d075_q95_fai": FROZEN_D075_Q95_FAI,
        "resistance_formula": "R = 1 + c * lambda_f_1125 / frozen_D075_q95",
        "lambda_p_max": float(morphology["lambda_p"].max()),
        "lambda_f_1125_median_occupied": float(occupied["lambda_f_1125"].median()),
        "lambda_f_1125_q95_core": float(core["lambda_f_1125"].quantile(0.95)),
        "r1_c1_range": [float(morphology["r1_c1"].min()), float(morphology["r1_c1"].max())],
        "inherited_build_resolution_qa": build_qa,
    }
    with report_json.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
