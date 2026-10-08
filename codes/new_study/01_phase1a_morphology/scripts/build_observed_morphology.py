"""Build the observed-height Paris morphology grid at 100 m.

This is the non-imputed baseline. All footprints contribute to plan-area
fraction. Height and frontal-area metrics use only buildings with a valid
observed BD TOPO height. Per-cell height coverage is retained explicitly.

Grid conventions
----------------
* EPSG:2154.
* An origin aligned to a 1 km lattice, so 100 m cells aggregate exactly to the
  future 1 km WRF comparison grid.
* Direction is meteorological inflow direction: 0 degrees = from north,
  90 degrees = from east.
* Directional frontal area is symmetric at theta and theta+180 because it is
  calculated as footprint projected width times height; this is expected for
  the geometry-only resistance representation.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import get_coordinates
from shapely.geometry import box


DEFAULT_CLEAN = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\paris_buildings_2022_clean_observed.gpkg')
DEFAULT_OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\morphology')
DIRECTIONS = tuple(np.arange(0.0, 360.0, 22.5))
HEIGHT_BINS = (
    (0.0, 5.0, "00_05"),
    (5.0, 10.0, "05_10"),
    (10.0, 20.0, "10_20"),
    (20.0, 30.0, "20_30"),
    (30.0, 50.0, "30_50"),
    (50.0, np.inf, "50_plus"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-gpkg", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def direction_suffix(theta: float) -> str:
    value = int(round(theta * 10))
    return f"{value:04d}"


def projected_width(geometry, inflow_degrees: float) -> float:
    """Return summed component widths perpendicular to meteorological inflow."""
    theta = math.radians(inflow_degrees)
    # For a bearing measured clockwise from north, this unit vector is normal
    # to the flow axis. Its sign does not affect projection span.
    cross_x = math.cos(theta)
    cross_y = -math.sin(theta)

    if geometry.geom_type == "Polygon":
        components = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        components = list(geometry.geoms)
    else:
        return 0.0

    width = 0.0
    for polygon in components:
        coordinates = get_coordinates(polygon.exterior)
        if len(coordinates) == 0:
            continue
        projection = coordinates[:, 0] * cross_x + coordinates[:, 1] * cross_y
        width += float(projection.max() - projection.min())
    return width


def make_grid(boundary, resolution: int, origin_x: float, origin_y: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = boundary.bounds
    ix_min = math.floor((minx - origin_x) / resolution)
    iy_min = math.floor((miny - origin_y) / resolution)
    ix_max = math.ceil((maxx - origin_x) / resolution)
    iy_max = math.ceil((maxy - origin_y) / resolution)

    rows = []
    for iy in range(iy_min, iy_max):
        y0 = origin_y + iy * resolution
        for ix in range(ix_min, ix_max):
            x0 = origin_x + ix * resolution
            full = box(x0, y0, x0 + resolution, y0 + resolution)
            inside = full.intersection(boundary)
            if inside.is_empty or inside.area <= 0:
                continue
            rows.append(
                {
                    "grid_id": f"r{resolution}_x{ix:+05d}_y{iy:+05d}",
                    "ix": ix,
                    "iy": iy,
                    "x_center": x0 + resolution / 2,
                    "y_center": y0 + resolution / 2,
                    "cell_area_full_m2": float(resolution**2),
                    "analysis_area_m2": float(inside.area),
                    "paris_fraction": float(inside.area / resolution**2),
                    "edge_cell": bool(inside.area < resolution**2 * 0.999999),
                    "geometry": full,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:2154")


def representative_point_stats(
    buildings: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> pd.DataFrame:
    points = buildings[
        ["building_id", "height_valid", "height_m", "geometry"]
    ].copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    ).dropna(subset=["grid_id"])
    joined["height_observed"] = joined["height_m"].where(joined["height_valid"])
    grouped = joined.groupby("grid_id", observed=True)
    return grouped.agg(
        building_count=("building_id", "size"),
        height_valid_count=("height_valid", "sum"),
        h_mean=("height_observed", "mean"),
        h_std=("height_observed", lambda x: x.std(ddof=0)),
    )


def build_resolution(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    frontal_columns: list[str],
) -> tuple[gpd.GeoDataFrame, dict]:
    pieces = gpd.overlay(
        buildings[
            [
                "building_id",
                "footprint_area_m2",
                "height_valid",
                "height_m",
                *frontal_columns,
                "geometry",
            ]
        ],
        grid[["grid_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces["overlap_area_m2"] = pieces.geometry.area
    pieces = pieces[pieces["overlap_area_m2"] > 0].copy()
    pieces["footprint_fraction"] = np.minimum(
        1.0,
        pieces["overlap_area_m2"] / pieces["footprint_area_m2"].replace(0, np.nan),
    )

    by_grid = pieces.groupby("grid_id", observed=True)
    aggregates = by_grid.agg(
        footprint_area_sum_m2=("overlap_area_m2", "sum"),
        height_valid_footprint_m2=(
            "overlap_area_m2",
            lambda values: float(
                values[pieces.loc[values.index, "height_valid"].astype(bool)].sum()
            ),
        ),
    )

    # Union area avoids double counting overlapping footprint polygons.
    footprint_union = pieces[["grid_id", "geometry"]].dissolve(by="grid_id")
    aggregates["footprint_union_area_m2"] = footprint_union.geometry.area

    valid = pieces[pieces["height_valid"].astype(bool)].copy()
    valid["height_area_product"] = valid["height_m"] * valid["overlap_area_m2"]
    valid_grouped = valid.groupby("grid_id", observed=True)
    height_area = valid_grouped.agg(
        height_area_product=("height_area_product", "sum"),
        height_valid_area_for_weight=("overlap_area_m2", "sum"),
    )
    height_area["h_area_weighted"] = (
        height_area["height_area_product"] / height_area["height_valid_area_for_weight"]
    )
    aggregates = aggregates.join(height_area[["h_area_weighted"]], how="left")

    for lower, upper, label in HEIGHT_BINS:
        mask = valid["height_m"].ge(lower)
        if np.isfinite(upper):
            mask &= valid["height_m"].lt(upper)
        bin_area = valid.loc[mask].groupby("grid_id", observed=True)["overlap_area_m2"].sum()
        aggregates[f"hbin_{label}_area_m2"] = bin_area

    for column in frontal_columns:
        allocated = valid[column] * valid["footprint_fraction"]
        aggregates[column] = allocated.groupby(valid["grid_id"], observed=True).sum()

    point_stats = representative_point_stats(buildings, grid)
    result = grid.merge(aggregates.reset_index(), on="grid_id", how="left")
    result = result.merge(point_stats.reset_index(), on="grid_id", how="left")

    zero_columns = [
        "footprint_area_sum_m2",
        "footprint_union_area_m2",
        "height_valid_footprint_m2",
        "building_count",
        "height_valid_count",
        *frontal_columns,
        *[f"hbin_{label}_area_m2" for _, _, label in HEIGHT_BINS],
    ]
    for column in zero_columns:
        result[column] = result[column].fillna(0)

    result["building_count"] = result["building_count"].astype("int32")
    result["height_valid_count"] = result["height_valid_count"].astype("int32")
    # UCPs are defined per complete model-grid area. Using only the sliver of a
    # cell inside the administrative boundary would inflate edge-cell values.
    # paris_fraction and core_cell retain the boundary-completeness information.
    result["lambda_p"] = (
        result["footprint_union_area_m2"] / result["cell_area_full_m2"]
    )
    result["height_valid_footprint_share"] = np.where(
        result["footprint_area_sum_m2"] > 0,
        result["height_valid_footprint_m2"] / result["footprint_area_sum_m2"],
        np.nan,
    )
    for column in frontal_columns:
        suffix = column.removeprefix("frontal_area_")
        result[f"lambda_f_{suffix}"] = result[column] / result["cell_area_full_m2"]
        result = result.drop(columns=column)

    valid_area = result["height_valid_footprint_m2"].replace(0, np.nan)
    for _, _, label in HEIGHT_BINS:
        area_column = f"hbin_{label}_area_m2"
        result[f"hbin_{label}_frac_valid"] = result[area_column] / valid_area
        result = result.drop(columns=area_column)

    result["morphology_observed_only"] = True
    result["core_cell"] = result["paris_fraction"] >= 0.999999
    result["height_coverage_flag"] = pd.cut(
        result["height_valid_footprint_share"],
        bins=[-np.inf, 0.80, 0.90, 0.95, np.inf],
        labels=["poor_lt80", "moderate_80_90", "good_90_95", "excellent_ge95"],
    ).astype("string")

    result = gpd.GeoDataFrame(result, geometry="geometry", crs=grid.crs)
    occupied = result["footprint_union_area_m2"] > 0
    core_occupied = occupied & result["core_cell"]
    qa = {
        "n_cells": int(len(result)),
        "n_cells_with_buildings": int(occupied.sum()),
        "n_core_cells_with_buildings": int(core_occupied.sum()),
        "n_edge_cells": int(result["edge_cell"].sum()),
        "lambda_p_min": float(result["lambda_p"].min()),
        "lambda_p_median_occupied": float(
            result.loc[core_occupied, "lambda_p"].median()
        ),
        "lambda_p_p99": float(result["lambda_p"].quantile(0.99)),
        "lambda_p_max": float(result["lambda_p"].max()),
        "cells_lambda_p_gt_1": int((result["lambda_p"] > 1.000001).sum()),
        "height_valid_footprint_share_median_occupied": float(
            result.loc[core_occupied, "height_valid_footprint_share"].median()
        ),
        "height_valid_footprint_share_p05_occupied": float(
            result.loc[core_occupied, "height_valid_footprint_share"].quantile(0.05)
        ),
        "height_coverage_flag_counts": {
            str(k): int(v)
            for k, v in result["height_coverage_flag"].fillna("no_buildings").value_counts().items()
        },
    }
    return result, qa


def composite_map(grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, path: Path, resolution: int) -> None:
    occupied = grid[(grid["lambda_p"] > 0) & (grid["paris_fraction"] >= 0.5)]
    panels = [
        ("h_area_weighted", "Area-weighted mean height (m)", "viridis", 35),
        ("h_std", "Building-height SD (m)", "plasma", 12),
        ("lambda_p", "Plan-area fraction", "YlGnBu", 0.8),
        ("lambda_f_1125", "Frontal-area index, inflow 112.5°", "magma", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2), dpi=180)
    for ax, (column, title, cmap, fixed_max) in zip(axes.flat, panels):
        vmax = fixed_max
        if vmax is None:
            vmax = float(occupied[column].quantile(0.99)) if len(occupied) else 1.0
        display_grid = grid[grid["paris_fraction"] >= 0.5]
        display_grid.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            linewidth=0,
            legend=True,
            missing_kwds={"color": "#eeeeee"},
        )
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.55)
        ax.set_title(title, fontsize=10.5)
        ax.set_axis_off()
    fig.suptitle(f"Paris observed-height morphology baseline ({resolution} m)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clean = args.clean_gpkg.resolve()
    output = args.output_dir.resolve()
    if not clean.is_file():
        raise FileNotFoundError(clean)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    gpkg_output = output / "paris_morphology_observed_100m.gpkg"
    qa_output = output / "morphology_qa.json"
    if gpkg_output.exists() or qa_output.exists():
        raise FileExistsError("Refusing to overwrite an existing morphology run")

    buildings = gpd.read_file(clean, layer="paris_buildings_clean", engine="pyogrio")
    boundary = gpd.read_file(clean, layer="paris_boundary", engine="pyogrio")
    if buildings.crs is None or buildings.crs.to_epsg() != 2154:
        raise ValueError(f"Unexpected building CRS: {buildings.crs}")
    if (~buildings.geometry.is_valid).any() or buildings.geometry.is_empty.any():
        raise ValueError("Clean input contains invalid or empty geometries")
    if not buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Clean input contains non-polygon geometries")

    boundary_geom = boundary.geometry.union_all()
    minx, miny, _, _ = boundary_geom.bounds
    origin_x = math.floor(minx / 1000.0) * 1000.0
    origin_y = math.floor(miny / 1000.0) * 1000.0

    valid = buildings["height_valid"].astype(bool)
    frontal_columns = []
    for theta in DIRECTIONS:
        suffix = direction_suffix(theta)
        width_column = f"projected_width_{suffix}"
        frontal_column = f"frontal_area_{suffix}"
        buildings[width_column] = 0.0
        buildings.loc[valid, width_column] = buildings.loc[valid, "geometry"].map(
            lambda geom, angle=theta: projected_width(geom, angle)
        )
        buildings[frontal_column] = buildings[width_column] * buildings["height_m"].fillna(0)
        buildings = buildings.drop(columns=width_column)
        frontal_columns.append(frontal_column)

    qa = {
        "input_clean_gpkg": str(clean),
        "input_building_count": int(len(buildings)),
        "input_height_valid_count": int(valid.sum()),
        "crs": str(buildings.crs),
        "common_grid_origin_x": origin_x,
        "common_grid_origin_y": origin_y,
        "grid_alignment_m": 1000,
        "directions_degrees_meteorological_inflow": list(DIRECTIONS),
        "height_bins_m": [
            [lower, None if not np.isfinite(upper) else upper, label]
            for lower, upper, label in HEIGHT_BINS
        ],
        "height_policy": "observed only; no imputation",
        "lambda_p_policy": "all in-service footprint polygons; union area divided by full grid-cell area",
        "lambda_f_policy": "projected width * observed height, allocated by footprint overlap fraction and divided by full grid-cell area",
        "resolutions": {},
    }

    for resolution in (100,):
        grid = make_grid(boundary_geom, resolution, origin_x, origin_y)
        morphology, resolution_qa = build_resolution(buildings, grid, frontal_columns)
        layer = f"morphology_{resolution}m_observed"
        morphology.to_file(
            gpkg_output,
            layer=layer,
            driver="GPKG",
            engine="pyogrio",
            append=gpkg_output.exists(),
        )
        composite_map(
            morphology,
            boundary,
            figures / f"morphology_QA_{resolution}m.png",
            resolution,
        )
        qa["resolutions"][str(resolution)] = resolution_qa

    with qa_output.open("w", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
