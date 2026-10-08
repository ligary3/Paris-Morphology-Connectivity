"""Prepare and audit the June 2022 Paris BD TOPO building layer.

The raw GeoPackage is treated as immutable. The script:
1. extracts the official Paris commune (INSEE 75056),
2. selects in-service buildings intersecting Paris,
3. clips them to the commune boundary,
4. marks observed versus unreliable/missing heights without imputing them,
5. writes a clean GeoPackage and 100/250 m missingness grids,
6. produces machine-readable QA tables and diagnostic figures.

Height imputation and morphology/FAI calculations deliberately happen later,
after the missingness maps have been reviewed.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union


DEFAULT_RAW = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO\\1_DONNEES_LIVRAISON_2022-06-00173\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15.gpkg')
DEFAULT_OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v1')
EXPECTED_MD5 = "545bbac4dc944ef817b47af6f7adfd66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-gpkg", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_grid(boundary, resolution: int) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = boundary.bounds
    xmin = math.floor(minx / resolution) * resolution
    ymin = math.floor(miny / resolution) * resolution
    xmax = math.ceil(maxx / resolution) * resolution
    ymax = math.ceil(maxy / resolution) * resolution

    records: list[dict] = []
    row_id = 0
    y = ymin
    while y < ymax:
        x = xmin
        while x < xmax:
            geom = box(x, y, x + resolution, y + resolution)
            clipped = geom.intersection(boundary)
            if not clipped.is_empty and clipped.area > 0:
                records.append(
                    {
                        "grid_id": f"r{resolution}_{row_id:06d}",
                        "x_left": x,
                        "y_bottom": y,
                        "cell_area_m2": clipped.area,
                        "geometry": clipped,
                    }
                )
                row_id += 1
            x += resolution
        y += resolution
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:2154")


def missingness_grid(
    buildings: gpd.GeoDataFrame, boundary, resolution: int
) -> gpd.GeoDataFrame:
    grid = aligned_grid(boundary, resolution)
    points = buildings[["building_id", "height_valid", "geometry"]].copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    counts = (
        joined.dropna(subset=["grid_id"])
        .groupby("grid_id", observed=True)
        .agg(
            n_buildings=("building_id", "size"),
            n_height_valid=("height_valid", "sum"),
        )
        .reset_index()
    )
    grid = grid.merge(counts, on="grid_id", how="left")
    grid["n_buildings"] = grid["n_buildings"].fillna(0).astype("int32")
    grid["n_height_valid"] = grid["n_height_valid"].fillna(0).astype("int32")
    grid["n_height_missing"] = grid["n_buildings"] - grid["n_height_valid"]
    grid["height_missing_share"] = np.where(
        grid["n_buildings"] > 0,
        grid["n_height_missing"] / grid["n_buildings"],
        np.nan,
    )
    return grid


def write_missingness_map(
    grid: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, path: Path, resolution: int
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=180)
    grid.plot(
        column="height_missing_share",
        cmap="magma_r",
        vmin=0,
        vmax=0.5,
        linewidth=0,
        legend=True,
        legend_kwds={"label": "Share of buildings with missing/unreliable height"},
        missing_kwds={"color": "#eeeeee", "label": "No buildings"},
        ax=ax,
    )
    boundary_gdf.boundary.plot(ax=ax, color="#222222", linewidth=0.7)
    ax.set_title(f"Paris BD TOPO height missingness ({resolution} m grid)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_building_status_map(
    buildings: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, path: Path
) -> None:
    centroids = buildings[["height_valid", "geometry"]].copy()
    centroids.geometry = centroids.geometry.representative_point()
    valid = centroids[centroids["height_valid"]]
    missing = centroids[~centroids["height_valid"]]

    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=180)
    valid.plot(ax=ax, color="#b8b8b8", markersize=0.18, alpha=0.45)
    missing.plot(ax=ax, color="#d7301f", markersize=1.2, alpha=0.80)
    boundary_gdf.boundary.plot(ax=ax, color="#222222", linewidth=0.7)
    ax.set_title("Paris buildings with missing/unreliable height")
    ax.text(
        0.01,
        0.01,
        "Gray: observed height   Red: missing/unreliable height",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def grid_qa(grid: gpd.GeoDataFrame, resolution: int) -> dict:
    occupied = grid[grid["n_buildings"] > 0]
    return {
        "resolution_m": resolution,
        "n_cells_total": int(len(grid)),
        "n_cells_with_buildings": int(len(occupied)),
        "cells_zero_missing_pct": float(
            100 * (occupied["n_height_missing"] == 0).mean()
        )
        if len(occupied)
        else None,
        "cells_missing_share_gt_10pct": int(
            (occupied["height_missing_share"] > 0.10).sum()
        ),
        "cells_missing_share_gt_25pct": int(
            (occupied["height_missing_share"] > 0.25).sum()
        ),
        "cells_missing_share_gt_50pct": int(
            (occupied["height_missing_share"] > 0.50).sum()
        ),
        "median_missing_share_occupied": float(
            occupied["height_missing_share"].median()
        )
        if len(occupied)
        else None,
        "p95_missing_share_occupied": float(
            occupied["height_missing_share"].quantile(0.95)
        )
        if len(occupied)
        else None,
    }


def main() -> None:
    args = parse_args()
    raw = args.raw_gpkg.resolve()
    output = args.output_dir.resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)

    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    tables_dir = output / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    clean_path = output / "paris_buildings_2022_clean_observed.gpkg"
    grid_path = output / "paris_height_missingness_grids.gpkg"
    for target in (clean_path, grid_path):
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {target}. "
                "Use a new output directory for a new run."
            )

    raw_md5 = file_md5(raw)
    if raw_md5 != EXPECTED_MD5:
        raise ValueError(f"Raw MD5 mismatch: {raw_md5} != {EXPECTED_MD5}")

    paris = gpd.read_file(
        raw,
        layer="commune",
        where="code_insee = '75056'",
        engine="pyogrio",
    )
    if len(paris) != 1:
        raise ValueError(f"Expected one Paris commune feature, got {len(paris)}")
    if paris.crs is None or paris.crs.to_epsg() != 2154:
        raise ValueError(f"Unexpected CRS: {paris.crs}")
    paris_geom = unary_union(paris.geometry)

    source_columns = [
        "cleabs",
        "nature",
        "usage_1",
        "usage_2",
        "construction_legere",
        "etat_de_l_objet",
        "date_creation",
        "date_modification",
        "methode_d_acquisition_altimetri",
        "precision_altimetrique",
        "nombre_de_logements",
        "nombre_d_etages",
        "sources",
        "hauteur",
        "altitude_minimale_sol",
        "altitude_minimale_toit",
        "altitude_maximale_toit",
        "altitude_maximale_sol",
        "geometry",
    ]
    buildings = gpd.read_file(
        raw,
        layer="batiment",
        bbox=tuple(float(value) for value in paris.total_bounds),
        columns=source_columns,
        engine="pyogrio",
        use_arrow=False,
    )
    buildings = buildings[buildings.geometry.intersects(paris_geom)].copy()
    buildings = buildings[buildings["etat_de_l_objet"] == "En service"].copy()
    buildings.geometry = buildings.geometry.intersection(paris_geom)
    buildings = buildings[~buildings.geometry.is_empty].copy()
    buildings.geometry = buildings.geometry.force_2d()
    buildings = buildings[
        buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    buildings = buildings[buildings.geometry.area > 0].copy()

    buildings = buildings.rename(
        columns={
            "cleabs": "building_id",
            "usage_1": "usage_primary",
            "usage_2": "usage_secondary",
            "construction_legere": "is_lightweight",
            "etat_de_l_objet": "object_status",
            "date_creation": "created_at",
            "date_modification": "modified_at",
            "methode_d_acquisition_altimetri": "altimetry_method",
            "precision_altimetrique": "altimetry_precision_m",
            "nombre_de_logements": "n_dwellings",
            "nombre_d_etages": "n_floors",
            "sources": "source_records",
            "hauteur": "height_raw_m",
            "altitude_minimale_sol": "ground_z_min_m",
            "altitude_minimale_toit": "roof_z_min_m",
            "altitude_maximale_toit": "roof_z_max_m",
            "altitude_maximale_sol": "ground_z_max_m",
        }
    )
    buildings["footprint_area_m2"] = buildings.geometry.area
    unreliable_no_z = buildings["altimetry_method"].eq("Pas de Z")
    null_height = buildings["height_raw_m"].isna()
    nonpositive_height = buildings["height_raw_m"].fillna(0).le(0)
    buildings["height_valid"] = ~(
        unreliable_no_z | null_height | nonpositive_height
    )
    buildings["height_m"] = buildings["height_raw_m"].where(
        buildings["height_valid"]
    )
    buildings["height_source"] = np.select(
        [
            buildings["height_valid"],
            unreliable_no_z,
            null_height,
            nonpositive_height,
        ],
        ["observed", "missing_no_z", "missing_null", "missing_nonpositive"],
        default="missing_other",
    )
    buildings["height_imputed"] = False

    preferred_order = [
        "building_id",
        "nature",
        "usage_primary",
        "usage_secondary",
        "is_lightweight",
        "object_status",
        "height_raw_m",
        "height_m",
        "height_valid",
        "height_source",
        "height_imputed",
        "footprint_area_m2",
        "altimetry_method",
        "altimetry_precision_m",
        "n_dwellings",
        "n_floors",
        "source_records",
        "ground_z_min_m",
        "roof_z_min_m",
        "roof_z_max_m",
        "ground_z_max_m",
        "created_at",
        "modified_at",
        "geometry",
    ]
    buildings = buildings[preferred_order].sort_values("building_id").reset_index(drop=True)

    buildings.to_file(
        clean_path,
        layer="paris_buildings_clean",
        driver="GPKG",
        engine="pyogrio",
    )
    paris[["code_insee", "nom_officiel", "geometry"]].to_file(
        clean_path,
        layer="paris_boundary",
        driver="GPKG",
        engine="pyogrio",
        append=True,
    )

    grids: dict[int, gpd.GeoDataFrame] = {}
    grid_reports: list[dict] = []
    for resolution in (100, 250):
        grid = missingness_grid(buildings, paris_geom, resolution)
        grids[resolution] = grid
        grid_reports.append(grid_qa(grid, resolution))
        grid.to_file(
            grid_path,
            layer=f"missingness_{resolution}m",
            driver="GPKG",
            engine="pyogrio",
            append=grid_path.exists(),
        )
        write_missingness_map(
            grid,
            paris,
            figures_dir / f"height_missingness_{resolution}m.png",
            resolution,
        )
    write_building_status_map(
        buildings, paris, figures_dir / "building_height_status.png"
    )

    missing_by_source = (
        buildings.groupby(
            ["height_source", "altimetry_method", "nature", "usage_primary"],
            dropna=False,
        )
        .agg(
            n_buildings=("building_id", "size"),
            footprint_area_m2=("footprint_area_m2", "sum"),
        )
        .reset_index()
        .sort_values(["height_source", "n_buildings"], ascending=[True, False])
    )
    missing_by_source.to_csv(
        tables_dir / "height_status_by_attributes.csv", index=False, encoding="utf-8-sig"
    )

    valid_height = buildings.loc[buildings["height_valid"], "height_m"]
    qa = {
        "raw_file": str(raw),
        "raw_md5": raw_md5,
        "raw_md5_expected": EXPECTED_MD5,
        "raw_md5_match": raw_md5 == EXPECTED_MD5,
        "data_edition": "2022-06-15",
        "crs": str(buildings.crs),
        "paris_insee": "75056",
        "paris_boundary_area_km2": float(paris_geom.area / 1e6),
        "selection": "intersects Paris boundary; object_status == En service",
        "n_buildings_clean": int(len(buildings)),
        "n_unique_building_ids": int(buildings["building_id"].nunique()),
        "n_invalid_geometries": int((~buildings.geometry.is_valid).sum()),
        "n_empty_geometries": int(buildings.geometry.is_empty.sum()),
        "n_height_valid": int(buildings["height_valid"].sum()),
        "n_height_missing_or_unreliable": int((~buildings["height_valid"]).sum()),
        "height_valid_pct": float(100 * buildings["height_valid"].mean()),
        "height_missing_or_unreliable_pct": float(
            100 * (~buildings["height_valid"]).mean()
        ),
        "height_valid_quantiles_m": {
            str(q): float(valid_height.quantile(q))
            for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1)
        },
        "height_source_counts": {
            str(k): int(v) for k, v in buildings["height_source"].value_counts().items()
        },
        "altimetry_method_counts": {
            str(k): int(v)
            for k, v in buildings["altimetry_method"].fillna("<null>").value_counts().items()
        },
        "lightweight_building_count": int(buildings["is_lightweight"].fillna(False).sum()),
        "grid_missingness": grid_reports,
        "outputs": {
            "clean_buildings_gpkg": str(clean_path),
            "missingness_grids_gpkg": str(grid_path),
            "figures_dir": str(figures_dir),
            "tables_dir": str(tables_dir),
        },
    }
    with (output / "qa_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in qa.items()
            if isinstance(value, (str, int, float, bool))
        ]
    ).to_csv(tables_dir / "qa_summary_flat.csv", index=False, encoding="utf-8-sig")

    # ASCII escaping keeps the final console report portable on Windows shells
    # configured with a legacy code page; JSON files themselves remain UTF-8.
    print(json.dumps(qa, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
