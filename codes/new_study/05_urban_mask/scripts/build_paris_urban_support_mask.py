from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely


RAW_GPKG = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO\\1_DONNEES_LIVRAISON_2022-06-00173\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15.gpkg')
MORPHOLOGY_GPKG = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed\\paris_morphology_100m_imputed_v2.gpkg')
BUILDINGS_GPKG = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed\\paris_buildings_2022_targeted_imputed.gpkg')
OUTPUT_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\urban_mask_100m_v1')

BUFFER_DISTANCES_M = (10, 20, 30)
PRIMARY_BUFFER_M = 20
ACTIVATION_THRESHOLD = 0.20
URBAN_THRESHOLD = 0.50
N_WORKERS = 6


def valid_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].buffer(0)
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def load_layer(layer: str, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    frame = pyogrio.read_dataframe(RAW_GPKG, layer=layer, columns=columns)
    return valid_geometries(frame)


def local_union(tree: shapely.STRtree, geometries: np.ndarray, query_geometry, predicate: str = "intersects"):
    indices = tree.query(query_geometry, predicate=predicate)
    if len(indices) == 0:
        return shapely.GeometryCollection()
    return shapely.union_all(geometries[indices])


def plot_maps(result: gpd.GeoDataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    items = [
        ("lambda_p", "Building plan fraction", "YlOrBr", 0.0, 0.8),
        ("explicit_nonurban_fraction", "Explicit water/woodland/park fraction", "YlGnBu", 0.0, 1.0),
        ("urban_fraction_candidate", "Urban-support fraction (20 m buffer)", "magma", 0.0, 1.0),
        ("urban_support_class_code", "Support class: 0 nonurban, 1 mixed, 2 urban", "viridis", 0.0, 2.0),
        ("urban_fraction_b10", "Urban-support fraction (10 m)", "magma", 0.0, 1.0),
        ("urban_fraction_b30", "Urban-support fraction (30 m)", "magma", 0.0, 1.0),
    ]
    for axis, (column, title, cmap, vmin, vmax) in zip(axes.ravel(), items):
        result.plot(column=column, ax=axis, cmap=cmap, vmin=vmin, vmax=vmax, legend=True)
        axis.set_title(title, fontsize=12)
        axis.set_axis_off()
    fig.suptitle("Paris D075 100 m urban-support mask pilot", fontsize=16)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Non-empty output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(MORPHOLOGY_GPKG, layer="morphology_100m_imputed")
    grid = grid.sort_values("grid_id").reset_index(drop=True)
    if grid.crs.to_epsg() != 2154:
        raise ValueError(f"Expected EPSG:2154 morphology; found {grid.crs}")
    grid["raster_grid_id"] = np.arange(1, len(grid) + 1, dtype=np.int32)

    departments = load_layer("departement", ["code_insee", "nom_officiel"])
    paris = departments[departments["code_insee"].astype(str) == "75"].copy()
    if len(paris) != 1:
        raise ValueError(f"Expected one D075 polygon; found {len(paris)}")

    buildings = gpd.read_file(BUILDINGS_GPKG, layer="paris_buildings_targeted_imputed")
    buildings = valid_geometries(buildings.to_crs(grid.crs))
    water = load_layer("surface_hydrographique", ["nature"]).to_crs(grid.crs)
    vegetation = load_layer("zone_de_vegetation", ["nature"]).to_crs(grid.crs)
    woody = vegetation[vegetation["nature"].fillna("") != "Haie"].copy()
    activities = load_layer("zone_d_activite_ou_d_interet", ["nature", "nature_detaillee"]).to_crs(grid.crs)
    parks = activities[activities["nature_detaillee"].fillna("") == "Parc"].copy()

    paris_geometry = shapely.union_all(paris.geometry.array)
    grid_geometries = np.asarray(grid.geometry.array, dtype=object)
    denominators = grid["analysis_area_m2"].to_numpy(dtype=float)
    building_geometries = np.asarray(buildings.geometry.array, dtype=object)
    water_geometries = np.asarray(water.geometry.array, dtype=object)
    woody_geometries = np.asarray(woody.geometry.array, dtype=object)
    park_geometries = np.asarray(parks.geometry.array, dtype=object)
    building_tree = shapely.STRtree(building_geometries)
    water_tree = shapely.STRtree(water_geometries)
    woody_tree = shapely.STRtree(woody_geometries)
    park_tree = shapely.STRtree(park_geometries)

    output_columns = [
        "building_footprint_fraction_vector",
        "water_fraction",
        "woody_fraction",
        "park_fraction",
        "explicit_nonurban_fraction",
        "urban_fraction_b10",
        "urban_fraction_b20",
        "urban_fraction_b30",
    ]

    def compute_cell(index: int) -> tuple[float, ...]:
        cell = grid_geometries[index]
        cell_paris = shapely.intersection(cell, paris_geometry)
        denominator = denominators[index]
        if denominator <= 0 or shapely.is_empty(cell_paris):
            return tuple(0.0 for _ in output_columns)

        search_area = shapely.buffer(cell, max(BUFFER_DISTANCES_M))
        building_union = local_union(building_tree, building_geometries, search_area)
        water_union = local_union(water_tree, water_geometries, cell)
        woody_union = local_union(woody_tree, woody_geometries, cell)
        park_union = local_union(park_tree, park_geometries, cell)
        explicit_union = shapely.union_all([water_union, woody_union, park_union])
        explicit_in_cell = shapely.intersection(explicit_union, cell_paris)

        def area_fraction(geometry) -> float:
            area = float(shapely.area(shapely.intersection(geometry, cell_paris)))
            return float(np.clip(area / denominator, 0.0, 1.0))

        values = [
            area_fraction(building_union),
            area_fraction(water_union),
            area_fraction(woody_union),
            area_fraction(park_union),
            area_fraction(explicit_union),
        ]
        for distance in BUFFER_DISTANCES_M:
            support = shapely.difference(shapely.buffer(building_union, distance), explicit_in_cell)
            values.append(area_fraction(support))
        return tuple(values)

    records: list[tuple[float, ...]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for completed, values in enumerate(executor.map(compute_cell, range(len(grid))), start=1):
            records.append(values)
            if completed % 1000 == 0 or completed == len(grid):
                print(f"Processed {completed}/{len(grid)} grid cells", flush=True)

    result = grid.copy()
    value_array = np.asarray(records, dtype=float)
    for column_index, column in enumerate(output_columns):
        result[column] = value_array[:, column_index]

    result["urban_fraction_candidate"] = result[f"urban_fraction_b{PRIMARY_BUFFER_M}"]
    result["urban_mask_candidate"] = (
        (result["urban_fraction_candidate"] >= ACTIVATION_THRESHOLD)
        & (result["explicit_nonurban_fraction"] < 0.80)
    ).astype("int8")
    result["urban_support_class"] = pd.cut(
        result["urban_fraction_candidate"],
        bins=[-np.inf, ACTIVATION_THRESHOLD, URBAN_THRESHOLD, np.inf],
        labels=["nonurban", "mixed", "urban"],
        right=False,
    ).astype(str)
    class_map = {"nonurban": 0, "mixed": 1, "urban": 2}
    result["urban_support_class_code"] = result["urban_support_class"].map(class_map).astype("int8")
    result["mask_rule"] = "20m_building_envelope_minus_water_woody_Parc"
    result["mask_status"] = "PILOT_NOT_FINAL_WRF"

    output_gpkg = OUTPUT_DIR / "paris_urban_support_mask_100m_v1.gpkg"
    output_csv = OUTPUT_DIR / "paris_urban_support_mask_100m_v1.csv"
    output_png = OUTPUT_DIR / "paris_urban_support_mask_100m_QA.png"
    output_json = OUTPUT_DIR / "paris_urban_support_mask_100m_QA.json"
    result.to_file(output_gpkg, layer="urban_support_100m", driver="GPKG")
    result.drop(columns="geometry").to_csv(output_csv, index=False)
    plot_maps(result, output_png)

    core = result[result["paris_fraction"] >= 0.90].copy()
    sensitivity = {}
    primary_binary = core["urban_fraction_b20"] >= ACTIVATION_THRESHOLD
    for distance in (10, 30):
        alternate = core[f"urban_fraction_b{distance}"] >= ACTIVATION_THRESHOLD
        sensitivity[f"b{distance}_vs_b20_binary_agreement"] = float((alternate == primary_binary).mean())
        sensitivity[f"b{distance}_vs_b20_mean_abs_fraction_difference"] = float(
            (core[f"urban_fraction_b{distance}"] - core["urban_fraction_b20"]).abs().mean()
        )

    qa = {
        "status": "PILOT_NOT_FINAL_WRF_MASK",
        "input_grid_cells": int(len(result)),
        "D075_core_ge90pct_cells": int(len(core)),
        "area_method": "exact vector intersection in EPSG:2154",
        "parallel_workers": N_WORKERS,
        "spatial_strategy": "per-cell STRtree query within 30 m; local union and buffering",
        "primary_building_buffer_m": PRIMARY_BUFFER_M,
        "activation_threshold": ACTIVATION_THRESHOLD,
        "urban_threshold": URBAN_THRESHOLD,
        "explicit_exclusions": [
            "surface_hydrographique",
            "zone_de_vegetation excluding Haie",
            "zone_d_activite_ou_d_interet where nature_detaillee=Parc",
        ],
        "core_class_counts": {str(k): int(v) for k, v in core["urban_support_class"].value_counts().items()},
        "core_candidate_fraction_quantiles": {
            str(q): float(core["urban_fraction_candidate"].quantile(q)) for q in [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
        },
        "core_explicit_nonurban_quantiles": {
            str(q): float(core["explicit_nonurban_fraction"].quantile(q)) for q in [0, 0.5, 0.9, 0.95, 1]
        },
        "sensitivity": sensitivity,
        "limitations_before_final_wrf": [
            "20 m building-envelope distance is a pilot scale, not a physical transport distance",
            "BD TOPO park and woody layers do not constitute a complete impervious-surface product",
            "final activation and FRC_URB2D must be aggregated on the actual WRF d03 grid and checked against WPS land use",
        ],
        "outputs": {
            "gpkg": str(output_gpkg),
            "csv": str(output_csv),
            "figure": str(output_png),
        },
    }
    output_json.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
