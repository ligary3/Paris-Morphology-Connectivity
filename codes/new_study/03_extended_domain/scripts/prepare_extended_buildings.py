"""Prepare the full June-2022 BD TOPO delivery for the Circuit support domain.

The raw GeoPackage remains immutable. Buildings are cleaned once, labelled as
Paris core or peripheral computational support, and audited for height and
geometry quality. No height is imputed in this script.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon


DEFAULT_RAW = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15\\BDTOPO\\1_DONNEES_LIVRAISON_2022-06-00173\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15\\BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15.gpkg')
DEFAULT_OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')
EXPECTED_MD5 = "545bbac4dc944ef817b47af6f7adfd66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-gpkg", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delivery_footprint(raw: Path) -> gpd.GeoDataFrame:
    metadata = gpd.read_file(raw, layer="metadonnees_lot", engine="pyogrio")
    text = str(metadata.iloc[0]["etendue_geographique"])
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    latitudes = sorted(value for value in numbers if abs(value) > 40)
    longitudes = sorted(value for value in numbers if abs(value) <= 40)
    if len(latitudes) != 2 or len(longitudes) != 2:
        raise ValueError(f"Could not parse delivery extent: {text}")
    lon_min, lon_max = longitudes
    lat_min, lat_max = latitudes
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    corners = [
        transformer.transform(lon_min, lat_min),
        transformer.transform(lon_max, lat_min),
        transformer.transform(lon_max, lat_max),
        transformer.transform(lon_min, lat_max),
    ]
    return gpd.GeoDataFrame(
        [{"source": "BD TOPO metadata geographic extent", "geometry": Polygon(corners)}],
        geometry="geometry",
        crs="EPSG:2154",
    )


def summarize_subset(frame: gpd.GeoDataFrame) -> dict:
    valid = frame["height_valid"].astype(bool)
    area_total = float(frame["footprint_area_m2"].sum())
    missing_area = float(frame.loc[~valid, "footprint_area_m2"].sum())
    return {
        "n_buildings": int(len(frame)),
        "n_height_valid": int(valid.sum()),
        "n_height_missing": int((~valid).sum()),
        "height_valid_count_pct": float(valid.mean() * 100) if len(frame) else None,
        "footprint_area_total_m2": area_total,
        "missing_height_footprint_m2": missing_area,
        "missing_height_footprint_pct": float(missing_area / area_total * 100)
        if area_total
        else None,
        "missing_ge_500m2": int(
            ((~valid) & frame["footprint_area_m2"].ge(500)).sum()
        ),
        "missing_ge_1000m2": int(
            ((~valid) & frame["footprint_area_m2"].ge(1000)).sum()
        ),
        "missing_ge_5000m2": int(
            ((~valid) & frame["footprint_area_m2"].ge(5000)).sum()
        ),
    }


def make_qa_map(
    buildings: gpd.GeoDataFrame,
    paris: gpd.GeoDataFrame,
    footprint: gpd.GeoDataFrame,
    support: gpd.GeoDataFrame,
    output: Path,
) -> None:
    points = buildings[["height_valid", "in_paris_core", "geometry"]].copy()
    points.geometry = points.geometry.representative_point()
    # Plotting all points remains manageable but use tiny markers.
    valid = points[points["height_valid"]]
    missing = points[~points["height_valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=180)
    valid.plot(ax=axes[0], color="#bdbdbd", markersize=0.10, alpha=0.35)
    missing.plot(ax=axes[0], color="#d7301f", markersize=0.45, alpha=0.70)
    footprint.boundary.plot(ax=axes[0], color="#222222", linewidth=0.7)
    support.boundary.plot(ax=axes[0], color="#3182bd", linewidth=0.7)
    paris.boundary.plot(ax=axes[0], color="#08519c", linewidth=1.0)
    axes[0].set_title("Full delivery: observed (gray) and missing height (red)")
    axes[0].set_axis_off()

    peripheral = buildings[~buildings["in_paris_core"]].copy()
    peripheral["area_bin"] = pd.cut(
        peripheral["footprint_area_m2"],
        bins=[0, 20, 100, 500, 1000, 5000, np.inf],
        labels=["<20", "20–100", "100–500", "500–1000", "1000–5000", ">=5000"],
    )
    table = (
        peripheral.groupby("area_bin", observed=True)["height_valid"]
        .agg(["count", "mean"])
        .reset_index()
    )
    axes[1].bar(table["area_bin"].astype(str), (1 - table["mean"]) * 100, color="#756bb1")
    axes[1].set_ylabel("Missing/unreliable height (%)")
    axes[1].set_xlabel("Peripheral footprint-area class (m²)")
    axes[1].set_title("Peripheral missingness by building size")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw = args.raw_gpkg.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpkg_output = output / "extended_buildings_clean_observed.gpkg"
    qa_output = output / "extended_buildings_qa.json"
    table_output = output / "height_qa_core_periphery.csv"
    figure_output = output / "extended_building_height_QA.png"
    if any(path.exists() for path in [gpkg_output, qa_output, table_output, figure_output]):
        raise FileExistsError("Refusing to overwrite an existing extended-domain run")

    digest = file_md5(raw)
    if digest != EXPECTED_MD5:
        raise ValueError(f"Raw MD5 mismatch: {digest}")
    paris = gpd.read_file(
        raw, layer="commune", where="code_insee = '75056'", engine="pyogrio"
    )[["code_insee", "nom_officiel", "geometry"]]
    if len(paris) != 1:
        raise ValueError("Paris commune feature was not uniquely identified")
    paris_geom = paris.geometry.union_all()
    footprint = delivery_footprint(raw)
    support_geom = footprint.geometry.iloc[0].buffer(-500)
    if support_geom.is_empty:
        raise ValueError("The 500 m support-footprint erosion is empty")
    support = gpd.GeoDataFrame(
        [{"guard_m": 500, "geometry": support_geom}], geometry="geometry", crs="EPSG:2154"
    )

    source_columns = [
        "cleabs", "nature", "usage_1", "usage_2", "construction_legere",
        "etat_de_l_objet", "date_creation", "date_modification",
        "methode_d_acquisition_altimetri", "precision_altimetrique",
        "nombre_de_logements", "nombre_d_etages", "sources", "hauteur",
        "altitude_minimale_sol", "altitude_minimale_toit",
        "altitude_maximale_toit", "altitude_maximale_sol", "geometry",
    ]
    buildings = gpd.read_file(
        raw,
        layer="batiment",
        columns=source_columns,
        engine="pyogrio",
        use_arrow=False,
    )
    n_raw = len(buildings)
    buildings = buildings[buildings["etat_de_l_objet"].eq("En service")].copy()
    n_in_service = len(buildings)
    buildings.geometry = buildings.geometry.force_2d()
    invalid_before = int((~buildings.geometry.is_valid).sum())
    if invalid_before:
        bad = ~buildings.geometry.is_valid
        buildings.loc[bad, "geometry"] = buildings.loc[bad, "geometry"].make_valid()
    buildings = buildings[
        ~buildings.geometry.is_empty
        & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
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
    nonpositive = buildings["height_raw_m"].fillna(0).le(0)
    buildings["height_valid"] = ~(unreliable_no_z | null_height | nonpositive)
    buildings["height_m"] = buildings["height_raw_m"].where(buildings["height_valid"])
    buildings["height_source"] = np.select(
        [buildings["height_valid"], unreliable_no_z, null_height, nonpositive],
        ["observed", "missing_no_z", "missing_null", "missing_nonpositive"],
        default="missing_other",
    )
    representative = buildings.geometry.representative_point()
    buildings["in_paris_core"] = representative.within(paris_geom)
    buildings["intersects_paris_core"] = buildings.geometry.intersects(paris_geom)
    buildings["in_support_500m"] = representative.within(support_geom)
    buildings["domain_role"] = np.where(
        buildings["in_paris_core"], "paris_core", "peripheral_support"
    )

    buildings = buildings.sort_values("building_id").reset_index(drop=True)
    preferred = [
        "building_id", "domain_role", "in_paris_core", "intersects_paris_core",
        "in_support_500m", "nature", "usage_primary", "usage_secondary",
        "is_lightweight", "object_status", "height_raw_m", "height_m",
        "height_valid", "height_source", "footprint_area_m2",
        "altimetry_method", "altimetry_precision_m", "n_dwellings", "n_floors",
        "source_records", "ground_z_min_m", "roof_z_min_m", "roof_z_max_m",
        "ground_z_max_m", "created_at", "modified_at", "geometry",
    ]
    buildings = buildings[preferred]

    buildings.to_file(
        gpkg_output,
        layer="extended_buildings_clean",
        driver="GPKG",
        engine="pyogrio",
    )
    paris.to_file(gpkg_output, layer="paris_boundary", driver="GPKG", engine="pyogrio", append=True)
    footprint.to_file(gpkg_output, layer="delivery_footprint", driver="GPKG", engine="pyogrio", append=True)
    support.to_file(gpkg_output, layer="support_footprint_500m", driver="GPKG", engine="pyogrio", append=True)

    summaries = []
    for label, mask in [
        ("all_delivery", pd.Series(True, index=buildings.index)),
        ("paris_core", buildings["in_paris_core"]),
        ("peripheral_support", ~buildings["in_paris_core"]),
        ("computational_support_500m", buildings["in_support_500m"]),
    ]:
        summary = {"subset": label, **summarize_subset(buildings[mask])}
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(table_output, index=False, encoding="utf-8-sig")
    make_qa_map(buildings, paris, footprint, support, figure_output)

    qa = {
        "raw_file": str(raw),
        "raw_md5": digest,
        "raw_feature_count": n_raw,
        "in_service_count_before_geometry_clean": n_in_service,
        "clean_building_count": int(len(buildings)),
        "unique_building_ids": int(buildings["building_id"].nunique()),
        "invalid_geometry_before_repair": invalid_before,
        "invalid_geometry_after_clean": int((~buildings.geometry.is_valid).sum()),
        "empty_geometry_after_clean": int(buildings.geometry.is_empty.sum()),
        "delivery_footprint_area_km2": float(footprint.geometry.area.iloc[0] / 1e6),
        "support_500m_area_km2": float(support.geometry.area.iloc[0] / 1e6),
        "subsets": summaries,
        "decision_scope": "QA for Circuit computational support; not a new city-level building study",
    }
    with qa_output.open("w", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
