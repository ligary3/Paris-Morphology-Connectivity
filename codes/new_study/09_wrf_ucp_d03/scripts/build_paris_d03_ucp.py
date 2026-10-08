"""Aggregate Paris 100 m REAL/HOMO morphology to the validated WRF d03 grid.

This stage builds a standalone UCP database. It does not modify geo_em, wrfinput,
LU_INDEX, or WRF executables, and it does not download meteorological forcing.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.colors import ListedColormap
from pyproj import CRS, Transformer
from scipy.io import netcdf_file


ROOT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2')
INPUT_100M = ROOT / "urban_mask_100m_v1" / "paris_urban_support_mask_100m_v1.gpkg"
HOMO_TEMPLATE = ROOT / "real_homo_design_v2" / "homo_template.json"
OUTPUT_DIR = ROOT / "wrf_ucp_d03_v1"
GEO_EM_D03 = OUTPUT_DIR / "geo_em.d03.nc"
RAW_GPKG = (
    ROOT
    / ".."
    / ".."
    / "BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15"
    / "BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15"
    / "BDTOPO"
    / "1_DONNEES_LIVRAISON_2022-06-00173"
    / "BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15"
    / "BDT_3-0_GPKG_LAMB93_D075-ED2022-06-15.gpkg"
).resolve()

FAI_COLUMNS = [f"lambda_f_{int(angle * 10):04d}" for angle in np.arange(0, 360, 22.5)]
ANGLES = np.arange(0, 360, 22.5, dtype=np.float32)
ACTIVATION_THRESHOLD = 0.20
KAPPA = 0.4
CD = 1.2
ALPHA = 4.43
BETA = 1.0
FILL_VALUE = np.float32(-9999.0)


def scalar(value) -> float:
    return float(np.asarray(value).ravel()[0])


def load_geo_em() -> tuple[gpd.GeoDataFrame, dict, np.ndarray, np.ndarray]:
    if not GEO_EM_D03.exists():
        raise FileNotFoundError(GEO_EM_D03)
    with h5py.File(GEO_EM_D03, "r") as source:
        lat = np.asarray(source["XLAT_M"][0], dtype=float)
        lon = np.asarray(source["XLONG_M"][0], dtype=float)
        lu = np.asarray(source["LU_INDEX"][0], dtype=np.int16)
        landmask = np.asarray(source["LANDMASK"][0], dtype=np.int16)
        hgt = np.asarray(source["HGT_M"][0], dtype=float)
        attrs = {
            key: scalar(source.attrs[key])
            for key in ["DX", "DY", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON"]
        }
    if lat.shape != (120, 120) or lon.shape != lat.shape:
        raise ValueError(f"Unexpected d03 shape: {lat.shape}")

    wrf_crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={attrs['TRUELAT1']} +lat_2={attrs['TRUELAT2']} "
        f"+lat_0={attrs['CEN_LAT']} +lon_0={attrs['STAND_LON']} "
        "+a=6370000 +b=6370000 +units=m +no_defs"
    )
    to_wrf = Transformer.from_crs("EPSG:4326", wrf_crs, always_xy=True)
    x_observed, y_observed = to_wrf.transform(lon, lat)
    dx = attrs["DX"]
    dy = attrs["DY"]
    # XLAT/XLONG are stored as float32, so independently inverse-projecting
    # every center introduces metre-scale jitter and thin gaps between cells.
    # Fit the exact regular WRF lattice while retaining the geo_em placement.
    columns = np.arange(lon.shape[1], dtype=float)[None, :]
    rows = np.arange(lat.shape[0], dtype=float)[:, None]
    x0 = float(np.median(x_observed - columns * dx))
    y0 = float(np.median(y_observed - rows * dy))
    x = np.broadcast_to(x0 + columns * dx, lon.shape)
    y = np.broadcast_to(y0 + rows * dy, lat.shape)
    grid_regularity = {
        "float32_inverse_projection_max_x_residual_m": float(np.max(np.abs(x_observed - x))),
        "float32_inverse_projection_max_y_residual_m": float(np.max(np.abs(y_observed - y))),
        "regular_lattice_max_abs_dx_error_m": float(np.max(np.abs(np.diff(x, axis=1) - dx))),
        "regular_lattice_max_abs_dy_error_m": float(np.max(np.abs(np.diff(y, axis=0) - dy))),
    }
    boxes = shapely.box(x.ravel() - dx / 2, y.ravel() - dy / 2, x.ravel() + dx / 2, y.ravel() + dy / 2)
    frame = gpd.GeoDataFrame(
        {
            "wrf_id": [f"d03_i{i+1:03d}_j{j+1:03d}" for j in range(120) for i in range(120)],
            "i": np.tile(np.arange(1, 121, dtype=np.int16), 120),
            "j": np.repeat(np.arange(1, 121, dtype=np.int16), 120),
            "lat": lat.ravel(),
            "lon": lon.ravel(),
            "LU_INDEX": lu.ravel(),
            "LANDMASK": landmask.ravel(),
            "HGT_M": hgt.ravel(),
        },
        geometry=boxes,
        crs=wrf_crs,
    ).to_crs("EPSG:2154")
    frame["wrf_cell_area_m2"] = frame.geometry.area
    attrs["grid_regularity"] = grid_regularity
    attrs["wrf_crs_wkt"] = wrf_crs.to_wkt()
    return frame, attrs, lat, lon


def load_paris_boundary() -> shapely.Geometry:
    departments = pyogrio.read_dataframe(
        RAW_GPKG, layer="departement", columns=["code_insee", "nom_officiel"]
    )
    departments = departments.to_crs("EPSG:2154")
    paris = departments[departments["code_insee"].astype(str) == "75"]
    if len(paris) != 1:
        raise ValueError(f"Expected one D075 polygon, found {len(paris)}")
    return shapely.union_all(paris.geometry.array)


def macdonald(height: pd.Series, plan: pd.Series, frontal: pd.Series) -> tuple[pd.Series, pd.Series]:
    h = height.to_numpy(float)
    lp = plan.clip(1e-6, 0.999999).to_numpy(float)
    lf = frontal.clip(lower=1e-6).to_numpy(float)
    d_ratio = 1.0 + np.power(ALPHA, -lp) * (lp - 1.0)
    d_ratio = np.clip(d_ratio, 0.0, 0.999999)
    remaining = np.maximum(1.0 - d_ratio, 1e-9)
    exponent_base = 0.5 * BETA * CD / KAPPA**2 * remaining * lf
    z0_ratio = remaining * np.exp(-np.power(exponent_base, -0.5))
    return pd.Series(h * d_ratio, index=height.index), pd.Series(h * z0_ratio, index=height.index)


def aggregate(wrf: gpd.GeoDataFrame, paris_geometry, template: dict) -> tuple[gpd.GeoDataFrame, dict]:
    required = {
        "grid_id", "analysis_area_m2", "urban_fraction_candidate", "urban_support_class",
        "lambda_p", "h_area_weighted", *FAI_COLUMNS,
    }
    fine = gpd.read_file(INPUT_100M, layer="urban_support_100m", engine="pyogrio")
    missing = sorted(required.difference(fine.columns))
    if missing:
        raise ValueError(f"Missing 100 m fields: {missing}")
    fine = fine.to_crs("EPSG:2154").copy()
    fine["geometry"] = shapely.intersection(fine.geometry.array, paris_geometry)
    fine = fine[~shapely.is_empty(fine.geometry.array)].copy()
    fine["source_area_exact_m2"] = fine.geometry.area
    fine["scope"] = fine["urban_support_class"].isin(["urban", "mixed"])
    fine["urban_fraction_model"] = np.where(
        fine["scope"], fine["urban_fraction_candidate"].fillna(0).clip(0, 1), 0.0
    )

    candidates = wrf[wrf.intersects(paris_geometry)].copy()
    pieces = gpd.overlay(
        fine[[
            "grid_id", "scope", "urban_fraction_model", "lambda_p", "h_area_weighted",
            *FAI_COLUMNS, "geometry",
        ]],
        candidates[["wrf_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces["piece_area_m2"] = pieces.geometry.area
    pieces = pieces[pieces["piece_area_m2"] > 1e-6].copy()
    pieces["support_area_m2"] = pieces["urban_fraction_model"] * pieces["piece_area_m2"]
    pieces["footprint_area_m2"] = pieces["lambda_p"].fillna(0).clip(lower=0) * pieces["piece_area_m2"]
    pieces.loc[~pieces["scope"], "footprint_area_m2"] = 0.0
    pieces["volume_proxy_m3"] = (
        pieces["footprint_area_m2"] * pieces["h_area_weighted"].fillna(0).clip(lower=0)
    )
    for column in FAI_COLUMNS:
        pieces[f"frontal_{column}_m2"] = pieces[column].fillna(0).clip(lower=0) * pieces["piece_area_m2"]
        pieces.loc[~pieces["scope"], f"frontal_{column}_m2"] = 0.0

    sum_columns = [
        "piece_area_m2", "support_area_m2", "footprint_area_m2", "volume_proxy_m3",
        *[f"frontal_{column}_m2" for column in FAI_COLUMNS],
    ]
    grouped = pieces.groupby("wrf_id", observed=True)[sum_columns].sum()
    result = wrf.join(grouped, on="wrf_id")
    result[sum_columns] = result[sum_columns].fillna(0.0)
    result = result.rename(columns={"piece_area_m2": "D075_area_m2"})
    result["D075_fraction"] = result["D075_area_m2"] / result["wrf_cell_area_m2"]
    result["FRC_URB2D_candidate"] = result["support_area_m2"] / result["wrf_cell_area_m2"]
    result["urban_activation_candidate"] = result["FRC_URB2D_candidate"] >= ACTIVATION_THRESHOLD
    result["wps_urban_lu13"] = result["LU_INDEX"] == 13

    positive = result["support_area_m2"] > 0
    result["REAL_MH_raw_m"] = np.nan
    result["REAL_LP_raw"] = np.nan
    result["REAL_LF_scalar_raw"] = np.nan
    result.loc[positive, "REAL_MH_raw_m"] = (
        result.loc[positive, "volume_proxy_m3"] / result.loc[positive, "footprint_area_m2"].replace(0, np.nan)
    )
    result.loc[positive, "REAL_LP_raw"] = (
        result.loc[positive, "footprint_area_m2"] / result.loc[positive, "support_area_m2"]
    )
    for column in FAI_COLUMNS:
        suffix = column.removeprefix("lambda_f_")
        result[f"REAL_LF_{suffix}_raw"] = np.nan
        result.loc[positive, f"REAL_LF_{suffix}_raw"] = (
            result.loc[positive, f"frontal_{column}_m2"] / result.loc[positive, "support_area_m2"]
        )
    real_dir = [f"REAL_LF_{column.removeprefix('lambda_f_')}_raw" for column in FAI_COLUMNS]
    result["REAL_LF_scalar_raw"] = result[real_dir].mean(axis=1)
    real_d, real_z0 = macdonald(
        result.loc[positive, "REAL_MH_raw_m"],
        result.loc[positive, "REAL_LP_raw"],
        result.loc[positive, "REAL_LF_scalar_raw"],
    )
    result["REAL_ZD_raw_m"] = np.nan
    result["REAL_Z0_raw_m"] = np.nan
    result.loc[positive, "REAL_ZD_raw_m"] = real_d
    result.loc[positive, "REAL_Z0_raw_m"] = real_z0

    result["HOMO_MH_raw_m"] = np.where(positive, template["H_star_m"], np.nan)
    result["HOMO_LP_raw"] = np.where(positive, template["LP_star_conditional"], np.nan)
    for column in FAI_COLUMNS:
        suffix = column.removeprefix("lambda_f_")
        result[f"HOMO_LF_{suffix}_raw"] = np.where(
            positive, template["directional_FAI_star_conditional"][column], np.nan
        )
    homo_dir = [f"HOMO_LF_{column.removeprefix('lambda_f_')}_raw" for column in FAI_COLUMNS]
    result["HOMO_LF_scalar_raw"] = result[homo_dir].mean(axis=1)
    homo_d, homo_z0 = macdonald(
        result.loc[positive, "HOMO_MH_raw_m"],
        result.loc[positive, "HOMO_LP_raw"],
        result.loc[positive, "HOMO_LF_scalar_raw"],
    )
    result["HOMO_ZD_raw_m"] = np.nan
    result["HOMO_Z0_raw_m"] = np.nan
    result.loc[positive, "HOMO_ZD_raw_m"] = homo_d
    result.loc[positive, "HOMO_Z0_raw_m"] = homo_z0

    for prefix in ["REAL", "HOMO"]:
        result[f"{prefix}_MH_effective_m"] = result[f"{prefix}_MH_raw_m"].clip(lower=3.5)
        result[f"{prefix}_LP_effective"] = result[f"{prefix}_LP_raw"].clip(0.05, 1.0)
        result[f"{prefix}_LF_scalar_effective"] = result[f"{prefix}_LF_scalar_raw"].clip(0.05, 1.0)
        result[f"{prefix}_Z0_effective_m"] = result[f"{prefix}_Z0_raw_m"].clip(lower=0.1)

    scope_source = fine[fine["scope"]].copy()
    source_area = scope_source.geometry.area
    source_support = float(np.sum(scope_source["urban_fraction_model"] * source_area))
    source_footprint = float(np.sum(scope_source["lambda_p"].fillna(0) * source_area))
    source_volume = float(np.sum(scope_source["lambda_p"].fillna(0) * scope_source["h_area_weighted"].fillna(0) * source_area))
    aggregated = {
        "support_area_m2": float(result["support_area_m2"].sum()),
        "REAL_footprint_m2": float(result["footprint_area_m2"].sum()),
        "REAL_volume_m3": float(result["volume_proxy_m3"].sum()),
        "HOMO_footprint_m2": float((result["support_area_m2"] * template["LP_star_conditional"]).sum()),
        "HOMO_volume_m3": float((result["support_area_m2"] * template["LP_star_conditional"] * template["H_star_m"]).sum()),
    }
    audit = {
        "source_100m": {
            "support_area_m2": source_support,
            "footprint_m2": source_footprint,
            "volume_m3": source_volume,
        },
        "aggregated_d03": aggregated,
        "relative_mapping_error": {
            "support": (aggregated["support_area_m2"] - source_support) / source_support,
            "REAL_footprint": (aggregated["REAL_footprint_m2"] - source_footprint) / source_footprint,
            "REAL_volume": (aggregated["REAL_volume_m3"] - source_volume) / source_volume,
        },
        "REAL_vs_HOMO_relative_difference": {
            "footprint": (aggregated["HOMO_footprint_m2"] - aggregated["REAL_footprint_m2"]) / aggregated["REAL_footprint_m2"],
            "volume": (aggregated["HOMO_volume_m3"] - aggregated["REAL_volume_m3"]) / aggregated["REAL_volume_m3"],
        },
    }
    return result, audit


def write_netcdf(result: gpd.GeoDataFrame, lat: np.ndarray, lon: np.ndarray, output: Path) -> None:
    ordered = result.sort_values(["j", "i"])
    shape = lat.shape
    with netcdf_file(output, "w", version=2) as dataset:
        dataset.createDimension("south_north", shape[0])
        dataset.createDimension("west_east", shape[1])
        dataset.createDimension("direction", len(ANGLES))
        dataset.title = "Paris d03 standalone REAL/HOMO UCP candidate database"
        dataset.note = "Not a wrfinput file; LU_INDEX and geo_em are not modified"
        dataset.activation_threshold = ACTIVATION_THRESHOLD
        for name, values, dtype in [
            ("XLAT", lat, "f"), ("XLONG", lon, "f"),
            ("LU_INDEX", ordered["LU_INDEX"].to_numpy().reshape(shape), "h"),
            ("D075_FRACTION", ordered["D075_fraction"].to_numpy().reshape(shape), "f"),
            ("FRC_URB2D_CANDIDATE", ordered["FRC_URB2D_candidate"].to_numpy().reshape(shape), "f"),
            ("URBAN_ACTIVATION_CANDIDATE", ordered["urban_activation_candidate"].astype(np.int16).to_numpy().reshape(shape), "h"),
        ]:
            variable = dataset.createVariable(name, dtype, ("south_north", "west_east"))
            variable[:] = values
        direction = dataset.createVariable("DIRECTION_DEG", "f", ("direction",))
        direction[:] = ANGLES
        scalar_fields = {
            "REAL_MH_URB2D": "REAL_MH_effective_m",
            "REAL_LP_URB2D": "REAL_LP_effective",
            "REAL_LF_URB2D_S": "REAL_LF_scalar_effective",
            "REAL_ZD_URB2D": "REAL_ZD_raw_m",
            "REAL_Z0_URB2D": "REAL_Z0_effective_m",
            "HOMO_MH_URB2D": "HOMO_MH_effective_m",
            "HOMO_LP_URB2D": "HOMO_LP_effective",
            "HOMO_LF_URB2D_S": "HOMO_LF_scalar_effective",
            "HOMO_ZD_URB2D": "HOMO_ZD_raw_m",
            "HOMO_Z0_URB2D": "HOMO_Z0_effective_m",
        }
        for output_name, column in scalar_fields.items():
            values = ordered[column].to_numpy(float).reshape(shape)
            values = np.where(np.isfinite(values), values, FILL_VALUE)
            variable = dataset.createVariable(output_name, "f", ("south_north", "west_east"))
            variable._FillValue = FILL_VALUE
            variable[:] = values.astype(np.float32)
        for prefix in ["REAL", "HOMO"]:
            columns = [f"{prefix}_LF_{column.removeprefix('lambda_f_')}_raw" for column in FAI_COLUMNS]
            values = np.stack([ordered[column].to_numpy(float).reshape(shape) for column in columns])
            values = np.where(np.isfinite(values), values, FILL_VALUE)
            variable = dataset.createVariable(
                f"{prefix}_LF_URB2D_DIRECTIONAL", "f", ("direction", "south_north", "west_east")
            )
            variable._FillValue = FILL_VALUE
            variable[:] = values.astype(np.float32)


def make_figure(result: gpd.GeoDataFrame, output: Path) -> None:
    focus = result[result["D075_fraction"] > 0].copy()
    focus["activation_code"] = focus["urban_activation_candidate"].astype(int)
    panels = [
        ("FRC_URB2D_candidate", "Preserved urban fraction", "magma", 0, 1),
        ("REAL_MH_raw_m", "REAL mean height (m)", "viridis", 5, 27),
        ("HOMO_MH_raw_m", "HOMO mean height (m; uniform)", "viridis", 5, 27),
        ("REAL_LF_scalar_raw", "REAL scalar FAI", "plasma", 0, 1),
        ("HOMO_LF_scalar_raw", "HOMO scalar FAI (uniform)", "plasma", 0, 1),
        ("activation_code", "Activation: 0 no, 1 yes", ListedColormap(["#d9d9d9", "#2a9d8f"]), 0, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), constrained_layout=True)
    for axis, (column, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        focus.plot(column=column, ax=axis, cmap=cmap, vmin=vmin, vmax=vmax, legend=True, edgecolor="white", linewidth=0.3)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle("Paris REAL/HOMO morphology on the validated 1 km WRF d03 grid", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_existing = {"geo_em.d03.nc"}
    unexpected = {path.name for path in OUTPUT_DIR.iterdir()}.difference(expected_existing)
    if unexpected:
        raise FileExistsError(f"Output directory contains prior experiment files: {sorted(unexpected)}")
    template = json.loads(HOMO_TEMPLATE.read_text(encoding="utf-8"))
    wrf, attrs, lat, lon = load_geo_em()
    paris_geometry = load_paris_boundary()
    result, audit = aggregate(wrf, paris_geometry, template)

    active = result[result["urban_activation_candidate"]]
    d075 = result[result["D075_fraction"] > 0]
    qa = {
        "status": "D03_UCP_DATABASE_NOT_YET_WRFINPUT",
        "input_geo_em": str(GEO_EM_D03),
        "input_100m": str(INPUT_100M),
        "homo_template": str(HOMO_TEMPLATE),
        "d03_shape": [120, 120],
        "d03_resolution_m": attrs["DX"],
        "grid_regularity": attrs["grid_regularity"],
        "D075_intersecting_cells": int(len(d075)),
        "positive_support_cells": int((result["support_area_m2"] > 0).sum()),
        "activation_threshold": ACTIVATION_THRESHOLD,
        "active_candidate_cells": int(len(active)),
        "WPS_LU13_cells_within_D075": int(d075["wps_urban_lu13"].sum()),
        "activation_vs_WPS_LU13_agreement_within_D075": float(
            (d075["urban_activation_candidate"] == d075["wps_urban_lu13"]).mean()
        ),
        "constraints": {
            "REAL_z0_gt_mh": int((active["REAL_Z0_effective_m"] > active["REAL_MH_effective_m"]).sum()),
            "HOMO_z0_gt_mh": int((active["HOMO_Z0_effective_m"] > active["HOMO_MH_effective_m"]).sum()),
            "FRC_outside_0_1": int(((result["FRC_URB2D_candidate"] < 0) | (result["FRC_URB2D_candidate"] > 1)).sum()),
            "REAL_LP_outside_0_1": int(((active["REAL_LP_raw"] < 0) | (active["REAL_LP_raw"] > 1)).sum()),
        },
        "active_ranges": {
            column: [float(active[column].min()), float(active[column].median()), float(active[column].max())]
            for column in [
                "FRC_URB2D_candidate", "REAL_MH_raw_m", "REAL_LP_raw", "REAL_LF_scalar_raw",
                "REAL_ZD_raw_m", "REAL_Z0_raw_m", "HOMO_MH_raw_m", "HOMO_LP_raw",
                "HOMO_LF_scalar_raw", "HOMO_ZD_raw_m", "HOMO_Z0_raw_m",
            ]
        },
        "conservation_audit": audit,
        "raw_files_modified": False,
        "geo_em_modified": False,
        "wrfinput_written": False,
    }
    if max(abs(value) for value in audit["relative_mapping_error"].values()) > 1e-8:
        raise AssertionError(f"Mapping conservation failed: {audit['relative_mapping_error']}")
    if qa["constraints"]["FRC_outside_0_1"] or qa["constraints"]["REAL_LP_outside_0_1"]:
        raise AssertionError(f"Physical range check failed: {qa['constraints']}")
    if qa["constraints"]["REAL_z0_gt_mh"] or qa["constraints"]["HOMO_z0_gt_mh"]:
        raise AssertionError(f"z0/MH check failed: {qa['constraints']}")

    gpkg = OUTPUT_DIR / "paris_d03_real_homo_ucp.gpkg"
    result.to_file(gpkg, layer="d03_real_homo_ucp", driver="GPKG", engine="pyogrio")
    result.drop(columns="geometry").to_csv(
        OUTPUT_DIR / "paris_d03_real_homo_ucp.csv", index=False, encoding="utf-8-sig"
    )
    write_netcdf(result, lat, lon, OUTPUT_DIR / "paris_d03_real_homo_ucp.nc")
    make_figure(result, OUTPUT_DIR / "paris_d03_real_homo_ucp_QA.png")
    (OUTPUT_DIR / "paris_d03_real_homo_ucp_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
