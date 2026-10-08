"""Assign Cerema 2022 LCZ polygons to the existing Paris 100 m analysis grid."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio


ROOT = _release_path('E:\\lunwen4\\2')
LCZ_SHP = ROOT / "LCZ" / "Paris" / "Cerema_2022" / "extracted_vector" / "LCZ_SPOT_2022_Paris.shp"
GRID_GPKG = ROOT / "建筑形态" / "Paris" / "processed" / "phase1a_v2" / "extended_domain_v1" / "extended_morphology_multidirection_v1.gpkg"
CORE_GPKG = ROOT / "建筑形态" / "Paris" / "processed" / "phase1a_v2" / "targeted_imputed" / "paris_morphology_100m_imputed_v2.gpkg"
OUT = ROOT / "LCZ" / "Paris" / "Cerema_2022" / "processed_100m"
OUT_GPKG = OUT / "paris_cerema_lcz_100m.gpkg"

LCZ_LABELS = {
    1: "Compact high-rise", 2: "Compact mid-rise", 3: "Compact low-rise",
    4: "Open high-rise", 5: "Open mid-rise", 6: "Open low-rise",
    7: "Lightweight low-rise", 8: "Large low-rise", 9: "Sparsely built",
    10: "Heavy industry", 11: "Dense trees", 12: "Scattered trees",
    13: "Bush/scrub", 14: "Low plants", 15: "Bare rock/paved",
    16: "Bare soil/sand", 17: "Water",
}
LCZ_COLORS = {
    1: "#910613", 2: "#D9081C", 3: "#FF0A22", 4: "#C54F1E", 5: "#FF6628",
    6: "#FF985E", 7: "#FDED3F", 8: "#BBBBBB", 9: "#FFCBAB", 10: "#565656",
    11: "#006A18", 12: "#00A926", 13: "#628432", 14: "#B5DA7F", 15: "#000000",
    16: "#FCF7B1", 17: "#656BFA",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pyogrio.read_dataframe(GRID_GPKG, layer="extended_multidirection")
    if str(grid.crs).upper() != "EPSG:2154":
        raise ValueError(f"Unexpected grid CRS: {grid.crs}")

    bounds = tuple(grid.total_bounds)
    lcz = pyogrio.read_dataframe(
        LCZ_SHP,
        columns=["identifier", "hre", "are", "bur", "ror", "bsr", "war", "ver", "vhr", "lcz", "lcz_int"],
        bbox=bounds,
    )
    points = gpd.GeoDataFrame(
        grid[["grid_id", "x_center", "y_center", "analysis_core", "paris_fraction"]].copy(),
        geometry=gpd.points_from_xy(grid.x_center, grid.y_center),
        crs=grid.crs,
    )
    joined = gpd.sjoin(points, lcz, how="left", predicate="within")
    duplicate_count = int(joined.index.duplicated(keep=False).sum())
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()

    output = grid[["grid_id", "ix", "iy", "x_center", "y_center", "analysis_core", "paris_fraction", "geometry"]].copy()
    fields = ["identifier", "hre", "are", "bur", "ror", "bsr", "war", "ver", "vhr", "lcz", "lcz_int"]
    for field in fields:
        output[field] = joined[field].to_numpy()
    output["lcz_name"] = output.lcz_int.map(LCZ_LABELS)

    core = pyogrio.read_dataframe(CORE_GPKG, layer="morphology_100m_imputed")
    fai_columns = [column for column in core.columns if column.startswith("lambda_f_")]
    core["fai_scalar_16dir"] = core[fai_columns].mean(axis=1)
    morphology = core[["grid_id", "lambda_p", "h_area_weighted", "fai_scalar_16dir"]]
    output = output.merge(morphology, on="grid_id", how="left")

    pyogrio.write_dataframe(output, OUT_GPKG, layer="cerema_lcz_100m", driver="GPKG")

    assigned = output.lcz_int.notna()
    core_mask = output.analysis_core.astype(bool)
    built_lcz = output.lcz_int.between(1, 10)
    summary_table = (
        output.loc[core_mask & assigned]
        .groupby(["lcz_int", "lcz_name"], dropna=False)
        .agg(
            cells=("grid_id", "size"),
            cerema_hre_median=("hre", "median"),
            cerema_bur_median_pct=("bur", "median"),
            bdtopo_h_median=("h_area_weighted", "median"),
            bdtopo_lambda_p_median=("lambda_p", "median"),
            bdtopo_fai_scalar_median=("fai_scalar_16dir", "median"),
        )
        .reset_index()
    )
    summary_table.to_csv(OUT / "lcz_morphology_summary_d075.csv", index=False)

    diagnostic = {
        "status": "LCZ_100M_PREPARATION_PASS",
        "source_features_in_bbox": int(len(lcz)),
        "grid_cells": int(len(output)),
        "assigned_cells": int(assigned.sum()),
        "assigned_fraction": float(assigned.mean()),
        "analysis_core_cells": int(core_mask.sum()),
        "analysis_core_assigned": int((core_mask & assigned).sum()),
        "analysis_core_assigned_fraction": float(assigned[core_mask].mean()),
        "analysis_core_built_lcz_cells": int((core_mask & built_lcz).sum()),
        "duplicate_point_matches": duplicate_count,
        "method": "LCZ polygon containing each pre-existing 100 m grid-cell centroid; no extraction of 10.1 GB 1.5 m raster.",
        "crs": str(output.crs),
        "lcz_classes_core": {str(int(k)): int(v) for k, v in output.loc[core_mask & assigned, "lcz_int"].value_counts().sort_index().items()},
    }
    (OUT / "lcz_100m_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")

    core_plot = output[core_mask & assigned].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for klass, frame in core_plot.groupby("lcz_int"):
        frame.plot(ax=axes[0], color=LCZ_COLORS[int(klass)], linewidth=0)
    axes[0].set_title("Cerema 2022 LCZ on the 100 m D075 grid")
    axes[0].set_axis_off()

    built = core_plot[core_plot.lcz_int.between(1, 10) & core_plot.fai_scalar_16dir.notna()].copy()
    order = sorted(int(value) for value in built.lcz_int.unique())
    data = [built.loc[built.lcz_int == klass, "fai_scalar_16dir"].to_numpy() for klass in order]
    axes[1].boxplot(data, labels=[str(value) for value in order], showfliers=False)
    axes[1].set_xlabel("LCZ class")
    axes[1].set_ylabel("BD TOPO scalar FAI (16-direction mean)")
    axes[1].set_title("Within-class 3-D morphology variability")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("LCZ benchmark preparation: categorical classes versus detailed 3-D morphology")
    fig.savefig(OUT / "lcz_100m_and_fai_variability.png", dpi=240)
    plt.close(fig)
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
