"""Build 100 m morphology from the targeted-imputed height sensitivity data.

The grid, direction convention, denominators, and morphology equations are
identical to ``build_observed_morphology.py``. This script also compares the
imputed product against the canonical observed-only baseline.
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

from build_observed_morphology import (
    DIRECTIONS,
    HEIGHT_BINS,
    build_resolution,
    direction_suffix,
    make_grid,
    projected_width,
)


DEFAULT_IMPUTED = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed\\paris_buildings_2022_targeted_imputed.gpkg')
DEFAULT_OBSERVED = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\morphology_v2\\paris_morphology_observed_100m.gpkg')
DEFAULT_OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imputed-gpkg", type=Path, default=DEFAULT_IMPUTED)
    parser.add_argument("--observed-morphology", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def comparison_metrics(merged: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    occupied = merged["footprint_union_area_m2"].gt(0)
    core = merged["core_cell"].astype(bool)
    use = merged[occupied & core]
    for column in columns:
        observed = use[f"{column}_observed"]
        imputed = use[f"{column}_imputed"]
        valid = observed.notna() & imputed.notna()
        difference = imputed[valid] - observed[valid]
        denominator = observed[valid].abs().replace(0, np.nan)
        records.append(
            {
                "variable": column,
                "n": int(valid.sum()),
                "observed_median": float(observed[valid].median()),
                "imputed_median": float(imputed[valid].median()),
                "mean_difference": float(difference.mean()),
                "median_absolute_difference": float(difference.abs().median()),
                "p95_absolute_difference": float(difference.abs().quantile(0.95)),
                "median_absolute_percent_difference": float(
                    (difference.abs() / denominator).median() * 100
                ),
                "pearson_r": float(observed[valid].corr(imputed[valid], method="pearson")),
                "spearman_r": float(observed[valid].corr(imputed[valid], method="spearman")),
            }
        )
    return pd.DataFrame(records)


def top_decile_overlap(observed: pd.Series, imputed: pd.Series) -> dict:
    valid = observed.notna() & imputed.notna()
    observed = observed[valid]
    imputed = imputed[valid]
    obs_top = set(observed.index[observed >= observed.quantile(0.90)])
    imp_top = set(imputed.index[imputed >= imputed.quantile(0.90)])
    intersection = len(obs_top & imp_top)
    union = len(obs_top | imp_top)
    return {
        "observed_top_n": len(obs_top),
        "imputed_top_n": len(imp_top),
        "intersection_n": intersection,
        "jaccard": float(intersection / union) if union else np.nan,
        "observed_recall_pct": float(intersection / len(obs_top) * 100) if obs_top else np.nan,
    }


def make_comparison_figure(
    merged: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, path: Path
) -> None:
    display = merged[merged["paris_fraction"].ge(0.5)].copy()
    panels = [
        ("h_area_weighted", "Δ area-weighted height (m)", "RdBu_r"),
        ("h_std", "Δ height SD (m)", "RdBu_r"),
        ("lambda_f_1125", "Δ FAI, inflow 112.5°", "PuOr_r"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=180)
    for ax, (column, title, cmap) in zip(axes, panels):
        delta = display[f"{column}_imputed"] - display[f"{column}_observed"]
        display[f"delta_{column}"] = delta
        vmax = float(delta.abs().quantile(0.99))
        display.plot(
            column=f"delta_{column}",
            ax=ax,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            linewidth=0,
            legend=True,
            missing_kwds={"color": "#eeeeee"},
        )
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.5)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Targeted-imputed minus observed-only morphology", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    imputed_path = args.imputed_gpkg.resolve()
    observed_path = args.observed_morphology.resolve()
    output = args.output_dir.resolve()
    # v2 is the first complete comparison build. The unsuffixed file may exist
    # from an interrupted pre-comparison run and must not be consumed.
    gpkg_output = output / "paris_morphology_100m_imputed_v2.gpkg"
    metrics_output = output / "observed_vs_imputed_morphology_metrics.csv"
    cells_output = output / "observed_vs_imputed_morphology_cells.csv"
    qa_output = output / "morphology_imputed_qa.json"
    figure_output = output / "observed_vs_imputed_morphology_QA.png"
    if any(p.exists() for p in [gpkg_output, metrics_output, cells_output, qa_output]):
        raise FileExistsError("Refusing to overwrite existing imputed morphology outputs")
    output.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(
        imputed_path,
        layer="paris_buildings_targeted_imputed",
        engine="pyogrio",
    )
    boundary = gpd.read_file(imputed_path, layer="paris_boundary", engine="pyogrio")
    observed = gpd.read_file(
        observed_path,
        layer="morphology_100m_observed",
        engine="pyogrio",
    )
    if buildings["height_analysis_m"].isna().any():
        raise ValueError("Imputed building product contains missing analysis heights")

    # Reuse the exact morphology functions by placing the analysis height in the
    # expected column and marking all buildings as height-available.
    buildings["height_m"] = buildings["height_analysis_m"].astype(float)
    buildings["height_valid"] = True
    boundary_geom = boundary.geometry.union_all()
    minx, miny, _, _ = boundary_geom.bounds
    origin_x = math.floor(minx / 1000.0) * 1000.0
    origin_y = math.floor(miny / 1000.0) * 1000.0

    frontal_columns = []
    for theta in DIRECTIONS:
        suffix = direction_suffix(theta)
        frontal_column = f"frontal_area_{suffix}"
        widths = buildings.geometry.map(
            lambda geom, angle=theta: projected_width(geom, angle)
        )
        buildings[frontal_column] = widths * buildings["height_m"]
        frontal_columns.append(frontal_column)

    grid = make_grid(boundary_geom, 100, origin_x, origin_y)
    morphology, resolution_qa = build_resolution(buildings, grid, frontal_columns)
    morphology["morphology_observed_only"] = False
    morphology["morphology_targeted_imputed"] = True
    morphology.to_file(
        gpkg_output,
        layer="morphology_100m_imputed",
        driver="GPKG",
        engine="pyogrio",
    )

    compare_columns = [
        "h_mean",
        "h_area_weighted",
        "h_std",
        *[f"lambda_f_{direction_suffix(theta)}" for theta in DIRECTIONS],
    ]
    keep_common = [
        "grid_id",
        "footprint_union_area_m2",
        "paris_fraction",
        "core_cell",
        *compare_columns,
    ]
    merged = observed[keep_common + ["geometry"]].merge(
        morphology[keep_common].drop(columns=["footprint_union_area_m2", "paris_fraction", "core_cell"]),
        on="grid_id",
        suffixes=("_observed", "_imputed"),
        how="inner",
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=observed.crs)
    metrics = comparison_metrics(merged, compare_columns)
    metrics.to_csv(metrics_output, index=False, encoding="utf-8-sig")

    cell_columns = ["grid_id", "core_cell", "paris_fraction"]
    for column in compare_columns:
        cell_columns.extend([f"{column}_observed", f"{column}_imputed"])
    merged[cell_columns].to_csv(cells_output, index=False, encoding="utf-8-sig")
    make_comparison_figure(merged, boundary, figure_output)

    core_occupied = (
        merged["core_cell"].astype(bool)
        & merged["footprint_union_area_m2"].gt(0)
    )
    overlap = {}
    for theta in DIRECTIONS:
        column = f"lambda_f_{direction_suffix(theta)}"
        overlap[column] = top_decile_overlap(
            merged.loc[core_occupied, f"{column}_observed"],
            merged.loc[core_occupied, f"{column}_imputed"],
        )
    qa = {
        "input_imputed_buildings": str(imputed_path),
        "input_observed_morphology": str(observed_path),
        "output_imputed_morphology": str(gpkg_output),
        "grid_origin": [origin_x, origin_y],
        "resolution_m": 100,
        "directions_degrees_meteorological_inflow": list(DIRECTIONS),
        "height_bins_m": [
            [lower, None if not np.isfinite(upper) else upper, label]
            for lower, upper, label in HEIGHT_BINS
        ],
        "resolution_qa": resolution_qa,
        "top_decile_fai_overlap": overlap,
        "interpretation": "targeted-imputed sensitivity morphology; not observed truth",
    }
    with qa_output.open("w", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
