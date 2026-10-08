"""Compare 3 km and 5 km REAL Circuit current maps inside Paris D075."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')
MORPHOLOGY = BASE / "extended_morphology_resistance_1125.gpkg"
PILOT = BASE / "circuit_boundary_pilot"
ORIGIN_X = 643000
ORIGIN_Y = 6857000
CELL_SIZE = 100


def read_ascii(path: Path, value_name: str) -> pd.DataFrame:
    header: dict[str, float] = {}
    with path.open("r", encoding="ascii") as stream:
        for _ in range(6):
            key, value = stream.readline().split()[:2]
            header[key.lower()] = float(value)
    array = np.loadtxt(path, skiprows=6)
    nodata = header["nodata_value"]
    rows, cols = np.where(array != nodata)
    x_center = header["xllcorner"] + (cols + 0.5) * header["cellsize"]
    y_center = header["yllcorner"] + (header["nrows"] - rows - 0.5) * header["cellsize"]
    # Grid IDs refer to the lower-left cell index; centers sit at index + 0.5.
    ix = np.floor((x_center - ORIGIN_X) / CELL_SIZE).astype(int)
    iy = np.floor((y_center - ORIGIN_Y) / CELL_SIZE).astype(int)
    return pd.DataFrame({"ix": ix, "iy": iy, value_name: array[rows, cols]})


def overlap_at_fraction(frame: pd.DataFrame, fraction: float) -> dict:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    top3 = set(frame.nlargest(n, "current_3km").index)
    top5 = set(frame.nlargest(n, "current_5km").index)
    intersection = len(top3 & top5)
    union = len(top3 | top5)
    return {
        "fraction": fraction,
        "n_each": n,
        "n_intersection": intersection,
        "overlap_share": intersection / n,
        "jaccard": intersection / union,
    }


def main() -> None:
    report_path = PILOT / "boundary_pilot_comparison.json"
    output_gpkg = PILOT / "boundary_pilot_current_D075.gpkg"
    figure_path = PILOT / "boundary_pilot_current_D075.png"
    if any(path.exists() for path in [report_path, output_gpkg, figure_path]):
        raise FileExistsError("Refusing to overwrite existing comparison outputs")
    grid = gpd.read_file(
        MORPHOLOGY, layer="extended_morphology_r1_1125", engine="pyogrio"
    )
    paris = gpd.read_file(MORPHOLOGY, layer="paris_boundary", engine="pyogrio")
    current3 = read_ascii(PILOT / "real_3km" / "real_boundary_cum_curmap.asc", "current_3km")
    current5 = read_ascii(PILOT / "real_5km" / "real_boundary_cum_curmap.asc", "current_5km")
    result = grid.merge(current3, on=["ix", "iy"], how="left").merge(
        current5, on=["ix", "iy"], how="left"
    )
    core = result[result["analysis_core"]].dropna(
        subset=["current_3km", "current_5km"]
    ).copy()
    core["rank_pct_3km"] = core["current_3km"].rank(pct=True, method="average")
    core["rank_pct_5km"] = core["current_5km"].rank(pct=True, method="average")
    core["rank_pct_difference"] = core["rank_pct_5km"] - core["rank_pct_3km"]
    rho = float(spearmanr(core["current_3km"], core["current_5km"]).statistic)
    pearson_log = float(
        np.corrcoef(np.log1p(core["current_3km"]), np.log1p(core["current_5km"]))[0, 1]
    )
    overlaps = [overlap_at_fraction(core, fraction) for fraction in (0.10, 0.05, 0.01)]

    core.to_file(
        output_gpkg, layer="D075_current_3km_5km", driver="GPKG", engine="pyogrio"
    )
    paris.to_file(output_gpkg, layer="paris_boundary", driver="GPKG", engine="pyogrio", append=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), dpi=180)
    common_vmax = float(pd.concat([core["current_3km"], core["current_5km"]]).quantile(0.99))
    for ax, column, title in [
        (axes[0, 0], "current_3km", "REAL current: 3 km boundary"),
        (axes[0, 1], "current_5km", "REAL current: 5 km boundary"),
    ]:
        core.plot(column=column, ax=ax, cmap="magma", vmin=0, vmax=common_vmax, linewidth=0, legend=True)
        paris.boundary.plot(ax=ax, color="#31a354", linewidth=0.7)
        ax.set_title(title)
        ax.set_axis_off()
    max_abs = float(core["rank_pct_difference"].abs().quantile(0.99))
    core.plot(
        column="rank_pct_difference", ax=axes[1, 0], cmap="RdBu_r",
        vmin=-max_abs, vmax=max_abs, linewidth=0, legend=True,
    )
    paris.boundary.plot(ax=axes[1, 0], color="#222222", linewidth=0.7)
    axes[1, 0].set_title("Current percentile: 5 km minus 3 km")
    axes[1, 0].set_axis_off()
    axes[1, 1].hexbin(
        core["rank_pct_3km"], core["rank_pct_5km"], gridsize=50,
        mincnt=1, cmap="viridis",
    )
    axes[1, 1].plot([0, 1], [0, 1], color="#d7301f", linewidth=1)
    axes[1, 1].set_xlabel("3 km current percentile")
    axes[1, 1].set_ylabel("5 km current percentile")
    axes[1, 1].set_title(f"D075 rank agreement (Spearman = {rho:.3f})")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_D075_common_cells": int(len(core)),
        "spearman_current": rho,
        "pearson_log1p_current": pearson_log,
        "top_pathway_overlap": overlaps,
        "median_absolute_percentile_change": float(core["rank_pct_difference"].abs().median()),
        "p95_absolute_percentile_change": float(core["rank_pct_difference"].abs().quantile(0.95)),
        "current_median_3km": float(core["current_3km"].median()),
        "current_median_5km": float(core["current_5km"].median()),
        "interpretation_rule": "Boundary is practically stable when rank correlation is high and major top pathways remain spatially overlapping; this is a technical screening, not an inferential test.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
