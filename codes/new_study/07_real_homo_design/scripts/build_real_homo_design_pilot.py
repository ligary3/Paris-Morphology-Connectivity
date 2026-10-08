"""Build a 100 m REAL/HOMO morphology design pilot for Paris D075.

This script defines a counterfactual morphology template before the final WRF
grid exists. It does not aggregate to d03, write NetCDF, or modify raw inputs.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\urban_mask_100m_v1\\paris_urban_support_mask_100m_v1.gpkg')
OUTPUT_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\real_homo_design_v2')

KAPPA = 0.4
CD = 1.2
ALPHA = 4.43
BETA = 1.0
MIN_MEDOID_URBAN_FRACTION = 0.5
FAI_COLUMNS = [f"lambda_f_{int(angle * 10):04d}" for angle in np.arange(0, 360, 22.5)]
ANGLES = np.arange(0, 360, 22.5)


def macdonald_scalar(height: float, plan: float, frontal: float) -> tuple[float, float]:
    plan = float(np.clip(plan, 1e-6, 0.999999))
    frontal = float(max(frontal, 1e-6))
    displacement_ratio = 1.0 + ALPHA ** (-plan) * (plan - 1.0)
    displacement_ratio = float(np.clip(displacement_ratio, 0.0, 0.999999))
    remaining = max(1.0 - displacement_ratio, 1e-9)
    exponent_base = 0.5 * BETA * CD / KAPPA**2 * remaining * frontal
    roughness_ratio = remaining * np.exp(-(exponent_base ** -0.5))
    return height * displacement_ratio, height * roughness_ratio


def weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    return float(np.nansum(values.to_numpy(float) * weights.to_numpy(float)))


def build_template(grid: gpd.GeoDataFrame) -> tuple[dict, gpd.GeoDataFrame, dict]:
    work = grid.copy()
    required = {
        "grid_id", "analysis_area_m2", "urban_fraction_candidate", "urban_support_class",
        "lambda_p", "h_area_weighted", "footprint_union_area_m2", *FAI_COLUMNS,
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work["urban_fraction"] = work["urban_fraction_candidate"].fillna(0.0).clip(0.0, 1.0)
    work["analysis_area"] = work["analysis_area_m2"].fillna(0.0).clip(lower=0.0)
    work["homo_scope"] = (
        work["urban_support_class"].isin(["urban", "mixed"])
        & (work["urban_fraction"] > 0)
        & (work["analysis_area"] > 0)
    )
    scope = work.loc[work["homo_scope"]].copy()
    if scope.empty:
        raise ValueError("No urban/mixed cells available for HOMO template")

    support_area = weighted_sum(scope["urban_fraction"], scope["analysis_area"])
    real_footprint = weighted_sum(scope["lambda_p"].fillna(0), scope["analysis_area"])
    lp_star = real_footprint / support_area

    # Use the same area × plan-fraction footprint proxy in both template
    # estimation and the conservation audit. This keeps the counterfactual
    # building-volume proxy exactly conserved, including partial edge cells.
    height_weights = scope["analysis_area"] * scope["lambda_p"].fillna(0).clip(lower=0)
    valid_height = scope["h_area_weighted"].notna() & (height_weights > 0)
    real_volume = weighted_sum(
        scope.loc[valid_height, "h_area_weighted"], height_weights.loc[valid_height]
    )
    height_footprint = float(height_weights.loc[valid_height].sum())
    h_star = real_volume / height_footprint

    fai_star = {}
    for column in FAI_COLUMNS:
        total = weighted_sum(scope[column].fillna(0), scope["analysis_area"])
        fai_star[column] = total / support_area
    scalar_fai_star = float(np.mean(list(fai_star.values())))
    displacement_star, z0_star = macdonald_scalar(h_star, lp_star, scalar_fai_star)

    # Conditional urban morphology for comparison and medoid selection.
    denom = scope["urban_fraction"].replace(0, np.nan)
    scope["lp_conditional"] = scope["lambda_p"] / denom
    for column in FAI_COLUMNS:
        scope[f"{column}_conditional"] = scope[column] / denom
    scope["scalar_fai_conditional"] = scope[[f"{c}_conditional" for c in FAI_COLUMNS]].mean(axis=1)

    medoid_columns = ["h_area_weighted", "lp_conditional"] + [f"{c}_conditional" for c in FAI_COLUMNS]
    medoid_pool = scope.loc[
        (scope["urban_fraction"] >= MIN_MEDOID_URBAN_FRACTION)
        & (scope["lambda_p"] > 0)
        & scope[medoid_columns].notna().all(axis=1)
    ].copy()
    if medoid_pool.empty:
        raise ValueError("No valid high-support cells for joint medoid")
    matrix = medoid_pool[medoid_columns].to_numpy(float)
    median_profile = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale <= 1e-12] = 1.0
    distances = np.sqrt(np.sum(((matrix - median_profile) / scale) ** 2, axis=1))
    medoid_row = medoid_pool.iloc[int(np.argmin(distances))]

    template = {
        "status": "100M_DESIGN_TEMPLATE_NOT_FINAL_WRF_INPUT",
        "scope_definition": "D075 urban/mixed support cells with urban_fraction_candidate > 0",
        "n_all_cells": int(len(work)),
        "n_homogenized_cells": int(len(scope)),
        "n_medoid_candidate_cells": int(len(medoid_pool)),
        "urban_support_area_m2": support_area,
        "H_star_m": h_star,
        "LP_star_conditional": lp_star,
        "directional_FAI_star_conditional": fai_star,
        "scalar_FAI_star_conditional": scalar_fai_star,
        "Macdonald_displacement_star_m": displacement_star,
        "Macdonald_z0_star_m": z0_star,
        "joint_medoid_reference": {
            "grid_id": str(medoid_row["grid_id"]),
            "urban_fraction": float(medoid_row["urban_fraction"]),
            "H_m": float(medoid_row["h_area_weighted"]),
            "LP_conditional": float(medoid_row["lp_conditional"]),
            "directional_FAI_conditional": {
                column: float(medoid_row[f"{column}_conditional"]) for column in FAI_COLUMNS
            },
            "scalar_FAI_conditional": float(medoid_row["scalar_fai_conditional"]),
            "robust_distance_to_joint_median": float(np.min(distances)),
        },
        "scalar_FAI_rule": "arithmetic mean of the 16 conditional directional FAI values",
        "z0_d_rule": "Macdonald parameters derived from H*, LP*, and scalar FAI*; not independently averaged",
    }

    # Generate a reference table only for cells changed by the counterfactual.
    reference = scope.copy()
    reference["H_REAL_m"] = reference["h_area_weighted"]
    reference["H_HOMO_m"] = h_star
    reference["LP_REAL_full"] = reference["lambda_p"]
    reference["LP_REAL_cond"] = reference["lp_conditional"]
    reference["LP_HOMO_full"] = reference["urban_fraction"] * lp_star
    reference["LP_HOMO_cond"] = lp_star
    for column in FAI_COLUMNS:
        suffix = column.removeprefix("lambda_f_")
        reference[f"FAI_REAL_{suffix}"] = reference[column]
        reference[f"FAI_HOMO_{suffix}"] = reference["urban_fraction"] * fai_star[column]
    reference["FAI_REAL_scalar"] = reference[FAI_COLUMNS].mean(axis=1)
    reference["FAI_HOMO_scalar"] = reference["urban_fraction"] * scalar_fai_star
    reference["D_HOMO_m"] = displacement_star
    reference["Z0_HOMO_m"] = z0_star
    reference["joint_medoid_reference"] = reference["grid_id"].astype(str).eq(str(medoid_row["grid_id"]))

    # Conservation audit within the homogenized scope.
    area = reference["analysis_area"]
    real_volume_a = weighted_sum(reference["LP_REAL_full"].fillna(0) * reference["H_REAL_m"].fillna(0), area)
    homo_volume_a = weighted_sum(reference["LP_HOMO_full"] * reference["H_HOMO_m"], area)
    audit = {
        "building_footprint_proxy": {
            "REAL_m2": weighted_sum(reference["LP_REAL_full"].fillna(0), area),
            "HOMO_m2": weighted_sum(reference["LP_HOMO_full"], area),
        },
        "building_volume_proxy": {"REAL_m3": real_volume_a, "HOMO_m3": homo_volume_a},
        "directional_frontal_area_proxy": {},
    }
    for column in FAI_COLUMNS:
        suffix = column.removeprefix("lambda_f_")
        audit["directional_frontal_area_proxy"][suffix] = {
            "REAL_m2": weighted_sum(reference[f"FAI_REAL_{suffix}"].fillna(0), area),
            "HOMO_m2": weighted_sum(reference[f"FAI_HOMO_{suffix}"], area),
        }
    for group in [audit["building_footprint_proxy"], audit["building_volume_proxy"]]:
        keys = list(group)
        group["relative_difference"] = (group[keys[1]] - group[keys[0]]) / group[keys[0]]
    for values in audit["directional_frontal_area_proxy"].values():
        values["relative_difference"] = (values["HOMO_m2"] - values["REAL_m2"]) / values["REAL_m2"]
    return template, reference, audit


def plot_concept(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#555555"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.4, 4.7, 3.1, 1.35, "REAL 100 m morphology\nspatially varying H, LP, FAI(θ)", "#d9ecf7")
    box(4.4, 4.7, 4.2, 1.35, "Conservation template\nH*, LP*, 16-direction FAI*\nurban fraction remains spatially varying", "#fff2cc")
    box(9.5, 4.7, 3.1, 1.35, "HOMO 100 m reference\nwithin D075 urban support", "#f4cccc")
    ax.annotate("", xy=(4.35, 5.38), xytext=(3.55, 5.38), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(9.45, 5.38), xytext=(8.65, 5.38), arrowprops=dict(arrowstyle="->", lw=1.8))

    box(1.0, 1.2, 3.2, 1.3, "Preserved\nwater, parks, urban support,\nurban fraction, citywide totals", "#d9ead3")
    box(4.9, 1.2, 3.2, 1.3, "Circuit branch\ndirectional FAI*(θ)\n→ resistance", "#eadcf8")
    box(8.8, 1.2, 3.2, 1.3, "WRF branch\nH*, LP*, scalar FAI*\n→ d*, z0*", "#eadcf8")
    ax.annotate("", xy=(2.6, 2.55), xytext=(5.2, 4.65), arrowprops=dict(arrowstyle="->", lw=1.4, color="#666666"))
    ax.annotate("", xy=(6.5, 2.55), xytext=(6.5, 4.65), arrowprops=dict(arrowstyle="->", lw=1.4, color="#666666"))
    ax.annotate("", xy=(10.4, 2.55), xytext=(7.8, 4.65), arrowprops=dict(arrowstyle="->", lw=1.4, color="#666666"))
    ax.text(6.5, 6.65, "REAL/HOMO morphology design (before final geogrid)", ha="center", fontsize=15, weight="bold")
    ax.text(6.5, 0.35, "No 1 km WRF grid, NetCDF, HOMO wrfinput, or ERA5 is produced here.", ha="center", fontsize=9, color="#555555")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distributions(reference: gpd.GeoDataFrame, template: dict, output: Path) -> None:
    medoid = template["joint_medoid_reference"]
    panels = [
        ("H_REAL_m", "Building height H (m)", template["H_star_m"], medoid["H_m"]),
        ("LP_REAL_cond", "Conditional plan fraction LP", template["LP_star_conditional"], medoid["LP_conditional"]),
        ("scalar_fai_conditional", "Conditional scalar FAI", template["scalar_FAI_star_conditional"], medoid["scalar_FAI_conditional"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (column, title, primary, medoid_value) in zip(axes, panels):
        values = reference[column].replace([np.inf, -np.inf], np.nan).dropna()
        upper = values.quantile(0.995)
        ax.hist(values.clip(upper=upper), bins=35, color="#7aa6c2", alpha=0.82, edgecolor="white")
        ax.axvline(primary, color="#c0392b", linewidth=2.2, label="Conservation template")
        ax.axvline(medoid_value, color="#6c3483", linewidth=1.8, linestyle="--", label="Joint medoid")
        ax.set_title(title)
        ax.set_ylabel("100 m urban-support cells")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("REAL distributions and HOMO reference values", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_directional(reference: gpd.GeoDataFrame, template: dict, output: Path) -> None:
    conditional_columns = [f"{c}_conditional" for c in FAI_COLUMNS]
    q25 = reference[conditional_columns].quantile(0.25).to_numpy(float)
    q75 = reference[conditional_columns].quantile(0.75).to_numpy(float)
    primary = np.array([template["directional_FAI_star_conditional"][c] for c in FAI_COLUMNS])
    medoid = np.array([template["joint_medoid_reference"]["directional_FAI_conditional"][c] for c in FAI_COLUMNS])
    angles_closed = np.r_[ANGLES, 360.0]
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.fill_between(angles_closed, np.r_[q25, q25[0]], np.r_[q75, q75[0]], color="#b8d8e8", alpha=0.65, label="REAL cell IQR")
    ax.plot(angles_closed, np.r_[primary, primary[0]], color="#c0392b", linewidth=2.4, marker="o", ms=4, label="Conservation template")
    ax.plot(angles_closed, np.r_[medoid, medoid[0]], color="#6c3483", linewidth=1.7, linestyle="--", marker="s", ms=3.5, label="Joint medoid")
    ax.axvline(112.5, color="#333333", linewidth=1.1, linestyle=":", label="Phase I direction 112.5°")
    ax.set_xticks(ANGLES)
    ax.set_xticklabels([f"{a:g}°" for a in ANGLES], rotation=45, ha="right")
    ax.set_ylabel("Conditional directional FAI")
    ax.set_title("Paris 16-direction HOMO FAI template")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Non-empty output directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT_DIR / "figures"
    figures.mkdir()

    grid = gpd.read_file(INPUT, layer="urban_support_100m")
    template, reference, audit = build_template(grid)
    template["conservation_audit"] = audit

    (OUTPUT_DIR / "homo_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    primary = {
        "template_type": "conservation_template",
        "grid_id": "not_applicable",
        "urban_fraction": 1.0,
        "H_m": template["H_star_m"],
        "LP_conditional": template["LP_star_conditional"],
        "scalar_FAI_conditional": template["scalar_FAI_star_conditional"],
        "displacement_m": template["Macdonald_displacement_star_m"],
        "z0_m": template["Macdonald_z0_star_m"],
    }
    medoid = template["joint_medoid_reference"]
    medoid_d, medoid_z0 = macdonald_scalar(medoid["H_m"], medoid["LP_conditional"], medoid["scalar_FAI_conditional"])
    reference_row = {
        "template_type": "joint_medoid_reference",
        "grid_id": medoid["grid_id"],
        "urban_fraction": medoid["urban_fraction"],
        "H_m": medoid["H_m"],
        "LP_conditional": medoid["LP_conditional"],
        "scalar_FAI_conditional": medoid["scalar_FAI_conditional"],
        "displacement_m": medoid_d,
        "z0_m": medoid_z0,
    }
    for column in FAI_COLUMNS:
        suffix = column.removeprefix("lambda_f_")
        primary[f"FAI_{suffix}_conditional"] = template["directional_FAI_star_conditional"][column]
        reference_row[f"FAI_{suffix}_conditional"] = medoid["directional_FAI_conditional"][column]
    rows.extend([primary, reference_row])
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "homo_template.csv", index=False)

    keep = [
        "grid_id", "urban_support_class", "urban_fraction", "analysis_area",
        "H_REAL_m", "H_HOMO_m", "LP_REAL_full", "LP_REAL_cond", "LP_HOMO_full",
        "LP_HOMO_cond", "FAI_REAL_scalar", "FAI_HOMO_scalar", "D_HOMO_m", "Z0_HOMO_m",
        "joint_medoid_reference", "scalar_fai_conditional",
        *[f"FAI_REAL_{c.removeprefix('lambda_f_')}" for c in FAI_COLUMNS],
        *[f"FAI_HOMO_{c.removeprefix('lambda_f_')}" for c in FAI_COLUMNS],
        *[f"{c}_conditional" for c in FAI_COLUMNS], "geometry",
    ]
    reference[keep].to_file(
        OUTPUT_DIR / "urban_cells_reference.gpkg", layer="real_homo_100m_reference", driver="GPKG"
    )

    plot_concept(figures / "REAL_vs_HOMO_concept.png")
    plot_distributions(reference, template, figures / "parameter_distribution.png")
    plot_directional(reference, template, figures / "directional_FAI_template.png")
    print(json.dumps(template, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
