"""Create a practical height-complete building layer for the Circuit support domain.

Paris-core imputations are copied from the canonical D075 product. Remaining
missing peripheral heights use a transparent hierarchical grouped-median rule.
The goal is to prevent false low-resistance holes, not to develop a new height
prediction method.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_buildings_clean_observed.gpkg')
DEFAULT_PARIS = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed\\paris_buildings_2022_targeted_imputed.gpkg')
DEFAULT_OUTPUT_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1')

AREA_BINS = [0, 20, 100, 500, 1000, 5000, np.inf]
AREA_LABELS = ["lt20", "20_100", "100_500", "500_1000", "1000_5000", "ge5000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gpkg", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paris-gpkg", type=Path, default=DEFAULT_PARIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def add_group_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["area_bin"] = pd.cut(
        out["footprint_area_m2"], AREA_BINS, labels=AREA_LABELS, include_lowest=True
    ).astype("string").fillna("unknown")
    floors = pd.to_numeric(out["n_floors"], errors="coerce")
    out["floor_bin"] = np.select(
        [floors.le(0) | floors.isna(), floors.le(2), floors.le(5), floors.le(10)],
        ["unknown", "1_2", "3_5", "6_10"],
        default="ge11",
    )
    for column in ["usage_primary", "nature", "is_lightweight"]:
        out[column] = out[column].astype("string").fillna("unknown")
    return out


def grouped_lookup(
    target: pd.DataFrame,
    donors: pd.DataFrame,
    group_columns: list[str],
    minimum_donors: int,
) -> tuple[pd.Series, pd.Series]:
    stats = (
        donors.groupby(group_columns, dropna=False)["height_m"]
        .agg(["median", "count"])
        .reset_index()
    )
    stats.loc[stats["count"] < minimum_donors, "median"] = np.nan
    keys = pd.MultiIndex.from_frame(target[group_columns])
    table = stats.set_index(group_columns)
    medians = table["median"].reindex(keys)
    counts = table["count"].reindex(keys)
    return (
        pd.Series(medians.to_numpy(), index=target.index, dtype=float),
        pd.Series(counts.to_numpy(), index=target.index, dtype=float),
    )


def apply_hierarchy(
    target: pd.DataFrame, donors: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series]:
    result = pd.Series(np.nan, index=target.index, dtype=float)
    method = pd.Series(pd.NA, index=target.index, dtype="string")
    donor_n = pd.Series(np.nan, index=target.index, dtype=float)
    levels = [
        (["usage_primary", "is_lightweight", "area_bin", "floor_bin"], 20, "usage_light_area_floor"),
        (["usage_primary", "is_lightweight", "area_bin"], 30, "usage_light_area"),
        (["nature", "area_bin"], 30, "nature_area"),
        (["area_bin"], 30, "area"),
    ]
    for columns, minimum, label in levels:
        pending = result.isna()
        if not pending.any():
            break
        estimates, counts = grouped_lookup(target.loc[pending], donors, columns, minimum)
        usable = estimates.notna()
        chosen = estimates.index[usable]
        result.loc[chosen] = estimates.loc[chosen]
        method.loc[chosen] = label
        donor_n.loc[chosen] = counts.loc[chosen]
    pending = result.isna()
    if pending.any():
        result.loc[pending] = float(donors["height_m"].median())
        method.loc[pending] = "overall_median"
        donor_n.loc[pending] = float(len(donors))
    return result, method, donor_n


def validation_summary(frame: pd.DataFrame) -> dict:
    eligible = frame[(~frame["in_paris_core"]) & frame["height_valid"]].copy()
    sample_n = min(10000, len(eligible) // 5)
    validation = eligible.sample(sample_n, random_state=20220718)
    donors = frame[frame["height_valid"] & ~frame.index.isin(validation.index)].copy()
    predicted, _, _ = apply_hierarchy(validation, donors)
    observed = validation["height_m"].astype(float)
    error = predicted - observed
    return {
        "n_holdout": int(sample_n),
        "observed_mean_m": float(observed.mean()),
        "predicted_mean_m": float(predicted.mean()),
        "mean_bias_m": float(error.mean()),
        "mae_m": float(error.abs().mean()),
        "median_absolute_error_m": float(error.abs().median()),
        "spearman": float(observed.corr(predicted, method="spearman")),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_gpkg = output_dir / "extended_buildings_targeted_imputed.gpkg"
    report_json = output_dir / "extended_height_imputation_qa.json"
    method_csv = output_dir / "extended_height_imputation_methods.csv"
    if any(path.exists() for path in [output_gpkg, report_json, method_csv]):
        raise FileExistsError("Refusing to overwrite an existing imputation run")

    buildings = gpd.read_file(
        args.input_gpkg,
        layer="extended_buildings_clean",
        engine="pyogrio",
    )
    paris = gpd.read_file(
        args.paris_gpkg,
        layer="paris_buildings_targeted_imputed",
        columns=["building_id", "height_analysis_m", "height_analysis_source"],
        engine="pyogrio",
    ).drop(columns="geometry")
    paris = paris.drop_duplicates("building_id").set_index("building_id")
    buildings = add_group_fields(buildings)
    buildings["height_analysis_m"] = buildings["height_m"].astype(float)
    buildings["height_analysis_source"] = np.where(
        buildings["height_valid"], "observed_bdtopo", pd.NA
    )
    buildings["imputation_method"] = pd.NA
    buildings["imputation_donor_n"] = np.nan

    # Preserve every canonical Paris value that can be matched by building ID.
    paris_height = buildings["building_id"].map(paris["height_analysis_m"])
    paris_source = buildings["building_id"].map(paris["height_analysis_source"])
    use_paris = buildings["in_paris_core"] & paris_height.notna()
    buildings.loc[use_paris, "height_analysis_m"] = paris_height.loc[use_paris]
    paris_was_imputed = use_paris & ~buildings["height_valid"]
    buildings.loc[paris_was_imputed, "height_analysis_source"] = "paris_canonical_imputed"
    buildings.loc[paris_was_imputed, "imputation_method"] = paris_source.loc[paris_was_imputed]

    donors = buildings[buildings["height_valid"]].copy()
    target_mask = buildings["height_analysis_m"].isna()
    estimates, methods, donor_counts = apply_hierarchy(buildings.loc[target_mask], donors)
    buildings.loc[target_mask, "height_analysis_m"] = estimates
    buildings.loc[target_mask, "height_analysis_source"] = "peripheral_group_imputed"
    buildings.loc[target_mask, "imputation_method"] = methods
    buildings.loc[target_mask, "imputation_donor_n"] = donor_counts
    buildings["height_imputed"] = ~buildings["height_valid"]
    buildings["review_large_missing"] = (~buildings["height_valid"]) & buildings[
        "footprint_area_m2"
    ].ge(1000)

    if buildings["height_analysis_m"].isna().any():
        raise ValueError("Height-complete product still contains missing analysis heights")
    if not np.allclose(
        buildings.loc[buildings["height_valid"], "height_analysis_m"],
        buildings.loc[buildings["height_valid"], "height_m"],
    ):
        raise ValueError("Observed heights changed during imputation")

    validation = validation_summary(buildings)
    method_table = (
        buildings.loc[buildings["height_imputed"]]
        .groupby(["domain_role", "height_analysis_source", "imputation_method"], dropna=False)
        .agg(
            n_buildings=("building_id", "size"),
            footprint_area_m2=("footprint_area_m2", "sum"),
            median_imputed_height_m=("height_analysis_m", "median"),
        )
        .reset_index()
    )
    method_table.to_csv(method_csv, index=False, encoding="utf-8-sig")

    # Temporary grouping fields are not part of the canonical deliverable.
    buildings = buildings.drop(columns=["area_bin", "floor_bin"])
    buildings.to_file(
        output_gpkg,
        layer="extended_buildings_imputed",
        driver="GPKG",
        engine="pyogrio",
    )
    for layer in ["paris_boundary", "delivery_footprint", "support_footprint_500m"]:
        feature = gpd.read_file(args.input_gpkg, layer=layer, engine="pyogrio")
        feature.to_file(output_gpkg, layer=layer, driver="GPKG", engine="pyogrio", append=True)

    report = {
        "input_buildings": int(len(buildings)),
        "observed_heights_unchanged": True,
        "paris_canonical_imputed_n": int(paris_was_imputed.sum()),
        "peripheral_or_unmatched_imputed_n": int(target_mask.sum()),
        "large_missing_ge_1000m2_flagged_n": int(buildings["review_large_missing"].sum()),
        "analysis_height_min_m": float(buildings["height_analysis_m"].min()),
        "analysis_height_median_m": float(buildings["height_analysis_m"].median()),
        "analysis_height_max_m": float(buildings["height_analysis_m"].max()),
        "validation": validation,
        "purpose": "Prevent missing-height buildings from creating false low-resistance paths; not a height-prediction contribution.",
    }
    with report_json.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
