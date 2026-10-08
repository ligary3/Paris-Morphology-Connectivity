"""Create and validate a targeted height-imputed Paris building product.

The observed BD TOPO height is never changed. Only buildings whose cleaned
``height_valid`` flag is false receive ``height_imputed_m``. The imputer is a
transparent hierarchical donor method using spatial proximity, footprint area,
building use/nature, lightweight status, and floor count when available.

Validation uses pseudo-missing observed buildings under two protocols:

* random stratified holdout (closest to interpolation of scattered gaps), and
* 1 km spatial-block holdout (a stricter spatial-transfer stress test).

All random operations use a fixed seed. The script refuses to overwrite output.
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
from scipy.spatial import cKDTree


DEFAULT_CLEAN = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\paris_buildings_2022_clean_observed.gpkg')
DEFAULT_MORPH = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\morphology_v2\\paris_morphology_observed_100m.gpkg')
DEFAULT_OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\targeted_imputed')
SEED = 20220718
VALIDATION_N = 12000
K_SPATIAL_CANDIDATES = 160
K_FINAL_DONORS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-gpkg", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--observed-morphology", type=Path, default=DEFAULT_MORPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-n", type=int, default=VALIDATION_N)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 1e-12)
    cumulative = np.cumsum(weights)
    cutoff = q * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, cutoff), len(values) - 1)])


def clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("<missing>").astype(str)


def prepare_features(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    points = frame.geometry.representative_point()
    features = pd.DataFrame(index=frame.index)
    features["x"] = points.x
    features["y"] = points.y
    features["log_area"] = np.log1p(frame["footprint_area_m2"].clip(lower=0))
    features["floors"] = frame["n_floors"].where(frame["n_floors"].gt(0))
    features["usage"] = clean_category(frame["usage_primary"])
    features["nature"] = clean_category(frame["nature"])
    features["lightweight"] = frame["is_lightweight"].fillna(False).astype(bool)
    return features


class HierarchicalDonorImputer:
    """Interpretable local donor imputer with documented fallbacks."""

    def __init__(self, donors: pd.DataFrame, heights: pd.Series):
        self.donors = donors.reset_index(drop=True).copy()
        self.heights = heights.reset_index(drop=True).astype(float).to_numpy()
        self.tree = cKDTree(self.donors[["x", "y"]].to_numpy())
        self.global_median = float(np.median(self.heights))
        self.group_medians: dict[tuple, float] = {}
        for keys in [
            ["usage", "nature", "lightweight"],
            ["usage", "lightweight"],
            ["lightweight"],
        ]:
            temp = self.donors[keys].copy()
            temp["height"] = self.heights
            for key, value in temp.groupby(keys, observed=True)["height"].median().items():
                if not isinstance(key, tuple):
                    key = (key,)
                self.group_medians[(tuple(keys), key)] = float(value)

    def baseline_group_median(self, row: pd.Series) -> float:
        for keys in [
            ("usage", "nature", "lightweight"),
            ("usage", "lightweight"),
            ("lightweight",),
        ]:
            key = tuple(row[k] for k in keys)
            found = self.group_medians.get((keys, key))
            if found is not None:
                return found
        return self.global_median

    def predict_one(self, row: pd.Series) -> dict:
        n_query = min(K_SPATIAL_CANDIDATES, len(self.donors))
        distances, indexes = self.tree.query([row["x"], row["y"]], k=n_query)
        distances = np.atleast_1d(distances).astype(float)
        indexes = np.atleast_1d(indexes).astype(int)
        candidates = self.donors.iloc[indexes].copy()
        candidates["distance_m"] = distances
        candidates["height"] = self.heights[indexes]

        exact = (
            candidates["usage"].eq(row["usage"])
            & candidates["nature"].eq(row["nature"])
            & candidates["lightweight"].eq(row["lightweight"])
        )
        usage_light = (
            candidates["usage"].eq(row["usage"])
            & candidates["lightweight"].eq(row["lightweight"])
        )
        same_light = candidates["lightweight"].eq(row["lightweight"])
        if exact.sum() >= 8:
            pool = candidates.loc[exact].copy()
            fallback = "usage_nature_lightweight"
        elif usage_light.sum() >= 8:
            pool = candidates.loc[usage_light].copy()
            fallback = "usage_lightweight"
        elif same_light.sum() >= 8:
            pool = candidates.loc[same_light].copy()
            fallback = "lightweight"
        else:
            pool = candidates
            fallback = "nearest_all"

        area_delta = np.abs(pool["log_area"].to_numpy() - float(row["log_area"]))
        score = pool["distance_m"].to_numpy() / 600.0 + area_delta / 0.70
        floor_used = bool(pd.notna(row["floors"]) and float(row["floors"]) > 0)
        if floor_used:
            donor_floor = pool["floors"].to_numpy(dtype=float)
            floor_known = np.isfinite(donor_floor)
            floor_penalty = np.full(len(pool), 0.85)
            floor_penalty[floor_known] = (
                np.abs(donor_floor[floor_known] - float(row["floors"])) / 2.0
            )
            score += floor_penalty

        selected_order = np.argsort(score)[: min(K_FINAL_DONORS, len(pool))]
        selected = pool.iloc[selected_order]
        selected_score = score[selected_order]
        weights = 1.0 / (0.20 + selected_score)
        values = selected["height"].to_numpy(dtype=float)
        prediction = weighted_quantile(values, weights, 0.50)
        q25 = weighted_quantile(values, weights, 0.25)
        q75 = weighted_quantile(values, weights, 0.75)

        # Guard only against implausible donor tails; retain tall-building values.
        lower = max(1.0, float(np.quantile(self.heights, 0.001)))
        upper = float(np.quantile(self.heights, 0.9995))
        prediction = float(np.clip(prediction, lower, upper))
        return {
            "prediction": prediction,
            "q25": q25,
            "q75": q75,
            "donor_n": int(len(selected)),
            "nearest_donor_m": float(selected["distance_m"].min()),
            "median_donor_m": float(selected["distance_m"].median()),
            "fallback": fallback,
            "floor_constraint_used": floor_used,
            "baseline_group_median": self.baseline_group_median(row),
        }

    def predict(self, targets: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [self.predict_one(row) for _, row in targets.iterrows()],
            index=targets.index,
        )


def size_tier(area: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [area.ge(1000), area.ge(500)],
            ["P1_ge1000", "P2_500_1000"],
            default="P3_lt500",
        ),
        index=area.index,
        dtype="string",
    )


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)
    strata = frame["tier"].astype(str) + "|" + frame["floor_available"].astype(str)
    indexes: list[int] = []
    for _, ids in frame.groupby(strata).groups.items():
        ids = np.asarray(list(ids))
        share = len(ids) / len(frame)
        take = min(len(ids), max(40, int(round(n * share))))
        indexes.extend(rng.choice(ids, size=take, replace=False).tolist())
    if len(indexes) > n:
        indexes = rng.choice(np.asarray(indexes), size=n, replace=False).tolist()
    return pd.Index(indexes)


def validation_metrics(results: pd.DataFrame, protocol: str) -> list[dict]:
    records: list[dict] = []
    groups = [("all", results)]
    groups += [(str(k), v) for k, v in results.groupby("tier", observed=True)]
    groups += [
        ("floor_available", results[results["floor_available"]]),
        ("floor_unavailable", results[~results["floor_available"]]),
    ]
    for group_name, group in groups:
        if group.empty:
            continue
        for method, column in [
            ("hierarchical_donor", "prediction"),
            ("group_median_baseline", "baseline_group_median"),
        ]:
            error = group[column] - group["observed_height"]
            records.append(
                {
                    "protocol": protocol,
                    "group": group_name,
                    "method": method,
                    "n": int(len(group)),
                    "mae_m": float(error.abs().mean()),
                    "median_ae_m": float(error.abs().median()),
                    "rmse_m": float(np.sqrt(np.mean(error**2))),
                    "bias_m": float(error.mean()),
                    "within_3m_pct": float((error.abs() <= 3).mean() * 100),
                    "within_5m_pct": float((error.abs() <= 5).mean() * 100),
                }
            )
    return records


def run_random_validation(
    features: pd.DataFrame,
    heights: pd.Series,
    sample_index: pd.Index,
) -> pd.DataFrame:
    donor_mask = ~features.index.isin(sample_index)
    model = HierarchicalDonorImputer(features.loc[donor_mask], heights.loc[donor_mask])
    predicted = model.predict(features.loc[sample_index])
    predicted["observed_height"] = heights.loc[sample_index]
    return predicted


def run_spatial_block_validation(
    features: pd.DataFrame,
    heights: pd.Series,
    sample_index: pd.Index,
) -> pd.DataFrame:
    sampled = features.loc[sample_index].copy()
    sampled["spatial_fold"] = (
        (np.floor(sampled["x"] / 1000).astype(int) * 3)
        + np.floor(sampled["y"] / 1000).astype(int)
    ) % 5
    output = []
    all_folds = (
        (np.floor(features["x"] / 1000).astype(int) * 3)
        + np.floor(features["y"] / 1000).astype(int)
    ) % 5
    for fold in range(5):
        target_ids = sampled.index[sampled["spatial_fold"].eq(fold)]
        donor_mask = all_folds.ne(fold)
        model = HierarchicalDonorImputer(features.loc[donor_mask], heights.loc[donor_mask])
        predicted = model.predict(features.loc[target_ids])
        predicted["observed_height"] = heights.loc[target_ids]
        predicted["spatial_fold"] = fold
        output.append(predicted)
    return pd.concat(output).sort_index()


def attach_grid_impact(
    buildings: gpd.GeoDataFrame, morphology_path: Path
) -> gpd.GeoDataFrame:
    grid = gpd.read_file(
        morphology_path,
        layer="morphology_100m_observed",
        engine="pyogrio",
    )
    points = buildings[["building_id", "geometry"]].copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        grid[["grid_id", "height_valid_footprint_share", "core_cell", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    joined = joined.drop_duplicates("building_id")
    impact = joined.set_index("building_id")
    buildings["grid_100m_id"] = buildings["building_id"].map(impact["grid_id"])
    buildings["grid_height_valid_footprint_share_observed"] = buildings[
        "building_id"
    ].map(impact["height_valid_footprint_share"])
    buildings["grid_core_cell"] = (
        buildings["building_id"].map(impact["core_cell"]).fillna(False).astype(bool)
    )
    buildings["high_impact_cell"] = (
        buildings["grid_height_valid_footprint_share_observed"].lt(0.80)
        & buildings["grid_core_cell"]
    )
    return buildings


def make_figures(
    validation: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    sample = validation.sample(min(6000, len(validation)), random_state=SEED)
    axes[0].hexbin(
        sample["observed_height"],
        sample["prediction"],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    limit = float(np.quantile(sample[["observed_height", "prediction"]].to_numpy(), 0.995))
    axes[0].plot([0, limit], [0, limit], color="#d62728", lw=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Observed height (m)")
    axes[0].set_ylabel("Pseudo-imputed height (m)")
    axes[0].set_title("Random pseudo-missing validation")

    missing = buildings[~buildings["height_valid"].astype(bool)].copy()
    missing.plot(
        column="height_imputed_m",
        ax=axes[1],
        markersize=np.clip(missing["footprint_area_m2"] / 50, 0.4, 12),
        cmap="magma",
        vmin=0,
        vmax=float(missing["height_imputed_m"].quantile(0.99)),
        legend=True,
    )
    boundary.boundary.plot(ax=axes[1], color="#202020", linewidth=0.5)
    axes[1].set_axis_off()
    axes[1].set_title("Targeted imputed heights (missing objects only)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clean_path = args.clean_gpkg.resolve()
    morphology_path = args.observed_morphology.resolve()
    output = args.output_dir.resolve()
    gpkg_output = output / "paris_buildings_2022_targeted_imputed.gpkg"
    validation_output = output / "pseudo_missing_predictions.csv"
    metrics_output = output / "imputation_validation_metrics.csv"
    summary_output = output / "imputation_qa.json"
    figure_output = output / "imputation_validation_QA.png"
    if any(p.exists() for p in [gpkg_output, validation_output, metrics_output, summary_output]):
        raise FileExistsError("Refusing to overwrite an existing imputation run")
    output.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(clean_path, layer="paris_buildings_clean", engine="pyogrio")
    boundary = gpd.read_file(clean_path, layer="paris_boundary", engine="pyogrio")
    if buildings.crs is None or buildings.crs.to_epsg() != 2154:
        raise ValueError(f"Unexpected CRS: {buildings.crs}")
    buildings = attach_grid_impact(buildings, morphology_path)
    features = prepare_features(buildings)
    valid_mask = buildings["height_valid"].astype(bool)
    observed_features = features.loc[valid_mask].copy()
    observed_heights = buildings.loc[valid_mask, "height_m"].astype(float)
    observed_features["tier"] = size_tier(
        buildings.loc[valid_mask, "footprint_area_m2"]
    )
    observed_features["floor_available"] = observed_features["floors"].notna()

    sample_index = stratified_sample(observed_features, args.validation_n, args.seed)
    random_validation = run_random_validation(
        observed_features.drop(columns=["tier", "floor_available"]),
        observed_heights,
        sample_index,
    )
    spatial_validation = run_spatial_block_validation(
        observed_features.drop(columns=["tier", "floor_available"]),
        observed_heights,
        sample_index,
    )
    for validation in [random_validation, spatial_validation]:
        validation["tier"] = observed_features.loc[validation.index, "tier"]
        validation["floor_available"] = observed_features.loc[
            validation.index, "floor_available"
        ]
        validation["building_id"] = buildings.loc[validation.index, "building_id"]
        validation["footprint_area_m2"] = buildings.loc[
            validation.index, "footprint_area_m2"
        ]
    random_validation["protocol"] = "random_stratified"
    spatial_validation["protocol"] = "spatial_block_1km"
    all_validation = pd.concat([random_validation, spatial_validation], ignore_index=True)
    all_validation.to_csv(validation_output, index=False, encoding="utf-8-sig")

    metrics = validation_metrics(random_validation, "random_stratified")
    metrics += validation_metrics(spatial_validation, "spatial_block_1km")
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_output, index=False, encoding="utf-8-sig")

    final_model = HierarchicalDonorImputer(
        features.loc[valid_mask], buildings.loc[valid_mask, "height_m"]
    )
    missing_mask = ~valid_mask
    imputed = final_model.predict(features.loc[missing_mask])
    buildings["height_imputed_m"] = np.nan
    buildings.loc[missing_mask, "height_imputed_m"] = imputed["prediction"]
    buildings["height_analysis_m"] = buildings["height_m"]
    buildings.loc[missing_mask, "height_analysis_m"] = imputed["prediction"]
    buildings["height_analysis_source"] = "observed_bdtopo"
    buildings.loc[missing_mask, "height_analysis_source"] = np.where(
        imputed["floor_constraint_used"],
        "imputed_local_type_area_floor_donor",
        "imputed_local_type_area_donor",
    )
    buildings["imputation_priority"] = "observed"
    buildings.loc[missing_mask, "imputation_priority"] = size_tier(
        buildings.loc[missing_mask, "footprint_area_m2"]
    )
    buildings["imputation_q25_m"] = np.nan
    buildings["imputation_q75_m"] = np.nan
    buildings["imputation_iqr_m"] = np.nan
    buildings["imputation_donor_n"] = pd.Series(pd.NA, index=buildings.index, dtype="Int32")
    buildings["imputation_nearest_donor_m"] = np.nan
    buildings["imputation_median_donor_m"] = np.nan
    buildings["imputation_fallback"] = pd.Series(pd.NA, index=buildings.index, dtype="string")
    buildings["floor_constraint_used"] = False
    for target, source in [
        ("imputation_q25_m", "q25"),
        ("imputation_q75_m", "q75"),
        ("imputation_nearest_donor_m", "nearest_donor_m"),
        ("imputation_median_donor_m", "median_donor_m"),
        ("imputation_fallback", "fallback"),
        ("floor_constraint_used", "floor_constraint_used"),
    ]:
        buildings.loc[missing_mask, target] = imputed[source]
    buildings.loc[missing_mask, "imputation_donor_n"] = imputed["donor_n"].astype("Int32")
    buildings.loc[missing_mask, "imputation_iqr_m"] = imputed["q75"] - imputed["q25"]
    buildings["review_required"] = missing_mask & (
        buildings["footprint_area_m2"].ge(1000)
        | buildings["high_impact_cell"]
        | buildings["imputation_iqr_m"].gt(10)
        | buildings["imputation_median_donor_m"].gt(1500)
    )

    if buildings.loc[missing_mask, "height_analysis_m"].isna().any():
        raise ValueError("Some missing-height buildings were not imputed")
    if (buildings.loc[valid_mask, "height_analysis_m"] != buildings.loc[valid_mask, "height_m"]).any():
        raise ValueError("Observed heights were unexpectedly changed")

    buildings.to_file(
        gpkg_output,
        layer="paris_buildings_targeted_imputed",
        driver="GPKG",
        engine="pyogrio",
    )
    boundary.to_file(
        gpkg_output,
        layer="paris_boundary",
        driver="GPKG",
        engine="pyogrio",
        append=True,
    )
    make_figures(random_validation, buildings, boundary, figure_output)

    random_all = metrics_frame[
        metrics_frame["protocol"].eq("random_stratified")
        & metrics_frame["group"].eq("all")
        & metrics_frame["method"].eq("hierarchical_donor")
    ].iloc[0]
    spatial_all = metrics_frame[
        metrics_frame["protocol"].eq("spatial_block_1km")
        & metrics_frame["group"].eq("all")
        & metrics_frame["method"].eq("hierarchical_donor")
    ].iloc[0]
    summary = {
        "input_clean_gpkg": str(clean_path),
        "input_observed_morphology": str(morphology_path),
        "seed": args.seed,
        "validation_sample_n": int(len(sample_index)),
        "n_buildings": int(len(buildings)),
        "n_observed_unchanged": int(valid_mask.sum()),
        "n_imputed": int(missing_mask.sum()),
        "priority_counts_missing": {
            str(k): int(v)
            for k, v in buildings.loc[missing_mask, "imputation_priority"].value_counts().items()
        },
        "high_impact_cell_count_missing": int(
            buildings.loc[missing_mask, "high_impact_cell"].sum()
        ),
        "review_required_count_missing": int(
            buildings.loc[missing_mask, "review_required"].sum()
        ),
        "floor_constraint_count_missing": int(
            buildings.loc[missing_mask, "floor_constraint_used"].sum()
        ),
        "imputed_height_summary_m": {
            str(k): float(v)
            for k, v in buildings.loc[missing_mask, "height_imputed_m"]
            .describe(percentiles=[0.05, 0.50, 0.95, 0.99])
            .items()
        },
        "validation_random_all": random_all.to_dict(),
        "validation_spatial_block_all": spatial_all.to_dict(),
        "method": {
            "type": "hierarchical weighted-median donor",
            "features": [
                "representative-point distance",
                "log footprint area",
                "usage_primary",
                "nature",
                "is_lightweight",
                "positive n_floors when available",
            ],
            "object_priority": "P1 >=1000 m2; P2 500-1000 m2; P3 <500 m2",
            "review_required": "P1 or core-cell observed height footprint coverage <80%, or donor IQR >10 m, or median donor distance >1500 m",
            "interpretation": "sensitivity product; imputed values are not observed truth",
        },
    }
    with summary_output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
