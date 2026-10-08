"""Permutation check: is observed Circuit NAV stronger than a shuffled extra predictor?"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


WORKSPACE = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
ROOT = WORKSPACE / "new_study/17_transport_metric_candidates/analysis_strict_upwind_pbl"
DIRECTIONS = (45.0, 67.5, 90.0, 112.5)
N_PERMUTATIONS = 100


def load_cross():
    path = WORKSPACE / "new_study/16_multidirection_circuit/scripts/test_cross_direction_added_value.py"
    spec = importlib.util.spec_from_file_location("cross", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tag(direction):
    return f"{int(round(direction * 10)):04d}"


def main():
    cross = load_cross()
    data = np.load(ROOT / "strict_model_inputs.npz")
    rng = np.random.default_rng(20260904)
    direction_results = []
    null_combined = np.zeros(N_PERMUTATIONS)
    observed_changes = []
    for direction in DIRECTIONS:
        dtag = tag(direction)
        baseline = data[f"baseline_{dtag}"]
        network = data[f"network_{dtag}"]
        xy = data[f"xy_{dtag}"]
        y = data[f"y_10_1000_{dtag}"]
        groups = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(xy)
        base = cross.rmse_cv(baseline, y, groups)
        actual = cross.rmse_cv(np.column_stack([baseline, network]), y, groups)
        observed = (actual - base) / base
        observed_changes.append(observed)
        null = np.empty(N_PERMUTATIONS)
        for index in range(N_PERMUTATIONS):
            shuffled = rng.permutation(network[:, 0])[:, None]
            shuffled_rmse = cross.rmse_cv(np.column_stack([baseline, shuffled]), y, groups)
            null[index] = (shuffled_rmse - base) / base
        null_combined += null / len(DIRECTIONS)
        direction_results.append({
            "direction": direction,
            "observed_relative_rmse_change": float(observed),
            "null_median": float(np.median(null)),
            "one_sided_p": float((1 + np.sum(null <= observed)) / (N_PERMUTATIONS + 1)),
        })
    observed_combined = float(np.mean(observed_changes))
    result = {
        "status": "PERMUTATION_COMPLETE",
        "metric": "10-1000 m mean along-flow transport",
        "n_permutations": N_PERMUTATIONS,
        "direction_results": direction_results,
        "combined_observed_mean_relative_rmse_change": observed_combined,
        "combined_null_median": float(np.median(null_combined)),
        "combined_one_sided_p": float((1 + np.sum(null_combined <= observed_combined)) / (N_PERMUTATIONS + 1)),
        "scope": "Simple cell-label permutation diagnostic; spatial block CV remains the primary robustness control.",
    }
    (ROOT / "permutation_nav_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "permutation_nav_null.npz", combined=null_combined)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
