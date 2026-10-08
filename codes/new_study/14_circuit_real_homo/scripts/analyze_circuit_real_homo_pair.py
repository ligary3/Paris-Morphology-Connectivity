"""QA and summarize the matched 112.5-degree REAL/HOMO Circuit pair."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\circuit_real_homo_1125_v1')
OUT = ROOT / "analysis"


def read_ascii(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    header: dict[str, float] = {}
    with path.open("r", encoding="ascii") as stream:
        for _ in range(6):
            key, value = stream.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(stream)
    return data, header


def values_for_cells(array: np.ndarray, header: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    xll, yll, size = header["xllcorner"], header["yllcorner"], header["cellsize"]
    nrows = int(header["nrows"])
    cols = np.rint((frame["x_center"].to_numpy() - (xll + size / 2)) / size).astype(int)
    south_rows = np.rint((frame["y_center"].to_numpy() - (yll + size / 2)) / size).astype(int)
    rows = nrows - 1 - south_rows
    return array[rows, cols]


def overlap(a: np.ndarray, b: np.ndarray, top_fraction: float) -> float:
    n = len(a)
    count = max(1, int(np.ceil(n * top_fraction)))
    ia = set(np.argpartition(a, -count)[-count:])
    ib = set(np.argpartition(b, -count)[-count:])
    return len(ia & ib) / count


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(ROOT / "cellwise_input_audit.csv")
    real_grid, real_header = read_ascii(ROOT / "real" / "real_1125_cum_curmap.asc")
    homo_grid, homo_header = read_ascii(ROOT / "homo" / "homo_1125_cum_curmap.asc")
    if real_header != homo_header or real_grid.shape != homo_grid.shape:
        raise ValueError("REAL/HOMO output rasters are not aligned")
    frame["current_real"] = values_for_cells(real_grid, real_header, frame)
    frame["current_homo"] = values_for_cells(homo_grid, homo_header, frame)
    frame["delta_current"] = frame["current_real"] - frame["current_homo"]
    frame["delta_resistance"] = frame["resistance_real"] - frame["resistance_homo"]

    core = frame["analysis_core"].astype(bool) & (frame["current_real"] > -9000) & (frame["current_homo"] > -9000)
    treated = frame["grid_id"].isin(
        frame.loc[np.abs(frame["delta_resistance"]) > 0, "grid_id"]
    ) & core
    c = frame.loc[core].copy()
    tc = frame.loc[treated].copy()
    rho = float(spearmanr(c["current_real"], c["current_homo"]).statistic)
    delta = c["delta_current"].to_numpy()
    scale = float(np.mean(np.abs(c["current_homo"])))
    audit = {
        "status": "CIRCUIT_REAL_HOMO_PASS",
        "n_analysis_core_cells": int(core.sum()),
        "n_treated_core_cells": int(treated.sum()),
        "spearman_current_real_homo": rho,
        "mean_current_real": float(c["current_real"].mean()),
        "mean_current_homo": float(c["current_homo"].mean()),
        "delta_current_mean": float(delta.mean()),
        "delta_current_median": float(np.median(delta)),
        "delta_current_mean_abs": float(np.mean(np.abs(delta))),
        "delta_current_rms": float(np.sqrt(np.mean(delta ** 2))),
        "delta_current_p05": float(np.percentile(delta, 5)),
        "delta_current_p95": float(np.percentile(delta, 95)),
        "relative_mean_abs_delta_vs_homo": float(np.mean(np.abs(delta)) / scale),
        "top_pathway_overlap": {
            "top_10pct": overlap(c["current_real"].to_numpy(), c["current_homo"].to_numpy(), 0.10),
            "top_05pct": overlap(c["current_real"].to_numpy(), c["current_homo"].to_numpy(), 0.05),
            "top_01pct": overlap(c["current_real"].to_numpy(), c["current_homo"].to_numpy(), 0.01),
        },
        "spearman_delta_current_vs_delta_resistance_treated": float(
            spearmanr(tc["delta_current"], tc["delta_resistance"]).statistic
        ),
        "interpretation_scope": "Morphology-induced network contrast only; not yet evidence of atmospheric transport or network added value.",
    }
    (OUT / "circuit_real_homo_diagnostics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    frame.to_csv(OUT / "circuit_real_homo_cellwise.csv", index=False)

    report = f"""# Matched REAL–HOMO Circuit diagnostic

## Decision

**PASS.** The 112.5° pair used an identical 5 km computational domain, electrode geometry,
Q95 scaling and solver configuration. Only the predefined D075 HOMO cells changed resistance.

## Main results inside D075

- REAL–HOMO current-map Spearman: **{rho:.4f}**
- Mean absolute network change / mean absolute HOMO current: **{audit['relative_mean_abs_delta_vs_homo'] * 100:.2f}%**
- Top-10% pathway overlap: **{audit['top_pathway_overlap']['top_10pct'] * 100:.1f}%**
- Top-5% pathway overlap: **{audit['top_pathway_overlap']['top_05pct'] * 100:.1f}%**
- Top-1% pathway overlap: **{audit['top_pathway_overlap']['top_01pct'] * 100:.1f}%**
- Delta-current 5th–95th percentile: **{audit['delta_current_p05']:.6g} to {audit['delta_current_p95']:.6g}**

The result quantifies how real internal morphology reorganizes network current relative to the
matched homogenized Paris. It does not by itself show that the atmosphere follows Circuit.
The next experiment aggregates this network contrast to the exact WRF d03 grid and tests whether
it improves prediction of REAL–HOMO WRF transport beyond local and neighborhood morphology.
"""
    (OUT / "CIRCUIT_REAL_HOMO_REPORT.md").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    x, y = c["x_center"] / 1000.0, c["y_center"] / 1000.0
    lo = float(min(np.percentile(c["current_real"], 1), np.percentile(c["current_homo"], 1)))
    hi = float(max(np.percentile(c["current_real"], 99), np.percentile(c["current_homo"], 99)))
    for ax, column, title in zip(axes[:2], ("current_real", "current_homo"), ("REAL current", "HOMO current")):
        p = ax.scatter(x, y, c=c[column], s=4, cmap="viridis", vmin=lo, vmax=hi, linewidths=0)
        fig.colorbar(p, ax=ax, shrink=0.78)
        ax.set_title(title)
        ax.set_aspect("equal")
    limit = float(np.percentile(np.abs(c["delta_current"]), 99))
    p = axes[2].scatter(x, y, c=c["delta_current"], s=4, cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), linewidths=0)
    fig.colorbar(p, ax=axes[2], shrink=0.78)
    axes[2].set_title("REAL − HOMO current")
    axes[2].set_aspect("equal")
    for ax in axes:
        ax.set_xlabel("Lambert-93 Easting (km)")
        ax.set_ylabel("Northing (km)")
    fig.suptitle("Paris 112.5° inflow: matched Circuit counterfactual")
    fig.savefig(OUT / "circuit_real_homo_maps.png", dpi=240)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
