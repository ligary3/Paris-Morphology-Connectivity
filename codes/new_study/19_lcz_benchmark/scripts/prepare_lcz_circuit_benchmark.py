"""Prepare direction-matched LCZ categorical Circuit benchmark cases for Paris."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = _release_path('E:\\lunwen4\\2')
MORPH_MULTI = ROOT / "建筑形态" / "Paris" / "processed" / "phase1a_v2" / "extended_domain_v1" / "extended_morphology_multidirection_v1.gpkg"
MORPH_1125 = ROOT / "建筑形态" / "Paris" / "processed" / "phase1a_v2" / "extended_domain_v1" / "extended_morphology_resistance_1125.gpkg"
REFERENCE = ROOT / "建筑形态" / "Paris" / "processed" / "phase1a_v2" / "real_homo_design_v2" / "urban_cells_reference.gpkg"
LCZ = ROOT / "LCZ" / "Paris" / "Cerema_2022" / "processed_100m" / "paris_cerema_lcz_100m.gpkg"
OUTPUT = ROOT / "LCZ" / "Paris" / "Cerema_2022" / "circuit_benchmark_v1"
DIRECTIONS = (22.5, 45.0, 67.5, 90.0, 112.5)
Q95 = 0.976565954884937
NODATA = -9999.0


def suffix(direction: float) -> str:
    return f"{int(round(direction * 10)):04d}"


def projection_bounds(geometry, bearing: float) -> tuple[float, float]:
    theta = math.radians(bearing)
    dx, dy = math.sin(theta), math.cos(theta)
    parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    coordinates = np.vstack([np.asarray(part.exterior.coords) for part in parts])
    projected = coordinates[:, 0] * dx + coordinates[:, 1] * dy
    return float(projected.min()), float(projected.max())


def grid_array(frame: pd.DataFrame, column: str):
    ix0, ix1 = int(frame.ix.min()), int(frame.ix.max())
    iy0, iy1 = int(frame.iy.min()), int(frame.iy.max())
    result = np.full((iy1 - iy0 + 1, ix1 - ix0 + 1), NODATA)
    result[iy1 - frame.iy.to_numpy(int), frame.ix.to_numpy(int) - ix0] = frame[column].to_numpy(float)
    return result, float(frame.x_center.min() - 50), float(frame.y_center.min() - 50)


def write_ascii(path: Path, values: np.ndarray, xll: float, yll: float) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ncols {values.shape[1]}\nnrows {values.shape[0]}\nxllcorner {xll:.3f}\nyllcorner {yll:.3f}\ncellsize 100\nNODATA_value -9999\n")
        np.savetxt(stream, values, fmt="%.9g")


def config(habitat: Path, focal: Path, output: Path) -> str:
    return f"""[Circuitscape mode]
data_type = raster
scenario = pairwise
[Habitat raster or graph]
habitat_file = {habitat}
habitat_map_is_resistances = True
[Connection scheme for raster habitat data]
connect_four_neighbors_only = False
connect_using_avg_resistances = True
[Short circuit regions (aka polygons)]
use_polygons = False
polygon_file = None
[Options for pairwise and one-to-all and all-to-one modes]
point_file = {focal}
use_included_pairs = False
included_pairs_file = None
[Calculation options]
solver = cg+amg
[Output options]
output_file = {output}
write_cur_maps = True
write_cum_cur_map_only = True
write_volt_maps = False
write_max_cur_maps = False
set_null_currents_to_nodata = True
set_focal_node_currents_to_zero = False
compress_grids = False
log_transform_maps = False
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    grid = gpd.read_file(MORPH_MULTI, layer="extended_multidirection", engine="pyogrio")
    extra = gpd.read_file(MORPH_1125, layer="extended_morphology_r1_1125", engine="pyogrio")[["grid_id", "lambda_f_1125"]]
    grid = grid.merge(extra, on="grid_id", how="left")
    paris = gpd.read_file(MORPH_MULTI, layer="paris_boundary", engine="pyogrio").geometry.union_all()
    lcz = gpd.read_file(LCZ, layer="cerema_lcz_100m", engine="pyogrio")[["grid_id", "lcz_int"]]
    ref = gpd.read_file(REFERENCE, layer="real_homo_100m_reference", engine="pyogrio")[["grid_id", "FAI_REAL_scalar", "urban_support_class"]]
    grid = grid.merge(lcz, on="grid_id", how="left").merge(ref, on="grid_id", how="left")
    replace = grid.FAI_REAL_scalar.notna()

    class_proxy = (
        grid.loc[replace & grid.lcz_int.between(1, 10)]
        .groupby("lcz_int")["FAI_REAL_scalar"]
        .median()
        .to_dict()
    )
    grid["fai_lcz_base"] = 0.0
    built = replace & grid.lcz_int.between(1, 10)
    grid.loc[built, "fai_lcz_base"] = grid.loc[built, "lcz_int"].map(class_proxy)

    audits = []
    for direction in DIRECTIONS:
        tag = suffix(direction)
        real_column = f"lambda_f_{tag}"
        transport = (direction + 180) % 360
        theta = math.radians(transport)
        flow = grid.x_center * math.sin(theta) + grid.y_center * math.cos(theta)
        p0, p1 = projection_bounds(paris, transport)
        domain_mask = flow.between(p0 - 5000, p1 + 5000) & grid.support_core_cell.astype(bool)
        domain = grid.loc[domain_mask].copy()
        domain["flow_s_m"] = flow[domain_mask]
        domain["fai_real"] = domain[real_column]
        domain["fai_lcz"] = domain.fai_real
        treated = domain.FAI_REAL_scalar.notna()
        raw_proxy = domain.loc[treated, "fai_lcz_base"].to_numpy(float)
        scale = float(domain.loc[treated, "fai_real"].sum() / raw_proxy.sum())
        domain.loc[treated, "fai_lcz"] = raw_proxy * scale
        domain["resistance_lcz"] = 1 + domain.fai_lcz / Q95
        domain["focal"] = NODATA
        s0, s1 = domain.flow_s_m.min(), domain.flow_s_m.max()
        domain.loc[domain.flow_s_m <= s0 + 300, "focal"] = 1
        domain.loc[domain.flow_s_m >= s1 - 300, "focal"] = 2

        run = OUTPUT / tag
        run.mkdir()
        resistance, xll, yll = grid_array(domain, "resistance_lcz")
        focal, _, _ = grid_array(domain, "focal")
        habitat_path, focal_path = run / "resistance.asc", run / "focal.asc"
        write_ascii(habitat_path, resistance, xll, yll)
        write_ascii(focal_path, focal, xll, yll)
        ini = run / "circuitscape.ini"
        ini.write_text(config(habitat_path, focal_path, run / f"lcz_{tag}.out"), encoding="utf-8")
        domain[["grid_id", "ix", "iy", "x_center", "y_center", "analysis_core", "lcz_int", "fai_real", "fai_lcz", "resistance_lcz"]].to_csv(run / "cells.csv", index=False)

        comparison = domain.loc[treated, ["fai_real", "fai_lcz"]].dropna()
        audits.append({
            "direction": direction,
            "tag": tag,
            "transport": transport,
            "n_domain": int(len(domain)),
            "n_replaced_urban_support": int(treated.sum()),
            "lcz_proxy_scale": scale,
            "fai_sum_relative_difference": float((comparison.fai_lcz.sum() - comparison.fai_real.sum()) / comparison.fai_real.sum()),
            "spearman_real_directional_fai_vs_lcz_proxy": float(spearmanr(comparison.fai_real, comparison.fai_lcz).statistic),
            "rmse_real_directional_fai_vs_lcz_proxy": float(np.sqrt(np.mean((comparison.fai_real - comparison.fai_lcz) ** 2))),
            "outside_treatment_max_abs_fai_change": float(np.max(np.abs(domain.loc[~treated, "fai_real"] - domain.loc[~treated, "fai_lcz"]))),
        })

    report = {
        "status": "LCZ_CIRCUIT_INPUTS_READY",
        "definition": "Within the same 8,229 D075 urban-support cells used by HOMO, replace directional 3-D FAI by one city-calibrated scalar FAI median per built LCZ class. Natural LCZ classes receive zero building FAI. A direction-specific global scale preserves total D075 FAI; outside treatment cells remains REAL.",
        "class_fai_proxy_unscaled": {str(int(key)): float(value) for key, value in class_proxy.items()},
        "q95_frozen": Q95,
        "directions": audits,
    }
    (OUTPUT / "input_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
