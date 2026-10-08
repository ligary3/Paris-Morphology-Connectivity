"""Prepare a matched 112.5-degree REAL/HOMO Circuitscape pair on the frozen 5 km domain."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import numpy as np


EXTENDED = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_morphology_resistance_1125.gpkg')
HOMO_REFERENCE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\real_homo_design_v2\\urban_cells_reference.gpkg')
OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\circuit_real_homo_1125_v1')
Q95 = 0.976565954884937
NODATA = -9999.0
CELL_SIZE = 100
ELECTRODE_WIDTH_M = 300


def write_ascii(path: Path, array: np.ndarray, xll: float, yll: float) -> None:
    header = (
        f"ncols {array.shape[1]}\n"
        f"nrows {array.shape[0]}\n"
        f"xllcorner {xll:.3f}\n"
        f"yllcorner {yll:.3f}\n"
        f"cellsize {CELL_SIZE}\n"
        f"NODATA_value {NODATA:g}\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        np.savetxt(stream, array, fmt="%.9g", delimiter=" ")


def build_array(frame: gpd.GeoDataFrame, value_column: str) -> tuple[np.ndarray, float, float]:
    ix_min, ix_max = int(frame["ix"].min()), int(frame["ix"].max())
    iy_min, iy_max = int(frame["iy"].min()), int(frame["iy"].max())
    array = np.full((iy_max - iy_min + 1, ix_max - ix_min + 1), NODATA, dtype=float)
    rows = iy_max - frame["iy"].to_numpy(dtype=int)
    cols = frame["ix"].to_numpy(dtype=int) - ix_min
    array[rows, cols] = frame[value_column].to_numpy(dtype=float)
    return array, float(frame["x_center"].min() - 50), float(frame["y_center"].min() - 50)


def config_text(habitat: Path, focal: Path, output_base: Path) -> str:
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
output_file = {output_base}
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
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    full = gpd.read_file(EXTENDED, layer="extended_morphology_r1_1125", engine="pyogrio")
    domain = full.loc[full["domain_5km"]].copy()
    reference = gpd.read_file(HOMO_REFERENCE, layer="real_homo_100m_reference", engine="pyogrio")
    if reference["grid_id"].duplicated().any():
        raise ValueError("Duplicate HOMO grid_id")
    homo_lookup = reference.set_index("grid_id")["FAI_HOMO_1125"]
    matched = domain["grid_id"].isin(homo_lookup.index)
    if int(matched.sum()) != len(reference):
        raise ValueError(f"Only {matched.sum()} of {len(reference)} HOMO cells occur in 5 km domain")

    domain["lambda_f_real"] = domain["lambda_f_1125"]
    domain["lambda_f_homo"] = domain["lambda_f_real"]
    domain.loc[matched, "lambda_f_homo"] = domain.loc[matched, "grid_id"].map(homo_lookup)
    domain["resistance_real"] = 1.0 + domain["lambda_f_real"] / Q95
    domain["resistance_homo"] = 1.0 + domain["lambda_f_homo"] / Q95

    s_min, s_max = float(domain["flow_s_m"].min()), float(domain["flow_s_m"].max())
    domain["focal_id"] = NODATA
    source = domain["flow_s_m"] <= s_min + ELECTRODE_WIDTH_M
    ground = domain["flow_s_m"] >= s_max - ELECTRODE_WIDTH_M
    domain.loc[source, "focal_id"] = 1
    domain.loc[ground, "focal_id"] = 2

    arrays = {}
    for name, column in (("real", "resistance_real"), ("homo", "resistance_homo")):
        run_dir = OUTPUT / name
        run_dir.mkdir()
        resistance, xll, yll = build_array(domain, column)
        focal, fxll, fyll = build_array(domain, "focal_id")
        if resistance.shape != focal.shape or (xll, yll) != (fxll, fyll):
            raise ValueError("Raster mismatch")
        habitat = run_dir / "resistance_r1.asc"
        focal_path = run_dir / "focal_regions.asc"
        write_ascii(habitat, resistance, xll, yll)
        write_ascii(focal_path, focal, xll, yll)
        (run_dir / "circuitscape.ini").write_text(
            config_text(habitat, focal_path, run_dir / f"{name}_1125.out"), encoding="utf-8"
        )
        arrays[name] = resistance

    changed = domain.loc[matched]
    unchanged = domain.loc[~matched]
    audit = {
        "status": "READY",
        "direction_inflow_deg": 112.5,
        "direction_transport_deg": 292.5,
        "q95_frozen": Q95,
        "domain": "frozen 5 km boundary-pilot domain",
        "n_domain_cells": int(len(domain)),
        "n_homogenized_d075_cells": int(matched.sum()),
        "n_source_cells": int(source.sum()),
        "n_ground_cells": int(ground.sum()),
        "outside_d075_max_abs_resistance_change": float(
            np.abs(unchanged["resistance_real"] - unchanged["resistance_homo"]).max()
        ),
        "d075_mean_fai_real": float(changed["lambda_f_real"].mean()),
        "d075_mean_fai_homo": float(changed["lambda_f_homo"].mean()),
        "d075_sum_fai_real": float(changed["lambda_f_real"].sum()),
        "d075_sum_fai_homo": float(changed["lambda_f_homo"].sum()),
        "raster_shape": list(arrays["real"].shape),
        "real_homo_focal_identical": True,
    }
    domain.drop(columns="geometry").to_csv(OUTPUT / "cellwise_input_audit.csv", index=False)
    (OUTPUT / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
