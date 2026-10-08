"""Prepare matched 3 km and 5 km REAL Circuitscape boundary-pilot inputs."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
from pathlib import Path

import geopandas as gpd
import numpy as np


INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_morphology_resistance_1125.gpkg')
OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\circuit_boundary_pilot')
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
        np.savetxt(stream, array, fmt="%.8g", delimiter=" ")


def build_array(frame: gpd.GeoDataFrame, value_column: str, fill: float) -> tuple[np.ndarray, float, float]:
    ix_min, ix_max = int(frame["ix"].min()), int(frame["ix"].max())
    iy_min, iy_max = int(frame["iy"].min()), int(frame["iy"].max())
    array = np.full((iy_max - iy_min + 1, ix_max - ix_min + 1), fill, dtype=float)
    # ESRI ASCII rows are written north to south.
    rows = iy_max - frame["iy"].to_numpy(dtype=int)
    cols = frame["ix"].to_numpy(dtype=int) - ix_min
    array[rows, cols] = frame[value_column].to_numpy(dtype=float)
    xll = float(frame.loc[frame["ix"].idxmin(), "x_center"] - CELL_SIZE / 2)
    yll = float(frame.loc[frame["iy"].idxmin(), "y_center"] - CELL_SIZE / 2)
    return array, xll, yll


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
    grid = gpd.read_file(INPUT, layer="extended_morphology_r1_1125", engine="pyogrio")
    report: dict[str, object] = {
        "input": str(INPUT),
        "resistance": "r1_c1",
        "cell_size_m": CELL_SIZE,
        "electrode_width_m": ELECTRODE_WIDTH_M,
        "neighbors": 8,
        "connection_resistance": "average adjacent cell resistances",
        "runs": [],
    }
    for distance in (3, 5):
        run_dir = OUTPUT / f"real_{distance}km"
        run_dir.mkdir()
        domain = grid[grid[f"domain_{distance}km"]].copy()
        s_min, s_max = float(domain["flow_s_m"].min()), float(domain["flow_s_m"].max())
        domain["focal_id"] = NODATA
        source = domain["flow_s_m"] <= s_min + ELECTRODE_WIDTH_M
        ground = domain["flow_s_m"] >= s_max - ELECTRODE_WIDTH_M
        domain.loc[source, "focal_id"] = 1
        domain.loc[ground, "focal_id"] = 2
        if source.sum() == 0 or ground.sum() == 0 or (source & ground).any():
            raise ValueError(f"Invalid electrodes for {distance} km run")

        resistance, xll, yll = build_array(domain, "r1_c1", NODATA)
        focal, focal_xll, focal_yll = build_array(domain, "focal_id", NODATA)
        if (xll, yll) != (focal_xll, focal_yll) or resistance.shape != focal.shape:
            raise ValueError("Resistance and focal rasters are not aligned")
        habitat_path = run_dir / "resistance_r1.asc"
        focal_path = run_dir / "focal_regions.asc"
        config_path = run_dir / "circuitscape.ini"
        output_base = run_dir / "real_boundary.out"
        write_ascii(habitat_path, resistance, xll, yll)
        write_ascii(focal_path, focal, xll, yll)
        config_path.write_text(
            config_text(habitat_path, focal_path, output_base), encoding="utf-8"
        )
        report["runs"].append(
            {
                "distance_km": distance,
                "n_domain_cells": int(len(domain)),
                "n_source_cells": int(source.sum()),
                "n_ground_cells": int(ground.sum()),
                "raster_rows": int(resistance.shape[0]),
                "raster_cols": int(resistance.shape[1]),
                "resistance_min": float(domain["r1_c1"].min()),
                "resistance_max": float(domain["r1_c1"].max()),
                "config": str(config_path),
            }
        )
    (OUTPUT / "boundary_pilot_inputs.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
