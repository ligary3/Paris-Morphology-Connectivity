"""Prepare matched REAL/HOMO Circuit pairs for four event-stage inflow directions."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np


INPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\extended_morphology_multidirection_v1.gpkg')
REFERENCE = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\real_homo_design_v2\\urban_cells_reference.gpkg')
OUTPUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\extended_domain_v1\\circuit_multidirection_pairs_v1')
DIRECTIONS = (22.5, 45.0, 67.5, 90.0)
Q95 = 0.976565954884937
NODATA = -9999.0
CELL_SIZE = 100


def suffix(direction: float) -> str:
    return f"{int(round(direction * 10)):04d}"


def projection_bounds(geometry, bearing: float) -> tuple[float, float]:
    theta = math.radians(bearing)
    dx, dy = math.sin(theta), math.cos(theta)
    parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    coordinates = np.vstack([np.asarray(part.exterior.coords) for part in parts])
    projected = coordinates[:, 0] * dx + coordinates[:, 1] * dy
    return float(projected.min()), float(projected.max())


def array(frame, column):
    ix0, ix1, iy0, iy1 = int(frame.ix.min()), int(frame.ix.max()), int(frame.iy.min()), int(frame.iy.max())
    result = np.full((iy1 - iy0 + 1, ix1 - ix0 + 1), NODATA)
    result[iy1 - frame.iy.to_numpy(int), frame.ix.to_numpy(int) - ix0] = frame[column].to_numpy(float)
    return result, float(frame.x_center.min() - 50), float(frame.y_center.min() - 50)


def write_ascii(path, values, xll, yll):
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ncols {values.shape[1]}\nnrows {values.shape[0]}\nxllcorner {xll:.3f}\nyllcorner {yll:.3f}\ncellsize 100\nNODATA_value -9999\n")
        np.savetxt(stream, values, fmt="%.9g")


def config(habitat, focal, output):
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


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    grid = gpd.read_file(INPUT, layer="extended_multidirection", engine="pyogrio")
    paris = gpd.read_file(INPUT, layer="paris_boundary", engine="pyogrio").geometry.union_all()
    ref = gpd.read_file(REFERENCE, layer="real_homo_100m_reference", engine="pyogrio").set_index("grid_id")
    audits = []
    for direction in DIRECTIONS:
        tag = suffix(direction)
        transport = (direction + 180.0) % 360.0
        theta = math.radians(transport)
        flow = grid.x_center * math.sin(theta) + grid.y_center * math.cos(theta)
        p0, p1 = projection_bounds(paris, transport)
        mask = flow.between(p0 - 5000, p1 + 5000) & grid.support_core_cell.astype(bool)
        domain = grid.loc[mask].copy()
        domain["flow_s_m"] = flow[mask]
        real_col, homo_col = f"lambda_f_{tag}", f"FAI_HOMO_{tag}"
        domain["fai_real"] = domain[real_col]
        domain["fai_homo"] = domain["fai_real"]
        matched = domain.grid_id.isin(ref.index)
        domain.loc[matched, "fai_homo"] = domain.loc[matched, "grid_id"].map(ref[homo_col])
        domain["resistance_real"] = 1 + domain.fai_real / Q95
        domain["resistance_homo"] = 1 + domain.fai_homo / Q95
        s0, s1 = domain.flow_s_m.min(), domain.flow_s_m.max()
        domain["focal"] = NODATA
        domain.loc[domain.flow_s_m <= s0 + 300, "focal"] = 1
        domain.loc[domain.flow_s_m >= s1 - 300, "focal"] = 2
        direction_dir = OUTPUT / tag
        direction_dir.mkdir()
        for case, column in (("real", "resistance_real"), ("homo", "resistance_homo")):
            run = direction_dir / case
            run.mkdir()
            resistance, xll, yll = array(domain, column)
            focal, _, _ = array(domain, "focal")
            habitat_path, focal_path = run / "resistance.asc", run / "focal.asc"
            write_ascii(habitat_path, resistance, xll, yll)
            write_ascii(focal_path, focal, xll, yll)
            (run / "circuitscape.ini").write_text(config(habitat_path, focal_path, run / f"{case}_{tag}.out"), encoding="utf-8")
        domain[["grid_id", "ix", "iy", "x_center", "y_center", "analysis_core", "fai_real", "fai_homo", "resistance_real", "resistance_homo"]].to_csv(direction_dir / "cells.csv", index=False)
        audits.append({"direction": direction, "tag": tag, "transport": transport, "n_domain": len(domain), "n_homo": int(matched.sum()), "fai_sum_relative_difference": float((domain.loc[matched, 'fai_homo'].sum() - domain.loc[matched, 'fai_real'].sum()) / domain.loc[matched, 'fai_real'].sum())})
    (OUTPUT / "input_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
