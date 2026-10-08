"""Safely inject the frozen Paris d03 activation/UCP templates.

`pre-real` operates on copied met_em.d03 files before real.exe. It writes the
land-use activation and all distributed-drag fields, including ZD_URB2D which
real.exe consumes while constructing the model height.

`post-real` operates on a copied wrfinput_d03 after real.exe. It restores the
target FRC_URB2D in the D075 treatment core (real.exe recomputes this field) and
verifies that activation and persistent UCP fields survived.

Inputs are never edited in place: --output must not already exist.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import argparse
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np


DEFAULT_TEMPLATE = _release_path('/mnt/e/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')

URBAN_CLASS = 13
FIELD_META = {
    "MH_URB2D": ("m", "Mean Building Height"),
    "BUILD_AREA_FRACTION": ("dimensionless", "Building Plan Area Density"),
    "LF_URB2D_S": ("dimensionless", "Scalar Frontal Area Index"),
    "ZD_URB2D": ("m", "Zero-plane Displacement"),
    "Z0_URB2D": ("m", "Urban Roughness Length for Momentum"),
}


def load_templates(path: Path, mode: str) -> tuple[dict[str, np.ndarray], np.lib.npyio.NpzFile]:
    package = np.load(path)
    required = {
        "LU_INDEX_TARGET", "PRIMARY_ACTIVATION", "TREATMENT_CORE",
        "FRC_PATCH_MASK", "FRC_URB2D_TARGET",
        *{f"{mode}_{name}" for name in FIELD_META},
    }
    missing = sorted(required.difference(package.files))
    if missing:
        raise KeyError(f"Template package is missing: {missing}")
    fields = {name: np.asarray(package[f"{mode}_{name}"], dtype=np.float32) for name in FIELD_META}
    return fields, package


def copy_new(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def spatial_dims(ds: netCDF4.Dataset) -> tuple[str, str, str]:
    for candidate in (("Time", "south_north", "west_east"), ("Time", "south_north_stag", "west_east_stag")):
        if all(name in ds.dimensions for name in candidate):
            return candidate
    raise KeyError("Cannot find WRF Time/south_north/west_east dimensions")


def get_or_create(ds: netCDF4.Dataset, name: str) -> netCDF4.Variable:
    if name in ds.variables:
        var = ds.variables[name]
        if var.dimensions != spatial_dims(ds):
            raise ValueError(f"{name} has unexpected dimensions {var.dimensions}")
        return var
    units, description = FIELD_META[name]
    var = ds.createVariable(name, "f4", spatial_dims(ds))
    var.units = units
    var.description = description
    var.MemoryOrder = "XY "
    var.stagger = ""
    var.FieldType = np.int32(104)
    return var


def patch_landuse(ds: netCDF4.Dataset, target: np.ndarray, frc_target: np.ndarray, core: np.ndarray) -> int:
    lu_var = ds.variables["LU_INDEX"]
    land_var = ds.variables["LANDUSEF"]
    lu = np.asarray(lu_var[0], dtype=np.int16)
    land = np.asarray(land_var[0], dtype=np.float64)
    if lu.shape != (120, 120) or land.shape[1:] != (120, 120):
        raise ValueError(f"Unexpected d03 shape: LU {lu.shape}; LANDUSEF {land.shape}")

    changed = core & (lu != target)
    for j, i in np.argwhere(changed):
        old = land[:, j, i].copy()
        nonurban_sum = old.sum() - old[URBAN_CLASS - 1]
        if target[j, i] == URBAN_CLASS:
            urb = float(np.clip(frc_target[j, i], 0.01, 0.99))
            old[URBAN_CLASS - 1] = 0.0
            if old.sum() <= 0:
                old[0] = 1.0
            old *= (1.0 - urb) / old.sum()
            old[URBAN_CLASS - 1] = urb
        else:
            old[URBAN_CLASS - 1] = 0.0
            if nonurban_sum <= 0:
                old[int(target[j, i]) - 1] = 1.0
            else:
                old /= old.sum()
        land[:, j, i] = old
        lu[j, i] = target[j, i]

    if not np.allclose(land.sum(axis=0), 1.0, atol=2e-6):
        raise ValueError("Patched LANDUSEF does not sum to one")
    lu_var[0] = lu
    land_var[0] = land.astype(np.float32)
    return int(changed.sum())


def pre_real(args: argparse.Namespace) -> None:
    fields, package = load_templates(args.template, args.mode)
    copy_new(args.input, args.output)
    target = np.asarray(package["LU_INDEX_TARGET"], dtype=np.int16)
    core = np.asarray(package["TREATMENT_CORE"], dtype=bool)
    frc = np.asarray(package["FRC_URB2D_TARGET"], dtype=np.float32)

    with netCDF4.Dataset(args.output, "r+") as ds:
        changed = patch_landuse(ds, target, frc, core)
        for name, values in fields.items():
            var = get_or_create(ds, name)
            for time_index in range(len(ds.dimensions["Time"])):
                var[time_index] = values
        ds.setncattr("PARIS_UCP_STAGE", "pre-real")
        ds.setncattr("PARIS_UCP_MODE", args.mode)
        ds.setncattr("PARIS_UCP_TEMPLATE", args.template.name)

    with netCDF4.Dataset(args.output) as ds:
        for name, values in fields.items():
            if not np.allclose(ds.variables[name][0], values, atol=1e-6, rtol=0):
                raise AssertionError(f"Post-write mismatch: {name}")
        if not np.array_equal(ds.variables["LU_INDEX"][0].astype(np.int16), target):
            raise AssertionError("Post-write mismatch: LU_INDEX")
    print(json.dumps({"stage": "pre-real", "mode": args.mode, "output": str(args.output), "landuse_cells_changed": changed}, indent=2))


def post_real(args: argparse.Namespace) -> None:
    fields, package = load_templates(args.template, args.mode)
    copy_new(args.input, args.output)
    target = np.asarray(package["LU_INDEX_TARGET"], dtype=np.int16)
    core = np.asarray(package["FRC_PATCH_MASK"], dtype=bool)
    frc_target = np.asarray(package["FRC_URB2D_TARGET"], dtype=np.float32)

    with netCDF4.Dataset(args.output, "r+") as ds:
        if "FRC_URB2D" not in ds.variables or "IVGTYP" not in ds.variables:
            raise KeyError("wrfinput must contain FRC_URB2D and IVGTYP")
        frc = np.asarray(ds.variables["FRC_URB2D"][0], dtype=np.float32)
        frc[core] = frc_target[core]
        ds.variables["FRC_URB2D"][0] = frc
        ds.setncattr("PARIS_UCP_STAGE", "post-real-validated")
        ds.setncattr("PARIS_UCP_MODE", args.mode)

    with netCDF4.Dataset(args.output) as ds:
        ivgtyp = np.asarray(ds.variables["IVGTYP"][0], dtype=np.int16)
        if not np.array_equal(ivgtyp[core], target[core]):
            bad = int(np.count_nonzero(ivgtyp[core] != target[core]))
            raise AssertionError(f"IVGTYP activation mismatch in {bad} treatment-core cells")
        if not np.allclose(ds.variables["FRC_URB2D"][0][core], frc_target[core], atol=1e-6, rtol=0):
            raise AssertionError("FRC_URB2D mismatch after post-real patch")
        # ZD_URB2D is input-stream-only and may not be present in wrfinput.
        for name in ("MH_URB2D", "BUILD_AREA_FRACTION", "LF_URB2D_S", "Z0_URB2D"):
            if name not in ds.variables:
                raise KeyError(f"real.exe did not preserve required field {name}")
            if not np.allclose(ds.variables[name][0], fields[name], atol=1e-6, rtol=0):
                raise AssertionError(f"real.exe altered or lost {name}")
    print(json.dumps({"stage": "post-real", "mode": args.mode, "output": str(args.output), "frc_cells_restored": int(core.sum())}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre-real", "post-real"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("REAL", "HOMO"), required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stage == "pre-real":
        pre_real(arguments)
    else:
        post_real(arguments)
