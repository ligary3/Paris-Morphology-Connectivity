from __future__ import annotations

from release_paths import resolve_path as _release_path

import hashlib
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


REAL_CASE = _release_path('/home/gary/wrf-work/cases/paris_real_init_v1')
HOMO_CASE = _release_path('/home/gary/wrf-work/cases/paris_homo_init_v1')
REAL_D03 = REAL_CASE / "wrfinput_d03.REAL.final"
HOMO_D03 = HOMO_CASE / "wrfinput_d03.HOMO.final"
TEMPLATE = _release_path('/mnt/e/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')
OUT = _release_path('/mnt/d/研究生/代码/论文4/lunwen4/new_study/13_wrf_homo_pair/homo_pair_input_qa.json')
PERSISTENT_FIELDS = (
    "MH_URB2D",
    "BUILD_AREA_FRACTION",
    "LF_URB2D_S",
    "Z0_URB2D",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def as_array(variable: nc.Variable) -> np.ndarray:
    return np.ma.filled(variable[:], np.nan)


def different(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return ~np.isclose(a, b, atol=1.0e-12, rtol=0.0, equal_nan=True)
    return a != b


def dilate(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            src_j0 = max(0, -dj)
            src_j1 = min(mask.shape[0], mask.shape[0] - dj)
            src_i0 = max(0, -di)
            src_i1 = min(mask.shape[1], mask.shape[1] - di)
            dst_j0 = src_j0 + dj
            dst_j1 = src_j1 + dj
            dst_i0 = src_i0 + di
            dst_i1 = src_i1 + di
            result[dst_j0:dst_j1, dst_i0:dst_i1] |= mask[src_j0:src_j1, src_i0:src_i1]
    return result


def spatial_change_mask(diff: np.ndarray, dimensions: tuple[str, ...]) -> tuple[np.ndarray, str] | None:
    y_candidates = [name for name in dimensions if name in ("south_north", "south_north_stag")]
    x_candidates = [name for name in dimensions if name in ("west_east", "west_east_stag")]
    if len(y_candidates) != 1 or len(x_candidates) != 1:
        return None
    y_name, x_name = y_candidates[0], x_candidates[0]
    y_axis, x_axis = dimensions.index(y_name), dimensions.index(x_name)
    reduce_axes = tuple(index for index in range(diff.ndim) if index not in (y_axis, x_axis))
    spatial = np.any(diff, axis=reduce_axes) if reduce_axes else diff
    if y_axis > x_axis:
        spatial = spatial.T
    return np.asarray(spatial, dtype=bool), f"{y_name},{x_name}"


def allowed_mask(active: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    halo = dilate(active)
    if spatial_shape == halo.shape:
        return halo
    if spatial_shape == (halo.shape[0], halo.shape[1] + 1):
        result = np.zeros(spatial_shape, dtype=bool)
        result[:, :-1] |= halo
        result[:, 1:] |= halo
        return result
    if spatial_shape == (halo.shape[0] + 1, halo.shape[1]):
        result = np.zeros(spatial_shape, dtype=bool)
        result[:-1, :] |= halo
        result[1:, :] |= halo
        return result
    raise ValueError(f"unsupported spatial shape: {spatial_shape}")


def main() -> None:
    package = np.load(TEMPLATE)
    active = package["PRIMARY_ACTIVATION"].astype(bool)
    core = package["TREATMENT_CORE"].astype(bool)
    treatment_active = active & core
    treatment_inactive = (~active) & core
    background_active = active & (~core)

    common_files = (
        "wrfinput_d01",
        "wrfinput_d02",
        "wrfbdy_d01",
        "wrflowinp_d01",
        "wrflowinp_d02",
        "wrflowinp_d03",
    )
    common_hashes: dict[str, object] = {}
    for name in common_files:
        real_hash = sha256(REAL_CASE / name)
        homo_hash = sha256(HOMO_CASE / name)
        common_hashes[name] = {
            "real": real_hash,
            "homo": homo_hash,
            "exact_equal": real_hash == homo_hash,
        }

    result: dict[str, object] = {
        "counts": {
            "treatment_active": int(treatment_active.sum()),
            "treatment_inactive": int(treatment_inactive.sum()),
            "background_active": int(background_active.sum()),
            "all_active": int(active.sum()),
        },
        "common_file_hashes": common_hashes,
        "persistent_fields": {},
        "d03_changed_variables": {},
    }

    with nc.Dataset(REAL_D03) as real, nc.Dataset(HOMO_D03) as homo:
        result["homo_stage"] = homo.getncattr("PARIS_UCP_STAGE")
        result["homo_mode"] = homo.getncattr("PARIS_UCP_MODE")

        for field in PERSISTENT_FIELDS:
            real_values = np.asarray(real.variables[field][0])
            homo_values = np.asarray(homo.variables[field][0])
            expected_homo = package[f"HOMO_{field}"]
            diff = different(real_values, homo_values)
            result["persistent_fields"][field] = {
                "homo_max_abs_error_vs_template": float(np.max(np.abs(homo_values - expected_homo))),
                "changed_total": int(diff.sum()),
                "changed_treatment_active": int(diff[treatment_active].sum()),
                "changed_outside_treatment_active": int(diff[~treatment_active].sum()),
                "real_homo_background_exact_equal": bool(np.array_equal(real_values[~core], homo_values[~core])),
                "real_homo_inactive_core_exact_equal": bool(
                    np.array_equal(real_values[treatment_inactive], homo_values[treatment_inactive])
                ),
            }

        for field in ("IVGTYP", "FRC_URB2D"):
            real_values = np.asarray(real.variables[field][:])
            homo_values = np.asarray(homo.variables[field][:])
            result[f"{field.lower()}_exact_equal"] = bool(np.array_equal(real_values, homo_values))

        mh = np.asarray(homo.variables["MH_URB2D"][0])
        z0 = np.asarray(homo.variables["Z0_URB2D"][0])
        result["homo_z0_lt_mh_all_active"] = bool(np.all(z0[active] < mh[active]))

        common_variables = sorted(set(real.variables).intersection(homo.variables))
        for name in common_variables:
            a = as_array(real.variables[name])
            b = as_array(homo.variables[name])
            if a.shape != b.shape:
                result["d03_changed_variables"][name] = {"shape_mismatch": [a.shape, b.shape]}
                continue
            diff = different(a, b)
            if not np.any(diff):
                continue
            spatial_info = spatial_change_mask(diff, real.variables[name].dimensions)
            item: dict[str, object] = {
                "changed_values": int(diff.sum()),
                "dimensions": list(real.variables[name].dimensions),
            }
            if spatial_info is None:
                item["nonspatial_difference"] = True
            else:
                spatial, dimension_kind = spatial_info
                allowed = allowed_mask(treatment_active, spatial.shape)
                item.update(
                    {
                        "changed_spatial_cells": int(spatial.sum()),
                        "outside_one_cell_halo": int(np.count_nonzero(spatial & ~allowed)),
                        "spatial_dimensions": dimension_kind,
                    }
                )
            result["d03_changed_variables"][name] = item

    changed_union = np.zeros_like(treatment_active)
    for field in PERSISTENT_FIELDS:
        with nc.Dataset(REAL_D03) as real, nc.Dataset(HOMO_D03) as homo:
            changed_union |= different(
                np.asarray(real.variables[field][0]), np.asarray(homo.variables[field][0])
            )

    derived_leakage = {
        name: item.get("outside_one_cell_halo", 0)
        for name, item in result["d03_changed_variables"].items()
        if isinstance(item, dict) and item.get("outside_one_cell_halo", 0) != 0
    }
    nonspatial = [
        name
        for name, item in result["d03_changed_variables"].items()
        if isinstance(item, dict) and item.get("nonspatial_difference")
    ]
    requirements = {
        "common_parent_and_boundary_files_exact_equal": all(
            item["exact_equal"] for item in common_hashes.values()
        ),
        "homo_template_exact": all(
            item["homo_max_abs_error_vs_template"] <= 1.0e-6
            for item in result["persistent_fields"].values()
        ),
        "persistent_changes_confined_to_85_cells": all(
            item["changed_outside_treatment_active"] == 0
            for item in result["persistent_fields"].values()
        ) and np.array_equal(changed_union, treatment_active),
        "activation_and_fraction_exact_equal": result["ivgtyp_exact_equal"] and result["frc_urb2d_exact_equal"],
        "homo_z0_lt_mh": result["homo_z0_lt_mh_all_active"],
        "no_derived_difference_beyond_one_cell_halo": len(derived_leakage) == 0,
        "no_nonspatial_variable_difference": len(nonspatial) == 0,
    }
    result["derived_leakage"] = derived_leakage
    result["nonspatial_changed_variables"] = nonspatial
    result["requirements"] = requirements
    result["status"] = "PASS" if all(requirements.values()) else "REVIEW_REQUIRED"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(requirements.values()):
        raise AssertionError(json.dumps(requirements, indent=2))


if __name__ == "__main__":
    main()
