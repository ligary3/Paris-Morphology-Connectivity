from __future__ import annotations

from release_paths import resolve_path as _release_path

import hashlib
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


CASE = _release_path('/home/gary/wrf-work/cases/paris_real_init_v1')
RAW = CASE / "wrfinput_d03"
FINAL = CASE / "wrfinput_d03.REAL.final"
TEMPLATE = _release_path('/mnt/e/lunwen4/2/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')
OUT = _release_path('/mnt/d/研究生/代码/论文4/lunwen4/new_study/12_wrf_real_init/real_wrfinput_qa.json')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read2d(dataset: nc.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset.variables[name][:])
    while values.ndim > 2:
        values = values[0]
    return values


def main() -> None:
    package = np.load(TEMPLATE)
    active = package["PRIMARY_ACTIVATION"].astype(bool)
    core = package["TREATMENT_CORE"].astype(bool)
    treatment_active = active & core
    treatment_inactive = (~active) & core
    background_active = active & (~core)

    fields = [
        "MH_URB2D",
        "BUILD_AREA_FRACTION",
        "LF_URB2D_S",
        "Z0_URB2D",
    ]

    result: dict[str, object] = {
        "raw_sha256": sha256(RAW),
        "final_sha256": sha256(FINAL),
        "counts": {
            "treatment_active": int(treatment_active.sum()),
            "treatment_inactive": int(treatment_inactive.sum()),
            "background_active": int(background_active.sum()),
            "all_active": int(active.sum()),
        },
        "fields": {},
    }

    with nc.Dataset(RAW) as raw, nc.Dataset(FINAL) as final:
        result["post_real_stage"] = final.getncattr("PARIS_UCP_STAGE")
        result["post_real_mode"] = final.getncattr("PARIS_UCP_MODE")
        result["zd_handling"] = (
            "ZD_URB2D is a pre-real input only; WRF v4.7.1 distributed-drag "
            "diagnoses displacement internally and does not persist ZD_URB2D in wrfinput."
        )
        frc_raw = read2d(raw, "FRC_URB2D")
        frc_final = read2d(final, "FRC_URB2D")
        frc_target = package["FRC_URB2D_TARGET"]
        ivgtyp = read2d(final, "IVGTYP")
        urban_class = ivgtyp == 13
        result["urban_activation"] = {
            "ivgtyp_13_cells": int(urban_class.sum()),
            "active_mask_cells": int(active.sum()),
            "ivgtyp_vs_active_mismatch": int(np.count_nonzero(urban_class != active)),
        }
        result["frc"] = {
            "changed_cells_raw_to_final": int(np.count_nonzero(~np.isclose(frc_raw, frc_final))),
            "core_target_max_abs_error": float(np.max(np.abs(frc_final[core] - frc_target[core]))),
            "outside_core_max_abs_change": float(np.max(np.abs(frc_final[~core] - frc_raw[~core]))),
            "positive_all": int(np.count_nonzero(frc_final > 0)),
            "positive_treatment": int(np.count_nonzero(frc_final[treatment_active] > 0)),
            "zero_treatment_inactive": int(np.count_nonzero(np.isclose(frc_final[treatment_inactive], 0))),
        }

        for field in fields:
            actual = read2d(final, field)
            expected = package[f"REAL_{field}"]
            active_values = actual[active]
            max_error = float(np.max(np.abs(actual - expected)))
            result["fields"][field] = {
                "max_abs_error_vs_template": max_error,
                "active_min": float(active_values.min()),
                "active_median": float(np.median(active_values)),
                "active_max": float(active_values.max()),
                "nonfinite_count": int(np.count_nonzero(~np.isfinite(actual))),
                "positive_cells": int(np.count_nonzero(actual > 0)),
            }
            if max_error > 1.0e-5:
                raise AssertionError(f"{field} differs from REAL template: {max_error}")

        mh = read2d(final, "MH_URB2D")
        z0 = read2d(final, "Z0_URB2D")
        result["z0_lt_mh_all_active"] = bool(np.all(z0[active] < mh[active]))
        if not result["z0_lt_mh_all_active"]:
            raise AssertionError("Z0_URB2D is not below MH_URB2D on all active cells")
        if result["frc"]["core_target_max_abs_error"] > 1.0e-6:
            raise AssertionError("FRC core differs from target")
        if result["frc"]["outside_core_max_abs_change"] > 1.0e-6:
            raise AssertionError("post-real changed FRC outside D075 treatment core")
        if result["urban_activation"]["ivgtyp_vs_active_mismatch"] != 0:
            raise AssertionError("IVGTYP urban activation differs from the frozen active mask")

    result["status"] = "REAL_WRFINPUT_QA_PASS"
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
