from __future__ import annotations

from release_paths import resolve_path as _release_path

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
UCP_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_ucp_d03_v1')
ACT_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_activation_pilot_v1')
OUT = ACT_DIR / "injection_templates"

GEO = UCP_DIR / "geo_em.d03.nc"
UCP = UCP_DIR / "paris_d03_real_homo_ucp.csv"
ACT = ACT_DIR / "paris_d03_activation_candidates.csv"

URBAN_CLASS = 13

# Neutral d03 background based on the WRF URBPARM.TBL high-density (type 2)
# geometry: ZR=7.5 m, roof width=9.4 m, road width=9.4 m. Plan and
# frontal indices assume a square repeating building/road unit. d and z0 are
# the same Macdonald transform used for the D075 templates.
BACKGROUND = {
    "source": "WRF-4.7.1 URBPARM.TBL urban type 2 geometry",
    "MH_URB2D": 7.5,
    "BUILD_AREA_FRACTION": 0.25,
    "LF_URB2D_S": 0.19946808510638295,
    "ZD_URB2D": 3.6227697249985926,
    "Z0_URB2D": 0.7764714653356897,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def reshape(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(120, 120)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ucp = pd.read_csv(UCP)
    act = pd.read_csv(ACT)
    if not np.array_equal(ucp["wrf_id"].to_numpy(), act["wrf_id"].to_numpy()):
        raise ValueError("UCP and activation rows do not align")

    with h5py.File(GEO, "r") as src:
        lu = np.asarray(src["LU_INDEX"][0], dtype=np.int16)

    primary = act["activation_primary"].astype(bool).to_numpy()
    core = act["d075_core_majority"].astype(bool).to_numpy()
    treatment_active = primary & core
    background_active = primary & ~core
    treatment_inactive = ~primary & core

    lu_target = lu.reshape(-1).copy()
    replacement = act["dominant_nonurban_class"].to_numpy(dtype=np.int16)
    lu_target[treatment_inactive] = replacement[treatment_inactive]
    lu_target[treatment_active] = URBAN_CLASS

    arrays: dict[str, np.ndarray] = {
        "LU_INDEX_TARGET": reshape(lu_target),
        "PRIMARY_ACTIVATION": reshape(primary.astype(np.int8)),
        "TREATMENT_CORE": reshape(core.astype(np.int8)),
        "FRC_PATCH_MASK": reshape(core.astype(np.int8)),
        "FRC_URB2D_TARGET": reshape(np.where(treatment_active, ucp["FRC_URB2D_candidate"], 0.0)),
    }

    mapping = {
        "MH_URB2D": ("REAL_MH_effective_m", "HOMO_MH_effective_m"),
        "BUILD_AREA_FRACTION": ("REAL_LP_effective", "HOMO_LP_effective"),
        "LF_URB2D_S": ("REAL_LF_scalar_effective", "HOMO_LF_scalar_effective"),
        "ZD_URB2D": ("REAL_ZD_raw_m", "HOMO_ZD_raw_m"),
        "Z0_URB2D": ("REAL_Z0_effective_m", "HOMO_Z0_effective_m"),
    }

    for field, (real_col, homo_col) in mapping.items():
        for mode, column in (("REAL", real_col), ("HOMO", homo_col)):
            values = np.zeros(len(ucp), dtype=np.float32)
            values[background_active] = BACKGROUND[field]
            values[treatment_active] = ucp.loc[treatment_active, column].to_numpy(dtype=np.float32)
            arrays[f"{mode}_{field}"] = reshape(values)

    # Core validity checks. Background equality is the key counterfactual QA.
    for mode in ("REAL", "HOMO"):
        mh = arrays[f"{mode}_MH_URB2D"]
        lp = arrays[f"{mode}_BUILD_AREA_FRACTION"]
        lf = arrays[f"{mode}_LF_URB2D_S"]
        zd = arrays[f"{mode}_ZD_URB2D"]
        z0 = arrays[f"{mode}_Z0_URB2D"]
        active2d = arrays["PRIMARY_ACTIVATION"].astype(bool)
        if not np.all(np.isfinite(mh[active2d])):
            raise ValueError(f"{mode}: non-finite MH")
        if not np.all((mh[active2d] > 0) & (lp[active2d] > 0) & (lp[active2d] < 1)):
            raise ValueError(f"{mode}: invalid MH/LP")
        if not np.all((lf[active2d] > 0) & (zd[active2d] >= 0) & (z0[active2d] > 0) & (z0[active2d] < mh[active2d])):
            raise ValueError(f"{mode}: invalid LF/ZD/Z0")

    outside = ~arrays["TREATMENT_CORE"].astype(bool)
    for field in mapping:
        if not np.array_equal(arrays[f"REAL_{field}"][outside], arrays[f"HOMO_{field}"][outside]):
            raise ValueError(f"REAL/HOMO background differs for {field}")

    np.savez_compressed(OUT / "paris_d03_real_homo_injection_templates.npz", **arrays)

    rows = []
    for field in mapping:
        for mode in ("REAL", "HOMO"):
            values = arrays[f"{mode}_{field}"]
            rows.append({
                "mode": mode,
                "field": field,
                "active_min": float(values[arrays["PRIMARY_ACTIVATION"].astype(bool)].min()),
                "active_median": float(np.median(values[arrays["PRIMARY_ACTIVATION"].astype(bool)])),
                "active_max": float(values[arrays["PRIMARY_ACTIVATION"].astype(bool)].max()),
                "treatment_median": float(np.median(values[arrays["TREATMENT_CORE"].astype(bool) & arrays["PRIMARY_ACTIVATION"].astype(bool)])),
            })
    pd.DataFrame(rows).to_csv(OUT / "injection_template_field_summary.csv", index=False)

    summary = {
        "version": "paris_d03_real_homo_injection_templates_v1",
        "d03_cells": 14400,
        "primary_active_cells_global": int(primary.sum()),
        "treatment_core_cells": int(core.sum()),
        "treatment_active_cells": int(treatment_active.sum()),
        "treatment_inactive_cells": int(treatment_inactive.sum()),
        "common_background_active_cells": int(background_active.sum()),
        "real_homo_background_exact_equal": True,
        "background_template": BACKGROUND,
        "inputs": {"geo": sha256(GEO), "ucp": sha256(UCP), "activation": sha256(ACT)},
    }
    (OUT / "injection_template_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
