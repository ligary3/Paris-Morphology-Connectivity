"""Export the assembled REAL d03 surface fields for station validation (run in WSL)."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import importlib.util
from pathlib import Path

import numpy as np


WORKSPACE = _release_path('/mnt/d/研究生/代码/论文4/lunwen4')
OUTPUT = _release_path('/mnt/e/lunwen4/2/气象观测/Paris_WRF_validation/processed/wrf_real_surface_97h.npz')


def main():
    source = WORKSPACE / "new_study/13_wrf_homo_pair/scripts/analyze_full96h_regimes.py"
    spec = importlib.util.spec_from_file_location("full96h", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    times, data, lat, lon, metadata = module.load_segments(module.REAL_SEGMENTS, 3, module.D03_FIELDS)
    u, v = module.earth_wind(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        times=np.asarray(times), lat=lat, lon=lon,
        u10_earth=u.astype(np.float32), v10_earth=v.astype(np.float32),
        t2_k=data["T2"].astype(np.float32), pblh_m=data["PBLH"].astype(np.float32),
    )
    print("records", len(times), "shape", u.shape, "output", OUTPUT)


if __name__ == "__main__":
    main()
