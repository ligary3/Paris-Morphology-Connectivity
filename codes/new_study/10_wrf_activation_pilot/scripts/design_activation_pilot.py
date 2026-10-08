from __future__ import annotations

from release_paths import resolve_path as _release_path

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _release_path('D:\\研究生\\代码\\论文4\\lunwen4')
UCP_DIR = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_ucp_d03_v1')
GEO = UCP_DIR / "geo_em.d03.nc"
UCP_CSV = UCP_DIR / "paris_d03_real_homo_ucp.csv"
OUT = _release_path('E:\\lunwen4\\2\\建筑形态\\Paris\\processed\\phase1a_v2\\wrf_activation_pilot_v1')
REPORT = ROOT / "new_study" / "10_wrf_activation_pilot" / "WRF_ACTIVATION_PILOT_REPORT.md"

URBAN_CLASS = 13
FRC_THRESHOLD = 0.20
D075_CORE_THRESHOLD = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(copy=True)
    return series.astype(int).to_numpy() != 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UCP_CSV)
    if len(df) != 120 * 120:
        raise ValueError(f"Expected 14,400 d03 cells, got {len(df):,}")

    with h5py.File(GEO, "r") as src:
        lu = np.asarray(src["LU_INDEX"][0], dtype=np.int16)
        landusef = np.asarray(src["LANDUSEF"][0], dtype=np.float64)
        mminlu = src.attrs["MMINLU"].decode() if isinstance(src.attrs["MMINLU"], bytes) else str(src.attrs["MMINLU"])

    if lu.shape != (120, 120) or landusef.shape != (21, 120, 120):
        raise ValueError(f"Unexpected WPS arrays: LU={lu.shape}, LANDUSEF={landusef.shape}")
    if not np.allclose(landusef.sum(axis=0), 1.0, atol=2e-6):
        raise ValueError("LANDUSEF does not sum to one")

    # CSV uses one-based i/j; NumPy arrays are [j, i].
    j0 = df["j"].to_numpy(dtype=int) - 1
    i0 = df["i"].to_numpy(dtype=int) - 1
    if not np.array_equal(lu[j0, i0], df["LU_INDEX"].to_numpy(dtype=int)):
        raise ValueError("CSV and geo_em LU_INDEX are not aligned")

    d075 = df["D075_fraction"].to_numpy() > 0
    d075_core = df["D075_fraction"].to_numpy() >= D075_CORE_THRESHOLD
    wps = lu[j0, i0] == URBAN_CLASS
    strict = as_bool(df["urban_activation_candidate"])

    # Primary scientific mask: refine only majority-D075 cells. Edge fragments and
    # the rest of d03 remain common background, preventing the administrative
    # boundary from becoming an artificial rural wall.
    primary = wps.copy()
    primary[d075_core] = strict[d075_core]

    # For cells deactivated inside the treatment core, retain the strongest
    # pre-existing non-urban MODIS class rather than inventing a new class.
    nonurban = landusef.copy()
    nonurban[URBAN_CLASS - 1, :, :] = -1.0
    dominant_nonurban = np.argmax(nonurban, axis=0).astype(np.int16) + 1
    replacement = dominant_nonurban[j0, i0]

    action = np.full(len(df), "unchanged_background", dtype=object)
    action[d075_core & wps & ~primary] = "deactivate_to_dominant_nonurban"
    action[d075_core & ~wps & primary] = "activate_as_urban"
    action[d075_core & (wps == primary)] = "core_unchanged"

    df_out = df[[
        "wrf_id", "i", "j", "lat", "lon", "LU_INDEX", "D075_fraction",
        "FRC_URB2D_candidate", "urban_activation_candidate", "wps_urban_lu13",
    ]].copy()
    df_out["d075_core_majority"] = d075_core
    df_out["activation_wps"] = wps
    df_out["activation_strict_d075"] = strict
    df_out["activation_primary"] = primary
    df_out["dominant_nonurban_class"] = replacement
    df_out["primary_action"] = action
    df_out["real_homo_treatment_cell"] = d075_core
    df_out["background_ucp_required"] = primary & ~d075_core
    df_out.to_csv(OUT / "paris_d03_activation_candidates.csv", index=False)

    arrays = {
        "activation_wps": wps.reshape(120, 120),
        "activation_strict_d075": strict.reshape(120, 120),
        "activation_primary": primary.reshape(120, 120),
        "d075_fraction": df["D075_fraction"].to_numpy().reshape(120, 120),
        "d075_core_majority": d075_core.reshape(120, 120),
        "frc_d075_candidate": df["FRC_URB2D_candidate"].to_numpy().reshape(120, 120),
        "dominant_nonurban_class": replacement.reshape(120, 120),
    }
    np.savez_compressed(OUT / "paris_d03_activation_candidates.npz", **arrays)

    conflict = df_out[d075 & (wps != strict)].copy()
    conflict.to_csv(OUT / "paris_d03_wps_vs_strict_conflicts.csv", index=False)
    changed = df_out[d075_core & (wps != primary)].copy()
    changed.to_csv(OUT / "paris_d03_primary_changed_cells.csv", index=False)

    ys, xs = np.where(arrays["d075_fraction"] > 0)
    pad = 2
    yslice = slice(max(0, ys.min() - pad), min(120, ys.max() + pad + 1))
    xslice = slice(max(0, xs.min() - pad), min(120, xs.max() + pad + 1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (arrays["activation_wps"], "A  WPS MODIS urban (LU=13)"),
        (arrays["activation_strict_d075"], "B  Strict D075 support (FRC >= 0.20)"),
        (arrays["activation_primary"], "C  Primary: refine majority-D075 cells"),
        ((arrays["activation_primary"].astype(int) - arrays["activation_wps"].astype(int)), "D  Primary minus WPS"),
    ]
    for ax, (data, title) in zip(axes.flat, panels):
        if title.startswith("D"):
            im = ax.imshow(data[yslice, xslice], origin="lower", cmap="bwr", vmin=-1, vmax=1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_ticks([-1, 0, 1], labels=["deactivate", "same", "activate"])
        else:
            im = ax.imshow(data[yslice, xslice], origin="lower", cmap="Greys", vmin=0, vmax=1)
        contour = arrays["d075_fraction"][yslice, xslice]
        ax.contour(contour, levels=[0.01, 0.5], colors=["#F59E0B", "#2563EB"], linewidths=[1.0, 1.5])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("d03 i (zoomed)")
        ax.set_ylabel("d03 j (zoomed)")
    fig.suptitle("Paris d03 urban activation pilot\norange: D075 intersection; blue: majority-D075 core", fontsize=14)
    fig.savefig(OUT / "paris_d03_activation_pilot.png", dpi=220)
    plt.close(fig)

    summary = {
        "version": "paris_d03_activation_pilot_v1",
        "mminlu": mminlu,
        "urban_class": URBAN_CLASS,
        "frc_threshold": FRC_THRESHOLD,
        "d075_core_threshold": D075_CORE_THRESHOLD,
        "d03_cells": int(len(df)),
        "d075_intersection_cells": int(d075.sum()),
        "d075_majority_core_cells": int(d075_core.sum()),
        "wps_urban_in_d075": int((wps & d075).sum()),
        "strict_active_in_d075": int((strict & d075).sum()),
        "primary_active_in_d075": int((primary & d075).sum()),
        "wps_vs_strict_conflicts_in_d075": int((d075 & (wps != strict)).sum()),
        "primary_changes_in_majority_core": int((d075_core & (wps != primary)).sum()),
        "primary_deactivations": int((d075_core & wps & ~primary).sum()),
        "primary_activations": int((d075_core & ~wps & primary).sum()),
        "outside_or_edge_background_urban_cells": int((primary & ~d075_core).sum()),
        "geo_sha256": sha256(GEO),
        "ucp_csv_sha256": sha256(UCP_CSV),
    }
    (OUT / "paris_d03_activation_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-01
- Verification Status: VERIFIED
- Version Label: paris_d03_activation_pilot_v1

# Paris WRF d03城市激活策略Pilot

## 结论

正式REAL/HOMO处理区采用“**D075占格面积至少50%的多数覆盖格**”。在该区域内，`FRC_URB2D_candidate >= 0.20`才激活distributed-drag；D075边缘碎片和d03外围维持共同背景，不参与REAL–HOMO处理。

这不是把D075行政边界直接作为WRF边界，而是只限制反事实处理区域。外围城市仍需共同UCP背景，不能全部改成农村。

## 三种候选

| 方案 | D075内激活格 | 用途 |
|---|---:|---|
| WPS原始 `LU_INDEX=13` | {summary['wps_urban_in_d075']} | benchmark |
| strict D075 `FRC>=0.20` | {summary['strict_active_in_d075']} | 敏感性；会误处理边缘碎片 |
| primary majority-core | {summary['primary_active_in_d075']} | **正式方案** |

primary只相对WPS修改{summary['primary_changes_in_majority_core']}格：停用{summary['primary_deactivations']}格，新增{summary['primary_activations']}格。被停用格使用原`LANDUSEF`中占比最大的非城市类别。

## 为什么不能只写FRC

WRF v4.7.1源码显示：

1. `real.exe`在distributed-drag路径中按`1 - SHDAVG/100`重算`FRC_URB2D`；
2. SLUCM是否激活由`IVGTYP == ISURBAN`决定；
3. `ZD_URB2D`在`real.exe`阶段加入模式地形，因此必须在`real.exe`之前存在；
4. `FRC_URB2D`应在`real.exe`之后按目标值再次写入并回读验证。

因此正式写入采用两段式：

```text
met_em之前/之中：土地利用激活 + MH/LP/LF/Z0/ZD
        ↓ real.exe
wrfinput之后：恢复目标FRC + 全字段回读验证
```

## 输出

- `paris_d03_activation_candidates.csv`
- `paris_d03_activation_candidates.npz`
- `paris_d03_wps_vs_strict_conflicts.csv`
- `paris_d03_primary_changed_cells.csv`
- `paris_d03_activation_pilot.png`
- `paris_d03_activation_pilot_summary.json`

## 尚未执行

- 未修改原`geo_em.d03.nc`；
- 未下载ERA5；
- 未生成`met_em`或`wrfinput`；
- 未把GloUCP外围参数转换为distributed-drag字段。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
