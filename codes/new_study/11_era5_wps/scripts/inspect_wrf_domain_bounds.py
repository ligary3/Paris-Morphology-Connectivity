
from release_paths import resolve_path as _release_path
from pathlib import Path

import netCDF4


CASE = _release_path('/home/gary/wrf-work/cases/paris_geogrid_v1')

for domain in (1, 2, 3):
    path = CASE / f"geo_em.d0{domain}.nc"
    with netCDF4.Dataset(path) as ds:
        lat = ds.variables["XLAT_M"][0]
        lon = ds.variables["XLONG_M"][0]
        print(
            f"d0{domain}: north={float(lat.max()):.4f}, west={float(lon.min()):.4f}, "
            f"south={float(lat.min()):.4f}, east={float(lon.max()):.4f}"
        )
