#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/data/ERA5/paris_20220716_20"
WPS_DIR="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/src/WPS-4.7.0"
GEOGRID_CASE="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_geogrid_v1"
CASE_DIR="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_wps_era5_v1"
STAGE_DIR="$CASE_DIR/grib_stage"
EXPECTED_WPS_TIMES=97

pressure_files=("$DATA_DIR"/era5_pressure_202207{16,17,18,19,20}.grib)
single_files=("$DATA_DIR"/era5_single_202207{16,17,18,19,20}.grib)
all_files=("${pressure_files[@]}" "${single_files[@]}")

missing=0
for path in "${all_files[@]}"; do
    if [[ ! -s "$path" ]]; then
        echo "MISSING_OR_EMPTY $path" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "ERA5_INPUT_CHECK_FAILED" >&2
    exit 2
fi

mkdir -p "$CASE_DIR" "$STAGE_DIR" "$CASE_DIR/metgrid"
rm -f "$STAGE_DIR"/*.grib
for path in "${all_files[@]}"; do
    ln -sf "$path" "$STAGE_DIR/$(basename "$path")"
done

cp "$GEOGRID_CASE/namelist.wps" "$CASE_DIR/namelist.wps"
for domain in 01 02 03; do
    ln -sf "$GEOGRID_CASE/geo_em.d${domain}.nc" "$CASE_DIR/geo_em.d${domain}.nc"
done
ln -sf "$WPS_DIR/ungrib.exe" "$CASE_DIR/ungrib.exe"
ln -sf "$WPS_DIR/metgrid.exe" "$CASE_DIR/metgrid.exe"
ln -sf "$WPS_DIR/ungrib/Variable_Tables/Vtable.ERA-interim.pl" "$CASE_DIR/Vtable"
ln -sf "$WPS_DIR/metgrid/METGRID.TBL.ARW" "$CASE_DIR/metgrid/METGRID.TBL"

cd "$CASE_DIR"
rm -f GRIBFILE.??? ERA5:* met_em.d0*.nc ungrib.log metgrid.log
"$WPS_DIR/link_grib.csh" "$STAGE_DIR"/*.grib

echo "RUN_UNGRIB $(date --iso-8601=seconds)"
./ungrib.exe
grep -Eq "Successful completion.*ungrib" ungrib.log

era5_count=$(find . -maxdepth 1 -type f -name 'ERA5:*' | wc -l)
echo "ERA5_INTERMEDIATE_COUNT $era5_count"
if [[ "$era5_count" -ne "$EXPECTED_WPS_TIMES" ]]; then
    echo "UNEXPECTED_ERA5_TIME_COUNT expected=$EXPECTED_WPS_TIMES actual=$era5_count" >&2
    exit 3
fi

echo "RUN_METGRID $(date --iso-8601=seconds)"
./metgrid.exe
grep -Eq "Successful completion.*metgrid" metgrid.log

for domain in 01 02 03; do
    count=$(find . -maxdepth 1 -type f -name "met_em.d${domain}.*.nc" | wc -l)
    echo "MET_EM_D${domain}_COUNT $count"
    if [[ "$count" -ne "$EXPECTED_WPS_TIMES" ]]; then
        echo "UNEXPECTED_METGRID_TIME_COUNT domain=$domain expected=$EXPECTED_WPS_TIMES actual=$count" >&2
        exit 4
    fi
done

first_file=$(find . -maxdepth 1 -type f -name 'met_em.d03.*.nc' | sort | head -n 1)
last_file=$(find . -maxdepth 1 -type f -name 'met_em.d03.*.nc' | sort | tail -n 1)
echo "FIRST_D03 $first_file"
echo "LAST_D03 $last_file"
echo "WPS_ERA5_PIPELINE_PASS"
