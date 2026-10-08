#!/usr/bin/env bash
set -euo pipefail

ROOT="${PARIS_CODE_ROOT:?set PARIS_CODE_ROOT}"
WPS_CASE="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_wps_era5_v1"
WRF_RUN="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/src/WRF-4.7.1/run"
CASE_DIR="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_real_init_v1"
PYTHON="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/venvs/cdsapi/bin/python"
INJECTOR="$ROOT/new_study/10_wrf_activation_pilot/scripts/wrf_ucp_injection.py"
TEMPLATE="${PARIS_DATA_ROOT:?set PARIS_DATA_ROOT}/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz"
NAMELIST="$ROOT/new_study/12_wrf_real_init/namelist.input.real_full"

if [[ -e "$CASE_DIR" ]]; then
    echo "REFUSING_EXISTING_CASE $CASE_DIR" >&2
    exit 2
fi
mkdir -p "$CASE_DIR"

for source in "$WRF_RUN"/*; do
    name=$(basename "$source")
    [[ "$name" == "namelist.input" ]] && continue
    ln -s "$source" "$CASE_DIR/$name"
done
cp "$NAMELIST" "$CASE_DIR/namelist.input"

for domain in 01 02; do
    for source in "$WPS_CASE"/met_em.d${domain}.*.nc; do
        ln -s "$source" "$CASE_DIR/$(basename "$source")"
    done
done

count=0
for source in "$WPS_CASE"/met_em.d03.*.nc; do
    output="$CASE_DIR/$(basename "$source")"
    "$PYTHON" "$INJECTOR" pre-real \
        --input "$source" \
        --output "$output" \
        --mode REAL \
        --template "$TEMPLATE"
    count=$((count + 1))
done

if [[ "$count" -ne 97 ]]; then
    echo "UNEXPECTED_D03_PRE_REAL_COUNT $count" >&2
    exit 3
fi

echo "REAL_CASE_PREPARED $CASE_DIR"
echo "D03_PRE_REAL_COUNT $count"
