#!/usr/bin/env bash
set -euo pipefail

wrf_run="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/src/WRF-4.7.1/run"
lookup_source="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_real_smoke6h_v1"

prepare_case() {
  local mode="$1"
  local suffix="$2"
  local run_hours="$3"
  local end_hour="$4"
  local previous="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${mode,,}_core48h_v1"
  local case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${mode,,}_${suffix}_v1"

  if [[ -e "${case_dir}" ]]; then
    echo "ERROR: refusing to overwrite ${case_dir}" >&2
    exit 2
  fi
  mkdir -p "${case_dir}"

  for item in "${wrf_run}"/*; do
    name=$(basename "${item}")
    case "${name}" in
      namelist.input|freezeH2O.dat|qr_acr_qg_V4.dat|qr_acr_qsV2.dat) continue ;;
    esac
    ln -s "${item}" "${case_dir}/${name}"
  done

  cp "${previous}/namelist.input" "${case_dir}/namelist.input"
  sed -i -E \
    -e 's/^[[:space:]]*run_days[[:space:]]*=.*/ run_days                            = 0,/' \
    -e "s/^[[:space:]]*run_hours[[:space:]]*=.*/ run_hours                           = ${run_hours},/" \
    -e 's/^[[:space:]]*start_day[[:space:]]*=.*/ start_day                           = 19,   19,   19,/' \
    -e 's/^[[:space:]]*start_hour[[:space:]]*=.*/ start_hour                          = 00,   00,   00,/' \
    -e 's/^[[:space:]]*end_day[[:space:]]*=.*/ end_day                             = 19,   19,   19,/' \
    -e "s/^[[:space:]]*end_hour[[:space:]]*=.*/ end_hour                            = ${end_hour},   ${end_hour},   ${end_hour},/" \
    -e 's/^[[:space:]]*restart[[:space:]]*=.*/ restart                             = .true.,/' \
    "${case_dir}/namelist.input"

  if [[ "${suffix}" == "final24h" ]]; then
    sed -i -E \
      -e 's/^[[:space:]]*run_days[[:space:]]*=.*/ run_days                            = 1,/' \
      -e 's/^[[:space:]]*run_hours[[:space:]]*=.*/ run_hours                           = 0,/' \
      -e 's/^[[:space:]]*end_day[[:space:]]*=.*/ end_day                             = 20,   20,   20,/' \
      -e 's/^[[:space:]]*end_hour[[:space:]]*=.*/ end_hour                            = 00,   00,   00,/' \
      "${case_dir}/namelist.input"
  fi

  for domain in 01 02 03; do
    ln -s "${previous}/wrfrst_d${domain}_2022-07-19_00:00:00" \
      "${case_dir}/wrfrst_d${domain}_2022-07-19_00:00:00"
    ln -s "${previous}/wrflowinp_d${domain}" "${case_dir}/wrflowinp_d${domain}"
    ln -s "${previous}/wrfinput_d${domain}" "${case_dir}/wrfinput_d${domain}"
  done
  ln -s "${previous}/wrfbdy_d01" "${case_dir}/wrfbdy_d01"
  for lookup in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do
    ln -s "${lookup_source}/${lookup}" "${case_dir}/${lookup}"
  done
  echo "PREPARED=${case_dir}"
}

prepare_case "REAL" "restart19_test1h" 1 01
prepare_case "REAL" "final24h" 0 00
prepare_case "HOMO" "final24h" 0 00
