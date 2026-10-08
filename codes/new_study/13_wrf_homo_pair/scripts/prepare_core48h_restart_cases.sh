#!/usr/bin/env bash
set -euo pipefail

wrf_run="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/src/WRF-4.7.1/run"
lookup_source="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_real_smoke6h_v1"
namelist_source="${PARIS_CODE_ROOT:?set PARIS_CODE_ROOT}/new_study/13_wrf_homo_pair/namelist.input.core48h_restart"

prepare_case() {
  local mode="$1"
  local previous="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${mode,,}_pair24h_v1"
  local case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${mode,,}_core48h_v1"

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
  cp "${namelist_source}" "${case_dir}/namelist.input"
  for domain in 01 02 03; do
    ln -s "${previous}/wrfrst_d${domain}_2022-07-17_00:00:00" \
      "${case_dir}/wrfrst_d${domain}_2022-07-17_00:00:00"
    ln -s "${previous}/wrflowinp_d${domain}" "${case_dir}/wrflowinp_d${domain}"
    ln -s "${previous}/wrfinput_d${domain}" "${case_dir}/wrfinput_d${domain}"
  done
  ln -s "${previous}/wrfbdy_d01" "${case_dir}/wrfbdy_d01"
  for lookup in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do
    ln -s "${lookup_source}/${lookup}" "${case_dir}/${lookup}"
  done
  echo "${mode}_CORE48H_CASE=${case_dir}"
}

prepare_case "REAL"
prepare_case "HOMO"

