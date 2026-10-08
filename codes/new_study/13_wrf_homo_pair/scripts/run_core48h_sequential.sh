#!/usr/bin/env bash
set -euo pipefail

log_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/logs"
state_file="${log_dir}/paris_core48h.state"
mkdir -p "${log_dir}"

run_case() {
  local mode="$1"
  local case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${mode,,}_core48h_v1"
  local stem="paris_${mode,,}_core48h"
  test -x "${case_dir}/wrf.exe"
  for domain in 01 02 03; do
    test -e "${case_dir}/wrfrst_d${domain}_2022-07-17_00:00:00"
  done
  printf '%s_RUNNING=%s\n' "${mode}" "$(date --iso-8601=seconds)" >> "${state_file}"
  cd "${case_dir}"
  set +e
  /usr/bin/time -v -o "${log_dir}/${stem}.time.log" \
    mpirun -np 4 ./wrf.exe > "${log_dir}/${stem}.stdout.log" 2>&1
  status=$?
  set -e
  printf '%s_EXIT=%s\n' "${mode}" "${status}" >> "${state_file}"
  printf '%s_FINISHED=%s\n' "${mode}" "$(date --iso-8601=seconds)" >> "${state_file}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

: > "${state_file}"
printf 'CORE48H_START=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
run_case "REAL"
run_case "HOMO"
printf 'CORE48H_FINISHED=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
