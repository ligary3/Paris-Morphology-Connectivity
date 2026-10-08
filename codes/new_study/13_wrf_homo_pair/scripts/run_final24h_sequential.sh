#!/usr/bin/env bash
set -euo pipefail

log_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/logs"
state_file="${log_dir}/paris_core48h.state"
mkdir -p "${log_dir}"

configure_final_day() {
  local case_dir="$1"
  local namelist="${case_dir}/namelist.input"

  test -f "${namelist}"
  for domain in 01 02 03; do
    test -s "${case_dir}/wrfrst_d${domain}_2022-07-19_00:00:00"
  done

  if [[ ! -f "${namelist}.pre_final24h" ]]; then
    cp -p "${namelist}" "${namelist}.pre_final24h"
  fi

  sed -i -E \
    -e 's/^[[:space:]]*run_days[[:space:]]*=.*/ run_days                            = 1,/' \
    -e 's/^[[:space:]]*run_hours[[:space:]]*=.*/ run_hours                           = 0,/' \
    -e 's/^[[:space:]]*start_day[[:space:]]*=.*/ start_day                           = 19,   19,   19,/' \
    -e 's/^[[:space:]]*start_hour[[:space:]]*=.*/ start_hour                          = 00,   00,   00,/' \
    -e 's/^[[:space:]]*end_day[[:space:]]*=.*/ end_day                             = 20,   20,   20,/' \
    -e 's/^[[:space:]]*end_hour[[:space:]]*=.*/ end_hour                            = 00,   00,   00,/' \
    -e 's/^[[:space:]]*restart[[:space:]]*=.*/ restart                             = .true.,/' \
    "${namelist}"
}

run_case() {
  local mode="$1"
  local lower_mode="${mode,,}"
  local case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_${lower_mode}_final24h_v1"
  local stem="paris_${lower_mode}_final24h"

  test -x "${case_dir}/wrf.exe"
  printf 'FINAL24H_%s_RUNNING=%s\n' "${mode}" "$(date --iso-8601=seconds)" >> "${state_file}"
  cd "${case_dir}"
  set +e
  /usr/bin/time -v -o "${log_dir}/${stem}.time.log" \
    mpirun -np 4 ./wrf.exe > "${log_dir}/${stem}.stdout.log" 2>&1
  status=$?
  set -e
  printf 'FINAL24H_%s_EXIT=%s\n' "${mode}" "${status}" >> "${state_file}"
  printf 'FINAL24H_%s_FINISHED=%s\n' "${mode}" "$(date --iso-8601=seconds)" >> "${state_file}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

grep -q '^CORE48H_FINISHED=' "${state_file}"
printf 'FINAL24H_START=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
run_case "REAL"
run_case "HOMO"
printf 'FULL96H_FINISHED=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
