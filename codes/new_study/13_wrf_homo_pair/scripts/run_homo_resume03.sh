#!/usr/bin/env bash
set -euo pipefail

case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_homo_resume03_v1"
log_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/logs"
state_file="${log_dir}/paris_core48h.state"

for domain in 01 02 03; do
  test -s "${case_dir}/wrfrst_d${domain}_2022-07-19_03:00:00"
done

printf 'HOMO_RESUME03_RUNNING=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
cd "${case_dir}"
set +e
/usr/bin/time -v -o "${log_dir}/paris_homo_resume03.time.log" \
  mpirun -np 4 ./wrf.exe > "${log_dir}/paris_homo_resume03.stdout.log" 2>&1
status=$?
set -e
printf 'HOMO_RESUME03_EXIT=%s\n' "${status}" >> "${state_file}"
printf 'HOMO_RESUME03_FINISHED=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
if [[ "${status}" -eq 0 ]]; then
  printf 'FULL96H_FINISHED=%s\n' "$(date --iso-8601=seconds)" >> "${state_file}"
fi
exit "${status}"
