#!/usr/bin/env bash
set -euo pipefail

case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_real_init_v1"
log_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/logs"
mkdir -p "${log_dir}"

echo "CASE=${case_dir}"
for domain in 01 02 03; do
  count=$(find "${case_dir}" -maxdepth 1 -name "met_em.d${domain}.*.nc" | wc -l)
  echo "met_em.d${domain}=${count}"
  if [[ "${count}" -ne 97 ]]; then
    echo "ERROR: expected 97 met_em files for d${domain}" >&2
    exit 2
  fi
done

test -x "${case_dir}/real.exe"
test -f "${case_dir}/namelist.input"

rm -f "${case_dir}"/rsl.out.* "${case_dir}"/rsl.error.*

cd "${case_dir}"
echo "REAL_START=$(date --iso-8601=seconds)"
/usr/bin/time -v -o "${log_dir}/paris_real_init.time.log" \
  mpirun -np 5 ./real.exe > "${log_dir}/paris_real_init.stdout.log" 2>&1
status=$?
echo "REAL_END=$(date --iso-8601=seconds)"
echo "REAL_EXIT=${status}"
exit "${status}"
