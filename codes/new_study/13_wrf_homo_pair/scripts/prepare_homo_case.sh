#!/usr/bin/env bash
set -euo pipefail

root="${PARIS_CODE_ROOT:?set PARIS_CODE_ROOT}"
wps_case="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_wps_era5_v1"
wrf_run="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/src/WRF-4.7.1/run"
case_dir="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/cases/paris_homo_init_v1"
python_bin="${PARIS_WRF_ROOT:?set PARIS_WRF_ROOT}/venvs/cdsapi/bin/python"
injector="${root}/new_study/10_wrf_activation_pilot/scripts/wrf_ucp_injection.py"
template="${PARIS_DATA_ROOT:?set PARIS_DATA_ROOT}/建筑形态/Paris/processed/phase1a_v2/wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz"
namelist="${root}/new_study/12_wrf_real_init/namelist.input.real_full"

if [[ -e "${case_dir}" ]]; then
  echo "ERROR: refusing to overwrite existing HOMO case: ${case_dir}" >&2
  exit 2
fi
mkdir -p "${case_dir}"

for source in "${wrf_run}"/*; do
  name=$(basename "${source}")
  [[ "${name}" == "namelist.input" ]] && continue
  ln -s "${source}" "${case_dir}/${name}"
done
cp "${namelist}" "${case_dir}/namelist.input"

for domain in 01 02; do
  for source in "${wps_case}"/met_em.d${domain}.*.nc; do
    ln -s "${source}" "${case_dir}/$(basename "${source}")"
  done
done

count=0
for source in "${wps_case}"/met_em.d03.*.nc; do
  output="${case_dir}/$(basename "${source}")"
  "${python_bin}" "${injector}" pre-real \
    --input "${source}" \
    --output "${output}" \
    --mode HOMO \
    --template "${template}"
  count=$((count + 1))
done

if [[ "${count}" -ne 97 ]]; then
  echo "ERROR: expected 97 HOMO d03 pre-real files, found ${count}" >&2
  exit 3
fi

echo "HOMO_CASE_PREPARED=${case_dir}"
echo "D03_PRE_REAL_COUNT=${count}"
