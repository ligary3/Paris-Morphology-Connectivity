# Workflow and scientific entry points

Run archived Python scripts through `python codes/run.py <path-relative-to-codes> [arguments]`. This sets import paths and the working directory expected by cross-stage imports. Consult each script's argument parser before using it on a new data layout.

| Component | Entry points under `codes/new_study/` | Notes |
|---|---|---|
| Building processing | `01_phase1a_morphology/scripts/prepare_paris_buildings.py`, `impute_paris_building_heights.py`, `build_imputed_morphology.py` | Inspect stage-specific defaults; early v1/observed-only products are retained as provenance |
| Peripheral building context | `03_extended_domain/scripts/prepare_extended_buildings.py`, `impute_extended_heights.py`, `build_extended_multidirection_morphology.py` in stage 16 | Peripheral morphology is unchanged between scenarios |
| Urban/non-urban mask | `05_urban_mask/scripts/build_paris_urban_support_mask.py` | Buffer-distance sensitivity is distinct from grid-resolution sensitivity |
| Shared morphology template | `07_real_homo_design/scripts/build_real_homo_design_pilot.py` | Conservation template is the experiment; the medoid is a reference comparison |
| WRF UCP construction | `09_wrf_ucp_d03/scripts/build_paris_d03_ucp.py`, `10_wrf_activation_pilot/scripts/build_injection_templates.py` | Requires real `geo_em.d03.nc` coordinates |
| Two-stage injection | `10_wrf_activation_pilot/scripts/wrf_ucp_injection.py` | `pre-real` writes activation/UCP; `post-real` restores FRC after real.exe |
| ERA5 / WPS | `11_era5_wps/scripts/download_era5_wrf.py` and WPS configuration template | CDS credentials stay in the user's local configuration; accept dataset terms separately |
| Four-direction networks | `14_circuit_real_homo/scripts/prepare_circuit_real_homo_pair.py`, `16_multidirection_circuit/scripts/prepare_multidirection_pairs.py`, `run_circuitscape_many.jl` | Network solves use Julia Circuitscape 5.15.0 |
| WRF response extraction | `13_wrf_homo_pair/scripts/analyze_full96h_regimes.py`, `17_transport_metric_candidates/scripts/export_3d_transport_metrics.py` | Existing completed outputs, not a fresh WRF integration |
| Main spatial prediction | `17_transport_metric_candidates/scripts/strict_upwind_context_test.py` | Final baseline: local + 3.5 km neighborhood + 5 km upwind sector, four directions, ten seeds |
| Permutation control | `17_transport_metric_candidates/scripts/permutation_nav_test.py` | Reuses the frozen primary model inputs |
| Enhanced / nonlinear controls | `21_review_controls/run_models.py` | Nested spatial inner tuning; additional within-cell FAI distribution descriptors |
| Resistance scale | `21_review_controls/prepare_q.py`, `run_q.jl`, `analyze_q.py` | 0.5q/q/2q; preparation + solve + analysis |
| LCZ benchmark | `19_lcz_benchmark/scripts/prepare_paris_lcz_100m.py`, `prepare_lcz_circuit_benchmark.py`, `compare_lcz_vs_3d_nav.py` | Local 3D class calibration; same HOMO reference; no LCZ–WRF integration |
| Weather validation | `18_wrf_observation_validation/scripts/validate_surface_stations.py`, `validate_trappes_profiles.py` | ISD and IGRA inputs required |
| Air-quality association | `20_air_quality_environment/scripts/prepare_airparif_event.py`, `analyze_air_quality_link.py` | Association, not a chemical-transport counterfactual |
| O₃ station exclusion | `21_review_controls/observations.py` | Recomputes demeaning per station subset; includes available/common-hour comparisons |

The earlier `15_network_added_value` and `16_multidirection_circuit/test_cross_direction_added_value.py` scripts contain reusable utilities and pilot comparisons. Do not confuse their early direction sets/seed counts with the final four-direction strict test in stage 17.

## WRF execution

WRF 4.7.1, WPS 4.7.0 and Julia/Circuitscape are separate external software installations, not Python requirements. Domain configuration and the 96 h reference namelist are in `configs/`. Archived initialization and restart shell scripts preserve the original local case layout; review them before use. The `namelist.input.real_full` is a 96 h reference configuration, not evidence that the recorded integrations were executed as one uninterrupted operating-system job. Both scenarios' output chronology and restart files are what establish coverage.

The two-stage UCP procedure must not be replaced with one injection:

```text
met_em → pre-real activation/UCP injection → real.exe
       → post-real FRC restoration → wrfinput → wrf.exe
```

Background wind direction comes from the common d02 Paris-surroundings field. Hourly WRF wind vectors are projected onto the associated transport-to vector. Wind-from and transport-to differ by 180°; Circuit selects the nearest 22.5° wind-from library direction.
