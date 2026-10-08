# Distributed data and external sources

This repository distributes **derived statistical inputs and results**, not the original third-party datasets or full WRF output.

## Files

- `analysis_inputs/strict_model_inputs.npz`: direction-specific 12-feature baseline, Circuit predictor, projected WRF response and EPSG:2154 coordinates of 85 treatment cells. Keys follow `<quantity>_<direction*10 padded to four digits>`; targets are `y_10_1000_*` and alternative vertical-range targets.
- `analysis_inputs/enhanced_model_inputs.npz`: saved enhanced 20-feature inputs and matching 13/21-feature network comparisons.
- `results/primary/`: original direction/seed/vertical-metric RMSE tests and summaries.
- `results/controls/`: original enhanced/nonlinear metrics, paired comparisons and tuning metadata.
- `results/network/`: four-direction REAL–HOMO similarity and q-sensitivity results.
- `results/observations/`: original surface/profile validation statistics, lag results and O₃ station-exclusion sensitivities.

`rmse_baseline` and `rmse_circuit` are in m/s. In the archived primary table, `relative_rmse_change = (RMSE_circuit − RMSE_baseline)/RMSE_baseline`, so negative values mean improvement. The standalone reproduction reports `reduction_pct = 100*(RMSE_baseline − RMSE_circuit)/RMSE_baseline`, so positive values mean improvement. This sign convention is explicit, not a change in results.

All coordinates describe the public study grid/monitoring network, not personal locations. Timestamps are UTC. O₃/NO₂ source observations were mass concentrations; Ox is derived after molar conversion. Consult column names and original metadata for association and interval definitions.

## Access routes

Obtain BD TOPO, ERA5, Cerema LCZ, NOAA ISD/IGRA and Airparif observations from the official links in the main README under their providers' applicable conditions. This release does not grant rights to redistribute those source datasets. WRF/WPS/Circuitscape retain their upstream software licenses. API credentials must remain local and must never be committed.

No article DOI, data DOI, embargo, or unrestricted raw-data availability is asserted by this code release.
