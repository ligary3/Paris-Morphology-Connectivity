# Release provenance and verification

This is a curated publication companion, not a copy of the entire working directory. Unrelated notebooks, manuscripts, authoring plugins, credentials, binary software, large WRF files, failed-run logs and deployment troubleshooting scripts are excluded.

`source_manifest.csv` records the included source files using repository-relative provenance and SHA-256 checksums. Archived scientific Python files are changed only to relocate absolute `Path(...)` constants and to replace a plotting-authoring-only geometry recorder with a release-local recorder. Shell path substitutions use environment variables. Numerical algorithms, data selection and stored statistical values are not changed.

The manuscript figure notebook embeds the shared helpers, data-loading block and final revision functions from the existing figure scripts. It is for data figures; the PowerPoint workflow and generated REAL–HOMO conceptual illustration are assets, not falsely presented as numerical plotting code.

Verification scope:

- Python source and notebook code syntax.
- Frozen array dimensions, four directions and 85 treatment cells.
- Existing 40-comparison primary RMSE summary and reference records.
- Release file checksums and credential/private-file scan.
- Lightweight numerical reproduction of the packaged primary spatial-CV test, when listed in `release_checks.json` as completed.

No fresh WRF integration, ERA5 download, full building processing, full figure rendering or new scientific sensitivity experiment is implied. The release checks file records which checks actually ran. Full-workflow dependency ranges are installation guidance, not a claim of a tested lockfile for every geospatial component.
