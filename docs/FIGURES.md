# Final manuscript figures

The current [framework](../images/framework.png) and [study-area map](../images/study_area.png) are the figures used by the manuscript. The [editable PowerPoint](../images/framework.pptx) can be used without a JavaScript environment.

## Figure 1
Run `node codes/figures/build_workflow.mjs` with Node.js and the authoring runtime `@oai/artifact-tool` available, or set `ARTIFACT_TOOL_MODULE` to the module's import URL. This runtime is optional for the scientific analyses. The six raster panels under `images/workflow_assets` are embedded unchanged. The REAL and HOMO building illustrations are conceptual, not measurements.

## Figure 2
Install the full Python dependencies. Copy the four files in `datas/study_area` into a local folder and set `PARIS_MAP_CACHE` to that folder. Add the Natural Earth `ne_50m_admin_0_countries.zip` file and the prepared DEM at `dem/paris_glo30_lamb93_50m.tif`. Set `PARIS_NATURAL_EARTH` to the cache containing the 110m physical land and cultural boundary shapefiles. Set `PARIS_BDTOPO_GPKG` to the June 2022 D075 BD TOPO GeoPackage. Run `python codes/figures/render_study_area.py`. Both scripts write to `outputs/figures/`. No WRF runs are required.
