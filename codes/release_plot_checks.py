"""Record panel geometry for exported figures.

This is a release-local, dependency-free geometry recorder. It does not claim
to reproduce the original authoring environment's collision/alignment audit.
"""
import json
from pathlib import Path

def require_matplotlib_panel_alignment(fig, axes, panel_ids=None, json_out=None, overlay_svg=None, strict=False, **kwargs):
    records = [{'panel': str(i if panel_ids is None else panel_ids[i]),
                'bounds': list(ax.get_position().bounds)} for i,ax in enumerate(axes)]
    if json_out is not None:
        target=Path(json_out)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps({'check':'geometry_record_only','panels':records},indent=2),encoding='utf-8')
    return records
