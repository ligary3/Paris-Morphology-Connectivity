"""Relocate archived paths without changing numerical experiment logic."""
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY / 'codes'
DATA_ROOT = Path(os.environ.get('PARIS_DATA_ROOT', REPOSITORY / 'datas' / 'external'))
WRF_ROOT = Path(os.environ.get('PARIS_WRF_ROOT', REPOSITORY / 'datas' / 'wrf-work'))

def resolve_path(value):
    text = str(value).replace('\\', '/')
    roots = (
        ('D:/研究生/代码/论文4/lunwen4', CODE_ROOT),
        ('/mnt/d/研究生/代码/论文4/lunwen4', CODE_ROOT),
        ('E:/lunwen4/2', DATA_ROOT),
        ('/mnt/e/lunwen4/2', DATA_ROOT),
        ('/home/gary/wrf-work', WRF_ROOT),
        ('C:/Users/12169/.local/share/cartopy', Path(os.environ.get('PARIS_CARTOPY_CACHE', REPOSITORY / 'datas' / 'cartopy'))),
        ('C:/Users/12169/.codex/skills/nature-figure/scripts', CODE_ROOT),
    )
    for prefix, root in roots:
        if text == prefix or text.startswith(prefix + '/'):
            return root / text[len(prefix):].lstrip('/')
    return Path(value)
