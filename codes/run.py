"""Run an archived Python entry point with its original relative layout."""
import argparse
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('script', help='Path relative to codes/, e.g. new_study/07_real_homo_design/scripts/build_real_homo_design_pilot.py')
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path(__file__).resolve().parent
target = (root / args.script).resolve()
if not target.is_relative_to(root) or not target.is_file():
    parser.error('Choose an existing Python script inside codes/.')
sys.path[:0] = [str(target.parent), str(root)]
os.chdir(root)
sys.argv = [str(target)] + args.arguments
runpy.run_path(str(target), run_name='__main__')
