#!/usr/bin/env python3
"""Refresh question-led reports and other run notebooks without rebuilding simulation catalogues."""
import argparse
from pathlib import Path
from parity_study.tests.notebooks import write_run_notebook, write_overview_notebook
from parity_study.paths import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=Path, help='One saved run directory')
    group.add_argument('--all', action='store_true', help='All saved runs in the configured scratch output directory')
    p.add_argument('--rebuild', action='store_true',
                   help='Replace source cells with the current template; archive the old notebook in its matching output folder')
    p.add_argument('--overview', action='store_true', help='Also create or refresh an optional overview notebook')
    a = p.parse_args()
    results = ROOT/'results'
    runs = [a.run.resolve()] if a.run else sorted(r for r in OUTPUT_ROOT.iterdir() if (r/'config.json').is_file())
    for run in runs:
        write_run_notebook(run, rebuild=a.rebuild)
    if a.overview:
        write_overview_notebook(results)
