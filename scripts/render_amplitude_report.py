#!/usr/bin/env python3
"""Render only the amplitude notebook from a completed, frozen experiment."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import parity_study.tests.amplitude_notebooks
import parity_study.tests.notebooks
import parity_study.tests.study_notebooks
from parity_study.tests.notebooks import write_run_notebook
from parity_study.tests.amplitude_scan import verify
from parity_study.storage import atomic_json,digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
run=parser.parse_args().run.resolve()
config,protocol,path=verify(run)
report=json.loads((path.parent/'analysis.json').read_text())
if not report['complete'] or report['protocol_sha256']!=digest(path):
    raise ValueError('Complete, matching analysis is required before rendering')
notebook_path=write_run_notebook(run,rebuild=True)
atomic_json(path.parent/'report_provenance.json',dict(run=str(run),
    manifest_sha256=digest(run/'manifest.json'),protocol_sha256=digest(path),
    notebook_sha256=digest(notebook_path),report_job=os.environ.get('SLURM_JOB_ID'),
    report_code_sha256={str(p):digest(p) for p in [Path(__file__).resolve(),
        Path(parity_study.tests.amplitude_notebooks.__file__),Path(parity_study.tests.notebooks.__file__),
        Path(parity_study.tests.study_notebooks.__file__)]},
    executed_at=datetime.now(ZoneInfo('Europe/Zurich')).isoformat()))
