#!/usr/bin/env python3
"""Execute the fiducial notebook, retaining report code and previous provenance."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import os

from parity_study.paths import OUTPUT_ROOT, selection_output, independent_ic_output
from parity_study.storage import atomic_json, digest
from parity_study.tests.notebooks import write_run_notebook

project = Path(__file__).resolve().parents[1]
out = OUTPUT_ROOT/'field_PVdisp'
run = OUTPUT_ROOT/'20261121_mesh_ic_L1000_N256_fixedIC_10'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--selection-dir',type=Path,default=selection_output(out))
parser.add_argument('--independent-ic-dir',type=Path,default=independent_ic_output(out))
args = parser.parse_args()
selection = args.selection_dir.resolve()
relative_selection = selection.relative_to(out.resolve())
confirmation = json.loads((selection/'confirmation.json').read_text())
if digest(selection/'protocol.json') != confirmation['protocol_sha256']:
    raise ValueError('Selection summary does not match its protocol')
independent = args.independent_ic_dir.resolve() if args.independent_ic_dir else None
independent_report = None
if independent is not None:
    relative_independent = independent.relative_to(out.resolve())
    independent_report = json.loads((independent/'confirmation.json').read_text())
    if digest(independent/'protocol.json') != independent_report['protocol_sha256']:
        raise ValueError('Independent-IC summary does not match its protocol')
    if not independent_report['independent_ic_audit']['passed']:
        raise ValueError('Independent-IC audit has not passed')
stamp = datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y%m%d_%H%M%S_%f')
snapshot = out/'report_code'/stamp
snapshot.mkdir(parents=True)
shutil.copytree(project/'src/parity_study', snapshot/'parity_study',
                ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(Path(__file__), snapshot/Path(__file__).name)
sources = {str(p.relative_to(project)):digest(p) for p in (project/'src/parity_study').glob('*.py')}
atomic_json(snapshot/'manifest.json', sources)
provenance = out/'report_provenance.json'
old = json.loads(provenance.read_text()) if provenance.exists() else {}
if provenance.exists():
    shutil.copy2(provenance, snapshot/'previous_report_provenance.json')
if (out/'figures').exists():
    shutil.copytree(out/'figures',snapshot/'previous_figures')
pointer = out/'active_selection.json'
old_pointer = json.loads(pointer.read_text()) if pointer.exists() else None
atomic_json(pointer,dict(directory=relative_selection.as_posix(),protocol_sha256=confirmation['protocol_sha256']))
independent_pointer = out/'active_independent_ic.json'
old_independent_pointer = json.loads(independent_pointer.read_text()) if independent_pointer.exists() else None
if independent is not None:
    atomic_json(independent_pointer,dict(directory=relative_independent.as_posix(),
                                        protocol_sha256=independent_report['protocol_sha256']))
try:
    path = write_run_notebook(run, rebuild=True)
except Exception:
    if old_pointer is None:
        pointer.unlink()
    else:
        atomic_json(pointer,old_pointer)
    if old_independent_pointer is None:
        independent_pointer.unlink(missing_ok=True)
    else:
        atomic_json(independent_pointer,old_independent_pointer)
    raise
for name, expected in sources.items():
    if digest(project/name) != expected:
        raise RuntimeError(f'Report source changed during execution: {name}')
extra = dict(independent_ic_run=independent_report['confirmation_run'],
             independent_ic_realizations=independent_report['primary']['n'],
             independent_ic_protocol_sha256=independent_report['protocol_sha256']) if independent_report else {}
atomic_json(provenance, dict(old, source_run=run.name,
    confirmation_run=confirmation['confirmation_run'],
    confirmation_realizations=confirmation['primary']['n'],
    selection_protocol_sha256=confirmation['protocol_sha256'],
    report_job=os.environ.get('SLURM_JOB_ID'), code_snapshot=str(snapshot),
    notebook_sha256=digest(path), executed_at=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(), **extra))
