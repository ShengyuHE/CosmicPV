#!/usr/bin/env python3
"""Freeze source, configuration and DIVE, then submit validation -> array -> summary."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np
from parity_study.storage import atomic_json, digest
from parity_study.catalogue import check_amplitude_convention
from parity_study.paths import OUTPUT_ROOT, expose_output
from parity_study.scheduler import resource_args
from parity_study.seeds import seed_start

ROOT = Path(__file__).resolve().parents[1]


def resource(name):
    """Support the project inputs relocated by the user into logs/."""
    direct = ROOT/name
    return direct if direct.exists() else ROOT/'logs'/name


def submit(args):
    cfg = json.loads(args.config.read_text())
    seed_start(cfg)
    spectrum = cfg['field'].get('spectrum', {})
    if spectrum.get('kind') == 'table' and 'path' in spectrum:
        source = (args.config.parent/spectrum['path']).resolve()
        values = np.loadtxt(source, delimiter=spectrum.get('delimiter'), ndmin=2)
        if values.shape[1] != 2:
            raise ValueError('Input P(k) table must have exactly two columns: k, P(k)')
        cfg['field']['spectrum'] = dict(kind='table', k=values[:,0].tolist(), p=values[:,1].tolist(),
                                        source=str(source), source_sha256=digest(source))
    convention = cfg['field'].get('amplitude_convention')
    if convention is None:
        raise ValueError("Set field.amplitude_convention explicitly to 'rms' (A is an RMS length) "
                         "or 'notebook' (the original per-mode power amplitude).")
    check_amplitude_convention(convention)
    model = cfg['field'].get('model', 'poisson_helical')
    if model == 'mesh_za_helical':
        from parity_study.initial_conditions import validate_field
        validate_field(cfg['field'])
        if cfg['positive']['seeds'] or cfg['null'].get('include_original_L1_A1'):
            raise ValueError('Use separate legacy runs for molecule/original-Poisson controls')
    elif model != 'poisson_helical':
        raise ValueError(f'Unknown catalogue model: {model}')
    now = datetime.now(ZoneInfo('Europe/Zurich'))
    name = args.name or f"{now:%Y%m%d_%H%M%S}_{args.config.stem}"
    if Path(name).name != name:
        raise ValueError('Run name must be a single directory name')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run = OUTPUT_ROOT/name
    run.mkdir(exist_ok=False)
    expose_output(ROOT, run)
    code = run/'code'
    code.mkdir()
    for folder in ['src','tests','slurm','scripts','patches']:
        shutil.copytree(resource(folder),code/folder,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    (code/'bin').mkdir()
    shutil.copy2(ROOT/'build/dive/DIVE',code/'bin/DIVE')
    shutil.copy2(ROOT/'pyproject.toml',code/'pyproject.toml')
    (code/'tp4a-pv-delaunay').mkdir()
    shutil.copy2(ROOT/'tp4a-pv-delaunay/parity_injection.ipynb',code/'tp4a-pv-delaunay/parity_injection.ipynb')
    shutil.copy2(resource('vendor')/'sources.json',code/'sources.json')
    (code/'vendor-source').mkdir()
    for filename in ['DIVE.cpp','CMakeLists.txt','LICENSE.txt']:
        shutil.copy2(resource('vendor')/'DIVE'/filename,code/'vendor-source'/filename)
    (run/'logs').mkdir()
    atomic_json(run/'config.json',cfg)
    hashes={str(p.relative_to(run)):digest(p) for p in sorted(code.rglob('*')) if p.is_file()}
    hashes['config.json']=digest(run/'config.json')
    manifest=dict(created=now.isoformat(),project=str(ROOT),python=sys.version,
                  executable=str(ROOT/'.venv/bin/python'),
                  packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','finufft','matplotlib','pytest','nbformat','nbclient','ipykernel']},
                  notebook_commit=subprocess.check_output(['git','-C',str(ROOT/'tp4a-pv-delaunay'),'rev-parse','HEAD'],text=True).strip(),
                  output_root=str(OUTPUT_ROOT),
                  dive_source=json.loads((resource('vendor')/'sources.json').read_text()),sha256=hashes)
    atomic_json(run/'manifest.json',manifest)
    if args.prepare_only:
        print(run)
        return
    sch=cfg['scheduler']
    ntasks=math.ceil(max(cfg[x]['seeds'] for x in ['null','scan','positive'])/sch['seeds_per_task'])
    jobs={}
    def sbatch(stage, extra):
        cmd=['sbatch','--parsable','--job-name',f'pv-{stage}',
             '--chdir',str(ROOT),'--output',str(run/'logs'/f'{stage}-%A_%a.out'),
             '--error',str(run/'logs'/f'{stage}-%A_%a.err'),*extra,
             str(code/'slurm/job.sh'),str(ROOT),str(run),stage]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        jobs[stage]=job
        atomic_json(run/'jobs.json',jobs)
        print(f'{stage}: {job}',flush=True)
        return job
    validation=sbatch('validate',resource_args(sch, 'validate'))
    production=sbatch('ensemble',resource_args(sch, 'ensemble') + [
                    '--array',f'0-{ntasks-1}%{sch["concurrency"]}',
                    '--dependency',f'afterok:{validation}','--kill-on-invalid-dep=yes'])
    summary=sbatch('summary',resource_args(sch, 'summary') + [
                      '--dependency',f'afterany:{production}'])
    if cfg.get('purpose') == 'amplitude_scan':
        sbatch('amplitude-report',resource_args(sch, 'summary') + [
            '--dependency',f'afterok:{summary}','--kill-on-invalid-dep=yes'])
    elif cfg.get('purpose') == 'best_configuration_confirmation':
        sbatch('confirmation-report',resource_args(sch, 'summary') + [
            '--dependency',f'afterok:{summary}','--kill-on-invalid-dep=yes'])
    print(f'run: {run}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=resource('configs')/'ic-smoke.json')
    p.add_argument('--name')
    p.add_argument('--prepare-only',action='store_true')
    submit(p.parse_args())
