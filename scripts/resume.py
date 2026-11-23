#!/usr/bin/env python3
"""Resubmit only seed blocks with missing records, using the original source snapshot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
from parity_study.storage import load_config, check_manifest, atomic_json
from parity_study.summarize import expected_cases
from parity_study.scheduler import resource_args
from parity_study.seeds import seed_ids, seed_start

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
run=a.run.resolve()
check_manifest(run)
config=load_config(run)
report=json.loads((run/'validation/report.json').read_text())
if not report.get('passed'):
    raise SystemExit('Validation must pass before resuming production.')
missing=set()
for case,n in expected_cases(config).items():
    missing.update(seed for seed in seed_ids(config, n) if not (run/'records'/case/f'seed_{seed:04d}.npz').exists())
blocks=sorted({(s-seed_start(config))//config['scheduler']['seeds_per_task'] for s in missing})
print('Incomplete blocks:',blocks)
if not blocks or a.dry_run:
    raise SystemExit(0)
# Refuse overlapping writes while any of this run's jobs remain active.
job_ids=[]
for path in [run/'jobs.json',*run.glob('resume-*.json')]:
    job_ids.extend(str(v) for k,v in json.loads(path.read_text()).items() if k in ('validate','ensemble','summary'))
active=subprocess.run(['squeue','--noheader','--jobs',','.join(job_ids),'--format=%i %T'],capture_output=True,text=True)
if active.stdout.strip():
    raise SystemExit('This run still has active jobs; wait for them before resuming:\n'+active.stdout)
s=config['scheduler']
cmd=['sbatch','--parsable','--job-name','pv-resume','--partition',s['partition'],
     '--cpus-per-task',str(s['cpus']),'--mem',s['memory'],'--time',s['time'],
     '--array',','.join(map(str,blocks))+f'%{s["concurrency"]}',
     '--output',str(run/'logs/resume-%A_%a.out'),'--error',str(run/'logs/resume-%A_%a.err'),
     str(run/'code/slurm/job.sh'),str(ROOT),str(run),'ensemble']
job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
summary=subprocess.check_output(['sbatch','--parsable','--job-name','pv-summary',
     *resource_args(s, 'summary'),'--dependency',f'afterany:{job}',
     '--output',str(run/'logs/summary-%A_%a.out'),'--error',str(run/'logs/summary-%A_%a.err'),
     str(run/'code/slurm/job.sh'),str(ROOT),str(run),'summary'],text=True).strip().split(';')[0]
now=datetime.now(ZoneInfo('Europe/Zurich'))
atomic_json(run/f'resume-{now:%Y%m%d_%H%M%S}.json',dict(ensemble=job,summary=summary,blocks=blocks))
print('Submitted ensemble',job,'and summary',summary)
