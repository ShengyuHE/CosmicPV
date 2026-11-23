#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
vendor = Path('vendor') if Path('vendor').is_dir() else Path('logs/vendor')
source=json.loads((vendor/'sources.json').read_text())['DIVE']
repo=vendor/'DIVE'
if not repo.exists():
    subprocess.run(['git','clone',source['url'],str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',source['commit']],check=True)
actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if actual != source['commit']:
    raise SystemExit(f'Existing {repo} has a different revision; inspect it before rebuilding.')
PY
bash scripts/build_dive.sh
