import hashlib
import json
import os
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_config(run):
    return json.loads((Path(run)/'config.json').read_text())


def check_manifest(run):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(run/name) != expected:
            raise RuntimeError(f'Run input changed: {name}. Prepare a new run.')
    return manifest
