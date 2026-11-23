#!/usr/bin/env python3
"""Copy compatible completed mesh measurements into a larger frozen ensemble."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from parity_study.initial_conditions import displace_mesh
from parity_study.storage import atomic_json, check_manifest, digest, load_config
from parity_study.seeds import seed_ids
from parity_study.summarize import expected_cases


def reuse(run):
    config = load_config(run)
    if 'reuse_from' not in config:
        return
    source = Path(config['reuse_from']).resolve()
    old = load_config(source)
    check_manifest(run)
    check_manifest(source)
    if config['field'] != old['field'] or config['field'].get('model') != 'mesh_za_helical':
        raise ValueError('Reusing a complete ensemble requires identical mesh field parameters')
    for name in ('initial_conditions.py', 'catalogue.py', 'delaunay.py'):
        rel = Path('code/src/parity_study')/name
        if digest(run/rel) != digest(source/rel):
            raise ValueError(f'Construction source differs: {name}')
    if digest(run/'code/bin/DIVE') != digest(source/'code/bin/DIVE'):
        raise ValueError('DIVE executables differ')
    if not json.loads((source/'validation/fixed_ic_ensemble.json').read_text())['passed']:
        raise ValueError('Source ensemble has not passed its fixed-IC audit')
    if config['null']['A'] != old['null']['A'] or config['scan']['amplitudes'] != old['scan']['amplitudes']:
        raise ValueError('Catalogue amplitudes differ')
    copied = []
    def copy(rel):
        a, b = source/rel, run/rel
        sha = digest(a)
        b.parent.mkdir(parents=True, exist_ok=True)
        if b.exists():
            if digest(b) != sha:
                raise ValueError(f'Refusing to replace a different existing file: {b}')
        else:
            shutil.copy2(a, b)
        if digest(b) != sha:
            raise IOError(f'Copy checksum failed: {b}')
        copied.append(dict(path=str(rel), sha256=sha))
    cases = expected_cases(config)
    maximum = max(config['null']['seeds'], config['scan']['seeds'])
    old_seeds = set(seed_ids(old, max(old['null']['seeds'],old['scan']['seeds'])))
    reused = []
    for seed in seed_ids(config, maximum):
        if seed not in old_seeds:
            continue
        rel = Path('fields')/f'main_{seed:04d}.npz'
        with np.load(source/rel) as z:
            arrays = {k:z[k] for k in ('q','za','plus','minus')}
            meta = json.loads(str(z['metadata_json']))
        if meta['field'] != config['field'] or meta['helical_seed'] != seed:
            raise ValueError('Source field cache does not match its seed or parameters')
        copy(rel)
        for case,count in cases.items():
            if seed not in seed_ids(config,count):
                continue
            rel = Path('records')/case/f'seed_{seed:04d}.npz'
            with np.load(source/rel) as z:
                record = json.loads(str(z['metadata_json']))
                positions = z['catalogue']
            p = record['parameters']
            if record['seed'] != seed or p['density_seed'] != config['field']['density_seed']:
                raise ValueError('Source catalogue metadata does not match')
            expected, _ = displace_mesh(arrays,A=p['A'],eps=p['eps'],L=config['field']['L'])
            np.testing.assert_array_equal(positions,expected)
            copy(rel)
        atomic_json(run/'completed'/f'seed_{seed:04d}.json',dict(seed=seed,completed=True,reused_from=str(source)))
        reused.append(seed)
    # Reuse the costly validation field; the validation stage reruns its geometry tests.
    copy(Path('fields/validation_mesh_0042.npz'))
    atomic_json(run/'reused_inputs.json',dict(source=str(source),reused_seeds=reused,files=copied))
    print(f'Reused {len(reused)} seeds and {len(copied)} cached files from {source.name}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    reuse(parser.parse_args().run.resolve())
