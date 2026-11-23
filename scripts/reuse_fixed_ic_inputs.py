#!/usr/bin/env python3
"""Reuse compatible density/helical components in a newly prepared fixed-IC run."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from parity_study.initial_conditions import displace_mesh
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest, load_config
from parity_study.summarize import expected_cases


def reuse(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    check_manifest(source)
    check_manifest(target)
    if source == target or (target/'fields').exists() or (target/'records').exists():
        raise ValueError('Target must be a new run without fields or records')
    old, config = load_config(source), load_config(target)
    field = config['field']
    fixed_seed = field.get('density_seed')
    if field.get('model') != 'mesh_za_helical' or fixed_seed is None:
        raise ValueError('Target requires mesh initial conditions and a fixed density_seed')
    source_shape = {k:v for k,v in old['field'].items() if k != 'density_seed'}
    target_shape = {k:v for k,v in field.items() if k != 'density_seed'}
    if source_shape != target_shape:
        raise ValueError('Source and target field parameters differ beyond density_seed')
    if digest(source/'code/bin/DIVE') != digest(target/'code/bin/DIVE'):
        raise ValueError('Reused catalogue measurements require the same DIVE executable')
    density_path = source/'fields'/f'main_{fixed_seed:04d}.npz'
    with np.load(density_path) as z:
        density_meta = json.loads(str(z['metadata_json']))
        base = {k:z[k] for k in ('q', 'za', 'delta_mesh')}
    if density_meta['field'] != old['field'] or density_meta.get('density_seed', density_meta['seed']) != fixed_seed:
        raise ValueError('Source density cache does not match the requested density seed')
    density_hash = digest(density_path)
    report = dict(source_run=str(source), source_manifest_sha256=digest(source/'manifest.json'),
                  density_seed=fixed_seed, density_source=str(density_path),
                  density_source_sha256=density_hash, fields=[], records=[])
    maximum = max(config['null']['seeds'], config['scan']['seeds'])
    cases = expected_cases(config)
    specifications = {'null_za': (0., 0.), 'null_symmetric': (config['null']['A'], 0.)}
    specifications.update({f'scan_A{A:g}_{suffix}': (A, eps)
                           for A in config['scan']['amplitudes']
                           for suffix, eps in (('plus', 1.), ('minus', -1.))})
    from parity_study.seeds import seed_ids
    for seed in seed_ids(config, maximum):
        path = source/'fields'/f'main_{seed:04d}.npz'
        if not path.exists():
            continue
        started = time.monotonic()
        with np.load(path) as z:
            helical_meta = json.loads(str(z['metadata_json']))
            if helical_meta['field'] != old['field'] or helical_meta['seed'] != seed:
                raise ValueError(f'Incompatible source cache: {path}')
            np.testing.assert_array_equal(z['q'], base['q'])
            arrays = dict(base, plus=z['plus'], minus=z['minus'])
        source_hash = digest(path)
        provenance = dict(density=dict(path=str(density_path), sha256=density_hash),
                          helical=dict(path=str(path), sha256=source_hash))
        meta = dict(density_meta, field=field, seed=seed, density_seed=fixed_seed,
                    helical_seed=seed, fixed_density=True, plus=helical_meta['plus'],
                    minus=helical_meta['minus'], reused_components=provenance,
                    rng='scalar: SeedSequence(density_seed).spawn(3)[0]; '
                        'helical: SeedSequence(seed).spawn(3)[1:3]',
                    generation_seconds=time.monotonic()-started)
        atomic_npz(target/'fields'/path.name, **arrays,
                   metadata_json=np.array(json.dumps(meta, allow_nan=False)))
        report['fields'].append(dict(seed=seed, **provenance))
        # Only a source catalogue with this same density AND helical seed is reusable.
        if helical_meta.get('density_seed', seed) != fixed_seed:
            continue
        for case, count in cases.items():
            record = source/'records'/case/f'seed_{seed:04d}.npz'
            if seed not in seed_ids(config, count) or not record.exists():
                continue
            with np.load(record) as z:
                saved = {k:z[k] for k in z.files if k != 'metadata_json'}
                rm = json.loads(str(z['metadata_json']))
            params = rm['parameters']
            if rm['seed'] != seed or rm['case'] != case or rm['L'] != field['L']:
                raise ValueError(f'Incompatible catalogue record: {record}')
            if (params['A'], params['eps']) != specifications[case]:
                continue
            expected, _ = displace_mesh(arrays, A=params['A'], eps=params['eps'], L=field['L'])
            np.testing.assert_array_equal(saved['catalogue'], expected)
            rm['parameters'].update(density_seed=fixed_seed, helical_seed=seed, fixed_density=True)
            rm['reused_from'] = dict(path=str(record), sha256=digest(record))
            atomic_npz(target/'records'/case/record.name, **saved,
                       metadata_json=np.array(json.dumps(rm, allow_nan=False)))
            report['records'].append(dict(case=case, seed=seed, **rm['reused_from']))
    atomic_json(target/'reused_inputs.json', report)
    print(f"Reused {len(report['fields'])} field caches and {len(report['records'])} catalogue records.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    reuse(args.source, args.run)
