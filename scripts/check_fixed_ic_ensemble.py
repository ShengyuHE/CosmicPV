#!/usr/bin/env python3
"""Verify that a completed ensemble varies only its helical displacement."""
import argparse
import json
from pathlib import Path
import numpy as np
from parity_study.initial_conditions import displace_mesh
from parity_study.storage import atomic_json, check_manifest, load_config
from parity_study.summarize import expected_cases
from parity_study.seeds import seed_ids, seed_start
import hashlib


def check(run):
    run = Path(run).resolve()
    check_manifest(run)
    config = load_config(run)
    field = config['field']
    fixed_seed = field.get('density_seed')
    if field.get('model') != 'mesh_za_helical' or fixed_seed is None:
        raise ValueError('This audit requires a fixed-density mesh ensemble')
    n = max(config['null']['seeds'], config['scan']['seeds'])
    cases = expected_cases(config)
    with np.load(run/'fields'/f'main_{seed_start(config):04d}.npz') as z:
        reference = {k:z[k] for k in ('q', 'za', 'delta_mesh')}
    rows, helical_hashes, record_count = [], {'plus':set(), 'minus':set()}, 0
    for seed in seed_ids(config, n):
        with np.load(run/'fields'/f'main_{seed:04d}.npz') as z:
            arrays = {k:z[k] for k in ('q', 'za', 'delta_mesh', 'plus', 'minus')}
            meta = json.loads(str(z['metadata_json']))
        assert meta['density_seed'] == fixed_seed and meta['helical_seed'] == seed
        assert meta['field'] == field and meta['fixed_density']
        np.testing.assert_array_equal(arrays['q'], reference['q'])
        np.testing.assert_array_equal(arrays['delta_mesh'], reference['delta_mesh'])
        za_error = float(np.max(abs(arrays['za']-reference['za'])))
        # Parallel NUFFT summation may change its last few floating-point bits.
        assert za_error < 1e-12
        for hand in helical_hashes:
            helical_hashes[hand].add(hashlib.sha256(arrays[hand].tobytes()).hexdigest())
        for case, count in cases.items():
            if seed not in seed_ids(config, count):
                continue
            with np.load(run/'records'/case/f'seed_{seed:04d}.npz') as z:
                rm = json.loads(str(z['metadata_json']))
                positions = z['catalogue']
            params = rm['parameters']
            assert params['density_seed'] == fixed_seed and params['helical_seed'] == seed
            expected, _ = displace_mesh(arrays, A=params['A'], eps=params['eps'], L=field['L'])
            np.testing.assert_array_equal(positions, expected)
            record_count += 1
        rows.append(dict(helical_seed=seed, density_seed=fixed_seed,
                         density_identical=True, lattice_identical=True,
                         max_za_difference=za_error))
    assert all(len(values) == n for values in helical_hashes.values())
    assert len(list((run/'records/null_za').glob('seed_*.npz'))) == 1
    report = dict(passed=True, density_seed=fixed_seed, displacement_seeds=n,
                  checked_catalogues=record_count,
                  density_and_lattice='exactly identical across displacement seeds',
                  za_tolerance=1e-12, max_za_difference=max(r['max_za_difference'] for r in rows),
                  distinct_helical_fields={k:len(v) for k,v in helical_hashes.items()}, rows=rows)
    atomic_json(run/'validation/fixed_ic_ensemble.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    check(parser.parse_args().run)
