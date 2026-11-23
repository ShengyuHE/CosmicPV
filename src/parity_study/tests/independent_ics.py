"""Test a previously frozen shape selection across independent Gaussian ICs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parity_study.initial_conditions import displace_mesh
from parity_study.seeds import seed_ids
from parity_study.tests.selections import measure, selection_mask, write_measurements
from parity_study.storage import atomic_json, check_manifest, digest, load_config
from parity_study.summarize import expected_cases, mean_error
from parity_study.mesh_cases import case_parameters


def prepare(parent_path, reference_path, out, first=100, count=25):
    """Freeze the existing cut and a sample size before any new results exist."""
    parent = json.loads(parent_path.read_text())
    reference = json.loads(reference_path.read_text())
    if first <= max(parent['discovery_seeds'] + parent['confirmation_seeds']
                    + reference.get('extension', {}).get('added_seeds', [])):
        raise ValueError('Use fresh seeds beyond the previous ensembles')
    if count < 2:
        raise ValueError('At least two independent initial conditions are required')
    path = out/'protocol.json'
    if path.exists():
        raise FileExistsError(path)
    field = dict(parent['field'], density_seed=None)
    cuts = [c for c in parent['cuts'] if c['id'] in ('all', parent['primary_selection'])]
    protocol = dict(experiment='independent_initial_conditions',
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol=str(parent_path.resolve()), parent_protocol_sha256=digest(parent_path),
        fixed_ic_reference=str(reference_path.resolve()), fixed_ic_reference_sha256=digest(reference_path),
        field=field, A=parent['A'], cuts=cuts, primary_selection=parent['primary_selection'],
        expected_sign=parent['expected_sign'], seeds=list(range(first, first+count)),
        primary_test='One frozen selection; two-sided paired Student t, alpha=0.05, '
                     'with the same sign as discovery required for replication.',
        secondary_test='All-tetrahedron baseline and epsilon=0/ZA-only controls are descriptive.',
        sampling='One independent Gaussian density and one independent helical pair per seed. '
                 'Scalar/positive/negative fields use distinct SeedSequence child streams. '
                 'Within each seed all helicities share density, ZA, lattice and the helical pair.',
        uncertainty='Across independent joint density/helical realizations; includes both sources of scatter.',
        stopping_rule=f'Analyze all {count} planned realizations without adapting the cut or sample size.')
    atomic_json(path, protocol)
    return path


def verify(config, protocol, path):
    if digest(path) != config['selection_protocol']['sha256']:
        raise ValueError('Independent-IC protocol changed after submission')
    if protocol.get('experiment') != 'independent_initial_conditions':
        raise ValueError('Wrong experiment protocol')
    parent_path = Path(protocol['parent_protocol'])
    if digest(parent_path) != protocol['parent_protocol_sha256']:
        raise ValueError('Original selection protocol changed')
    parent = json.loads(parent_path.read_text())
    expected_cuts = [c for c in parent['cuts'] if c['id'] in ('all', parent['primary_selection'])]
    if (protocol['cuts'] != expected_cuts or protocol['primary_selection'] != parent['primary_selection']
            or protocol['expected_sign'] != parent['expected_sign'] or protocol['A'] != parent['A']):
        raise ValueError('The frozen selection, sign and amplitude must stay unchanged')
    expected_field = dict(parent['field'], density_seed=None)
    if config['field'] != expected_field or protocol['field'] != expected_field:
        raise ValueError('Only the fixed-density setting may change')
    ref_path = Path(protocol['fixed_ic_reference'])
    if digest(ref_path) != protocol['fixed_ic_reference_sha256']:
        raise ValueError('Fixed-IC reference result changed')
    reference = json.loads(ref_path.read_text())
    previous = set(parent['discovery_seeds'] + parent['confirmation_seeds']
                   + reference.get('extension', {}).get('added_seeds', []))
    seeds = list(seed_ids(config, config['scan']['seeds']))
    if previous.intersection(seeds):
        raise ValueError('Independent-IC seeds overlap previous ensembles')
    if (seeds != protocol['seeds'] or config['null']['seeds'] != len(seeds)
            or config['scan']['amplitudes'] != [protocol['A']]
            or config['null']['A'] != protocol['A'] or config['positive']['seeds']
            or config.get('reuse_from') or config['purpose'] != 'independent_ic_confirmation'):
        raise ValueError('Run differs from the frozen independent-IC experiment')
    return seeds


def audit(run, config, seeds, cuts):
    """Check independent fields and exact matched particle maps for every catalogue."""
    field = config['field']
    if field.get('density_seed') is not None:
        raise ValueError('Independent-IC audit requires varying density')
    hashes = {k: set() for k in ('delta_mesh', 'za', 'plus', 'minus')}
    rows, za_signals = [], [[] for _ in cuts]
    reference_q = None
    expected = expected_cases(config)
    parameters = case_parameters(config)
    for case, count in expected.items():
        actual = {p.name for p in (run/'records'/case).glob('seed_*.npz')}
        planned = {f'seed_{s:04d}.npz' for s in seed_ids(config, count)}
        if actual != planned:
            raise ValueError(f'Incomplete or unexpected catalogues in {case}')
    for seed in seeds:
        with np.load(run/'fields'/f'main_{seed:04d}.npz') as z:
            arrays = {k: z[k] for k in ('q', 'za', 'plus', 'minus', 'delta_mesh')}
            meta = json.loads(str(z['metadata_json']))
        if (meta['field'] != field or meta['density_seed'] != seed or meta['helical_seed'] != seed
                or meta['fixed_density']):
            raise ValueError(f'Incorrect field seeds: {seed}')
        if reference_q is None:
            reference_q = arrays['q'].copy()
        np.testing.assert_array_equal(arrays['q'], reference_q)
        for key in hashes:
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f'Nonfinite {key}: {seed}')
            hashes[key].add(hashlib.sha256(arrays[key].tobytes()).hexdigest())
        for case in expected:
            with np.load(run/'records'/case/f'seed_{seed:04d}.npz') as z:
                meta = json.loads(str(z['metadata_json']))
                params = meta['parameters']
                expected_A, expected_eps = parameters[case]
                if (meta['seed'] != seed or meta['case'] != case or params['density_seed'] != seed
                        or params['helical_seed'] != seed or params['fixed_density']
                        or params['A'] != expected_A or params['eps'] != expected_eps):
                    raise ValueError(f'Mismatched catalogue: {case}/{seed}')
                positions, _ = displace_mesh(arrays, A=expected_A, eps=expected_eps, L=field['L'])
                np.testing.assert_array_equal(z['catalogue'], positions)
                if case == 'null_za':
                    tets = z['tetrahedra']
                    for i, cut in enumerate(cuts):
                        values = tets[selection_mask(tets, cut, field['L']/field['particle_grid']), 1]
                        if not len(values):
                            raise ValueError('Empty ZA-only selection')
                        za_signals[i].append(float(values.mean()))
        rows.append(dict(seed=seed, density_seed=seed, helical_seed=seed,
                         matched_catalogues=len(expected), maps_exact=True))
    if any(len(v) != len(seeds) for v in hashes.values()):
        raise ValueError('Repeated density, ZA or helical fields across independent ICs')
    report = dict(passed=True, independent_initial_conditions=len(seeds),
                  checked_catalogues=len(seeds)*len(expected), lattice_identical=True,
                  distinct_fields={k: len(v) for k,v in hashes.items()}, rows=rows)
    atomic_json(run/'validation/independent_ic_ensemble.json', report)
    return report, np.asarray(za_signals)


def confirm(run):
    run = Path(run).resolve()
    check_manifest(run)
    config = load_config(run)
    if not json.loads((run/'validation/report.json').read_text()).get('passed'):
        raise ValueError('Construction validation has not passed')
    path = Path(config['selection_protocol']['path'])
    protocol = json.loads(path.read_text())
    seeds = verify(config, protocol, path)
    audit_result, za_signals = audit(run, config, seeds, protocol['cuts'])
    result = measure(run, protocol['cuts'], seeds, protocol['A'])
    primary = next(r for r in result['summaries'] if r['selection'] == protocol['primary_selection'])
    same_sign = bool(np.sign(primary['mean']) == protocol['expected_sign'])
    replicated = bool(same_sign and primary['p_two_sided'] is not None and primary['p_two_sided'] < .05)
    controls = [dict(selection=cut['id'], epsilon0=mean_error(result['signals'][i, 1]),
                     za_only=mean_error(za_signals[i])) for i,cut in enumerate(protocol['cuts'])]
    report = dict(confirmation_run=str(run), protocol_sha256=digest(path), A=protocol['A'],
        primary_selection=protocol['primary_selection'], primary=primary,
        same_sign_as_discovery=same_sign, primary_confirmed=replicated,
        summaries=result['summaries'], controls=controls, independent_ic_audit=audit_result,
        conclusion=('The frozen shape selection replicates the helicity response across independent Gaussian ICs.'
                    if replicated else 'The frozen shape selection does not establish a same-sign helicity response across independent Gaussian ICs.'),
        scope=protocol['uncertainty'], primary_test=protocol['primary_test'])
    write_measurements(path.parent, 'confirmation', result)
    atomic_json(path.parent/'controls.json', controls)
    atomic_json(path.parent/'confirmation.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('independent_ic_audit', 'controls')}, indent=2), flush=True)


def comparison_figure(protocol, report):
    """Keep fixed-density and independent-IC estimates separate."""
    fixed = json.loads(Path(protocol['fixed_ic_reference']).read_text())
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ids = [c['id'] for c in protocol['cuts']]
    for offset, data, color, label in [(-.12, fixed, '#762a83', 'Fixed density: 25 displacement realizations'),
                                      (.12, report, '#1b7837', f"Independent density: {report['primary']['n']} realizations")]:
        rows = [next(r for r in data['summaries'] if r['selection'] == key) for key in ids]
        ax.errorbar([r['mean'] for r in rows], np.arange(len(rows))+offset,
                    xerr=[r['ci95_high']-r['mean'] for r in rows], fmt='o', capsize=4,
                    color=color, label=label)
    ax.axvline(0, color='k', lw=.7)
    ax.set(yticks=range(len(ids)), yticklabels=['All tetrahedra', 'Frozen shape selection'],
           xlabel=r'Catalogue difference $\overline{D}_B$',
           title=f"Displacement strength A={protocol['A']:g} · mean ± 95% interval")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed-start', type=int, default=100)
    p.add_argument('--seeds', type=int, default=25)
    p = sub.add_parser('confirm')
    p.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.parent, args.reference, args.out, args.seed_start, args.seeds))
    else:
        confirm(args.run)


if __name__ == '__main__':
    main()
