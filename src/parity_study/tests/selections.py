"""Exploratory parity-even cuts followed by a frozen, independent-seed test."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parity_study.storage import atomic_json, atomic_npz, digest, load_config, check_manifest
from parity_study.seeds import seed_ids
from parity_study.summarize import mean_error

EPSILONS = (-1, 0, 1)


def selection_mask(tetrahedra, cut, spacing):
    """All cuts are invariant under reflection, including exact endpoints."""
    if cut['variable'] == 'all':
        return np.ones(len(tetrahedra), dtype=bool)
    if cut['variable'] == 'abs_chi_hat':
        value = abs(tetrahedra[:, 1])
    elif cut['variable'] == 'size':
        value = np.cbrt(np.prod(tetrahedra[:, 2:5], axis=1)) / spacing
    elif cut['variable'] == 'shape':
        value = tetrahedra[:, 2] / tetrahedra[:, 4]
    else:
        raise ValueError(f"Unknown selection: {cut['variable']}")
    return ((value >= cut['low']) if cut['low'] is not None else np.ones(len(value), bool)) & (
        (value < cut['high']) if cut['high'] is not None else np.ones(len(value), bool))


def candidate_cuts(reference, spacing):
    cuts = [dict(id='all', label='All tetrahedra', variable='all', low=None, high=None)]
    for low, high in [(.45, .50), (.50, .55), (.45, .55)]:
        cuts.append(dict(id=f'abs_{low:.2f}_{high:.2f}', label=f'{low:.2f} ≤ |χ̂| < {high:.2f}',
                         variable='abs_chi_hat', low=low, high=high))
    # Geometry thresholds come only from the ZA-only control, without helicity labels.
    values = {'size':np.cbrt(np.prod(reference[:, 2:5], axis=1))/spacing,
              'shape':reference[:, 2]/reference[:, 4]}
    for variable, value in values.items():
        bounds = [None, *np.quantile(value, [1/3, 2/3]).tolist(), None]
        for i in range(3):
            cuts.append(dict(id=f'{variable}_{i+1}', label=f'{variable.capitalize()} third {i+1}',
                             variable=variable, low=bounds[i], high=bounds[i+1]))
    return cuts


def measure(run, cuts, seeds, A):
    """One independent observation per catalogue; tetrahedra are not replicates."""
    run = Path(run)
    config = load_config(run)
    field = config['field']
    spacing = field['L']/field['particle_grid']
    from parity_study.mesh_cases import symmetric_cases
    controls = {amplitude:name for name,amplitude in symmetric_cases(config)}
    if A not in controls:
        raise ValueError('Requested amplitude has no matching epsilon=0 control')
    cases = (f'scan_A{A:g}_minus', controls[A], f'scan_A{A:g}_plus')
    shape = (len(cuts), 3, len(seeds))
    signals, contributions = np.empty(shape), np.empty(shape)
    counts, totals = np.empty(shape, dtype=int), np.empty((3, len(seeds)), dtype=int)
    rows, hashes = [], {}
    for e, (eps, case) in enumerate(zip(EPSILONS, cases)):
        for j, seed in enumerate(seeds):
            path = run/'records'/case/f'seed_{seed:04d}.npz'
            hashes[str(path.relative_to(run))] = digest(path)
            with np.load(path) as z:
                meta = json.loads(str(z['metadata_json']))
                tets = z['tetrahedra']
                expected_mean = z['statistics'][list(z['statistic_names']).index('chi_hat')]
            params = meta['parameters']
            density_seed = field.get('density_seed')
            if density_seed is None:
                density_seed = seed
            if (meta['seed'] != seed or params['helical_seed'] != seed or params['eps'] != eps
                    or params['A'] != A or params['density_seed'] != density_seed):
                raise ValueError(f'Mismatched catalogue: {path}')
            if (not np.isfinite(tets).all() or (abs(tets[:, 1]) > 1+1e-10).any()
                    or (tets[:, 2:5] <= 0).any()
                    or not np.isclose(tets[:, 1].mean(), expected_mean, rtol=0, atol=1e-12)):
                raise ValueError(f'Invalid tetrahedra: {path}')
            totals[e, j] = len(tets)
            for c, cut in enumerate(cuts):
                chosen = tets[selection_mask(tets, cut, spacing), 1]
                if not len(chosen):
                    raise ValueError(f"Empty selection {cut['id']}: {path}")
                signals[c, e, j] = chosen.mean()
                contributions[c, e, j] = chosen.sum()/len(tets)
                counts[c, e, j] = len(chosen)
                rows.append(dict(selection=cut['id'], epsilon=eps, seed=seed,
                                 S=float(chosen.mean()), contribution=float(chosen.sum()/len(tets)),
                                 selected=len(chosen), total=len(tets), fraction=len(chosen)/len(tets)))
    paired = signals[:, 2] - signals[:, 0]
    summaries = [dict(selection=cut['id'], label=cut['label'],
                      retained_fraction=float((counts[c]/totals).mean()),
                      minimum_selected=int(counts[c].min()), **mean_error(paired[c]))
                 for c, cut in enumerate(cuts)]
    return dict(seeds=np.array(seeds), epsilons=np.array(EPSILONS), signals=signals,
                contributions=contributions, counts=counts, totals=totals, paired=paired,
                rows=rows, summaries=summaries, input_sha256=hashes)


def write_measurements(out, prefix, result):
    atomic_npz(out/f'{prefix}.npz', **{k:result[k] for k in (
        'seeds', 'epsilons', 'signals', 'contributions', 'counts', 'totals', 'paired')})
    for name, rows in [('catalogues', result['rows']), ('summary', result['summaries'])]:
        with (out/f'{prefix}_{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    atomic_json(out/f'{prefix}_inputs.json', result['input_sha256'])


def discover(run, out):
    check_manifest(run)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Frozen selection protocol already exists; use a new output directory')
    config = load_config(run)
    seeds = list(seed_ids(config, config['scan']['seeds']))
    field = config['field']
    if field.get('density_seed') is None:
        raise ValueError('This protocol requires a fixed Gaussian initial condition')
    reference_path = run/'records/null_za'/f'seed_{seeds[0]:04d}.npz'
    with np.load(reference_path) as z:
        reference = z['tetrahedra']
    cuts = candidate_cuts(reference, field['L']/field['particle_grid'])
    transform_checks = []
    transformed_reference = None
    for name, sign in [('reference', 1), ('inversion', -1), ('rotation90', 1), ('translation', 1)]:
        with np.load(run/'validation'/f'T1_{name}.npz') as z:
            tets = z['tetrahedra']
        values = []
        for cut in cuts:
            chosen = tets[selection_mask(tets, cut, field['L']/field['particle_grid']), 1]
            values.append((len(chosen), chosen.mean()))
        if transformed_reference is None:
            transformed_reference = values
        for cut, (n, mean), (n0, mean0) in zip(cuts, values, transformed_reference):
            error = float(abs(mean-sign*mean0))
            passed = bool(n == n0 and error < 1e-10)
            transform_checks.append(dict(transform=name, selection=cut['id'], passed=passed,
                                         selected=n, mean_error=error))
            if not passed:
                raise AssertionError(f"Selection transformation failed: {name} {cut['id']}")
    atomic_json(out/'transformation_checks.json', dict(passed=True, rows=transform_checks))
    A = config['null']['A']
    result = measure(run, cuts, seeds, A)
    eligible = [r for r in result['summaries'] if r['selection'] != 'all' and r['t'] is not None]
    winner = max(eligible, key=lambda r: abs(r['t']))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), discovery_run=str(run),
                    analysis_source_sha256=digest(Path(__file__)),
                    discovery_seeds=seeds, confirmation_seeds=list(range(max(seeds)+1, max(seeds)+11)),
                    A=A, field=field, cuts=cuts, primary_selection=winner['selection'],
                    expected_sign=int(np.sign(winner['mean'])),
                    ranking='Largest absolute paired Student t among nine cuts; baseline excluded.',
                    primary_test='Two-sided paired Student t on ten fresh displacement seeds, alpha=0.05; '
                                 'also require the same effect sign as discovery. Approximate normality across seeds.',
                    secondary_test='All nine cuts on fresh seeds, Bonferroni correction over nine; baseline descriptive.',
                    geometry_reference=str(reference_path), geometry_reference_sha256=digest(reference_path),
                    discovery_summaries=result['summaries'],
                    uncertainty='Conditional on density seed 0 and fixed ZA; excludes density cosmic variance.',
                    exploratory_warning='Chirality bands were suggested after viewing these discovery histograms; '
                                        'discovery p-values and confidence intervals are descriptive, not detection evidence.')
    write_measurements(out, 'discovery', result)
    atomic_json(protocol_path, protocol)
    print(json.dumps(protocol, indent=2), flush=True)


def verify_confirmation(config, protocol, protocol_path):
    planned = config['selection_protocol']
    if digest(protocol_path) != planned['sha256']:
        raise ValueError('Selection protocol changed after the confirmation run was prepared')
    seeds = list(seed_ids(config, config['scan']['seeds']))
    if set(seeds) & set(protocol['discovery_seeds']):
        raise ValueError('Discovery and confirmation seeds overlap')
    if seeds != protocol['confirmation_seeds'] or config['null']['seeds'] != len(seeds):
        raise ValueError('Confirmation seed range differs from frozen protocol')
    if config['field'] != protocol['field'] or config['scan']['amplitudes'] != [protocol['A']]:
        raise ValueError('Confirmation field or amplitude differs from frozen protocol')
    if 'extension' in protocol:
        extension = protocol['extension']
        parent_path = Path(extension['parent_protocol'])
        if digest(parent_path) != extension['parent_protocol_sha256']:
            raise ValueError('Original selection protocol changed')
        parent = json.loads(parent_path.read_text())
        for key in ('field', 'A', 'cuts', 'primary_selection', 'expected_sign', 'discovery_seeds'):
            if protocol[key] != parent[key]:
                raise ValueError(f'An ensemble extension cannot change {key}')
        previous = parent['confirmation_seeds']
        added = [s for s in seeds if s not in previous]
        if (not set(previous).issubset(seeds) or not added
                or previous != extension['previous_seeds'] or added != extension['added_seeds']):
            raise ValueError('Extension must preserve the old seeds and specify every added seed')
    return seeds


def confirm(run):
    check_manifest(run)
    config = load_config(run)
    protocol_path = Path(config['selection_protocol']['path'])
    protocol = json.loads(protocol_path.read_text())
    seeds = verify_confirmation(config, protocol, protocol_path)
    out = protocol_path.parent
    audit = json.loads((run/'validation/fixed_ic_ensemble.json').read_text())
    if not audit['passed']:
        raise ValueError('Fixed-initial-condition audit failed')
    # Verify the initial condition is also the same BETWEEN the two ensembles.
    source = Path(protocol['discovery_run'])
    with np.load(source/'fields'/f"main_{protocol['discovery_seeds'][0]:04d}.npz") as z:
        ref = {k:z[k] for k in ('q', 'za', 'delta_mesh')}
    with np.load(run/'fields'/f'main_{seeds[0]:04d}.npz') as z:
        for k in ('q', 'delta_mesh'):
            np.testing.assert_array_equal(z[k], ref[k])
        za_difference = float(abs(z['za']-ref['za']).max())
        assert za_difference < 1e-12
    result = measure(run, protocol['cuts'], seeds, protocol['A'])
    for row in result['summaries']:
        row['p_secondary_bonferroni'] = (min(1., 9*row['p_two_sided'])
            if row['selection'] != 'all' and row['p_two_sided'] is not None else None)
    primary = next(r for r in result['summaries'] if r['selection'] == protocol['primary_selection'])
    same_sign = bool(np.sign(primary['mean']) == protocol['expected_sign'])
    detected = bool(same_sign and primary['p_two_sided'] is not None and primary['p_two_sided'] < .05)
    report = dict(discovery_run=protocol['discovery_run'], confirmation_run=str(run),
                  protocol_sha256=digest(protocol_path), A=protocol['A'],
                  primary_selection=protocol['primary_selection'], primary=primary,
                  same_sign_as_discovery=same_sign, primary_confirmed=detected,
                  summaries=result['summaries'], fixed_ic_audit=audit,
                  between_ensembles_max_za_difference=za_difference,
                  conclusion=('The frozen selection has a same-sign response on independent displacement seeds.'
                              if detected else 'The frozen selection does not confirm a response on independent displacement seeds.'),
                  scope='Conditional on the one fixed Gaussian initial condition; not a test across density realizations.')
    if 'extension' in protocol:
        extension = protocol['extension']
        index = next(i for i,c in enumerate(protocol['cuts']) if c['id'] == protocol['primary_selection'])
        added = np.isin(result['seeds'], extension['added_seeds'])
        report.update(extension=extension, added_seeds_primary=mean_error(result['paired'][index, added]),
                      inference='The ensemble was extended after inspecting the original ten-seed result. '
                                'Combined p-values and intervals are nominal fixed-sample summaries, '
                                'not a new preplanned 25-seed confirmation. The added-seed estimate is saved separately.',
                      conclusion=(f'The frozen selection retains a same-sign response in the {len(seeds)}-realization ensemble.'
                                  if detected else f'The frozen selection is not significant in the {len(seeds)}-realization ensemble.'))
    write_measurements(out, 'confirmation', result)
    atomic_json(out/'confirmation.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('summaries','fixed_ic_audit')}, indent=2), flush=True)


def selection_comparison_figure(protocol, confirmation):
    """Compare independent ensembles; intervals are across seeds, not tetrahedra."""
    fig, ax = plt.subplots(figsize=(8.3, 5.4), constrained_layout=True)
    cuts = protocol['cuts']
    discovery_seeds = protocol['discovery_seeds']
    confirmation_seeds = protocol['confirmation_seeds']
    for offset, rows, color, label in [
            (-.13, protocol['discovery_summaries'], '#762a83',
             f'Discovery: {len(discovery_seeds)} seeds ({discovery_seeds[0]}–{discovery_seeds[-1]})'),
            (.13, confirmation['summaries'], '#1b7837',
             f"{'Extended' if 'extension' in protocol else 'Confirmation'}: {len(confirmation_seeds)} seeds "
             f'({confirmation_seeds[0]}–{confirmation_seeds[-1]})')]:
        mean = np.array([r['mean'] for r in rows])
        error = np.array([r['ci95_high']-r['mean'] for r in rows])
        ax.errorbar(mean, np.arange(len(rows))+offset, xerr=error, fmt='o', color=color,
                    ms=4, capsize=3, label=label)
    ax.axvline(0, color='k', lw=.7)
    ax.set(yticks=np.arange(len(cuts)),
           yticklabels=[c['label'] + ('  ★' if c['id'] == protocol['primary_selection'] else '') for c in cuts],
           xlabel=r'Selected catalogue difference $\overline{D}_B$',
           title=f"Displacement strength A={protocol['A']:g} · mean ± 95% interval\n★ Frozen primary selection")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    return fig


def selected_catalogue_figure(protocol, confirmation_arrays):
    from parity_study.tests.signal_plots import catalogue_signal_figure
    c = next(i for i, cut in enumerate(protocol['cuts']) if cut['id'] == protocol['primary_selection'])
    fig = catalogue_signal_figure(dict(A=protocol['A'], epsilons=confirmation_arrays['epsilons'],
        seeds=confirmation_arrays['seeds'], signals=confirmation_arrays['signals'][c]))
    seeds = confirmation_arrays['seeds']
    fig.axes[0].set(xlabel=r'Selected catalogue signal $S_{s,B}(\epsilon)$',
                   title=f"{protocol['cuts'][c]['label']} · {len(seeds)} realizations · A={protocol['A']:g}"
                         f"\nDisplacement seeds {seeds[0]}–{seeds[-1]}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['discover', 'confirm'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.stage == 'discover':
        if args.out is None:
            parser.error('--out is required for discovery')
        discover(args.run.resolve(), args.out)
    else:
        confirm(args.run.resolve())


if __name__ == '__main__':
    main()
