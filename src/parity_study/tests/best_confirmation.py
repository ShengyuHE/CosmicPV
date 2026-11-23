"""Fresh-IC confirmation of the configuration chosen by the amplitude scan."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from parity_study.tests.independent_ics import audit
from parity_study.seeds import seed_ids
from parity_study.tests.selections import measure, write_measurements
from parity_study.storage import atomic_json, check_manifest, digest, load_config
from parity_study.summarize import mean_error

PURPOSE = 'best_configuration_confirmation'
CONSTRUCTION_FILES = ('src/parity_study/initial_conditions.py',
                      'src/parity_study/catalogue.py',
                      'src/parity_study/delaunay.py', 'bin/DIVE')


def prepare(source, out, first=1000, count=50):
    from parity_study.tests.amplitude_scan import verify as verify_scan
    source, out = Path(source).resolve(), Path(out).resolve()
    _, previous, previous_path = verify_scan(source)
    analysis_path = previous_path.parent/'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    if not analysis['complete'] or analysis['protocol_sha256'] != digest(previous_path):
        raise ValueError('A complete matching amplitude analysis is required')
    selection_path = Path(previous['selection_protocol'])
    selection = json.loads(selection_path.read_text())
    old_seeds = sorted(set(previous['seeds'] + selection['discovery_seeds']
                           + selection['confirmation_seeds']))
    if count < 2 or first <= max(old_seeds):
        raise ValueError('Use at least two fresh seeds beyond the previous studies')
    path = out/'protocol.json'
    if path.exists():
        raise FileExistsError(path)
    protocol = dict(experiment=PURPOSE, created_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(source), source_manifest_sha256=digest(source/'manifest.json'),
        source_protocol=str(previous_path), source_protocol_sha256=digest(previous_path),
        source_analysis=str(analysis_path), source_analysis_sha256=digest(analysis_path),
        construction_sha256={p:digest(source/'code'/p) for p in CONSTRUCTION_FILES},
        field=previous['field'], cuts=previous['cuts'],
        primary_selection=previous['primary_selection'], expected_sign=previous['expected_sign'],
        A=analysis['optimization']['best_snr_A'], seeds=list(range(first, first+count)),
        length_unit=previous['length_unit'], epsilons=[-1,0,1],
        amplitude_choice='Best selected |mean D|/SEM in the completed amplitude scan; fixed before these ICs.',
        primary_test='One frozen amplitude and shape cut; two-sided paired Student t at alpha=0.05, '
                     'requiring the discovery sign. D_s = S_s(+1) - S_s(-1).',
        secondary_test='Unselected tetrahedra, epsilon=0 and ZA-only controls are descriptive.',
        sampling='Independent Gaussian density and independent helical pair for each seed; '
                 'the three helicities share that density, lattice and helical pair within a seed.',
        stopping_rule=f'Analyze all {count} new ICs without adapting A, the cut, or sample size.',
        histogram='One catalogue-mean chi_hat per IC and helicity; probability density normalized separately. '
                  'Gaussian curves use the sample mean and sample standard deviation (ddof=1), not SEM.',
        scope='Confirms the catalogue response at the chosen configuration; does not test a continuous '
              'amplitude optimum, Gaussianity, cosmological calibration or numerical convergence.')
    atomic_json(path, protocol)
    return path


def verify(run):
    run = Path(run).resolve()
    check_manifest(run)
    cfg = load_config(run)
    path = Path(cfg['confirmation_protocol']['path'])
    if digest(path) != cfg['confirmation_protocol']['sha256']:
        raise ValueError('Confirmation protocol changed after submission')
    p = json.loads(path.read_text())
    for key in ('source_protocol', 'source_analysis'):
        if digest(Path(p[key])) != p[key+'_sha256']:
            raise ValueError(f'Changed {key}')
    source = Path(p['source_run'])
    check_manifest(source)
    if digest(source/'manifest.json') != p['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    previous = json.loads(Path(p['source_protocol']).read_text())
    analysis = json.loads(Path(p['source_analysis']).read_text())
    selection_path = Path(previous['selection_protocol'])
    if digest(selection_path) != previous['selection_protocol_sha256']:
        raise ValueError('Original selection protocol changed')
    selection = json.loads(selection_path.read_text())
    if (p['experiment'] != PURPOSE or p['A'] != analysis['optimization']['best_snr_A']
            or any(p[k] != previous[k] for k in ('field','cuts','primary_selection','expected_sign'))):
        raise ValueError('Cannot change the chosen amplitude, field or frozen selection')
    seeds = list(seed_ids(cfg, cfg['scan']['seeds']))
    previous_seeds = previous['seeds'] + selection['discovery_seeds'] + selection['confirmation_seeds']
    if not seeds or min(seeds) <= max(previous_seeds):
        raise ValueError('Fresh confirmation seeds must follow previous study seeds')
    if (cfg['purpose'] != PURPOSE or seeds != p['seeds'] or len(seeds) < 2
            or cfg['field'] != p['field'] or cfg['field'].get('density_seed') is not None
            or cfg['null']['seeds'] != len(seeds) or cfg['null']['A'] != p['A']
            or cfg['null'].get('amplitudes', [p['A']]) != [p['A']]
            or cfg['scan']['amplitudes'] != [p['A']] or cfg['positive']['seeds']
            or cfg.get('reuse_from') or not cfg['save_tetrahedra']):
        raise ValueError('Run differs from the frozen confirmation configuration')
    for rel, sha in p['construction_sha256'].items():
        if digest(run/'code'/rel) != sha or digest(source/'code'/rel) != sha:
            raise ValueError(f'Catalogue construction changed: {rel}')
    return cfg, p, path


def distribution_summary(signals, cuts):
    signals = np.asarray(signals)
    if signals.ndim != 3 or signals.shape[:2] != (len(cuts),3) or signals.shape[2] < 2:
        raise ValueError('Require selection × three helicities × independent ICs')
    if not np.isfinite(signals).all():
        raise ValueError('Nonfinite catalogue signals')
    return [dict(selection=cut['id'], epsilon=eps, **mean_error(signals[c,e]))
            for c,cut in enumerate(cuts) for e,eps in enumerate((-1,0,1))]


def analyze(run):
    run = Path(run).resolve()
    cfg, p, path = verify(run)
    if not json.loads((run/'validation/report.json').read_text()).get('passed'):
        raise ValueError('Construction validation must pass first')
    audit_result, za = audit(run, cfg, p['seeds'], p['cuts'])
    result = measure(run, p['cuts'], p['seeds'], p['A'])
    primary = next(r for r in result['summaries'] if r['selection'] == p['primary_selection'])
    same_sign = bool(np.sign(primary['mean']) == p['expected_sign'])
    confirmed = bool(same_sign and primary['p_two_sided'] is not None and primary['p_two_sided'] < .05)
    distributions = distribution_summary(result['signals'], p['cuts'])
    controls = [dict(selection=cut['id'], epsilon0=mean_error(result['signals'][i,1]),
                     za_only=mean_error(za[i])) for i,cut in enumerate(p['cuts'])]
    out = path.parent
    write_measurements(out, 'confirmation', result)
    with (out/'distribution_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(distributions[0]))
        writer.writeheader()
        writer.writerows(distributions)
    inputs = {str(f.relative_to(run)):digest(f) for f in sorted((run/'records').glob('*/seed_*.npz'))}
    atomic_json(out/'input_sha256.json', inputs)
    report = dict(run=str(run), complete=True, A=p['A'], seeds=p['seeds'],
        protocol_sha256=digest(path), primary_selection=p['primary_selection'], primary=primary,
        primary_confirmed=confirmed, same_sign_as_discovery=same_sign,
        summaries=result['summaries'], distributions=distributions, controls=controls,
        independent_ic_audit=audit_result, primary_test=p['primary_test'], scope=p['scope'],
        catalogue_count=len(inputs),
        conclusion=('The fixed A and shape selection reproduce the signed catalogue response on fresh independent ICs.'
                    if confirmed else 'This fresh sample does not establish a same-sign catalogue response.'))
    atomic_json(out/'analysis.json', report)
    print(json.dumps(dict(primary=primary, confirmed=confirmed, catalogues=len(inputs))), flush=True)
    return report


def distribution_figure(report, signals, selection_index=1):
    """Empirical catalogue distributions with descriptive Gaussian curves."""
    values = np.asarray(signals)[selection_index]
    n = values.shape[1]
    means, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
    if not np.isfinite(values).all() or np.any(sd <= 0):
        raise ValueError('Gaussian curves require finite, nonconstant catalogue signals')
    pooled_sd = np.sqrt(np.mean(sd**2))
    width = 3.49*pooled_sd/n**(1/3)
    lo = min(values.min(), np.min(means-4*sd))
    hi = max(values.max(), np.max(means+4*sd))
    bins = np.linspace(lo,hi,max(10,int(np.ceil((hi-lo)/width)))+1)
    x = np.linspace(lo,hi,2400)
    fig, ax = plt.subplots(figsize=(9.5,4.8))
    colors = ('#2878b5','#777777','#d95335')
    for i,(eps,color) in enumerate(zip((-1,0,1),colors)):
        ax.hist(values[i],bins=bins,density=True,histtype='stepfilled',alpha=.24,color=color)
        ax.hist(values[i],bins=bins,density=True,histtype='step',linewidth=1.1,color=color)
        ax.plot(x,norm.pdf(x,loc=means[i],scale=sd[i]),color=color,lw=2,
                label=rf'$\epsilon={eps:+d}$: $\mu={means[i]:+.4f}$, $\sigma_S={sd[i]:.4f}$')
        ax.axvline(means[i],color=color,lw=.8,ls=':',alpha=.65)
    ax.set(xlabel=r'Catalogue signal $S_{s,B}(\epsilon)=\langle\widehat\chi\rangle_{t\in B}$',
           ylabel='Probability density',xlim=(lo,hi),ylim=(0,None),
           title=rf'Displacement strength $A={report["A"]:g}\,h^{{-1}}\mathrm{{Mpc}}$ | {n} independent ICs')
    ax.legend(fontsize=10)
    ax.grid(alpha=.18)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','analyze','report'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--first',type=int,default=1000)
    parser.add_argument('--count',type=int,default=50)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.source,args.out,args.first,args.count))
    elif args.action == 'verify':
        verify(args.run)
        print('Frozen confirmation configuration verified',flush=True)
    elif args.action == 'analyze':
        analyze(args.run)
    else:
        from parity_study.tests.notebooks import write_run_notebook
        _, _, path = verify(args.run)
        report = json.loads((path.parent/'analysis.json').read_text())
        if not report['complete'] or report['protocol_sha256'] != digest(path):
            raise ValueError('A complete matching analysis is required')
        notebook = write_run_notebook(args.run)
        atomic_json(path.parent/'report_provenance.json',dict(run=str(args.run.resolve()),
            notebook=str(notebook), notebook_sha256=digest(notebook),
            protocol_sha256=digest(path), manifest_sha256=digest(args.run/'manifest.json'),
            executed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
