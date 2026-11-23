"""Aggregate independent seed measurements, with paired differences for eps +/-1."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .storage import atomic_json, load_config
from .run import KEYS
from .seeds import seed_ids

PRIMARY_STATISTIC = 'chi_hat'


def chi_hat_scan(paired, amplitudes):
    """Primary scan: mean normalized chirality, corrected over planned amplitudes.

    Keep the complete family size for partial runs. Other saved statistics are
    diagnostics and do not choose the peak or enter this correction.
    """
    comparisons = len(amplitudes)
    rows = [dict(r, p_bonferroni=(min(1., comparisons*r['p_two_sided'])
                                if r['p_two_sided'] is not None else None))
            for r in paired if r['statistic'] == PRIMARY_STATISTIC]
    usable = [r for r in rows if r['p_bonferroni'] is not None]
    peaks = [r for r in usable if r['t'] is not None]
    return dict(statistic=PRIMARY_STATISTIC, comparisons=comparisons,
                available_comparisons=len(usable), rows=rows,
                peak=max(peaks, key=lambda r: abs(r['t'])) if peaks else None,
                bonferroni_p=min(r['p_bonferroni'] for r in usable) if usable else None)


def mean_error(values):
    values = np.asarray(values)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return dict(n=n, mean=mean, sd=None, sem=None, t=None, p_two_sided=None,
                    ci95_low=None, ci95_high=None)
    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(n)
    t = mean/sem if sem > 0 else None
    halfwidth = float(stats.t.ppf(.975, n-1)*sem)
    return dict(n=n, mean=mean, sd=sd, sem=float(sem), t=t,
                p_two_sided=float(2*stats.t.sf(abs(t), n-1)) if t is not None else None,
                ci95_low=mean-halfwidth, ci95_high=mean+halfwidth)


def expected_cases(config):
    cases = {}
    if config['null']['seeds']:
        first = 'null_za' if config['field'].get('model') == 'mesh_za_helical' else 'null_poisson'
        cases.update({first:config['null']['seeds'], 'null_symmetric':config['null']['seeds']})
        if first == 'null_za':
            from .mesh_cases import symmetric_cases
            cases.update({name:config['null']['seeds'] for name, _ in symmetric_cases(config)})
        if first == 'null_za' and config['field'].get('density_seed') is not None:
            cases[first] = 1
        if config['null'].get('include_original_L1_A1'):
            cases['null_notebook_L1_A1'] = config['null']['seeds']
    if config['scan']['seeds']:
        for A in config['scan']['amplitudes']:
            for suffix in ('plus','minus'):
                cases[f'scan_A{A:g}_{suffix}'] = config['scan']['seeds']
    if config['positive']['seeds']:
        for f in config['positive']['fractions']:
            cases[f'positive_f{f:g}'] = config['positive']['seeds']
    return cases


def summarize(run):
    config = load_config(run)
    fixed_density = (config['field'].get('model') == 'mesh_za_helical'
                     and config['field'].get('density_seed') is not None)
    convention = config['field'].get('amplitude_convention', 'notebook')
    dest = run/'summary'
    dest.mkdir(parents=True, exist_ok=True)
    records, rows, completeness, realization_rows = {}, [], {}, []
    for case, expected in expected_cases(config).items():
        group = {}
        for path in sorted((run/'records'/case).glob('seed_*.npz')):
            with np.load(path) as z:
                m = json.loads(str(z['metadata_json']))
                group[m['seed']] = (z['statistics'].copy(), m)
                realization_rows.append(dict(case=case, seed=m['seed'],
                    chi=float(z['statistics'][0]), chi_hat=float(z['statistics'][1]), sgn=float(z['statistics'][2]),
                    n_tetrahedra=m['n_tetrahedra'], L=m['L'],
                    rms_over_spacing=m['diagnostics'].get('rms_over_spacing'),
                    za_rms=m['diagnostics'].get('za_rms'),
                    helical_rms=m['diagnostics'].get('helical_rms'),
                    total_displacement_rms=m['diagnostics'].get('total_displacement_rms'),
                    delaunay_seconds=m['delaunay_seconds'],
                    volume_fraction=m['volume_fraction'], ambiguous_order_cells=m['ambiguous_order_cells']))
        missing = sorted(set(seed_ids(config, expected))-set(group))
        unexpected = sorted(set(group)-set(seed_ids(config, expected)))
        completeness[case] = dict(expected=expected, completed=len(group), missing_seeds=missing, unexpected_seeds=unexpected)
        records[case] = group
        if group:
            values = np.stack([group[s][0] for s in sorted(group)])
            for i, key in enumerate(KEYS):
                estimate = mean_error(values[:, i])
                if fixed_density and case.startswith('null_'):
                    # One fixed Gaussian realization need not be parity symmetric.
                    # Its control means are not tests of a zero ensemble mean.
                    estimate.update(t=None, p_two_sided=None)
                rows.append(dict(case=case, statistic=key, **estimate))
    paired = []
    if config['scan']['seeds']:
        for A in config['scan']['amplitudes']:
            plus, minus = (records[f'scan_A{A:g}_{s}'] for s in ('plus','minus'))
            seeds = sorted(set(plus)&set(minus))
            if not seeds:
                continue
            delta = np.stack([plus[s][0]-minus[s][0] for s in seeds])
            for i, key in enumerate(KEYS):
                paired.append(dict(A=A, statistic=key, **mean_error(delta[:,i])))
    molecular = []
    for case, group in records.items():
        if not case.startswith('positive_') or not group:
            continue
        metas = [x[1]['molecules'] for x in group.values()]
        for quantity in ['measured_mean','other_mean','molecule_contribution_to_all','injected_mean_handedness','n_identified']:
            values = [m[quantity] for m in metas if m[quantity] is not None]
            if values:
                molecular.append(dict(case=case, quantity=quantity, **mean_error(values)))
    primary_scan = chi_hat_scan(paired, config['scan']['amplitudes'])
    for name, table in [('ensemble_statistics',rows),('paired_differences',paired),
                        ('paired_chi_hat',primary_scan['rows']),
                        ('molecular_controls',molecular),('realizations',realization_rows)]:
        path = dest/f'{name}.csv'
        with path.open('w',newline='') as f:
            if table:
                w = csv.DictWriter(f,fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
    complete = all(not c['missing_seeds'] and not c['unexpected_seeds'] for c in completeness.values())
    report = dict(complete=complete, amplitude_convention=convention,
                  model=config['field'].get('model', 'poisson_helical'),
                  density_seed=config['field'].get('density_seed'),
                  ensemble=('helical displacements conditional on fixed density and ZA'
                            if fixed_density else 'independent density and displacement realizations'),
                  generated_at=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(timespec='seconds'),
                  completed_records=sum(c['completed'] for c in completeness.values()),
                  expected_records=sum(c['expected'] for c in completeness.values()),
                  completeness=completeness, ensemble_statistics=rows,
                  paired_differences=paired, molecular_controls=molecular,
                  primary_statistic=PRIMARY_STATISTIC, primary_scan=primary_scan,
                  inference='Errors are across independent seeds. Paired differences use matching seeds. '
                  'Student-t intervals/p-values are approximate if seed statistics are non-Gaussian. '
                  'The primary statistic is catalogue-mean chi_hat. Primary scan p_bonferroni '
                  'and bonferroni_p correct over all planned amplitudes in this run. '
                  'All p_two_sided values are uncorrected; chi and sgn are retained as diagnostics. '
                  'An individual |t|>2 is not a pipeline failure or a discovery.'
                  + (' Errors are conditional on one fixed density/ZA field; they exclude density cosmic variance. '
                     'The ZA-only control is measured once. Fixed-IC control means are not tested against zero.'
                     if fixed_density else ''))
    atomic_json(dest/'summary.json',report)
    if paired:
        fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
        data = [r for r in primary_scan['rows'] if r['sem'] is not None]
        if data:
            ax.errorbar([r['A'] for r in data],[r['mean'] for r in data],
                        yerr=[r['sem'] for r in data],fmt='o-',capsize=3)
        ax.axhline(0,color='k',lw=.7)
        xlabel = (r'Displacement strength $A$' if convention == 'rms'
                  else 'Discrete per-mode power amplitude A')
        ax.set(xscale='log',xlabel=xlabel,ylabel=r'$\overline{D}=\langle D_s\rangle_s$')
        fig.suptitle('Paired helicity contrast; errors are 1 SEM across seeds')
        fig.savefig(dest/'paired_differences.png',dpi=160)
        plt.close(fig)
    null_cases = [case for case,g in records.items() if case.startswith('null_') and len(g)>1]
    if null_cases:
        fig, axes = plt.subplots(1,len(null_cases),figsize=(4*len(null_cases),3),squeeze=False,constrained_layout=True)
        x=np.linspace(-4,4,200)
        for ax, case in zip(axes[0], null_cases):
            values=np.stack([v[0] for v in records[case].values()])
            i = KEYS.index(PRIMARY_STATISTIC)
            sd=values[:,i].std(ddof=1)
            if sd>0:
                ax.hist((values[:,i]-values[:,i].mean())/sd,bins='auto',density=True,alpha=.6)
            ax.plot(x,stats.norm.pdf(x),'k-',lw=1)
            ax.set(title=f'{case}: chi_hat',xlabel='Standardized seed statistic')
        fig.savefig(dest/'null_distributions.png',dpi=160)
        plt.close(fig)
    if molecular:
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        selected=[r for r in rows if r['case'].startswith('positive_') and r['statistic']=='chi_hat' and r['sem'] is not None]
        ax.errorbar([float(r['case'].split('_f')[1]) for r in selected],[r['mean'] for r in selected],
                    yerr=[r['sem'] for r in selected],fmt='o-',capsize=3,label='All tetrahedra')
        selected=[r for r in molecular if r['quantity']=='molecule_contribution_to_all' and r['sem'] is not None]
        ax.plot([float(r['case'].split('_f')[1]) for r in selected],[r['mean'] for r in selected],'s--',label='Identified molecule contribution')
        ax.axhline(0,color='k',lw=.7)
        ax.set(xlabel='Right-handed fraction',ylabel='Mean normalized chirality')
        ax.legend()
        fig.savefig(dest/'positive_control.png',dpi=160)
        plt.close(fig)
    print(f"{run.name}: {report['completed_records']}/{report['expected_records']} records; complete={complete}")
    for row in primary_scan['rows']:
        if row['t'] is not None:
            print(f" A={row['A']:g} {row['statistic']}: {row['mean']:+.5g} +/- {row['sem']:.3g}; t={row['t']:+.2f}")
    if primary_scan['bonferroni_p'] is not None:
        print(f" chi_hat scan: {primary_scan['comparisons']} planned amplitudes; "
              f"Bonferroni p={primary_scan['bonferroni_p']:.5g}")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--no-notebook',action='store_true',help='Refresh only the JSON, CSV, and PNG outputs')
    a=p.parse_args()
    run = a.run.resolve()
    if a.no_notebook:
        summarize(run)
    else:
        # The notebook refreshes the numerical summaries from saved records.
        from .tests.notebooks import write_run_notebook
        write_run_notebook(run)

if __name__=='__main__':
    main()
