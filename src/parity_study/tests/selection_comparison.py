"""Compare a requested wider Q interval using saved catalogue sums and counts."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.geometry import tetrahedron_geometry
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest
from parity_study.tests.selection_bins import paired_summary

CASES = ('scan_A16_minus', 'scan_A16_plus', 'null_symmetric', 'null_za')
SELECTIONS = (
    dict(id='all', label='All tetrahedra', parts=['all']),
    dict(id='reference_ratio', label='Frozen r cut', parts=['reference_ratio']),
    dict(id='Q_2', label='0.4 ≤ Q < 0.6', parts=['Q_2']),
    dict(id='Q_wide', label='0.4 ≤ Q < 0.8', parts=['Q_2', 'Q_3']),
)


def combine_disjoint(means, counts):
    """Mean of a union, weighting component means by their tetrahedron counts.

    The final axis indexes disjoint shape intervals within one catalogue.
    Empty component intervals contribute neither a sum nor a count.
    """
    means, counts = np.asarray(means, float), np.asarray(counts)
    if (means.ndim < 1 or means.shape != counts.shape or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts))
            or not np.isfinite(means[counts > 0]).all()):
        raise ValueError('Require matched means and nonnegative integer counts')
    total = counts.sum(axis=-1)
    sums = (np.where(counts > 0, means, 0)*counts).sum(axis=-1)
    result = np.full(np.shape(total), np.nan)
    np.divide(sums, total, out=result, where=total > 0)
    return result, total


def prepare(source, out, project):
    source, out, project = (Path(p).resolve() for p in (source, out, project))
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source/'protocol.json').read_text())
    expected = json.loads((source/'output_sha256.json').read_text())
    for name in ('catalogue_signals.csv', 'analysis.json'):
        if digest(source/name) != expected[name]:
            raise ValueError(f'Saved input changed: {name}')
    cuts = {c['id']: c for c in protocol['cuts']}
    assert (cuts['Q_2']['low'], cuts['Q_2']['high']) == (.4, .6)
    assert (cuts['Q_3']['low'], cuts['Q_3']['high']) == (.6, .8)
    target = project/'results/field_PVdisp_selection.ipynb'
    (out/'before').mkdir()
    shutil.copy2(target, out/'before'/target.name)
    shutil.copytree(project/'src', out/'code/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    (out/'code/tests').mkdir()
    shutil.copy2(project/'tests/test_selection_comparison.py', out/'code/tests')
    shutil.copy2(project/'slurm/selection_comparison.sh', out/'code/selection_comparison.sh')
    # These small inputs retain the sufficient statistics used for the union.
    (out/'inputs').mkdir()
    for name in ('protocol.json', 'catalogue_signals.csv', 'analysis.json'):
        shutil.copy2(source/name, out/'inputs'/name)
    direct_files = [f'records/{case}/seed_{seed:04d}_fraction_{f:g}.npz'
                    for seed in (protocol['seeds'][0], protocol['seeds'][len(protocol['seeds'])//2],
                                 protocol['seeds'][-1])
                    for f in protocol['fractions'] for case in CASES[:2]]
    config = dict(source=str(source), target=str(target), previous_notebook_sha256=digest(target),
                  seeds=protocol['seeds'], fractions=protocol['fractions'],
                  n_particles=protocol['Ngal'], cases=list(CASES), selections=SELECTIONS,
                  wide_interval=dict(variable='Q', low=.4, high=.8, upper_inclusive=False),
                  population_summary='Mean count and retained fraction over the two helicity catalogues in complete pairs.',
                  signal='Within each catalogue combine disjoint-bin chirality sums and counts, then form matched D = S(+1)-S(-1).',
                  uncertainty='Equal weights across available complete IC pairs; SEM = sample std(D) / sqrt(n_pairs).',
                  scope='Q range requested after inspecting these same 50 ICs; exploratory comparison, original r cut remains fiducial.',
                  output_link=f'saved/{source.name}/{out.name}',
                  direct_check_sha256={name: expected[name] for name in direct_files})
    atomic_json(out/'config.json', config)
    paths = [out/'config.json', *sorted((out/'code').rglob('*')), *sorted((out/'inputs').glob('*'))]
    atomic_json(out/'manifest.json', {'sha256': {
        str(p.relative_to(out)):digest(p) for p in paths if p.is_file()}})


def analyze(out):
    out = Path(out)
    check_manifest(out)
    config = json.loads((out/'config.json').read_text())
    with (out/'inputs/catalogue_signals.csv').open() as stream:
        raw = list(csv.DictReader(stream))
    index = {(int(r['seed']), float(r['fraction']), r['case'], r['selection']):r for r in raw}
    assert len(index) == len(raw), 'Duplicate catalogue measurements'
    selections, fractions, seeds = config['selections'], config['fractions'], config['seeds']
    signals = np.full((len(fractions), len(seeds), len(CASES), len(selections)), np.nan)
    counts = np.zeros(signals.shape, dtype=np.int64)
    totals = np.zeros(signals.shape[:-1], dtype=np.int64)
    catalogue_rows = []
    for fi, fraction in enumerate(fractions):
        for si, seed in enumerate(seeds):
            for ci, case in enumerate(CASES):
                total = int(index[seed, fraction, case, 'all']['total'])
                totals[fi, si, ci] = total
                for ti, cut in enumerate(selections):
                    parts = [index[seed, fraction, case, part] for part in cut['parts']]
                    assert all(int(r['total'])==total for r in parts)
                    signal, count = combine_disjoint([float(r['S']) for r in parts],
                                                     [int(r['selected']) for r in parts])
                    assert count <= total
                    signals[fi, si, ci, ti] = signal
                    counts[fi, si, ci, ti] = count
                    catalogue_rows.append(dict(seed=seed, fraction=fraction, case=case,
                        selection=cut['id'], S=float(signal) if count else None,
                        selected=int(count), total=total, retained_fraction=float(count/total)))
    direct_checks = []
    for name, expected in config['direct_check_sha256'].items():
        path = Path(config['source'])/name
        assert digest(path)==expected
        with np.load(path) as saved:
            tets = saved['tetrahedra']
            meta = json.loads(str(saved['metadata_json']))
        quality = tetrahedron_geometry(tets)['quality']
        keep = (quality >= .4) & (quality < .8)
        fi, si, ci = fractions.index(meta['fraction']), seeds.index(meta['seed']), CASES.index(meta['case'])
        assert int(keep.sum())==counts[fi, si, ci, 3]
        np.testing.assert_allclose(tets[keep,1].mean(), signals[fi,si,ci,3], rtol=1e-12, atol=1e-14)
        direct_checks.append(name)
    summaries, differences, controls = [], [], []
    coarse = json.loads((out/'inputs/analysis.json').read_text())
    for fi, fraction in enumerate(fractions):
        delta, n, mean, sem = paired_summary(signals[fi,:,0,:], signals[fi,:,1,:])
        differences.append(delta)
        for ti, cut in enumerate(selections):
            valid = np.isfinite(delta[:,ti])
            for ci in (2,3):
                values = signals[fi,:,ci,ti]
                values = values[np.isfinite(values)]
                controls.append(dict(fraction=fraction,selection=cut['id'],case=CASES[ci],n=len(values),
                    mean=float(values.mean()),sem=float(values.std(ddof=1)/np.sqrt(len(values)))))
            entry = dict(fraction=fraction,n_particles=round(config['n_particles']*fraction),
                selection=cut['id'],label=cut['label'],n_pairs=int(n[ti]),
                mean_D=float(mean[ti]),sem_D=float(sem[ti]),
                signal_to_sem=float(abs(mean[ti])/sem[ti]) if sem[ti]>0 else None,
                mean_selected=float(counts[fi,valid,:2,ti].mean()),
                retained_fraction=float((counts[fi,valid,:2,ti]/totals[fi,valid,:2]).mean()))
            summaries.append(entry)
            if cut['id']!='Q_wide':
                original = next(r for r in coarse['summaries'] if r['fraction']==fraction and r['selection']==cut['id'])
                np.testing.assert_allclose([entry['mean_D'],entry['sem_D']],
                                           [original['mean'],original['sem']],rtol=1e-12,atol=1e-14)
    gains = []
    for fraction in fractions:
        small = next(r for r in summaries if r['fraction']==fraction and r['selection']=='Q_2')
        wide = next(r for r in summaries if r['fraction']==fraction and r['selection']=='Q_wide')
        gains.append(dict(n_particles=wide['n_particles'],fraction=fraction,
            count_gain=wide['mean_selected']/small['mean_selected'],
            amplitude_ratio=abs(wide['mean_D']/small['mean_D']),sem_ratio=wide['sem_D']/small['sem_D'],
            sensitivity_gain=wide['signal_to_sem']/small['signal_to_sem']))
    atomic_npz(out/'catalogue_selections.npz',seeds=np.array(seeds),fractions=np.array(fractions),
        cases=np.array(CASES),selections=np.array([c['id'] for c in selections]),
        signals=signals,counts=counts,totals=totals,differences=np.array(differences))
    for name, rows in [('catalogue_signals',catalogue_rows),('summary',summaries),('gains',gains)]:
        with (out/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=dict(complete=True,config_sha256=digest(out/'config.json'),summaries=summaries,
                gains=gains,controls=controls,direct_checks=direct_checks,
                array_axes=dict(signals=['fraction','seed','case','selection'],
                    counts=['fraction','seed','case','selection'],totals=['fraction','seed','case'],
                    differences=['fraction','seed','selection']),scope=config['scope'])
    atomic_json(out/'analysis.json',report)
    print(json.dumps(gains,indent=2),flush=True)
    return report


def figure(out):
    import matplotlib.pyplot as plt
    out=Path(out)
    report=json.loads((out/'analysis.json').read_text())
    config=json.loads((out/'config.json').read_text())
    assert report['config_sha256']==digest(out/'config.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    colors=['#777777','#377eb8','#e08214','#238b45']
    for cut,color in zip(config['selections'],colors):
        rows=sorted((r for r in report['summaries'] if r['selection']==cut['id']),key=lambda r:r['n_particles'])
        x=[r['n_particles'] for r in rows]
        axes[0].errorbar(x,[r['mean_D'] for r in rows],yerr=[r['sem_D'] for r in rows],
                         color=color,marker='o',capsize=3,label=cut['label'])
        axes[1].plot(x,[r['mean_selected'] for r in rows],color=color,marker='o')
        axes[2].plot(x,[r['signal_to_sem'] for r in rows],color=color,marker='o')
    axes[0].axhline(0,color='gray',lw=.8)
    axes[0].set(title='Catalogue signal',ylabel=r'$\overline{D}\ \pm\ \mathrm{SEM}$')
    axes[0].legend(loc='lower left',fontsize=8)
    axes[1].set(yscale='log',title='Tetrahedra retained',ylabel='Mean selected count per catalogue')
    axes[2].set(title='Signal relative to uncertainty',ylabel=r'$|\overline{D}|\,/\,\mathrm{SEM}$')
    for ax in axes:
        ax.set_xscale('log',base=2)
        ax.set_xticks([8192,16384,32768],labels=['8,192','16,384','32,768'])
        ax.set_xlabel('Particles per catalogue'); ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Wider Q selection · {len(config["seeds"])} matched initial conditions · A = 16 h⁻¹ Mpc')
    (out/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(out/'figures'/f'Q_selection_comparison.{ext}',dpi=180)
    return fig


def summary_markdown(out):
    report=json.loads((Path(out)/'analysis.json').read_text())
    lines=['| Selection | Particles | Pairs | Mean tetrahedra | Retained | Mean D ± SEM | Signal / SEM |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['summaries']:
        lines.append(f"| {r['label']} | {r['n_particles']:,} | {r['n_pairs']} | {r['mean_selected']:,.0f} | "
                     f"{r['retained_fraction']:.2%} | {r['mean_D']:+.5f} ± {r['sem_D']:.5f} | {r['signal_to_sem']:.1f} |")
    lines.extend(['','**Wider Q interval compared with 0.4 ≤ Q < 0.6:**',''])
    for r in report['gains']:
        lines.append(f"- **{r['n_particles']:,} particles:** {r['count_gain']:.2f}× as many tetrahedra; "
                     f"{r['sensitivity_gain']:.2f}× the signal-to-SEM ratio.")
    return '\n'.join(lines)


def render(out):
    import nbformat
    from nbclient import NotebookClient
    out=Path(out).resolve()
    config=json.loads((out/'config.json').read_text())
    target=Path(config['target'])
    nb=nbformat.read(out/'before'/target.name,as_version=4)
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    section=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 4.'))
    cells=[md(r'''## 4. Does the wider $0.4\le Q<0.8$ selection help?

Compare the new interval with $0.4\le Q<0.6$, the frozen $r$ cut, and all tetrahedra, using the same 50 realizations. The wider cut combines the chirality sums and tetrahedron counts of the two adjacent intervals **within each catalogue**, before calculating $D_s$.

Counts and retained fractions below average over the two helicity catalogues. The sensitivity measure is $|\overline{D}|/\mathrm{SEM}$: a larger value means a larger mean signal relative to its uncertainty in this ensemble.
'''),code(f'''from pathlib import Path
import importlib.util, json, sys
from IPython.display import display, Markdown
COMPARE = Path({str(out)!r})
sys.path.insert(0, str(COMPARE/'code/src'))
spec = importlib.util.spec_from_file_location('pv_selection_comparison', COMPARE/'code/src/parity_study/tests/selection_comparison.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
%matplotlib inline
import matplotlib.pyplot as plt
fig = comparison.figure(COMPARE)
display(fig)
plt.close(fig)
display(Markdown(comparison.summary_markdown(COMPARE)))
'''),md(r'''The wider $Q$ cut improves sensitivity at 32,768 and 16,384 particles. At 8,192 particles, it retains more tetrahedra but dilutes the mean signal enough to reduce sensitivity. The frozen $r$ cut still gives the highest sensitivity at the fiducial 32,768 particles.

The wider $Q$ interval was chosen after viewing these same data: this comparison is exploratory. The frozen $r$ selection remains the fiducial cut.
''')]
    previous_links=nb.cells[-1].source[nb.cells[-1].source.index('[Fine-bin data'):]
    link=config['output_link']
    cells.append(md(previous_links+f'\n\n[Selection comparison data]({link}/summary.csv) · '
        f'[Per-catalogue signals]({link}/catalogue_signals.csv) · [Comparison plot]({link}/figures/) · '
        f'[Comparison configuration]({link}/config.json)'))
    # Execute the new comparison independently; existing figures and measurements stay intact.
    addition=nbformat.v4.new_notebook(cells=cells,metadata=nb.metadata)
    NotebookClient(addition,timeout=240,kernel_name='python3',
                   resources={'metadata':{'path':str(out)}}).execute()
    count=max(c.execution_count or 0 for c in nb.cells[:section] if c.cell_type=='code')
    for cell in addition.cells:
        if cell.cell_type=='code':
            count+=1; cell.execution_count=count
            cell.metadata['jupyter']={'source_hidden':True}
    nb.cells=nb.cells[:section]+addition.cells
    nbformat.validate(nb)
    assert digest(target)==config['previous_notebook_sha256'], 'Notebook changed during comparison'
    temporary=target.with_suffix('.tmp.ipynb'); nbformat.write(nb,temporary); temporary.replace(target)
    shutil.copy2(target,out/target.name)
    atomic_json(out/'report_provenance.json',dict(notebook=str(target),notebook_sha256=digest(target),
        previous_notebook_sha256=config['previous_notebook_sha256'],
        source=str(Path(__file__).resolve()),preserved_cells=list(range(section)),
        analysis_sha256=digest(out/'analysis.json'),config_sha256=digest(out/'config.json')))
    paths=[out/'analysis.json',out/'summary.csv',out/'catalogue_signals.csv',out/'gains.csv',
           out/'catalogue_selections.npz',out/'report_provenance.json',out/target.name,*sorted((out/'figures').glob('*'))]
    atomic_json(out/'output_sha256.json',{str(p.relative_to(out)):digest(p) for p in paths})
    print(f'Updated {target}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--project',type=Path)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.source,args.out,args.project)
    else:
        analyze(args.out)
        render(args.out)


if __name__=='__main__': main()
