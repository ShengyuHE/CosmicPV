"""Add fiducial-Q catalogue distributions to the existing confirmation notebook."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.geometry import tetrahedron_geometry
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest
from parity_study.summarize import mean_error
from parity_study.tests.best_confirmation import distribution_summary


def prepare(source, confirmation, out, project):
    source, confirmation, out, project = (Path(p).resolve() for p in
                                         (source, confirmation, out, project))
    expected = json.loads((source/'output_sha256.json').read_text())
    for name in ('analysis.json', 'catalogue_selections.npz'):
        assert digest(source/name) == expected[name]
    source_config = json.loads((source/'config.json').read_text())
    assert source_config['wide_interval'] == dict(variable='Q', low=.4, high=.8,
                                                upper_inclusive=False)
    out.mkdir(parents=True, exist_ok=False)
    target = project/'results/field_PVdisp_B.ipynb'
    (out/'before').mkdir()
    shutil.copy2(target, out/'before'/target.name)
    shutil.copytree(project/'src', out/'code/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    for folder, original, names in (
        ('selection', source, ('config.json', 'analysis.json', 'catalogue_selections.npz')),
        ('confirmation', confirmation, ('protocol.json', 'analysis.json', 'confirmation.npz'))):
        destination = out/'inputs'/folder
        destination.mkdir(parents=True)
        for name in names:
            shutil.copy2(original/name, destination/name)
    geometry = Path(source_config['source'])
    hashes = json.loads((geometry/'output_sha256.json').read_text())
    seeds = source_config['seeds']
    names = [f'records/null_symmetric/seed_{s:04d}_fraction_1.npz'
             for s in (seeds[0], seeds[len(seeds)//2], seeds[-1])]
    config = dict(source=str(source), confirmation=str(confirmation), geometry=str(geometry),
        target=str(target), previous_notebook_sha256=digest(target),
        selection=dict(id='Q_wide', variable='Q', low=.4, high=.8, upper_inclusive=False),
        fraction=1., n_particles=source_config['n_particles'], seeds=seeds,
        epsilons=[-1, 0, 1], cases=['scan_A16_minus', 'null_symmetric', 'scan_A16_plus'],
        direct_check_sha256={name: hashes[name] for name in names},
        scope='Fiducial Q interval chosen after viewing these same 50 ICs; descriptive distributions, not a new independent confirmation.',
        output_link=f'saved/field_PVdisp_B/{out.name}')
    atomic_json(out/'config.json', config)
    paths = [out/'config.json', *sorted((out/'inputs').rglob('*')), *sorted((out/'code').rglob('*'))]
    atomic_json(out/'manifest.json', dict(sha256={str(p.relative_to(out)): digest(p)
                                               for p in paths if p.is_file()}))


def analyze(out):
    out = Path(out)
    check_manifest(out)
    config = json.loads((out/'config.json').read_text())
    protocol = json.loads((out/'inputs/confirmation/protocol.json').read_text())
    original = json.loads((out/'inputs/confirmation/analysis.json').read_text())
    prior = json.loads((out/'inputs/selection/analysis.json').read_text())
    assert original['complete'] and original['independent_ic_audit']['passed']
    assert original['protocol_sha256'] == digest(out/'inputs/confirmation/protocol.json')
    source_config = json.loads((out/'inputs/selection/config.json').read_text())
    assert prior['complete'] and prior['config_sha256'] == digest(out/'inputs/selection/config.json')
    assert config['seeds'] == protocol['seeds'] == source_config['seeds']
    assert config['epsilons'] == protocol['epsilons']
    with np.load(out/'inputs/selection/catalogue_selections.npz') as z:
        seeds = z['seeds']
        assert seeds.tolist() == config['seeds']
        fi = z['fractions'].tolist().index(config['fraction'])
        ci = [z['cases'].tolist().index(case) for case in config['cases']]
        qi = z['selections'].tolist().index('Q_wide')
        signals = z['signals'][fi, :, :, qi][:, ci].T[None, :, :]
        counts = z['counts'][fi, :, :, qi][:, ci].T[None, :, :]
        totals = z['totals'][fi][:, ci].T
        # Check all 150 catalogue identities and helicity ordering against the original report.
        with np.load(out/'inputs/confirmation/confirmation.npz') as old:
            np.testing.assert_array_equal(old['seeds'], seeds)
            np.testing.assert_array_equal(old['epsilons'], config['epsilons'])
            for k, cut in enumerate(protocol['cuts']):
                name = 'all' if cut['id'] == 'all' else 'reference_ratio'
                assert cut['id'] == 'all' or cut['id'] == protocol['primary_selection']
                ti = z['selections'].tolist().index(name)
                np.testing.assert_allclose(z['signals'][fi, :, :, ti][:, ci].T,
                                           old['signals'][k], rtol=1e-12, atol=1e-14)
                np.testing.assert_array_equal(z['counts'][fi, :, :, ti][:, ci].T, old['counts'][k])
            np.testing.assert_array_equal(totals, old['totals'])
        paired = signals[:, 2] - signals[:, 0]
        np.testing.assert_allclose(paired[0], z['differences'][fi, :, qi])
    assert signals.shape == (1, 3, len(seeds)) and np.isfinite(signals).all()
    assert np.all(counts > 0) and np.all(counts <= totals)
    for name, sha in config['direct_check_sha256'].items():
        path = Path(config['geometry'])/name
        assert digest(path) == sha
        with np.load(path) as z:
            tets = z['tetrahedra']
            metadata = json.loads(str(z['metadata_json']))
        assert metadata['case'] == 'null_symmetric' and metadata['fraction'] == 1.
        quality = tetrahedron_geometry(tets)['quality']
        keep = (quality >= .4) & (quality < .8)
        si = config['seeds'].index(metadata['seed'])
        assert keep.sum() == counts[0, 1, si]
        np.testing.assert_allclose(tets[keep, 1].mean(), signals[0, 1, si], rtol=1e-12, atol=1e-14)
    distributions = distribution_summary(signals, [config['selection']])
    for ei, row in enumerate(distributions):
        row['mean_selected'] = float(counts[0, ei].mean())
        row['retained_fraction'] = float((counts[0, ei]/totals[ei]).mean())
    difference = mean_error(paired[0])
    previous = next(r for r in prior['summaries'] if r['fraction'] == 1 and r['selection'] == 'Q_wide')
    np.testing.assert_allclose([difference['mean'], difference['sem']],
                               [previous['mean_D'], previous['sem_D']], rtol=1e-12, atol=1e-14)
    rows = [dict(seed=int(seed), epsilon=eps, S=float(signals[0, ei, si]),
                 selected=int(counts[0, ei, si]), total=int(totals[ei, si]))
            for si, seed in enumerate(seeds) for ei, eps in enumerate(config['epsilons'])]
    for name, data in [('catalogue_signals', rows), ('distribution_summary', distributions)]:
        with (out/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    atomic_npz(out/'catalogue_signals.npz', seeds=seeds, epsilons=np.array(config['epsilons']),
               signals=signals, counts=counts, totals=totals, paired=paired)
    report = dict(complete=True, A=protocol['A'], selection=config['selection'],
        config_sha256=digest(out/'config.json'), distributions=distributions, paired=difference,
        n_particles=config['n_particles'], seeds=config['seeds'], scope=config['scope'],
        validation=dict(original_catalogues_matched=3*len(seeds),
                        epsilon0_direct_checks=len(config['direct_check_sha256']),
                        paired_summary_matches_saved_comparison=True))
    atomic_json(out/'analysis.json', report)
    print(json.dumps(dict(distributions=distributions, paired=difference), indent=2), flush=True)


def render(out):
    import nbformat
    from nbclient import NotebookClient
    out = Path(out).resolve()
    config = json.loads((out/'config.json').read_text())
    target = Path(config['target'])
    nb = nbformat.read(out/'before'/target.name, as_version=4)
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    header = md(r'''## 4. Fiducial $Q$ selection: catalogue-signal distributions

Use $B_Q: 0.4\le Q<0.8$, where $Q=12(3V)^{2/3}/\sum_{i<j}\ell_{ij}^2$ uses the tetrahedron volume $V$ and all six edges. This selection is applied on its own, without the $r$ cut.

Each histogram contains **50 catalogue signals**, $S_{s,B_Q}(\epsilon)=\mathrm{mean}_{t\in B_Q}\widehat\chi_t$, at 32,768 particles. The Gaussian curves use the catalogue scatter $\sigma_S$ defined in Section 2. These are the same ICs as above; the $Q$ interval was chosen after inspecting them, so this comparison is exploratory.
''')
    plot = code(fr'''Q_OUT = Path({str(out)!r})
q_report = json.loads((Q_OUT/'analysis.json').read_text())
assert q_report['complete'] and q_report['config_sha256'] == digest(Q_OUT/'config.json')
with np.load(Q_OUT/'catalogue_signals.npz') as z:
    q_signals = z['signals']
    assert z['seeds'].tolist() == seeds.tolist()
    assert z['epsilons'].tolist() == [-1, 0, 1]
fig = distribution_figure(q_report, q_signals, selection_index=0)
ax = fig.axes[0]
ax.set_xlabel(r'Catalogue signal $S_{{s,B_Q}}(\epsilon)=\langle\widehat\chi\rangle_{{t\in B_Q}}$')
ax.set_title(ax.get_title() + chr(10) + r'Fiducial $Q$ selection: $0.4\leq Q<0.8$')
fig.tight_layout()
(Q_OUT/'figures').mkdir(exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(Q_OUT/'figures'/f'catalogue_signal_distributions_Q.{{ext}}', dpi=180, bbox_inches='tight')
display(fig)
plt.close(fig)
show_table(q_report['distributions'], [('epsilon', 'ε'), ('n', 'Catalogues'),
    ('mean', 'Mean μ'), ('sd', 'Scatter σS'), ('sem', 'SEM'), ('mean_selected', 'Mean tetrahedra')])
q_difference = q_report['paired']
display(Markdown(r'**Matched helicity difference:** $D_{{s,B_Q}}=S_{{s,B_Q}}(+1)-S_{{s,B_Q}}(-1)$.'
    + f" Mean = **{{q_difference['mean']:+.6f}} ± {{q_difference['sem']:.6f}} SEM** across {{q_difference['n']}} ICs."))
''')
    plot.metadata['jupyter'] = {'source_hidden': True}
    # Execute only the new plot, using the original notebook's setup and histogram helper.
    mini = nbformat.v4.new_notebook(cells=[copy.deepcopy(nb.cells[1]), plot], metadata=nb.metadata)
    NotebookClient(mini, timeout=180, kernel_name='python3',
                   resources={'metadata': {'path': str(out)}}).execute()
    plot = mini.cells[1]
    plot.execution_count = max(c.execution_count or 0 for c in nb.cells if c.cell_type == 'code') + 1
    nb.cells[0].source += '\nSections 1–3 report the original frozen $r$ selection; Section 4 adds the fiducial $Q$ selection on the same catalogues.\n'
    link = config['output_link']
    links = md(f'[Q catalogue signals]({link}/catalogue_signals.csv) · '
        f'[Q Gaussian parameters]({link}/distribution_summary.csv) · [Q figures]({link}/figures/) · '
        f'[Q selection configuration]({link}/config.json)')
    nb.cells.extend([header, plot, links])
    nbformat.validate(nb)
    assert digest(target) == config['previous_notebook_sha256'], 'Notebook changed during reporting'
    temporary = target.with_suffix('.tmp.ipynb'); nbformat.write(nb, temporary); temporary.replace(target)
    shutil.copy2(target, out/target.name)
    atomic_json(out/'report_provenance.json', dict(notebook=str(target), notebook_sha256=digest(target),
        previous_notebook_sha256=config['previous_notebook_sha256'], preserved_cells=list(range(1, 8)),
        config_sha256=digest(out/'config.json'), analysis_sha256=digest(out/'analysis.json')))
    paths = [out/'analysis.json', out/'catalogue_signals.npz', out/'catalogue_signals.csv',
             out/'distribution_summary.csv', out/'report_provenance.json', out/target.name,
             *sorted((out/'figures').glob('*'))]
    atomic_json(out/'output_sha256.json', {str(p.relative_to(out)): digest(p) for p in paths})
    print(f'Updated {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.confirmation, args.out, args.project)
    else:
        analyze(args.out)
        render(args.out)


if __name__ == '__main__':
    main()
