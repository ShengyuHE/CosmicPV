"""Fine-bin catalogue PV response in r and Q, using saved tetrahedra."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.geometry import tetrahedron_geometry
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest

VARIABLES = ('r', 'Q')
CASES = ('scan_A16_minus', 'scan_A16_plus')
FIDUCIAL_Q_CUT = dict(low=0.4, high=0.8,
                      label='Fiducial Q selection: 0.40 ≤ Q < 0.80')


def binned_signal(shape, chi_hat, edges):
    """Conditional catalogue means, with [low, high) and a closed last bin."""
    shape, chi_hat, edges = map(np.asarray, (shape, chi_hat, edges))
    if (shape.ndim != 1 or chi_hat.shape != shape.shape or edges.ndim != 1
            or len(edges) < 2 or np.any(np.diff(edges) <= 0)
            or not all(np.isfinite(a).all() for a in (shape, chi_hat, edges))
            or np.any(shape < edges[0]) or np.any(shape > edges[-1])):
        raise ValueError('Require finite measurements inside increasing bin edges')
    count = np.histogram(shape, edges)[0]
    sums = np.histogram(shape, edges, weights=chi_hat)[0]
    means = np.full(len(count), np.nan)
    np.divide(sums, count, out=means, where=count > 0)
    return means, count


def paired_summary(minus, plus):
    """Equal IC weights and SEM of paired differences; missing bins stay missing."""
    minus, plus = np.asarray(minus, float), np.asarray(plus, float)
    if minus.shape != plus.shape or minus.ndim != 2:
        raise ValueError('Expected matched (realization, bin) arrays')
    difference = plus - minus
    valid = np.isfinite(difference)
    n = valid.sum(axis=0)
    mean = np.full(n.shape, np.nan)
    np.divide(np.where(valid, difference, 0).sum(axis=0), n, out=mean, where=n > 0)
    sem = np.full(n.shape, np.nan)
    residual = np.where(valid, difference - mean, 0)
    np.divide((residual**2).sum(axis=0), n*(n-1), out=sem, where=n > 1)
    return difference, n, mean, np.sqrt(sem)


def prepare(source, out, project, *, code_root=None):
    source, out, project = (Path(p).resolve() for p in (source, out, project))
    code_root = Path(code_root).resolve() if code_root else project
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source/'protocol.json').read_text())
    hashes = json.loads((source/'output_sha256.json').read_text())
    inputs = {f'records/{case}/seed_{seed:04d}_fraction_{fraction:g}.npz':
              hashes[f'records/{case}/seed_{seed:04d}_fraction_{fraction:g}.npz']
              for fraction in protocol['fractions'] for seed in protocol['seeds'] for case in CASES}
    target = project/'results/field_PVdisp_selection.ipynb'
    archive = out/'before'
    archive.mkdir()
    shutil.copy2(target, archive/target.name)
    shutil.copytree(code_root/'src', out/'code/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    (out/'code/tests').mkdir()
    shutil.copy2(code_root/'tests/test_selection_bins.py', out/'code/tests')
    shutil.copy2(code_root/'slurm/selection_bins.sh', out/'code/selection_bins.sh')
    config = dict(source=str(source), seeds=protocol['seeds'], fractions=protocol['fractions'],
                  edges=np.linspace(0., 1., 51).tolist(), variables=list(VARIABLES),
                  cases=list(CASES), epsilons=[-1, 1], n_particles=protocol['Ngal'],
                  cuts={'r': next(c for c in protocol['cuts'] if c['id']=='reference_ratio'),
                        'Q': next(c for c in protocol['cuts'] if c['id']=='Q_2')},
                  estimator='Equal-weight mean of S_B(+1)-S_B(-1) over the complete matched ICs available in each bin; S_B is mean chi_hat within bin.',
                  uncertainty='Sample standard deviation of paired IC differences / sqrt(n_pairs).',
                  display='Show every bin with at least one complete matched IC; SEM requires at least two; no imputation.',
                  scope=f"Exploratory fine bins of {len(protocol['seeds'])} saved ICs; no selection retuning.",
                  output_link=f'saved/{source.name}/{out.name}',
                  target=str(target), previous_notebook_sha256=digest(target),
                  protocol_sha256=digest(source/'protocol.json'),
                  coarse_analysis_sha256=digest(source/'analysis.json'), input_sha256=inputs)
    atomic_json(out/'config.json', config)
    paths = [out/'config.json', *sorted((out/'code').rglob('*'))]
    atomic_json(out/'manifest.json', {'sha256': {
        str(p.relative_to(out)): digest(p) for p in paths if p.is_file()}})
    return config


def analyze(out):
    out = Path(out)
    check_manifest(out)
    config = json.loads((out/'config.json').read_text())
    source = Path(config['source'])
    assert digest(source/'protocol.json') == config['protocol_sha256']
    assert digest(source/'analysis.json') == config['coarse_analysis_sha256']
    edges = np.array(config['edges'])
    fractions, seeds = config['fractions'], config['seeds']
    # Axes: density, IC, helicity (-1,+1), variable (r,Q), bin.
    signal = np.full((len(fractions), len(seeds), 2, 2, len(edges)-1), np.nan)
    count = np.zeros(signal.shape, dtype=np.int64)
    coarse = {seed: json.loads((source/'completed'/f'seed_{seed:04d}.json').read_text())['rows']
              for seed in seeds}
    checked = 0
    for fi, fraction in enumerate(fractions):
        for si, seed in enumerate(seeds):
            for ei, case in enumerate(CASES):
                name = f'records/{case}/seed_{seed:04d}_fraction_{fraction:g}.npz'
                path = source/name
                if digest(path) != config['input_sha256'][name]:
                    raise ValueError(f'Saved catalogue changed: {path}')
                with np.load(path) as saved:
                    tets = saved['tetrahedra']
                    meta = json.loads(str(saved['metadata_json']))
                assert (meta['seed'], meta['fraction'], meta['case'], meta['epsilon']) == (
                    seed, fraction, case, config['epsilons'][ei])
                shapes = (tets[:, 2]/tets[:, 4], tetrahedron_geometry(tets)['quality'])
                for vi, (variable, values) in enumerate(zip(VARIABLES, shapes)):
                    signal[fi, si, ei, vi], count[fi, si, ei, vi] = binned_signal(values, tets[:, 1], edges)
                    counts = count[fi, si, ei, vi]
                    assert counts.sum() == len(tets)
                    reconstructed = np.nansum(signal[fi, si, ei, vi]*counts)/len(tets)
                    np.testing.assert_allclose(reconstructed, tets[:, 1].mean(), atol=1e-13)
                    cut = config['cuts'][variable]
                    keep = (values >= cut['low']) & (values < cut['high'])
                    old = next(row for row in coarse[seed] if row['fraction']==fraction
                               and row['case']==case and row['selection']==cut['id'])
                    np.testing.assert_allclose(tets[keep, 1].mean(), old['S'], atol=1e-13)
                    assert int(keep.sum()) == old['selected']
                    checked += 1
        print(f'Fine bins measured for {round(config["n_particles"]*fraction):,} particles', flush=True)
    rows, differences = [], []
    for fi, fraction in enumerate(fractions):
        variables = []
        for vi, variable in enumerate(VARIABLES):
            delta, n, mean, sem = paired_summary(signal[fi, :, 0, vi], signal[fi, :, 1, vi])
            variables.append(delta)
            for bi in range(len(edges)-1):
                rows.append(dict(variable=variable, fraction=fraction,
                    n_particles=round(config['n_particles']*fraction),
                    low=float(edges[bi]), high=float(edges[bi+1]),
                    n_pairs=int(n[bi]), plotted=bool(n[bi]>0),
                    mean_D=float(mean[bi]) if np.isfinite(mean[bi]) else None,
                    sem_D=float(sem[bi]) if np.isfinite(sem[bi]) else None,
                    min_count=int(count[fi, :, :, vi, bi].min()),
                    mean_count=float(count[fi, :, :, vi, bi].mean())))
        differences.append(variables)
    atomic_npz(out/'catalogue_bins.npz', edges=edges, seeds=np.array(seeds),
               fractions=np.array(fractions), epsilons=np.array(config['epsilons']),
               variables=np.array(VARIABLES), signal=signal, count=count,
               difference=np.transpose(np.array(differences), (0, 2, 1, 3)))
    with (out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(complete=True, config_sha256=digest(out/'config.json'), rows=rows,
                  original_cut_checks=checked, catalogues_checked=len(config['input_sha256']),
                  array_axes=dict(signal=['fraction','seed','epsilon','variable','bin'],
                                  count=['fraction','seed','epsilon','variable','bin'],
                                  difference=['fraction','seed','variable','bin']))
    atomic_json(out/'analysis.json', report)
    return report


def plot_response(out, variable, *, omit_first_populated=0, figure_dir=None, highlight=None):
    """Plot catalogue differences, optionally hiding early populated bins per panel.

    Omission affects only the display; all measurements remain in the saved data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out = Path(out)
    if variable not in VARIABLES:
        raise ValueError(variable)
    if not isinstance(omit_first_populated, int) or omit_first_populated < 0:
        raise ValueError('omit_first_populated must be a nonnegative integer')
    config = json.loads((out/'config.json').read_text())
    report = json.loads((out/'analysis.json').read_text())
    assert report['config_sha256'] == digest(out/'config.json')
    fig, axes = plt.subplots(len(config['fractions']), 1, figsize=(10, 8.2),
                             sharex=True, sharey=True, constrained_layout=True)
    colors = ['#2c7fb8', '#e08214', '#756bb1']
    cut = config['cuts'][variable] if highlight is None else highlight
    if not 0 <= cut['low'] < cut['high'] <= 1:
        raise ValueError('Highlighted shape interval must lie within [0, 1]')
    for ax, fraction, color in zip(np.atleast_1d(axes), config['fractions'], colors):
        rows = [r for r in report['rows'] if r['variable']==variable and r['fraction']==fraction]
        low, high = np.array([r['low'] for r in rows]), np.array([r['high'] for r in rows])
        available = np.array([r['plotted'] for r in rows])
        omitted = np.flatnonzero(available)[:omit_first_populated]
        valid = available.copy()
        valid[omitted] = False
        pairs = np.array([r['n_pairs'] for r in rows])
        mean = np.array([np.nan if r['mean_D'] is None else r['mean_D'] for r in rows])
        sem = np.array([np.nan if r['sem_D'] is None else r['sem_D'] for r in rows])
        ax.axvspan(cut['low'], cut['high'], color='#f7cc64', alpha=.32, zorder=0)
        for left, right in zip(low[~available], high[~available]):
            ax.axvspan(left, right, facecolor='#eeeeee', hatch='///', edgecolor='#bbbbbb', lw=0, zorder=0)
        for left, right in zip(low[omitted], high[omitted]):
            ax.axvspan(left, right, facecolor='#eeeeee', lw=0, zorder=0)
        ax.bar(low[valid], mean[valid], width=(high-low)[valid], align='edge',
               facecolor=color, edgecolor=color, alpha=.62, linewidth=.6, zorder=2)
        has_sem = valid & np.isfinite(sem)
        ax.errorbar(((low+high)/2)[has_sem], mean[has_sem], yerr=sem[has_sem], fmt='none',
                    ecolor='#222222', elinewidth=.8, capsize=1.5, zorder=3)
        for x, n in zip(((low+high)/2)[valid & (pairs<len(config['seeds']))],
                        pairs[valid & (pairs<len(config['seeds']))]):
            ax.text(x, .96, str(n), transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5),
                    zorder=4)
        ax.axhline(0, color='#333333', lw=.8)
        ax.set_title(f"{round(config['n_particles']*fraction):,} particles · "
                     f"{len(config['seeds'])} initial conditions; available pairs per bin", loc='left', fontsize=11)
        ax.set(xlim=(0, 1), xticks=np.linspace(0, 1, 11))
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    label = (r'$r=\ell_{\min}/\ell_{\max}$ (three reference edges)' if variable=='r'
             else r'$Q=12(3V)^{2/3}/\sum_{i<j}\ell_{ij}^2$ (all six edges)')
    axes[-1].set_xlabel(label)
    fig.supylabel(r'$\overline{D}_B=\overline{S_B(+1)-S_B(-1)}$', fontsize=12)
    selection = cut.get('label',
        f'Frozen r selection: {cut["low"]:.5f} ≤ r < {cut["high"]:.5f}' if variable=='r'
        else f'Exploratory Q selection: {cut["low"]:.2f} ≤ Q < {cut["high"]:.2f}')
    handles = [Patch(facecolor='#f7cc64', alpha=.45, label=selection)]
    if omit_first_populated:
        handles.append(Patch(facecolor='#eeeeee',
                             label=f'Hidden: first {omit_first_populated} populated bins per panel'))
    if any(not r['plotted'] for r in report['rows'] if r['variable']==variable):
        handles.append(Patch(facecolor='#eeeeee', edgecolor='#bbbbbb', hatch='///',
                             label='Unavailable: no complete matched pair'))
    axes[0].legend(handles=handles, loc='upper left', fontsize=8, framealpha=.95)
    fig.suptitle(f'PV response across {variable} bins · width 0.02 · A = 16 h⁻¹ Mpc\n'
                 f'Mean paired difference ±1 SEM; small numbers give pair counts below {len(config["seeds"])}', fontsize=12)
    folder = Path(figure_dir) if figure_dir is not None else out/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'{variable}_fine_bins.{extension}', dpi=180)
    return fig


def build_notebook(out):
    import nbformat
    out = Path(out).resolve()
    config = json.loads((out/'config.json').read_text())
    target = Path(config['target'])
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[md(r'''# Which tetrahedron shapes carry the PV signal?

**Compare the catalogue helicity difference in fine bins of $r$ and $Q$.**
__N_IC__ Gaussian initial conditions (seeds __FIRST_SEED__–__LAST_SEED__), displacement strength $A=16\ h^{-1}\mathrm{Mpc}$,
$L=1000\ h^{-1}\mathrm{Mpc}$ and field mesh $256^3$.
Each row uses the same fields with 32,768, 16,384 or 8,192 particles; random subsets are matched between helicities.
__ENSEMBLE_DESCRIPTION__
'''), md(r'''## 1. Catalogue signal and uncertainty

Vertices are ordered by increasing sums of squared distances to the other vertices; vectors $\mathbf v_1,\mathbf v_2,\mathbf v_3$ start at the first vertex. The tetrahedron chirality is
$$\widehat\chi=\frac{\mathbf v_1\cdot(\mathbf v_2\times\mathbf v_3)}{|\mathbf v_1||\mathbf v_2||\mathbf v_3|}.$$
Exact ordering ties have $\widehat\chi=0$.

For shape bin $B$ and realization $s$,
$$S_{s,B}(\epsilon)=\frac{1}{N_{s,B}(\epsilon)}\sum_{t\in B}\widehat\chi_t(\epsilon),\qquad
D_{s,B}=S_{s,B}(+1)-S_{s,B}(-1),\qquad
\overline D_B=\frac1{n_B}\sum_{s\in\mathcal C_B}D_{s,B}.$$
$\epsilon=+1,-1$ denotes the two helicities; $N_{s,B}$ counts selected tetrahedra.
$\mathcal C_B$ contains the realizations with at least one tetrahedron in bin $B$ for **both helicities**, and $n_B=|\mathcal C_B|$ is their count (up to __N_IC__). **Each bar height is $\overline D_B$**. Each complete pair has equal weight; incomplete pairs are omitted from that bin. A negative bar means the positive-helicity catalogue has a lower mean chirality there.

Bins have width **0.02**, include their lower edge and exclude their upper edge (the last bin includes 1).
Error bars show one **standard error of the mean (SEM)** over those same complete pairs: $\mathrm{std}_{s\in\mathcal C_B}(D_{s,B},\mathrm{ddof}=1)/\sqrt{n_B}$.
Small numbers above bins give $n_B$ where it is below __N_IC__; unlabelled bars use all __N_IC__ pairs. Hatching means $n_B=0$; a single pair is shown without an SEM. Small populations can give noisy bars; per-bin counts are saved with the results.
'''), code(f'''from pathlib import Path
import json, sys
from IPython.display import display, Markdown
OUT = Path({str(out)!r})
sys.path.insert(0, str(OUT/'code/src'))
from parity_study.tests.selection_bins import plot_response, FIDUCIAL_Q_CUT
from parity_study.storage import digest
report = json.loads((OUT/'analysis.json').read_text())
config = json.loads((OUT/'config.json').read_text())
assert report['complete'] and report['config_sha256'] == digest(OUT/'config.json')
%matplotlib inline
import matplotlib.pyplot as plt
'''), md(r'''## 2. Response in $r$ bins

$r=\ell_{\min}/\ell_{\max}$ is the ratio of the shortest to longest of the **three edges from the canonical reference vertex**. It is dimensionless and unchanged by reflection.

Gold shading marks the frozen selection $0.6905111410\le r<0.7567675935$. The three rows share both axes for direct comparison.
The first three populated bins in each row are hidden for readability; their measurements remain in the saved data.
'''), code("fig = plot_response(OUT, 'r', omit_first_populated=3)\ndisplay(fig)\nplt.close(fig)"),
        md(r'''## 3. Response in $Q$ bins

$Q=12(3V)^{2/3}/\sum_{i<j}\ell_{ij}^{2}$ uses the **volume $V$ and all six edges**. $Q=1$ is regular; $Q\to0$ is degenerate. It is dimensionless and unchanged by reflection.

Gold shading marks the **fiducial $Q$ selection, $0.4\le Q<0.8$**. Each bar applies only its own $Q$ interval; no $r$ cut is also imposed.
'''), code("fig = plot_response(OUT, 'Q', highlight=FIDUCIAL_Q_CUT)\ndisplay(fig)\nplt.close(fig)"),
        md(r'''## 4. Comparison with the existing selections

The values below use each full selection directly. A wider selection must be recomputed from its tetrahedra: averaging the fine-bin bar heights would ignore their different populations.
'''), code('''source = Path(config['source'])
assert digest(source/'analysis.json') == config['coarse_analysis_sha256']
coarse = json.loads((source/'analysis.json').read_text())
labels = {'all': 'All tetrahedra', 'reference_ratio': 'Frozen r cut', 'Q_2': 'Q ∈ [0.4, 0.6) (exploratory)'}
lines = ['| Selection | Particles | Mean D ± SEM |', '|---|---:|---:|']
for row in coarse['summaries']:
    if row['selection'] in labels:
        lines.append(f"| {labels[row['selection']]} | {row['n_particles']:,} | "
                     f"{row['mean']:+.5f} ± {row['sem']:.5f} |")
display(Markdown('\\n'.join(lines)))
'''), md(r'''Larger $|\overline D_B|$ alone does not establish the best selection: its uncertainty also matters. The fiducial $Q$ interval, $0.4\le Q<0.8$, was chosen after viewing these data, so this comparison remains exploratory. The frozen $r$ cut remains the $r$-based reference.

[Fine-bin data and counts](__OUTPUT_LINK__/summary.csv) · [Per-realization arrays](__OUTPUT_LINK__/catalogue_bins.npz) · [Plots](__OUTPUT_LINK__/figures/) · [Configuration](__OUTPUT_LINK__/config.json)
''')], metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')))
    n = len(config['seeds'])
    description = (f'The ensemble includes the original ten shape-study realizations and {n-10} additional '
                   'realizations from the saved confirmation run. This combined comparison remains exploratory.'
                   if n > 10 and config['seeds'][:10] == list(range(1000,1010)) else
                   'Saved confirmation catalogues are reused for this exploratory shape comparison.')
    replacements = {'__N_IC__': str(n), '__FIRST_SEED__': str(config['seeds'][0]),
                    '__LAST_SEED__': str(config['seeds'][-1]), '__ENSEMBLE_DESCRIPTION__': description,
                    '__OUTPUT_LINK__': config.get('output_link', f'saved/{Path(config["source"]).name}/{out.name}')}
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            for token, value in replacements.items():
                cell.source = cell.source.replace(token, value)
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    return nb


def render(out):
    import nbformat
    from nbclient import NotebookClient
    out = Path(out).resolve()
    config = json.loads((out/'config.json').read_text())
    target = Path(config['target'])
    nb = build_notebook(out)
    NotebookClient(nb, timeout=240, kernel_name='python3',
                   resources={'metadata': {'path': str(out)}}).execute()
    nbformat.validate(nb)
    if digest(target) != config['previous_notebook_sha256']:
        raise RuntimeError('Notebook changed since preparation; refusing to overwrite concurrent edits')
    temporary = target.with_suffix('.tmp.ipynb')
    nbformat.write(nb, temporary)
    temporary.replace(target)
    shutil.copy2(target, out/target.name)
    atomic_json(out/'report_provenance.json', dict(notebook=str(target), notebook_sha256=digest(target),
        source=str(Path(__file__).resolve()), config_sha256=digest(out/'config.json'),
        analysis_sha256=digest(out/'analysis.json'), previous_notebook_sha256=config['previous_notebook_sha256']))
    files = [out/'analysis.json', out/'summary.csv', out/'catalogue_bins.npz',
             out/'report_provenance.json', out/target.name, *sorted((out/'figures').glob('*'))]
    atomic_json(out/'output_sha256.json', {str(p.relative_to(out)): digest(p) for p in files})
    print(f'Updated and executed {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--project', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.out, args.project)
    else:
        analyze(args.out)
        render(args.out)


if __name__ == '__main__':
    main()
