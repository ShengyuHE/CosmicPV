"""Fixed geometric cuts under matched thinning of existing PV catalogues.

Exploratory robustness study; reuses previously inspected initial conditions.
No new Gaussian fields, no per-catalogue quantile fitting, no new detection claim.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.delaunay import parity_stat
from parity_study.geometry import tetrahedron_geometry
from parity_study.storage import atomic_json, atomic_npz, digest, check_manifest
from parity_study.summarize import mean_error

CASES = ('scan_A16_minus','null_symmetric','scan_A16_plus','null_za')
EPSILONS = (-1,0,1,None)


def prepare(out,source,project, *, seed_start=1000, n_seeds=10):
    """Freeze the pilot design, input hashes, code and executable before analysis."""
    out,source,project=map(lambda p:Path(p).resolve(),(out,source,project))
    if (out/'protocol.json').exists(): raise FileExistsError('Pilot protocol already exists')
    out.mkdir(parents=True,exist_ok=True)
    check_manifest(source)
    field=json.loads((source/'config.json').read_text())['field']
    if field['L']!=1000 or field['N']!=256 or field['Ngal']!=32768:
        raise ValueError('This pilot is defined for the current L1000/N256/32768-particle run')
    if n_seeds < 2: raise ValueError('At least two independent initial conditions are required')
    seeds=list(range(seed_start,seed_start+n_seeds))
    files=[source/'records'/case/f'seed_{seed:04d}.npz' for seed in seeds for case in CASES]
    protocol=dict(experiment='fixed_geometric_selection_under_matched_thinning',
        source_run=str(source),L=field['L'],Ngal=field['Ngal'],A=16.,field=field,
        seeds=seeds,fractions=[1.,.5,.25],cuts=cuts(),
        shape_definition='Q = 12 (3 V)^(2/3) / sum of squares of all six edge lengths',
        size_definition='edge_rms = sqrt(sum of squares of all six edge lengths / 6)',
        thinning='Nested position-independent random indices; SeedSequence([IC_seed,20261122,437]); '
                 'same indices for all helicities and ZA-only within an IC.',
        ranking='Descriptive maximum over fixed Q bins of the minimum gain in |mean D|/SEM '
                'relative to all tetrahedra across the three densities; negative mean at all densities required.',
        scope='Exploratory reuse of previously inspected ICs; random thinning changes sampling geometry. '
              'Not fresh confirmation, not a test of denser particle lattices, and not a universal optimum.',
        physical_band='All positions are subsets of saved catalogues: fields, Fourier draws, and physical band remain identical.',
        input_sha256={str(path):digest(path) for path in files})
    code=out/'code'
    for folder in ('src','tests','scripts','slurm'):
        shutil.copytree(project/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for source_path,dest in [(project/'build/dive/DIVE','bin/DIVE'),
        (project/'pyproject.toml','pyproject.toml'),
        (project/'logs/patches/dive-parity.patch','patches/dive-parity.patch'),
        (project/'tp4a-pv-delaunay/parity_injection.ipynb','tp4a-pv-delaunay/parity_injection.ipynb')]:
        destination=code/dest; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_path,destination)
    shutil.copytree(project/'logs/vendor/DIVE',code/'vendor-source',ignore=shutil.ignore_patterns('.git'))
    atomic_json(out/'protocol.json',protocol)
    sha={str(path.relative_to(out)):digest(path) for path in sorted(code.rglob('*')) if path.is_file()}
    sha['protocol.json']=digest(out/'protocol.json')
    target = project/'results/field_PVdisp_selection.ipynb'
    if target.exists():
        (out/'before').mkdir(exist_ok=True)
        shutil.copy2(target, out/'before'/target.name)
        atomic_json(out/'notebook_input.json', dict(path=str(target), sha256=digest(target)))
        sha['notebook_input.json'] = digest(out/'notebook_input.json')
        sha[f'before/{target.name}'] = digest(out/'before'/target.name)
    atomic_json(out/'manifest.json',dict(project=str(project),sha256=sha))


def reuse_measurements(out, previous):
    """Reuse verified geometry in a new ensemble, preserving the prior run."""
    import ast
    out, previous = Path(out).resolve(), Path(previous).resolve()
    check_manifest(out)
    check_manifest(previous)
    protocol = json.loads((out/'protocol.json').read_text())
    old = json.loads((previous/'protocol.json').read_text())
    for key in ('source_run','field','L','Ngal','A','fractions','cuts','thinning'):
        if protocol[key] != old[key]:
            raise ValueError(f'Cannot reuse measurements with a different {key}')
    # Reuse is valid only for identical measurement definitions and executable.
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    current = functions(out/'code/src/parity_study/tests/selection_geometry.py')
    earlier = functions(previous/'code/src/parity_study/tests/selection_geometry.py')
    for name in ('thinning_indices','cuts','cut_mask','measure'):
        if current[name] != earlier[name]:
            raise ValueError(f'Measurement implementation changed: {name}')
    for name in ('code/bin/DIVE','code/src/parity_study/geometry.py','code/src/parity_study/delaunay.py'):
        if digest(out/name) != digest(previous/name):
            raise ValueError(f'Geometry implementation changed: {name}')
    old_hashes = json.loads((previous/'output_sha256.json').read_text())
    reused = []
    for seed in protocol['seeds']:
        if seed not in old['seeds']: continue
        record_path = previous/'completed'/f'seed_{seed:04d}.json'
        record = json.loads(record_path.read_text())
        assert record['protocol_sha256'] == digest(previous/'protocol.json')
        for name, expected in record['input_sha256'].items():
            assert protocol['input_sha256'][name] == expected
        for name, expected in record['index_sha256'].items():
            assert digest(previous/name) == expected
            (out/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous/name, out/name)
        for case in CASES:
            for fraction in protocol['fractions']:
                name = f'records/{case}/seed_{seed:04d}_fraction_{fraction:g}.npz'
                assert digest(previous/name) == old_hashes[name]
                (out/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(previous/name, out/name)
        record['protocol_sha256'] = digest(out/'protocol.json')
        record['reused_from'] = dict(run=str(previous), record_sha256=digest(record_path))
        atomic_json(out/'completed'/record_path.name, record)
        reused.append(seed)
    atomic_json(out/'reuse.json', dict(previous=str(previous), seeds=reused,
        previous_protocol_sha256=digest(previous/'protocol.json'),
        previous_output_hashes_sha256=digest(previous/'output_sha256.json')))
    return reused


def thinning_indices(count, seed, fractions):
    """One nested position-independent selection shared by all helicities."""
    order = np.random.default_rng(np.random.SeedSequence([seed,20261122,437])).permutation(count)
    return [np.sort(order[:int(round(count*fraction))]) for fraction in fractions]


def cuts():
    return [dict(id='all',label='All tetrahedra',variable='all',low=None,high=None),
            dict(id='reference_ratio',label='Original three-edge ratio',variable='reference_ratio',
                 low=.6905111410000623,high=.7567675934849439)] + [
        dict(id=f'Q_{i}',label=f'{low:.1f} ≤ Q '+('≤' if i==4 else '<')+f' {high:.1f}',
             variable='quality',low=low,high=high,upper_inclusive=i==4)
        for i,(low,high) in enumerate(zip([0.,.2,.4,.6,.8],[.2,.4,.6,.8,1.]))]


def cut_mask(tets, geometry, cut):
    if cut['variable']=='all': return np.ones(len(tets),bool)
    value = tets[:,2]/tets[:,4] if cut['variable']=='reference_ratio' else geometry['quality']
    upper = value<=cut['high'] if cut.get('upper_inclusive') else value<cut['high']
    return (value>=cut['low']) & upper


def measure(out,seed):
    out = Path(out)
    check_manifest(out)
    protocol = json.loads((out/'protocol.json').read_text())
    if seed not in protocol['seeds']: raise ValueError('Seed outside frozen pilot')
    source = Path(protocol['source_run'])
    subsets = thinning_indices(protocol['Ngal'],seed,protocol['fractions'])
    rows, hashes, indices_hashes, volume_errors, checks = [], {}, {}, [], []
    for index,indices in enumerate(subsets):
        path = out/'indices'/f'seed_{seed}_{index}.npz'
        atomic_npz(path,indices=indices)
        indices_hashes[str(path.relative_to(out))] = digest(path)
    for case,eps in zip(CASES,EPSILONS):
        path = source/'records'/case/f'seed_{seed:04d}.npz'
        expected = protocol['input_sha256'][str(path)]
        if digest(path)!=expected: raise ValueError(f'Source catalogue changed: {path}')
        hashes[str(path)] = expected
        with np.load(path) as saved:
            positions = saved['catalogue']
            original = saved['tetrahedra']
            meta = json.loads(str(saved['metadata_json']))
        assert len(positions)==protocol['Ngal']
        assert meta['parameters']['density_seed']==meta['parameters']['helical_seed']==seed
        assert meta['parameters']['A']==(0 if case=='null_za' else protocol['A'])
        assert meta['parameters']['eps']==(0 if eps is None else eps)
        for index,(fraction,indices) in enumerate(zip(protocol['fractions'],subsets)):
            dest = out/'records'/case/f'seed_{seed:04d}_fraction_{fraction:g}.npz'
            tets = parity_stat(positions[indices],L=protocol['L'],
                               executable=out/'code/bin/DIVE',geometry=True)
            g = tetrahedron_geometry(tets)
            volume_error = abs(g['volume'].sum()/protocol['L']**3-1)
            if volume_error > 1e-10: raise AssertionError('Periodic volumes do not cover the box')
            volume_errors.append(float(volume_error))
            if fraction==1:
                np.testing.assert_allclose(tets[:,:5],original,rtol=1e-12,atol=1e-12)
                checks.append(case)
            atomic_npz(dest,tetrahedra=tets,
                       metadata_json=np.array(json.dumps(dict(source=str(path),source_sha256=expected,
                           seed=seed,fraction=fraction,case=case,epsilon=eps,n_particles=len(indices),
                           index_file=str(out/'indices'/f'seed_{seed}_{index}.npz'),
                           columns=['chi','chi_hat','ref_lmin','ref_lmid','ref_lmax','volume',
                                    'edge_1','edge_2','edge_3','edge_4','edge_5','edge_6']))))
            for cut in protocol['cuts']:
                keep = cut_mask(tets,g,cut)
                if not keep.any(): raise ValueError(f"Empty {cut['id']} at seed={seed}, fraction={fraction}")
                selected = tets[keep,1]
                rows.append(dict(seed=seed,fraction=fraction,n_particles=len(indices),epsilon=eps,
                    case=case,selection=cut['id'],S=float(selected.mean()),
                    contribution=float(selected.sum()/len(tets)),selected=int(keep.sum()),
                    total=len(tets),retained_fraction=float(keep.mean()),
                    mean_edge_rms=float(g['edge_rms'][keep].mean()),
                    mean_quality=float(g['quality'][keep].mean())))
            print(f'seed={seed} case={case} fraction={fraction:g} tetrahedra={len(tets)}',flush=True)
    atomic_json(out/'completed'/f'seed_{seed:04d}.json',dict(seed=seed,rows=rows,input_sha256=hashes,
        index_sha256=indices_hashes,unthinned_catalogues_reproduced=checks,
        max_volume_error=max(volume_errors),protocol_sha256=digest(out/'protocol.json')))


def summarize(out):
    out=Path(out)
    check_manifest(out)
    protocol=json.loads((out/'protocol.json').read_text())
    rows=[]
    for seed in protocol['seeds']:
        record=json.loads((out/'completed'/f'seed_{seed:04d}.json').read_text())
        assert record['protocol_sha256']==digest(out/'protocol.json')
        assert record['unthinned_catalogues_reproduced']==list(CASES)
        assert record['max_volume_error']<1e-10
        for path,sha in record['index_sha256'].items():
            assert digest(out/path)==sha
        rows.extend(record['rows'])
    expected=len(protocol['seeds'])*len(protocol['fractions'])*len(CASES)*len(protocol['cuts'])
    if len(rows)!=expected: raise ValueError('Incomplete catalogue measurements')
    summaries=[]
    distributions=[]
    for fraction in protocol['fractions']:
        for cut in protocol['cuts']:
            signals=[]
            for case,eps in zip(CASES,EPSILONS):
                subset=[r for r in rows if r['fraction']==fraction and r['selection']==cut['id'] and r['case']==case]
                assert [r['seed'] for r in subset]==protocol['seeds']
                values=np.array([r['S'] for r in subset])
                signals.append(values)
                distributions.append(dict(fraction=fraction,selection=cut['id'],case=case,epsilon=eps,
                                          **mean_error(values)))
            selected=[r for r in rows if r['fraction']==fraction and r['selection']==cut['id'] and r['case']!='null_za']
            summaries.append(dict(fraction=fraction,n_particles=round(protocol['Ngal']*fraction),
                selection=cut['id'],label=cut['label'],**mean_error(signals[2]-signals[0]),
                retained_fraction=float(np.mean([r['retained_fraction'] for r in selected])),
                mean_edge_rms=float(np.mean([r['mean_edge_rms'] for r in selected]))))
    for row in summaries:
        baseline=next(r for r in summaries if r['fraction']==row['fraction'] and r['selection']=='all')
        row['gain_over_all']=float(abs(row['t'])/abs(baseline['t'])) if baseline['t'] else None
    ranking=[]
    for cut in protocol['cuts'][1:]:
        subset=[r for r in summaries if r['selection']==cut['id']]
        gains=[r['gain_over_all'] for r in subset]
        ranking.append(dict(selection=cut['id'],label=cut['label'],
            negative_at_all_densities=all(r['mean']<0 for r in subset),
            minimum_gain_over_all=min(gains) if all(x is not None for x in gains) else None))
    # Ranking is descriptive; do not convert this reuse of inspected data into confirmation.
    candidates=[r for r in ranking if r['selection'].startswith('Q_') and r['negative_at_all_densities']
                and r['minimum_gain_over_all'] is not None]
    candidate=max(candidates,key=lambda r:r['minimum_gain_over_all']) if candidates else None
    report=dict(complete=True,protocol_sha256=digest(out/'protocol.json'),summaries=summaries,
        distributions=distributions,ranking=ranking,exploratory_candidate=candidate,
        initial_conditions=len(protocol['seeds']),catalogues=len(protocol['seeds'])*len(CASES)*len(protocol['fractions']),
        scope=protocol['scope'])
    atomic_json(out/'analysis.json',report)
    for name,data in [('catalogue_signals',rows),('summary',summaries)]:
        with (out/f'{name}.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    return report


def robust_signal_cut(report):
    """Post-hoc candidate by minimum |D|/SEM, distinct from frozen relative gain."""
    candidates=[]
    for row in report['ranking']:
        if not row['selection'].startswith('Q_') or not row['negative_at_all_densities']:
            continue
        subset=[r for r in report['summaries'] if r['selection']==row['selection']]
        if all(r['t'] is not None for r in subset):
            candidates.append(dict(row,minimum_abs_t=min(abs(r['t']) for r in subset)))
    return max(candidates,key=lambda r:r['minimum_abs_t']) if candidates else None


def report_bins(out, project):
    """Summarize a complete geometry ensemble and execute its fine-bin notebook."""
    from parity_study.tests import selection_bins
    out = Path(out).resolve()
    expected = json.loads((out/'notebook_input.json').read_text())
    if digest(Path(expected['path'])) != expected['sha256']:
        raise RuntimeError('Notebook changed during geometry jobs; preserve concurrent edits')
    report = summarize(out)
    paths = [out/'analysis.json', out/'catalogue_signals.csv', out/'summary.csv',
             *sorted((out/'records').glob('*/*.npz'))]
    atomic_json(out/'output_sha256.json', {str(p.relative_to(out)):digest(p) for p in paths})
    fine = out/'fine_bins'
    selection_bins.prepare(out, fine, project, code_root=out/'code')
    measured = selection_bins.analyze(fine)
    selection_bins.render(fine)
    atomic_json(out/'completion_checks.json', dict(initial_conditions=report['initial_conditions'],
        catalogues=report['catalogues'], fine_bin_catalogues=measured['catalogues_checked'],
        original_selection_checks=measured['original_cut_checks'],
        notebook=str(Path(project)/'results/field_PVdisp_selection.ipynb')))


def figures(out):
    import matplotlib.pyplot as plt
    out=Path(out)
    protocol=json.loads((out/'protocol.json').read_text())
    report=json.loads((out/'analysis.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    colors=['#1b9e77','#d95f02','#7570b3']
    for fraction,color in zip(protocol['fractions'],colors):
        entries=[r for r in report['summaries'] if r['fraction']==fraction and r['selection'].startswith('Q_')]
        x=np.arange(.1,1.,.2)
        axes[0].errorbar(x,[r['mean'] for r in entries],yerr=[r['sem'] for r in entries],
                         color=color,marker='o',capsize=3,label=f"N = {round(protocol['Ngal']*fraction):,}")
        axes[1].plot(x,[r['gain_over_all'] for r in entries],color=color,marker='o')
        axes[2].plot(x,[r['retained_fraction'] for r in entries],color=color,marker='o')
    axes[0].axhline(0,color='gray',lw=.8)
    axes[0].set(ylabel=r'$\overline{D}_B=\overline{S_B(+1)-S_B(-1)}$',title='Catalogue helicity difference')
    axes[0].legend(fontsize=9)
    axes[1].axhline(1,color='gray',lw=.8,ls='--')
    axes[1].set(ylabel=r'$(|\overline{D}|/\mathrm{SEM})_B\,/\,(|\overline{D}|/\mathrm{SEM})_{\rm all}$',
                title='Signal-to-noise relative to all tetrahedra')
    axes[2].set(ylabel='Mean retained fraction',title='Population in each fixed shape bin')
    for ax in axes:
        ax.set(xlabel=r'Six-edge shape $Q$ (bin centre)',xticks=np.arange(.1,1,.2))
        ax.grid(alpha=.2)
    fig.suptitle('Fixed geometric bins; ten reused ICs; matched random thinning')
    comparison,ax=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    ids=['all','reference_ratio']
    robust=robust_signal_cut(report)
    if robust: ids.append(robust['selection'])
    for selection in ids:
        entries=[r for r in report['summaries'] if r['selection']==selection]
        ax.errorbar([r['n_particles'] for r in entries],[r['mean'] for r in entries],
                    yerr=[r['sem'] for r in entries],marker='o',capsize=3,label=entries[0]['label'])
    ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Particles per catalogue',ylabel=r'Mean matched difference $\overline{D}_B$',
           title='Same displacement fields, different sampling densities')
    ax.legend(); ax.grid(alpha=.2)
    directory=out/'figures'; directory.mkdir(exist_ok=True)
    for name,figure in [('fixed_shape_bins',fig),('density_comparison',comparison)]:
        for ext in ('png','pdf'):
            figure.savefig(directory/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    return fig,comparison


def render(out,project):
    import nbformat
    from nbclient import NotebookClient
    out=Path(out).resolve()
    report_source=out/'report_code/src' if (out/'report_code/manifest.json').exists() else out/'code/src'
    target=Path(project)/'results/field_PVdisp_selection.ipynb'
    if target.exists(): raise FileExistsError(target)
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[md(r'''# Does the tetrahedron selection transfer across particle densities?

**Exploratory study:** ten previously inspected ICs (1000–1009), $A=16\ h^{-1}\mathrm{Mpc}$, $L=1000\ h^{-1}\mathrm{Mpc}$, field mesh $256^3$. Nested random subsets retain 32,768, 16,384 or 8,192 particles. Within each IC the same particle indices are used for $\epsilon=-1,0,+1$ and the ZA-only control. The fields and Fourier band are unchanged.

This tests thinning of the existing displaced lattice, not convergence of a denser lattice or a new independent detection.
'''),md(r'''## 1. Fixed geometric definitions

The original cut is $0.6905111410\leq r_{\rm ref}<0.7567675935$, with $r_{\rm ref}=\ell_{\min}/\ell_{\max}$ for the **three reference-vertex edges**. It is already dimensionless. Its bounds came from one reference catalogue; their sensitivity need not transfer when Delaunay shapes change.

We also measure all six edge lengths $\ell_{ij}$ and the geometric volume $V$, then define
$$Q=\frac{12(3V)^{2/3}}{\sum_{i<j}\ell_{ij}^2},\qquad 0\leq Q\leq1.$$
$Q=1$ means regular; $Q\to0$ means degenerate. This [standard mean-ratio shape measure](https://people.math.sc.edu/Burkardt/cpp_src/tet_mesh_quality/tet_mesh_quality.html) is invariant under scaling, reflection, and vertex relabelling. It is a shape descriptor, not a complete description of tetrahedron geometry or a known optimal PV weight.

The five bins $[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]$ are fixed before this comparison. They are never refitted to a catalogue. Both chirality signs are retained. Physical size is recorded separately as $\ell_{\rm rms}=\sqrt{\sum\ell_{ij}^2/6}$.
'''),md(r'''## 2. Catalogue signal and uncertainty

With vertices ordered by sums of squared distances, $\widehat\chi=\mathbf v_1\cdot(\mathbf v_2\times\mathbf v_3)/(|\mathbf v_1||\mathbf v_2||\mathbf v_3|)$. Exact ordering ties have zero chirality weight; geometric volume is saved independently.

For a fixed selection $B$, $S_{s,B}(\epsilon)=\mathrm{mean}_{t\in B}\widehat\chi_t$. Each IC contributes one matched difference $D_{s,B}=S_{s,B}(+1)-S_{s,B}(-1)$. Error bars are one **standard error of the mean**, $\mathrm{SEM}=\mathrm{std}(D_s)/\sqrt{10}$. Tetrahedra are not independent replicates. The three thinning levels are correlated.
'''),code(f'''from pathlib import Path
import json,sys
import numpy as np
from IPython.display import display,Markdown
OUT=Path({str(out)!r})
sys.path.insert(0,{str(report_source)!r})
from parity_study.tests.selection_geometry import figures,robust_signal_cut
from parity_study.storage import digest
report=json.loads((OUT/'analysis.json').read_text())
assert report['complete'] and report['protocol_sha256']==digest(OUT/'protocol.json')
%matplotlib inline
import matplotlib.pyplot as plt
fig1,fig2=figures(OUT)
display(fig1); display(fig2)
plt.close('all')
'''),md(r'''## 3. What transfers, and what remains untested?

The same $Q$ bin has the same geometric meaning at every density. Equal geometry does not imply equal retained fraction, equal physical size, or equal PV response. The second panel divides $|\overline D|/\mathrm{SEM}$ by its all-tetrahedron value at the same density; values above one indicate a gain in this sample.
'''),code('''gain_candidate=report['exploratory_candidate']
candidate=robust_signal_cut(report)
if gain_candidate and gain_candidate['minimum_gain_over_all']>1:
    display(Markdown(f"**Candidate for independent follow-up:** {gain_candidate['label']}. "
        f"Its smallest signal-to-noise gain relative to all tetrahedra is "
        f"{gain_candidate['minimum_gain_over_all']:.2f}. This ranking is exploratory."))
else:
    display(Markdown("**No tested fixed Q bin improves signal-to-noise over all tetrahedra at every density.**"))
if candidate:
    display(Markdown(f"**Promising fixed-shape candidate: {candidate['label']}.** "
        f"The mean difference is negative at all three densities, with the smallest "
        f"absolute mean-to-SEM ratio equal to {candidate['minimum_abs_t']:.2f}. "
        "This trades sensitivity at the fiducial density for a clearer response in the sparse sample. "
        "This alternative ranking was considered after seeing the completed pilot; it requires independent confirmation."))
else:
    display(Markdown("No fixed Q bin keeps the previously observed negative difference at every tested density."))
lines=['| Selection | N particles | Mean D ± SEM | Retained | Mean edge RMS (h⁻¹ Mpc) |',
       '|---|---:|---:|---:|---:|']
show={'all','reference_ratio'}
if candidate: show.add(candidate['selection'])
for row in report['summaries']:
    if row['selection'] in show:
        lines.append(f"| {row['label']} | {row['n_particles']:,} | {row['mean']:+.5f} ± {row['sem']:.5f} "
                     f"| {row['retained_fraction']:.1%} | {row['mean_edge_rms']:.2f} |")
display(Markdown('\\n'.join(lines)))
original=[r for r in report['summaries'] if r['selection']=='reference_ratio']
display(Markdown('**Original-cut sensitivity gains relative to all tetrahedra:** '+
    ', '.join(f"{r['gain_over_all']:.2f} at {r['n_particles']:,} particles" for r in original)+'. '+
    'Its geometric definition is fixed, but its statistical advantage does not transfer across this thinning test.'))
display(Markdown('**Equal-helicity and ZA-only controls:** catalogue means ± one SEM for the displayed selections.'))
lines=['| Selection | N particles | ε=0 | ZA only |','|---|---:|---:|---:|']
for row in report['summaries']:
    if row['selection'] not in show: continue
    controls=[next(r for r in report['distributions'] if r['selection']==row['selection'] and
                  r['fraction']==row['fraction'] and r['case']==case) for case in ('null_symmetric','null_za')]
    values=[f"{r['mean']:+.5f} ± {r['sem']:.5f}" for r in controls]
    lines.append(f"| {row['label']} | {row['n_particles']:,} | {values[0]} | {values[1]} |")
display(Markdown('\\n'.join(lines)))
'''),md(r'''A general selection should be frozen in dimensionless shape coordinates and checked across sampling densities. If physical size is also selected, use a fixed physical band or a fixed ratio to a field correlation scale; dividing by particle spacing would change the physical scales being tested.

The displayed bin is a candidate, not an established replacement cut. Its boundaries are from the five predefined equal-width Q bins, but highlighting it for the largest minimum absolute signal-to-SEM ratio is a post-hoc choice. Confirm it on fresh ICs before replacing the fiducial selection. A joint fixed shape and physical-size selection is another possible follow-up. Random thinning also changes the sampling pattern of the original lattice; this study does not isolate particle count from that change.

A subsequent lattice-resolution study must hold $L$, the field mesh, $A$, the spectrum and its active Fourier band fixed: the current default $k_{\max}=\pi\,\min(N_{\rm mesh},N_{\rm particle,side})/L$ would otherwise change with particle count. The original confirmed fiducial cut is unchanged.

[Protocol](saved/selection_geometry_20261122/protocol.json) · [Catalogue signals](saved/selection_geometry_20261122/catalogue_signals.csv) · [Summary](saved/selection_geometry_20261122/summary.csv) · [Figures](saved/selection_geometry_20261122/figures/)
''')],metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')))
    for cell in nb.cells:
        if cell.cell_type=='code': cell.metadata['jupyter']={'source_hidden':True}
    NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    nbformat.write(nb,target)
    atomic_json(out/'report_provenance.json',dict(notebook=str(target),notebook_sha256=digest(target),
        protocol_sha256=digest(out/'protocol.json'),analysis_sha256=digest(out/'analysis.json'),
        report_source=str(report_source),report_module_sha256=digest(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','reuse','measure','report','summarize','report-bins'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--seed-start',type=int,default=1000)
    parser.add_argument('--n-seeds',type=int,default=10)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.out,args.source,args.project,
                                     seed_start=args.seed_start,n_seeds=args.n_seeds)
    elif args.action=='reuse': reuse_measurements(args.out,args.source)
    elif args.action=='measure': measure(args.out,args.seed)
    elif args.action=='summarize': summarize(args.out)
    elif args.action=='report-bins': report_bins(args.out,args.project)
    else:
        summarize(args.out)
        render(args.out,args.project)


if __name__=='__main__': main()
