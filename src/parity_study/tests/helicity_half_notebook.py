"""Five-helicity catalogue distributions using old and fresh matched ensembles."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest


def prepare(run, project):
    run, project = Path(run).resolve(), Path(project).resolve()
    check_manifest(run)
    p = json.loads((run/'protocol.json').read_text())
    out = run/'report'
    out.mkdir(exist_ok=False)
    (out/'code').mkdir()
    shutil.copy2(Path(__file__), out/'code/helicity_half_notebook.py')
    (out/'inputs/previous_Q').mkdir(parents=True)
    source = Path(p['source_Q_report'])
    for name, sha in p['source_sha256'].items():
        assert digest(source/name) == sha
        shutil.copy2(source/name, out/'inputs/previous_Q'/name)
    target = project/'results/field_PVdisp_B.ipynb'
    (out/'before').mkdir()
    shutil.copy2(target, out/'before'/target.name)
    config = dict(run=str(run), target=str(target), previous_notebook_sha256=digest(target),
        run_manifest_sha256=digest(run/'manifest.json'), protocol_sha256=digest(run/'protocol.json'),
        epsilons=[-1.,-.5,0.,.5,1.], original_epsilons=[-1.,0.,1.], new_epsilons=[-.5,.5],
        primary_selection=p['primary_selection'], n_per_helicity=len(p['seeds']),
        scope=p['scope'], output_link=f'saved/{run.name}/report')
    atomic_json(out/'config.json', config)
    paths = [out/'config.json', *sorted((out/'code').glob('*')), *sorted((out/'inputs').rglob('*'))]
    atomic_json(out/'manifest.json', dict(sha256={str(v.relative_to(out)):digest(v)
                                               for v in paths if v.is_file()}))


def analyze(out):
    from parity_study.summarize import mean_error
    out = Path(out)
    check_manifest(out)
    config = json.loads((out/'config.json').read_text())
    run = Path(config['run'])
    assert digest(run/'manifest.json') == config['run_manifest_sha256']
    assert digest(run/'protocol.json') == config['protocol_sha256']
    p = json.loads((run/'protocol.json').read_text())
    hashes = json.loads((run/'output_sha256.json').read_text())
    for name in ('summary/analysis.json','summary/signals.npz'):
        assert digest(run/name) == hashes[name]
    newer = json.loads((run/'summary/analysis.json').read_text())
    old = json.loads((out/'inputs/previous_Q/analysis.json').read_text())
    assert newer['complete'] and newer['audit']['passed'] and old['complete']
    assert newer['protocol_sha256'] == config['protocol_sha256']
    values, counts, totals, seeds = [], [], [], []
    with np.load(out/'inputs/previous_Q/catalogue_signals.npz') as previous, np.load(run/'summary/signals.npz') as new:
        assert previous['epsilons'].tolist() == [-1,0,1]
        assert new['epsilons'].tolist() == p['epsilons']
        assert previous['seeds'].tolist() == p['source_seeds']
        assert new['seeds'].tolist() == p['seeds']
        assert not set(previous['seeds']).intersection(new['seeds'])
        ni = new['selections'].tolist().index(config['primary_selection'])
        for eps in config['epsilons']:
            data, ci = (previous, 0) if eps in config['original_epsilons'] else (new, ni)
            ei = data['epsilons'].tolist().index(eps)
            values.append(data['signals'][ci,ei])
            counts.append(data['counts'][ci,ei])
            totals.append(data['totals'][ei])
            seeds.append(data['seeds'])
    values, counts, totals, seeds = map(np.array,(values,counts,totals,seeds))
    assert values.shape == (5,config['n_per_helicity']) and np.isfinite(values).all()
    assert np.all(counts > 0) and np.all(counts <= totals)
    distributions = [dict(epsilon=eps, **mean_error(values[i]),
        seed_start=int(seeds[i,0]),seed_end=int(seeds[i,-1]),
        mean_selected=float(counts[i].mean()),retained_fraction=float((counts[i]/totals[i]).mean()))
        for i,eps in enumerate(config['epsilons'])]
    differences = []
    for e, reference in ((.5,newer['primary']),(1.,old['paired'])):
        plus, minus = [config['epsilons'].index(eps) for eps in (e,-e)]
        np.testing.assert_array_equal(seeds[plus],seeds[minus])
        row = dict(abs_epsilon=e, **mean_error(values[plus]-values[minus]),
                   seed_start=int(seeds[plus,0]),seed_end=int(seeds[plus,-1]))
        np.testing.assert_allclose([row['mean'],row['sem']],[reference['mean'],reference['sem']],
                                   rtol=1e-12,atol=1e-14)
        differences.append(row)
    rows=[dict(epsilon=eps,seed=int(seeds[i,j]),S=float(values[i,j]),
               selected=int(counts[i,j]),total=int(totals[i,j]))
          for i,eps in enumerate(config['epsilons']) for j in range(values.shape[1])]
    atomic_npz(out/'five_helicities.npz',epsilons=np.array(config['epsilons']),seeds=seeds,
               signals=values,counts=counts,totals=totals)
    for name,data in [('catalogue_signals',rows),('distribution_summary',distributions),('paired_summary',differences)]:
        with (out/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    atomic_json(out/'analysis.json',dict(complete=True,A=p['A'],selection=dict(variable='Q',low=.4,high=.8),
        config_sha256=digest(out/'config.json'),distributions=distributions,paired=differences,
        new_IC_audit=newer['audit'],scope=p['scope'],new_result_sha256={name:hashes[name]
                  for name in ('summary/analysis.json','summary/signals.npz')}))


def figure(out):
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    out=Path(out)
    report=json.loads((out/'analysis.json').read_text())
    assert report['complete'] and report['config_sha256']==digest(out/'config.json')
    with np.load(out/'five_helicities.npz') as z:
        values,epsilons=z['signals'],z['epsilons']
    n=values.shape[1]
    means,sd=values.mean(axis=1),values.std(axis=1,ddof=1)
    if not np.isfinite(values).all() or np.any(sd<=0):
        raise ValueError('Require finite, nonconstant catalogue signals')
    width=3.49*np.sqrt(np.mean(sd**2))/n**(1/3)
    lo=min(values.min(),np.min(means-4*sd));hi=max(values.max(),np.max(means+4*sd))
    bins=np.linspace(lo,hi,max(10,int(np.ceil((hi-lo)/width)))+1)
    x=np.linspace(lo,hi,2400)
    fig,ax=plt.subplots(figsize=(10.5,6.1))
    colors=['#2878b5','#26a69a','#777777','#e6a128','#d95335']
    for i,(eps,color) in enumerate(zip(epsilons,colors)):
        ax.hist(values[i],bins=bins,density=True,histtype='stepfilled',alpha=.18,color=color)
        ax.hist(values[i],bins=bins,density=True,histtype='step',lw=1.1,color=color)
        eps_label=f'{eps:+g}' if eps else '0'
        ax.plot(x,norm.pdf(x,means[i],sd[i]),color=color,lw=2,
            label=rf'$\epsilon={eps_label}$: $\mu={means[i]:+.4f}$, $\sigma_S={sd[i]:.4f}$')
        ax.axvline(means[i],color=color,ls=':',lw=.7,alpha=.65)
    ax.set(xlabel=r'Catalogue signal $S_{s,B_Q}(\epsilon)=\langle\widehat\chi\rangle_{t\in B_Q}$',
        ylabel='Probability density',xlim=(lo,hi),ylim=(0,None),
        title=rf'Fiducial $Q$ selection: $0.4\leq Q<0.8$ · $A={report["A"]:g}\,h^{{-1}}\mathrm{{Mpc}}$'
              +f'\n{n} catalogue signals per helicity')
    ax.grid(alpha=.18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,fontsize=9,framealpha=.95)
    fig.tight_layout(rect=(0,.08,1,1))
    (out/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(out/'figures'/f'catalogue_signal_distributions_Q_five_helicities.{ext}',dpi=180,bbox_inches='tight')
    return fig


def table_markdown(out):
    report=json.loads((Path(out)/'analysis.json').read_text())
    lines=['| ε | Catalogues | IC seeds | Mean μ | Scatter σS | SEM |',
           '|---:|---:|---|---:|---:|---:|']
    for r in report['distributions']:
        lines.append(f"| {r['epsilon']:+g} | {r['n']} | {r['seed_start']}–{r['seed_end']} | "
                     f"{r['mean']:+.6f} | {r['sd']:.6f} | {r['sem']:.6f} |")
    lines.extend(['',r'**Matched difference:** $D_{s,e}=S_{s,B_Q}(+e)-S_{s,B_Q}(-e)$. SEM uses the scatter of matched differences within each ensemble.','',
                  '| e | Matched ICs | Mean D ± SEM |','|---:|---:|---:|'])
    for r in report['paired']:
        lines.append(f"| {r['abs_epsilon']:g} | {r['n']} | {r['mean']:+.6f} ± {r['sem']:.6f} |")
    return '\n'.join(lines)


def render(out):
    import nbformat
    from nbclient import NotebookClient
    out=Path(out).resolve()
    config=json.loads((out/'config.json').read_text())
    target=Path(config['target'])
    nb=nbformat.read(out/'before'/target.name,as_version=4)
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    index=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 4.'))
    cells=[md(r'''## 4. Fiducial $Q$ selection: catalogue-signal distributions

Use $B_Q:0.4\le Q<0.8$, where $Q=12(3V)^{2/3}/\sum_{i<j}\ell_{ij}^{2}$ uses the tetrahedron volume and all six edges. This selection is applied on its own, without the $r$ cut.

Each histogram contains **50 catalogue signals**, $S_{s,B_Q}(\epsilon)=\mathrm{mean}_{t\in B_Q}\widehat\chi_t$, at 32,768 particles and $A=16\ h^{-1}\mathrm{Mpc}$. Gaussian curves use the catalogue scatter $\sigma_S$ defined in Section 2.

- $\epsilon=-1,0,+1$: original IC seeds **1000–1049**; the $Q$ cut was chosen after inspecting this sample.
- $\epsilon=-0.5,+0.5$: **50 new ICs, seeds 1050–1099**, with $A$ and the $Q$ cut fixed before generation. Each pair shares its Gaussian density, ZA displacement and helical draws. These cases split the expected helical power 25:75 and 75:25 between the positive and negative components.

The two seed groups are independent; matched differences below pair helicities only within their own group.
'''),code(f'''from pathlib import Path
import importlib.util
from IPython.display import display, Markdown
%matplotlib inline
import matplotlib.pyplot as plt
Q_HALF = Path({str(out)!r})
spec = importlib.util.spec_from_file_location('pv_five_helicities', Q_HALF/'code/helicity_half_notebook.py')
five_helicities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five_helicities)
fig = five_helicities.figure(Q_HALF)
display(fig)
plt.close(fig)
display(Markdown(five_helicities.table_markdown(Q_HALF)))
''')]
    cells[1].metadata['jupyter']={'source_hidden':True}
    link=config['output_link'];runlink=f'saved/{Path(config["run"]).name}'
    cells.append(md(f'[Catalogue signals]({link}/catalogue_signals.csv) · '
        f'[Gaussian parameters]({link}/distribution_summary.csv) · [Matched differences]({link}/paired_summary.csv) · '
        f'[Figures]({link}/figures/) · [New-run protocol]({runlink}/protocol.json) · '
        f'[Validation]({runlink}/validation/ensemble.json)'))
    addition=nbformat.v4.new_notebook(cells=cells,metadata=nb.metadata)
    NotebookClient(addition,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    addition.cells[1].execution_count=max(c.execution_count or 0 for c in nb.cells[:index] if c.cell_type=='code')+1
    nb.cells=nb.cells[:index]+addition.cells
    nb.cells[0].source=nb.cells[0].source.replace(
        'Sections 1–3 report the original frozen $r$ selection; Section 4 adds the fiducial $Q$ selection on the same catalogues.',
        'Sections 1–3 report the original frozen $r$ selection. Section 4 compares the fiducial $Q$ cut across five helicities, including 50 additional independent ICs for $\\epsilon=\\pm0.5$.')
    nbformat.validate(nb)
    assert digest(target)==config['previous_notebook_sha256'], 'Notebook changed during computation'
    temporary=target.with_suffix('.tmp.ipynb');nbformat.write(nb,temporary);temporary.replace(target)
    shutil.copy2(target,out/target.name)
    atomic_json(out/'report_provenance.json',dict(notebook=str(target),notebook_sha256=digest(target),
        previous_notebook_sha256=config['previous_notebook_sha256'],preserved_cells=list(range(1,index)),
        analysis_sha256=digest(out/'analysis.json'),config_sha256=digest(out/'config.json')))
    paths=[out/'analysis.json',out/'five_helicities.npz',out/'catalogue_signals.csv',out/'distribution_summary.csv',
           out/'paired_summary.csv',out/'report_provenance.json',out/target.name,*sorted((out/'figures').glob('*'))]
    atomic_json(out/'output_sha256.json',{str(p.relative_to(out)):digest(p) for p in paths})
    print(f'Updated {target}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.run,args.project)
    else:
        analyze(args.out)
        render(args.out)


if __name__=='__main__': main()
