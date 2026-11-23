"""Matched density power before and after adding helical displacements."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np

from parity_study.paths import expose_output
from parity_study.power_spectrum import catalogue_power, density_mesh_modes, shell_average, wavenumbers
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest
from parity_study.tests.catalogue_power import EPSILONS, COLORS, error, write_csv


def prepare(project, source, out):
    project, source, out = (Path(p).resolve() for p in (project, source, out))
    check_manifest(source)
    old = json.loads((source/'config.json').read_text())
    hashes = json.loads((source/'output_sha256.json').read_text())
    assert digest(source/'power_spectra.npz') == hashes['power_spectra.npz']
    fields = {}
    for item in old['inputs']:
        run = Path(item['path']).parents[2]
        path = run/'fields'/f"main_{item['seed']:04d}.npz"
        if item['seed'] not in fields:
            reference = run/'records/null_za'/f"seed_{item['seed']:04d}.npz"
            fields[item['seed']] = dict(seed=item['seed'], path=str(path), sha256=digest(path),
                za_reference=str(reference) if reference.exists() else None,
                za_reference_sha256=digest(reference) if reference.exists() else None)
    target = project/'results/field_PVdisp_Pk.ipynb'
    out.mkdir(parents=True, exist_ok=False)
    (out/'inputs').mkdir()
    shutil.copy2(source/'power_spectra.npz', out/'inputs/final_power_spectra.npz')
    (out/'before').mkdir()
    shutil.copy2(target, out/'before'/target.name)
    cfg = dict(created_utc=datetime.now(timezone.utc).isoformat(), project=str(project), target=str(target),
        previous_notebook_sha256=digest(target), source=str(source),
        source_spectra_sha256=hashes['power_spectra.npz'], field=old['field'], A=old['A'],
        epsilons=old['epsilons'], n_per_case=old['n_per_case'], n_modes=old['n_modes'],
        tolerance=old['tolerance'], final_catalogues=old['inputs'], fields=list(fields.values()),
        comparison='Original Gaussian density versus ZA plus helical displacement, matched by IC seed.',
        baselines=dict(input='FFT of the actual saved delta_mesh; normalized Fourier-series coefficients.',
            za='x_ZA=(q+Psi_ZA) mod L, using the saved fields; no helical displacement.',
            final='x=(q+Psi_ZA+Psi_hel) mod L; reuse the previously measured point spectra.'),
        changes=dict(total='P_final/P_input-1', helical='P_final/P_ZA-1'),
        ratio_averaging='Form each ratio within one IC, then mean and SEM over five ICs.',
        plotted_range='Only complete shells within the initial density Fourier support.',
        full_band_summary='First average modes with 0<k<=k_IC,max per catalogue, then form matched ratios.',
        shot_noise='No Poisson subtraction for displaced lattice particles.',
        no_helical_baseline='A=0; epsilon=0 at A=16 still contains a transverse displacement.',
        scope='Five matched realizations per helicity; no new Gaussian draws or Delaunay computations.')
    atomic_json(out/'config.json', cfg)
    files = ['src/parity_study/__init__.py', 'src/parity_study/power_spectrum.py',
             'src/parity_study/storage.py', 'src/parity_study/paths.py',
             'src/parity_study/tests/__init__.py', 'src/parity_study/tests/catalogue_power.py',
             'src/parity_study/tests/catalogue_power_baseline.py', 'tests/test_power_spectrum.py',
             'slurm/catalogue_power_baseline.sh', 'requirements.txt']
    for name in files:
        dest = out/'code'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project/name, dest)
    for name in ('logs','validation','baselines','figures'):
        (out/name).mkdir()
    paths = [out/'config.json', out/'inputs/final_power_spectra.npz', out/'before'/target.name,
             *sorted((out/'code').rglob('*'))]
    atomic_json(out/'manifest.json', dict(project=str(project), sha256={
        str(p.relative_to(out)):digest(p) for p in paths if p.is_file()}))
    expose_output(project, out)
    print(f'Prepared {len(fields)} matched baselines; reusing 25 final spectra', flush=True)


def measure(out):
    out = Path(out)
    check_manifest(out)
    cfg = json.loads((out/'config.json').read_text())
    L, n = cfg['field']['L'], cfg['n_modes']
    with np.load(out/'inputs/final_power_spectra.npz') as z:
        previous = {key:z[key] for key in z.files}
    assert previous['epsilons'].tolist() == cfg['epsilons']
    kgrid = wavenumbers(n,L)
    kmax = np.pi*cfg['field']['particle_grid']/L
    active = (kgrid > 0) & (kgrid <= kmax)
    assert active.sum() == 17076
    edges = previous['edges']
    usable = edges[1:] <= kmax
    baselines, audits = {}, []
    for item in cfg['fields']:
        seed = item['seed']
        assert digest(item['path']) == item['sha256']
        with np.load(item['path']) as z:
            meta = json.loads(str(z['metadata_json']))
            assert meta['field'] == cfg['field'] and meta['density_seed'] == meta['helical_seed'] == seed
            delta, q, za, plus, minus = (z[key] for key in ('delta_mesh','q','za','plus','minus'))
        initial, fft_audit = density_mesh_modes(delta, n_modes=n)
        del delta
        assert abs(fft_audit['mean']) < 1e-12 and fft_audit['parseval_absolute_error'] < 1e-12
        np.testing.assert_allclose(np.sqrt(fft_audit['variance']), meta['sigma_delta_measured'], rtol=1e-12)
        initial3d = L**3*abs(initial)**2
        p_initial, k, counts = shell_average(initial3d,kgrid,edges)
        x_za = (q+za) % L
        if item['za_reference']:
            assert digest(item['za_reference']) == item['za_reference_sha256']
            with np.load(item['za_reference']) as z:
                np.testing.assert_array_equal(z['catalogue'], x_za)
        matched = []
        for cat in cfg['final_catalogues']:
            if cat['seed'] != seed:
                continue
            assert digest(cat['path']) == cat['sha256']
            eps = cat['epsilon']
            x = (q+(za+cfg['A']*(np.sqrt((1+eps)/2)*plus + np.sqrt((1-eps)/2)*minus))) % L
            with np.load(cat['path']) as z:
                np.testing.assert_array_equal(z['catalogue'], x)
            matched.append(eps)
        spec, za_coeff = catalogue_power(x_za,L=L,n_modes=n,edges=edges,tolerance=cfg['tolerance'],
                                        nthreads=int(os.environ.get('OMP_NUM_THREADS','1')))
        np.testing.assert_array_equal(spec['k'], previous['k'])
        np.testing.assert_array_equal(spec['nmodes'], previous['nmodes'])
        band_initial = float(initial3d[active].mean())
        band_za = float((L**3*abs(za_coeff[active])**2).mean())
        baseline = dict(P_input=p_initial,P_ZA=spec['power_raw'],band_input=band_initial,band_ZA=band_za)
        baselines[seed] = baseline
        atomic_npz(out/'baselines'/f'seed_{seed}.npz',seed=seed,k=k,edges=edges,nmodes=counts,
                   za_catalogue=x_za,**baseline)
        audits.append(dict(seed=seed, field_sha256=item['sha256'], matched_epsilons=matched,
            za_saved_catalogue_verified=bool(item['za_reference']), fft=fft_audit))
        print(f'Measured original Gaussian field and ZA baseline: seed={seed}',flush=True)
    seeds, final = previous['seeds'], previous['power_raw']
    p_input, p_za = [np.array([[baselines[int(s)][key] for s in row] for row in seeds])
                     for key in ('P_input','P_ZA')]
    band_input, band_za = [np.array([[baselines[int(s)][key] for s in row] for row in seeds])
                           for key in ('band_input','band_ZA')]
    band_final = previous['band_power'][:,:,0]
    relative_input = np.full_like(final,np.nan)
    relative_za = np.full_like(final,np.nan)
    relative_input[:,:,usable] = final[:,:,usable]/p_input[:,:,usable]-1
    relative_za[:,:,usable] = final[:,:,usable]/p_za[:,:,usable]-1
    band_rows = []
    for i,eps in enumerate(cfg['epsilons']):
        total = band_final[i]/band_input[i]-1
        helical = band_final[i]/band_za[i]-1
        placement = band_za[i]/band_input[i]-1
        band_rows.append(dict(epsilon=eps,n=5,seed_start=int(seeds[i,0]),seed_end=int(seeds[i,-1]),
            total_mean=float(total.mean()),total_sem=float(error(total)),
            helical_mean=float(helical.mean()),helical_sem=float(error(helical)),
            za_vs_input_mean=float(placement.mean()),za_vs_input_sem=float(error(placement))))
    atomic_npz(out/'comparison.npz',epsilons=np.array(cfg['epsilons']),seeds=seeds,k=previous['k'],
        edges=edges,nmodes=previous['nmodes'],usable=usable,P_final=final,P_input=p_input,P_ZA=p_za,
        input_ensemble=previous['density_input_ensemble'],relative_input=relative_input,relative_ZA=relative_za,
        band_final=band_final,band_input=band_input,band_ZA=band_za)
    rows = [dict(epsilon=eps,seed=int(seeds[i,j]),k=float(previous['k'][b]),
        P_input=float(p_input[i,j,b]),P_ZA=float(p_za[i,j,b]),P_final=float(final[i,j,b]),
        change_vs_input=float(relative_input[i,j,b]),change_vs_ZA=float(relative_za[i,j,b]))
        for i,eps in enumerate(cfg['epsilons']) for j in range(5) for b in np.flatnonzero(usable)]
    write_csv(out/'matched_power.csv',rows)
    write_csv(out/'band_summary.csv',band_rows)
    audit = dict(passed=True,original_fields=10,za_baselines=10,
        matched_final_catalogues=sum(len(r['matched_epsilons']) for r in audits),
        reused_final_spectra=25,field_checks=audits)
    assert audit['matched_final_catalogues'] == 25
    atomic_json(out/'validation/baselines.json',audit)
    atomic_json(out/'analysis.json',dict(complete=True,config_sha256=digest(out/'config.json'),
        kmax=kmax,full_band_nmodes=int(active.sum()),plotted_bins=int(usable.sum()),band_comparisons=band_rows))


def load_data(out):
    with np.load(Path(out)/'comparison.npz') as z:
        return {key:z[key] for key in z.files}


def spectra_figure(out):
    import matplotlib.pyplot as plt
    out = Path(out)
    data = load_data(out)
    keep, k = data['usable'], data['k'][data['usable']]
    fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax, indices, group in zip(axes,[[0,2,4],[1,3]],['1000–1004','1050–1054']):
        i = indices[0]
        ax.plot(k,data['input_ensemble'][keep],color='#999999',ls=':',lw=1.5,label='Input model (ensemble)')
        ax.plot(k,data['P_input'][i][:,keep].mean(axis=0),color='black',lw=2,label='Original Gaussian field (same ICs)')
        ax.plot(k,data['P_ZA'][i][:,keep].mean(axis=0),color='black',ls='--',lw=1.6,label='ZA only: no helical displacement')
        for i in indices:
            p = data['P_final'][i][:,keep]
            mean, sem = p.mean(axis=0),error(p)
            ax.plot(k,mean,color=COLORS[i],lw=1.6,label=rf'ZA + helical, $\epsilon={EPSILONS[i]:+g}$')
            ax.fill_between(k,mean-sem,mean+sem,color=COLORS[i],alpha=.10,linewidth=0)
        ax.set(xscale='log',yscale='log',xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',
               title=f'5 matched initial conditions: seeds {group}')
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r'$P(k)\ [(h^{-1}\mathrm{Mpc})^3]$')
    handles,labels = [],[]
    for ax in axes:
        h,l = ax.get_legend_handles_labels()
        for handle,label in zip(h,l):
            if label not in labels:
                handles.append(handle);labels.append(label)
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8.5,bbox_to_anchor=(.5,-.015))
    fig.suptitle('Original density power → ZA particles → ZA + helical displacement',fontsize=12)
    fig.tight_layout(rect=(0,.21,1,.93))
    for ext in ('png','pdf'):
        fig.savefig(out/'figures'/f'input_and_displaced_power.{ext}',dpi=180,bbox_inches='tight')
    return fig


def response_figure(out):
    import matplotlib.pyplot as plt
    out = Path(out)
    data = load_data(out)
    keep,k = data['usable'],data['k'][data['usable']]
    fig,axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,key,title,label in zip(axes,['relative_input','relative_ZA'],
            ['Total change from the original Gaussian field','Change caused by adding the helical displacement'],
            [r'$100\,[P_{\rm ZA+hel}/P_{\rm input}-1]$ [%]',r'$100\,[P_{\rm ZA+hel}/P_{\rm ZA}-1]$ [%]']):
        for i,eps in enumerate(EPSILONS):
            values = 100*data[key][i][:,keep]
            mean,sem = values.mean(axis=0),error(values)
            ax.plot(k,mean,color=COLORS[i],lw=1.4,label=rf'$\epsilon={eps:+g}$')
            ax.fill_between(k,mean-sem,mean+sem,color=COLORS[i],alpha=.14,linewidth=0)
        ax.axhline(0,color='black',ls='--',lw=.8)
        ax.set(xscale='log',xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel=label,title=title)
        ax.grid(alpha=.18)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=10)
    fig.suptitle(r'Matched power response: $A=16\ h^{-1}\mathrm{Mpc}$; shaded bands show SEM',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.93))
    for ext in ('png','pdf'):
        fig.savefig(out/'figures'/f'matched_power_response.{ext}',dpi=180,bbox_inches='tight')
    return fig


def summary_table(out):
    rows = json.loads((Path(out)/'analysis.json').read_text())['band_comparisons']
    lines = ['| ε | Change from original field | Change from ZA-only catalogue |',
             '|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['epsilon']:+g} | {100*r['total_mean']:+.3f}% ± {100*r['total_sem']:.3f}% | "
                     f"{100*r['helical_mean']:+.3f}% ± {100*r['helical_sem']:.3f}% |")
    return '\n'.join(lines)


def render(out):
    import nbformat
    from nbclient import NotebookClient
    out = Path(out).resolve()
    cfg = json.loads((out/'config.json').read_text())
    md,code = nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells = [md(r'''# Effect of helical displacement on the density power spectrum

**Question:** how does the point-catalogue $P(k)$ change relative to the original Gaussian field when we add the helical displacement?

## 1. Three stages, matched by initial condition

1. **Original Gaussian field:** $P_{\rm input}(k)$, measured from the saved density mesh $\delta(\mathbf q)$.
2. **ZA-only particles:** $\mathbf x_{\rm ZA}=\mathbf q+\boldsymbol\Psi_{\rm ZA}(\mathbf q)$, giving $P_{\rm ZA}(k)$.
3. **ZA + helical displacement:** $\mathbf x_{\rm ZA+hel}=\mathbf q+\boldsymbol\Psi_{\rm ZA}(\mathbf q)+\boldsymbol\Psi_{\rm hel}(\mathbf q)$, giving $P_{\rm ZA+hel}(k)$.

Positions are periodic. Each comparison uses the **same Gaussian field and initial particle lattice**. The no-helical-displacement baseline has $A=0$; the $\epsilon=0$ case below still has displacement strength $A=16$ and equal expected power in the two handed components.

Keep **5 realizations per helicity**: seeds 1000–1004 for $\epsilon=-1,0,+1$; seeds 1050–1054 for $\epsilon=-0.5,+0.5$. Use $L=1000\ h^{-1}\mathrm{Mpc}$, a $256^3$ initial density mesh, $32^3$ particles, and $A=16\ h^{-1}\mathrm{Mpc}$. All particles are used.
'''),code(f'''from pathlib import Path
import importlib.util
from IPython.display import display, Markdown
%matplotlib inline
import matplotlib.pyplot as plt
OUT = Path({str(out)!r})
spec = importlib.util.spec_from_file_location('matched_power_report', Path({str(Path(__file__).resolve())!r}))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
'''),md(r'''## 2. Original and displaced density power

For both meshes and particles, $P(k)=L^3\langle|\delta_{\mathbf k}|^2\rangle$ in each Fourier shell. The mesh uses normalized Fourier coefficients; particles use $\delta_{\mathbf k}=N_p^{-1}\sum_j e^{-i\mathbf k\cdot\mathbf x_j}$ for nonzero modes. Thus the spectra have the same units and normalization.

The black curves use the **actual matched initial conditions**. The dotted grey line is the analytic ensemble input spectrum. Coloured lines show the mean of five final-catalogue spectra; shading is **SEM** (sample standard deviation divided by $\sqrt{5}$).

Only complete shells inside the input band $k\le0.10053\ h\,\mathrm{Mpc}^{-1}$ are plotted, with width $2\pi/L$. The final partial shell at the cutoff is included in the broad-band table below. No Poisson shot-noise subtraction is applied to these displaced-lattice catalogues.
'''),code('fig = report.spectra_figure(OUT)\ndisplay(fig)\nplt.close(fig)'),
    md(r'''## 3. How much does adding the helical field change $P(k)$?

For each matched realization $s$, define
$$R_{{\rm total},s}(k)=\frac{P_{{\rm ZA+hel},s}(k)}{P_{{\rm input},s}(k)}-1,\qquad
R_{{\rm hel},s}(k)=\frac{P_{{\rm ZA+hel},s}(k)}{P_{{\rm ZA},s}(k)}-1.$$

The left panel measures the total change from the original field. The right panel isolates the effect of adding the helical displacement to the ZA catalogue. Zero means unchanged power; positive/negative values mean enhancement/suppression. Form ratios **within each IC first**, then plot their mean and SEM across five ICs.
'''),code('fig = report.response_figure(OUT)\ndisplay(fig)\nplt.close(fig)'),
    md(r'''## 4. Summary over the input Fourier band

First average power over all nonzero modes with $k\le0.10053\ h\,\mathrm{Mpc}^{-1}$ within each realization, then form the two matched ratios. The table gives the mean percentage change ± SEM over five realizations.
'''),code('display(Markdown(report.summary_table(OUT)))'),
    md(r'''The helical displacement is transverse, so $-\nabla\cdot\boldsymbol\Psi_{\rm hel}=0$: it does not change density at first order. The full particle map can change density power through higher-order terms. A power change relative to the original field measures this density response; the sign of parity violation is tested separately with chirality.

Ratios to the input field are restricted to its supported Fourier band. The earlier spectra at higher $k$ contain prominent particle-lattice structure and remain saved in the previous report.
''')]
    link = f'saved/{out.name}'
    cells.append(md(f'[Matched spectra and ratios]({link}/matched_power.csv) · [Band summary]({link}/band_summary.csv) · '
        f'[All arrays]({link}/comparison.npz) · [Plots]({link}/figures/) · [Configuration]({link}/config.json) · '
        f'[Validation]({link}/validation/baselines.json) · [Previous report]({link}/before/field_PVdisp_Pk.ipynb)'))
    for cell in cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden':True}
    nb = nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(
        display_name='Python 3 (project)',language='python',name='python3')))
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    nbformat.validate(nb)
    target = Path(cfg['target'])
    assert digest(target) == cfg['previous_notebook_sha256'], 'Target notebook changed during measurement'
    nbformat.write(nb,out/target.name)
    shutil.copy2(out/target.name,target)
    atomic_json(out/'report_provenance.json',dict(notebook_sha256=digest(target),
        source_sha256=digest(Path(__file__)),config_sha256=digest(out/'config.json'),data_sha256=digest(out/'comparison.npz')))
    files = [out/target.name,out/'comparison.npz',out/'analysis.json',out/'report_provenance.json',
             *out.glob('*.csv'),*(out/'baselines').glob('*.npz'),*(out/'figures').glob('*'),
             out/'validation/baselines.json',out/'validation/pytest.xml']
    atomic_json(out/'output_sha256.json',{str(p.relative_to(out)):digest(p) for p in files})
    print(f'Updated {target}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','render'])
    parser.add_argument('--out',required=True)
    parser.add_argument('--source')
    parser.add_argument('--project')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.project,args.source,args.out)
    elif args.action == 'run':
        measure(args.out)
        render(args.out)
    else:
        render(args.out)


if __name__ == '__main__':
    main()
