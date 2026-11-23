"""Five saved catalogues per helicity: periodic density power and report."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np

from parity_study.paths import expose_output
from parity_study.power_spectrum import catalogue_power, shell_average, wavenumbers
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest

EPSILONS = [-1., -.5, 0., .5, 1.]
COLORS = ['#2878b5', '#249b91', '#777777', '#d8a021', '#d95335']


def prepare(project, root, out):
    project, root, out = map(lambda p: Path(p).resolve(), (project, root, out))
    old = root/'20261122_mesh_ic_L1000_N256_A16_confirmation_50'
    new = root/'20261123_mesh_ic_L1000_N256_A16_epshalf_50'
    field = json.loads((old/'config.json').read_text())['field']
    assert field == json.loads((new/'protocol.json').read_text())['field']
    assert field['density_seed'] is None and field['L'] == 1000 and field['Ngal'] == 32768
    old_hashes = json.loads((root/'field_PVdisp_B/input_sha256.json').read_text())
    new_hashes = json.loads((new/'output_sha256.json').read_text())
    cases = ['scan_A16_minus', 'eps_m0p5', 'null_symmetric', 'eps_p0p5', 'scan_A16_plus']
    inputs = []
    for eps, case in zip(EPSILONS, cases):
        run, hashes, first = (new, new_hashes, 1050) if abs(eps) == .5 else (old, old_hashes, 1000)
        for seed in range(first, first+5):
            name = f'records/{case}/seed_{seed:04d}.npz'
            assert digest(run/name) == hashes[name]
            inputs.append(dict(epsilon=eps, case=case, seed=seed, path=str(run/name), sha256=hashes[name]))
    target = project/'results/field_PVdisp_Pk.ipynb'
    if target.exists():
        raise FileExistsError(target)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), project=str(project), target=str(target),
        field=field, A=16., epsilons=EPSILONS, n_per_case=5, inputs=inputs,
        n_modes=128, tolerance=1e-11,
        estimator='P_raw(k) = V mean_modes |sum_j exp(-i k.x_j)/N_p|^2; omit k=0.',
        shells='Width 2*pi/L, edges (0.5,1.5,...,63.5)*2*pi/L; both conjugate modes counted.',
        selection='All particles. No tetrahedron Q or r selection is applied to the point catalogue.',
        subtraction='No Poisson subtraction for this displaced lattice; V/N_p is shown as a reference only.',
        uncertainty='SEM = sample standard deviation across five ICs / sqrt(5); paired before averaging differences.',
        pairing='-1,0,+1 share seeds 1000-1004; -0.5,+0.5 share independent seeds 1050-1054.',
        sampling='First five seeds in each completed ensemble, fixed before inspecting power.',
        scope='A descriptive comparison; five realizations cannot estimate the full cross-bin covariance.',
        references=['https://finufft.readthedocs.io/en/latest/python.html',
                    'https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.fftpower.html'])
    atomic_json(out/'config.json', config)
    files = ['src/parity_study/__init__.py', 'src/parity_study/power_spectrum.py',
             'src/parity_study/storage.py', 'src/parity_study/paths.py',
             'src/parity_study/tests/__init__.py', 'src/parity_study/tests/catalogue_power.py',
             'tests/test_power_spectrum.py', 'slurm/catalogue_power.sh', 'requirements.txt']
    for name in files:
        dest = out/'code'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project/name, dest)
    for name in ('logs', 'validation', 'spectra', 'figures'):
        (out/name).mkdir()
    atomic_json(out/'manifest.json', dict(project=str(project), sha256={
        str(p.relative_to(out)): digest(p) for p in [out/'config.json', *sorted((out/'code').rglob('*'))] if p.is_file()}))
    expose_output(project, out)
    print(f'Prepared {len(inputs)} saved catalogues in {out}', flush=True)


def error(values, axis=0):
    return np.std(values, axis=axis, ddof=1)/np.sqrt(values.shape[axis])


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(out):
    out = Path(out)
    check_manifest(out)
    cfg = json.loads((out/'config.json').read_text())
    L, Np, n = cfg['field']['L'], cfg['field']['Ngal'], cfg['n_modes']
    kgrid = wavenumbers(n, L)
    k_ic = np.pi*cfg['field']['particle_grid']/L
    edges = 2*np.pi/L*np.arange(.5, n/2, 1.)
    bands = [(0., k_ic), (k_ic, 2*k_ic), (2*k_ic, edges[-1])]
    masks = [(kgrid > lo) & (kgrid <= hi) for lo, hi in bands]
    mode_check = np.array([(1,0,0),(-2,1,3),(8,0,0),(16,0,0),(25,-7,4),(40,12,-3),(62,0,0)])
    powers, band_powers, seeds, audits = [], [], [], []
    reflection_error = None
    for item in cfg['inputs']:
        path = Path(item['path'])
        assert digest(path) == item['sha256']
        with np.load(path) as z:
            x = z['catalogue']
            meta = json.loads(str(z['metadata_json']))
        assert meta['seed'] == item['seed'] and meta['case'] == item['case']
        assert x.shape == (Np, 3) and np.all((x >= 0) & (x < L))
        params = meta.get('parameters', meta)
        assert params['A'] == cfg['A'] and params.get('eps', params.get('epsilon')) == item['epsilon']
        spec, coeff = catalogue_power(x, L=L, n_modes=n, edges=edges,
            tolerance=cfg['tolerance'], nthreads=int(os.environ.get('OMP_NUM_THREADS', '1')))
        direct = np.exp(-2j*np.pi/L*(x@mode_check.T)).mean(axis=0)
        actual = coeff[tuple((mode_check % n).T)]
        discrepancy = float(abs(direct-actual).max())
        assert discrepancy < 2e-10
        if reflection_error is None:
            reflected, _ = catalogue_power((-x) % L, L=L, n_modes=n, edges=edges,
                tolerance=cfg['tolerance'], nthreads=int(os.environ.get('OMP_NUM_THREADS', '1')))
            reflection_error = float(np.max(abs(reflected['power_raw']/spec['power_raw']-1)))
            assert reflection_error < 1e-8
        p3d = L**3*abs(coeff)**2
        bp = np.array([p3d[m].mean() for m in masks])
        assert np.isfinite(spec['power_raw']).all() and np.all(spec['nmodes'] > 0)
        name = f"spectra/{item['case']}_seed_{item['seed']}.npz"
        atomic_npz(out/name, **spec, band_power=bp, epsilon=item['epsilon'], seed=item['seed'])
        powers.append(spec['power_raw']); band_powers.append(bp); seeds.append(item['seed'])
        audits.append(dict(**item, result=name, result_sha256=digest(out/name), direct_mode_max_error=discrepancy))
        print(f"Measured epsilon={item['epsilon']:+g} seed={item['seed']}", flush=True)
    power = np.asarray(powers).reshape(5,5,-1)
    band_power = np.asarray(band_powers).reshape(5,5,3)
    seeds = np.asarray(seeds).reshape(5,5)
    # Analytic ensemble density spectrum used by the ZA IC, with identical active modes.
    active = (kgrid >= 2*np.pi/L) & (kgrid <= k_ic)
    assert active.sum() == 17076
    shape = np.where(active, np.exp(-(kgrid*cfg['field']['spectrum']['smoothing'])**2), 0.)
    theory3d = L**3*cfg['field']['sigma_delta']**2*shape/shape.sum()
    theory, _, _ = shell_average(theory3d, kgrid, edges)
    differences, fractional, band_rows = [], [], []
    for e in [.5, 1.]:
        plus, minus = [EPSILONS.index(v) for v in (e,-e)]
        np.testing.assert_array_equal(seeds[plus], seeds[minus])
        diff = power[plus]-power[minus]
        frac = 2*diff/(power[plus]+power[minus])
        differences.append(diff); fractional.append(frac)
        bp, bm = band_power[plus], band_power[minus]
        bd, bf = bp-bm, 2*(bp-bm)/(bp+bm)
        for b,(lo,hi) in enumerate(bands):
            band_rows.append(dict(abs_epsilon=e, k_low_exclusive=lo, k_high_inclusive=hi,
                n_pairs=5, nmodes=int(masks[b].sum()),
                D_P_mean=float(bd[:,b].mean()), D_P_sem=float(error(bd[:,b])),
                fractional_mean=float(bf[:,b].mean()), fractional_sem=float(error(bf[:,b]))))
    differences, fractional = np.array(differences), np.array(fractional)
    atomic_npz(out/'power_spectra.npz', power_raw=power, k=spec['k'], edges=edges, nmodes=spec['nmodes'],
        epsilons=np.array(EPSILONS), seeds=seeds, mean=power.mean(axis=1), sem=error(power, axis=1),
        poisson_reference=L**3/Np, density_input_ensemble=theory, band_power=band_power,
        band_edges=np.array(bands), paired_abs_epsilon=np.array([.5,1.]),
        paired_difference=differences, paired_fractional_difference=fractional,
        paired_mean=differences.mean(axis=1), paired_sem=error(differences, axis=1),
        fractional_mean=fractional.mean(axis=1), fractional_sem=error(fractional, axis=1))
    rows = [dict(epsilon=eps, seed=int(seeds[i,j]), k=float(spec['k'][b]),
                 k_low=float(edges[b]), k_high=float(edges[b+1]), nmodes=int(spec['nmodes'][b]),
                 P_raw=float(power[i,j,b]), poisson_reference=L**3/Np)
            for i,eps in enumerate(EPSILONS) for j in range(5) for b in range(power.shape[-1])]
    write_csv(out/'catalogue_power.csv', rows)
    write_csv(out/'paired_band_summary.csv', band_rows)
    summary_rows = [dict(epsilon=eps, k=float(spec['k'][b]), n_catalogues=5,
                        P_mean=float(power[i,:,b].mean()), P_sem=float(error(power[i,:,b])))
                    for i,eps in enumerate(EPSILONS) for b in range(power.shape[-1])]
    write_csv(out/'mean_power.csv', summary_rows)
    write_csv(out/'paired_power.csv', [dict(abs_epsilon=e, k=float(spec['k'][b]), n_pairs=5,
        D_P_mean=float(differences[i,:,b].mean()), D_P_sem=float(error(differences[i,:,b])),
        fractional_mean=float(fractional[i,:,b].mean()), fractional_sem=float(error(fractional[i,:,b])))
        for i,e in enumerate([.5,1.]) for b in range(power.shape[-1])])
    atomic_json(out/'validation/catalogues.json', dict(passed=True, checked_catalogues=25,
        direct_modes_per_catalogue=len(mode_check), max_direct_error=max(r['direct_mode_max_error'] for r in audits),
        reflection_max_relative_error=reflection_error, inputs=audits))
    atomic_json(out/'analysis.json', dict(complete=True, config_sha256=digest(out/'config.json'),
        n_catalogues=25, n_per_case=5, k_input_max=k_ic, k_measurement_max=float(edges[-1]),
        band_comparisons=band_rows, scope=cfg['scope']))


def power_figure(out):
    import matplotlib.pyplot as plt
    out = Path(out)
    cfg = json.loads((out/'config.json').read_text())
    with np.load(out/'power_spectra.npz') as z:
        data = {k:z[k] for k in z.files}
    k, means, sem = data['k'], data['mean'], data['sem']
    k_ic = np.pi*cfg['field']['particle_grid']/cfg['field']['L']
    fig, axes = plt.subplots(1,2,figsize=(12,4.6))
    for ax in axes:
        for eps, color, mean, err in zip(EPSILONS,COLORS,means,sem):
            ax.plot(k, mean, color=color, lw=1.7, label=rf'$\epsilon={eps:+g}$')
            ax.fill_between(k, mean-err, mean+err, color=color, alpha=.12, linewidth=0)
        keep = data['density_input_ensemble'] > 0
        ax.plot(k[keep], data['density_input_ensemble'][keep], 'k--', lw=1.2,
                label=r'Input Gaussian density $P_\delta$')
        ax.axvline(k_ic, color='#666666', ls=':', lw=1.2)
        ax.set(xscale='log', yscale='log', xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',
               ylabel=r'$P_{\rm cat}(k)\ [(h^{-1}\mathrm{Mpc})^3]$')
        ax.grid(alpha=.18)
    axes[0].set_xlim(k[0]*.92, k_ic*1.05)
    low = k <= k_ic*1.05
    axes[0].set_ylim(min((means-sem)[:,low].min(), data['density_input_ensemble'][low].min())*.8,
                     max((means+sem)[:,low].max(),data['density_input_ensemble'][low].max())*1.2)
    axes[0].set_title('Scales within the input Fourier band')
    axes[1].axhline(float(data['poisson_reference']), color='#555555', ls='-.', lw=1.2,
                    label=r'Poisson reference $V/N_p$ (not subtracted)')
    axes[1].axvspan(k_ic, data['edges'][-1], color='#888888', alpha=.08)
    axes[1].set_xlim(k[0]*.92, data['edges'][-1])
    axes[1].set_title('Extended range: particle-discreteness diagnostic')
    fig.suptitle(r'Catalogue density power: $A=16\ h^{-1}\mathrm{Mpc}$; 5 catalogues per helicity', fontsize=12)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5,-.04))
    fig.tight_layout(rect=(0,.14,1,.94))
    for ext in ('png','pdf'):
        fig.savefig(out/'figures'/f'catalogue_power.{ext}', dpi=180, bbox_inches='tight')
    return fig


def difference_figure(out):
    import matplotlib.pyplot as plt
    out = Path(out)
    with np.load(out/'power_spectra.npz') as z:
        k, means, sem = z['k'], z['fractional_mean'], z['fractional_sem']
    cfg = json.loads((out/'config.json').read_text())
    k_ic = np.pi*cfg['field']['particle_grid']/cfg['field']['L']
    fig, axes = plt.subplots(1,2,figsize=(12,4.1), sharey=True)
    for i, (ax,e) in enumerate(zip(axes,[.5,1.])):
        keep = k < k_ic
        ax.errorbar(k[keep], 100*means[i,keep], yerr=100*sem[i,keep], color=COLORS[1 if i==0 else 0],
                    fmt='o-', ms=3.5, lw=1, capsize=2)
        ax.axhline(0, color='black', lw=.8)
        ax.set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',
               title=rf'$\epsilon=\pm{e:g}$: 5 matched pairs', xscale='log')
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r'Mean paired $D_P/\overline{P}_{\rm pair}$ [%]')
    fig.suptitle('Does reversing helicity change the density power? Error bars: SEM', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):
        fig.savefig(out/'figures'/f'paired_power_difference.{ext}',dpi=180,bbox_inches='tight')
    return fig


def band_table(out):
    report = json.loads((Path(out)/'analysis.json').read_text())
    lines = ['| Matched helicities | Mean fractional power difference ± SEM |', '|---|---:|']
    for row in report['band_comparisons']:
        if row['k_low_exclusive'] == 0:
            lines.append(f"| ±{row['abs_epsilon']:g} | {100*row['fractional_mean']:+.3f}% ± {100*row['fractional_sem']:.3f}% |")
    return '\n'.join(lines)


def render(out, allow_update=False):
    import nbformat
    from nbclient import NotebookClient
    out = Path(out).resolve()
    cfg = json.loads((out/'config.json').read_text())
    target = Path(cfg['target'])
    if target.exists():
        if not allow_update:
            raise FileExistsError(target)
        previous = json.loads((out/'report_provenance.json').read_text())
        assert digest(target) == previous['notebook_sha256'], 'Notebook changed outside this report'
        archive = out/'report_revision/before'
        archive.mkdir(parents=True, exist_ok=False)
        for path in [target, out/'report_provenance.json', out/'output_sha256.json']:
            shutil.copy2(path, archive/path.name)
        shutil.copytree(out/'figures', archive/'figures')
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    link = f'saved/{out.name}'
    cells = [md(r'''# Point-catalogue power spectra

**Question:** do the five helicity cases differ in their ordinary density power spectrum? Use **5 saved catalogues per case**, 25 total, at displacement strength $A=16\ h^{-1}\mathrm{Mpc}$.

## 1. Catalogues and estimator

$L=1000\ h^{-1}\mathrm{Mpc}$, IC mesh $256^3$, $N_p=32^3=32768$ particles; the particle map includes ZA and helical displacements. All particles enter this measurement; the tetrahedron $Q$ selection does not define a particle subset.

| Helicity $\epsilon$ | Gaussian IC seeds | Catalogues |
|---|---|---:|
| $-1,0,+1$ | 1000–1004, matched within each seed | 5 each |
| $-0.5,+0.5$ | 1050–1054, matched within each seed | 5 each |

For box volume $V=L^3$, positions $\mathbf x_j$ and nonzero wavevectors $\mathbf k=2\pi\mathbf n/L$,
$$\delta_{\mathbf k}=\frac{1}{N_p}\sum_j e^{-i\mathbf k\cdot\mathbf x_j},\qquad
P_{\rm cat}(k)=V\left\langle|\delta_{\mathbf k}|^2\right\rangle_{\text{modes in a shell}}.$$
The x-axis is wavenumber $k$ (larger $k$ means smaller spatial scales); the y-axis is density power, in volume units. Shell width is $2\pi/L=0.006283\ h\,\mathrm{Mpc}^{-1}$.

Fourier sums use [FINUFFT](https://finufft.readthedocs.io/en/latest/python.html) at tolerance $10^{-11}$, requesting a $128^3$ Fourier cube; this does not change the IC mesh. There is no particle-assignment window. The zero mode is excluded.

The [Poisson reference](https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.fftpower.html) is $V/N_p=30517.58\ (h^{-1}\mathrm{Mpc})^3$. It is **not subtracted**: these particles come from a displaced lattice, not independent Poisson sampling of a density field.
'''), code(f'''from pathlib import Path
import importlib.util
from IPython.display import display, Markdown
%matplotlib inline
import matplotlib.pyplot as plt
OUT = Path({str(out)!r})
spec = importlib.util.spec_from_file_location('saved_catalogue_power', Path({str(Path(__file__).resolve())!r}))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
'''), md(r'''## 2. Power spectra for the five helicities

Lines show the **mean of five catalogue spectra**; bands show **SEM**, the sample standard deviation across these five catalogues divided by $\sqrt{5}$. Fourier modes are not treated as independent catalogue realizations.

The dashed curve is the **ensemble input Gaussian-density spectrum**, not a prediction for the final displaced points. The vertical dotted line is the input cutoff and initial-particle Nyquist scale, $k_{\rm IC,max}=\pi(32)/L=0.10053\ h\,\mathrm{Mpc}^{-1}$. The shaded extended range shows small-scale structure of this discrete particle load; interpreting it as a continuum density spectrum requires a particle-resolution study.
'''), code('fig = report.power_figure(OUT)\ndisplay(fig)\nplt.close(fig)'),
    md(r'''## 3. Matched positive-minus-negative differences

For realization $s$ and helicity magnitude $e$,
$$D_{P,s,e}(k)=P_{s,+e}(k)-P_{s,-e}(k),\qquad
\overline P_{{\rm pair},s,e}(k)=\frac{P_{s,+e}(k)+P_{s,-e}(k)}{2}.$$
Plot the mean of $D_{P,s,e}/\overline P_{{\rm pair},s,e}$ as a percentage, with SEM across the **five matched pairs**. Zero means equal density power for the two signs. Only the input Fourier band is shown; all measured bins are saved.
'''), code('fig = report.difference_figure(OUT)\ndisplay(fig)\nplt.close(fig)\ndisplay(Markdown(report.band_table(OUT)))'),
    md(r'''The table averages power over **all nonzero modes with $k\le k_{\rm IC,max}$ within each catalogue**, then forms the paired fractional difference. It is a descriptive broad-band summary; neighbouring bins are correlated, and five realizations cannot determine their full covariance.

## 4. What this comparison tells us

**The broad-band difference between helicity signs is consistent with zero:** $-0.144\%\pm0.270\%$ for $\epsilon=\pm1$, and $+0.021\%\pm0.067\%$ for $\epsilon=\pm0.5$ (SEM). Individual-bin fluctuations remain; this five-pair check is not a joint significance test across all bins.

The lowest-$k$ offset between the half-helicity curves and the other curves compares **different Gaussian IC seeds**, so it cannot be attributed to helicity alone. The large high-$k$ peaks near $0.20$ and $0.28\ h\,\mathrm{Mpc}^{-1}$ coincide with the initial lattice scales $2\pi/d$ and $\sqrt{2}\,2\pi/d$, where $d=L/32=31.25\ h^{-1}\mathrm{Mpc}$. This supports interpreting them as residual lattice structure; a denser particle load is needed to assess convergence there.

The ordinary density power is **parity even**: reflecting all positions takes $\delta_{\mathbf k}$ to its complex conjugate and leaves $|\delta_{\mathbf k}|^2$ unchanged. Opposite-helicity ensembles therefore have the same expected density power in this parity-related construction. Their five-realization estimates can differ because the two helical components have independent random draws. The production check also measures a reflected catalogue and verifies its power is unchanged.

Similar $P(k)$ for the two signs is compatible with the detected chirality signal. Helical displacement is transverse, so it adds no density contrast at first order ($-\nabla\cdot\boldsymbol\Psi_{\rm hel}=0$); the full particle map can still alter density power through nonlinear terms. Equal power for different $|\epsilon|$ is not guaranteed by parity.
'''), md(f'[Configuration and catalogue hashes]({link}/config.json) · [Individual spectra]({link}/catalogue_power.csv) · '
             f'[Mean spectra]({link}/mean_power.csv) · [Paired differences]({link}/paired_power.csv) · '
             f'[Arrays]({link}/power_spectra.npz) · [Figures]({link}/figures/) · [Validation]({link}/validation/catalogues.json)')]
    for cell in cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    nb = nbformat.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(
        display_name='Python 3 (project)', language='python', name='python3')))
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata':{'path':str(out)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, out/target.name)
    shutil.copy2(out/target.name, target)
    atomic_json(out/'report_provenance.json', dict(notebook=str(target), notebook_sha256=digest(target),
        config_sha256=digest(out/'config.json'), source_sha256=digest(Path(__file__)),
        data_sha256=digest(out/'power_spectra.npz')))
    paths = [out/'analysis.json', out/'power_spectra.npz', out/target.name, out/'report_provenance.json',
             *out.glob('*.csv'), *(out/'spectra').glob('*.npz'), *(out/'figures').glob('*'),
             out/'validation/catalogues.json', out/'validation/pytest.xml']
    atomic_json(out/'output_sha256.json', {str(p.relative_to(out)):digest(p) for p in paths})
    print(f'Saved executed notebook: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','render'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--project')
    parser.add_argument('--root')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.project, args.root, args.out)
    elif args.action == 'run':
        measure(args.out)
        render(args.out)
    else:
        render(args.out, allow_update=True)


if __name__ == '__main__':
    main()
