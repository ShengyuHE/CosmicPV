"""Introduction to the current Gaussian-IC + ZA + helical displacement model."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from parity_study.initial_conditions import displace_mesh, spectrum_shape
from parity_study.paths import OUTPUT_ROOT, expose_output
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest

EPSILONS = [-1., 0., 1.]


def active_example(root=OUTPUT_ROOT):
    pointer = Path(root)/'intro_disp_PV/active_example.json'
    if not pointer.exists():
        return None
    record = json.loads(pointer.read_text())
    out = Path(record['directory'])
    if digest(out/'config.json') != record['config_sha256'] or digest(out/'examples.npz') != record['examples_sha256']:
        raise ValueError('Active fiducial introduction inputs changed')
    if 'report_source' in record and digest(record['report_source']) != record['report_source_sha256']:
        raise ValueError('Active fiducial introduction report source changed')
    return out


def prepare(project, source, out, seed=1000):
    project, source, out = (Path(p).resolve() for p in (project,source,out))
    field = json.loads((source/'config.json').read_text())['field']
    assert field['model'] == 'mesh_za_helical' and field['L'] == 1000 and field['N'] == 256
    assert field['particle_grid'] == 32 and field['amplitude_convention'] == 'rms'
    cache = source/'fields'/f'main_{seed:04d}.npz'
    catalogues = [source/'records'/case/f'seed_{seed:04d}.npz'
                  for case in ('scan_A16_minus','null_symmetric','scan_A16_plus')]
    target = project/'results/intro_PVdisp.ipynb'
    out.mkdir(parents=True,exist_ok=False)
    (out/'before').mkdir()
    shutil.copy2(target,out/'before'/target.name)
    cfg = dict(created_utc=datetime.now(timezone.utc).isoformat(),project=str(project),target=str(target),
        previous_notebook_sha256=digest(target),source=str(source),field=field,A=16.,seed=seed,
        epsilons=EPSILONS,field_cache=str(cache),field_cache_sha256=digest(cache),
        catalogues=[dict(path=str(path),sha256=digest(path)) for path in catalogues],
        plane_index=16,zoom_initial_bounds=[250.,750.],length_unit='h^-1 Mpc',
        scope='Illustrations from a saved production realization; no new fields or Delaunay catalogues.',
        displacement_power='H(k)=P_delta(k)*exp[-(k R_h)^2]/k^4; W_k=H(k)/sum_active_modes H(k); P_Psi,k=A^2 W_k.',
        input_model='x(q)=[q+Psi_ZA(q)+Psi_hel(q)] mod L; regular cell-centred lattice.',
        amplitude='Displacement strength A is the ensemble RMS vector magnitude; sampled realizations fluctuate.')
    atomic_json(out/'config.json',cfg)
    shutil.copytree(project/'src',out/'code/src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for name in ('tests/test_amplitude_scan.py','tests/test_initial_conditions.py','slurm/intro_fiducial.sh'):
        dest = out/'code'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/name,dest)
    for name in ('figures','validation','logs'):
        (out/name).mkdir()
    files = [out/'config.json',out/'before'/target.name,*sorted((out/'code').rglob('*'))]
    atomic_json(out/'manifest.json',dict(project=str(project),sha256={
        str(p.relative_to(out)):digest(p) for p in files if p.is_file()}))
    expose_output(project,out)
    print(f'Prepared fiducial introduction: {out}',flush=True)


def measure(out):
    out = Path(out)
    check_manifest(out)
    cfg = json.loads((out/'config.json').read_text())
    assert digest(cfg['field_cache']) == cfg['field_cache_sha256']
    with np.load(cfg['field_cache']) as z:
        arrays = {key:z[key] for key in ('q','za','plus','minus')}
        meta = json.loads(str(z['metadata_json']))
    field = cfg['field']
    assert meta['field'] == field and meta['density_seed'] == meta['helical_seed'] == cfg['seed']
    L,N,A = field['L'],field['N'],cfg['A']
    kmin,kmax = meta['kmin'],meta['kmax']
    axis = 2*np.pi*np.fft.fftfreq(N,d=L/N)
    if field['exclude_nyquist'] and N % 2 == 0:
        axis = np.delete(axis,N//2)
    axis = axis[abs(axis) <= kmax]
    waves = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    magnitude = np.linalg.norm(waves,axis=1)
    k = magnitude[(magnitude>=kmin)&(magnitude<=kmax)&(magnitude>0)]
    shape = spectrum_shape(k,field['spectrum'],np.ones(k.shape,bool))
    density_weights = shape/shape.sum()
    hel_shape = shape*np.exp(-(k*field['helical_smoothing'])**2)/k**4
    weights = hel_shape/hel_shape.sum()
    power = A**2*weights
    assert len(k) == meta['n_modes'] == 17076
    np.testing.assert_allclose(power.sum(),A**2,rtol=1e-14)
    expected_za = field['sigma_delta']*np.sqrt(np.sum(density_weights/k**2))
    np.testing.assert_allclose(expected_za,meta['expected_za_rms'],rtol=1e-14)
    edges = 2*np.pi/L*np.arange(.5,17.,1.)
    edges[-1] = np.nextafter(kmax,np.inf)
    counts = np.histogram(k,bins=edges)[0]
    shell_fraction = np.histogram(k,bins=edges,weights=weights)[0]
    np.testing.assert_allclose(shell_fraction.sum(),1.,rtol=1e-12)
    q = arrays['q']
    plane_z = (cfg['plane_index']+.5)*L/field['particle_grid']
    plane_mask = q[:,2] == plane_z
    assert plane_mask.sum() == field['particle_grid']**2
    plane = q[plane_mask]
    low,high = cfg['zoom_initial_bounds']
    track = plane_mask & np.all((q[:,:2]>=low)&(q[:,:2]<high),axis=1)
    x_za = (q+arrays['za']) % L
    slices,finals,motions,rows = [],[],[],[]
    for eps,reference in zip(cfg['epsilons'],cfg['catalogues']):
        helical = A*(np.sqrt((1+eps)/2)*arrays['plus']+np.sqrt((1-eps)/2)*arrays['minus'])
        final,diagnostics = displace_mesh(arrays,A=A,eps=eps,L=L)
        assert digest(reference['path']) == reference['sha256']
        with np.load(reference['path']) as z:
            np.testing.assert_array_equal(final,z['catalogue'])
        slices.append(helical[plane_mask]);finals.append(final[track]);motions.append(helical[track])
        rows.append(dict(epsilon=eps,expected_helical_RMS=A,
            measured_helical_RMS=diagnostics['helical_rms'],measured_ZA_RMS=diagnostics['za_rms'],
            measured_total_RMS=diagnostics['total_displacement_rms']))
    atomic_npz(out/'examples.npz',mode_k=k,mode_weight=weights,mode_power=power,
        shell_edges=edges,shell_fraction=shell_fraction,shell_counts=counts,
        plane=plane,slices=np.array(slices),epsilons=np.array(cfg['epsilons']),
        initial_points=q[track],za_points=x_za[track],final_points=np.array(finals),
        helical_motion=np.array(motions))
    with (out/'realization_RMS.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'mode_power.csv').open('w',newline='') as f:
        writer = csv.writer(f);writer.writerow(['k','W_k','A_squared_W_k'])
        writer.writerows(zip(k,weights,power))
    checks = dict(passed=True,matched_production_catalogues=3,active_modes=len(k),
        weight_sum=float(weights.sum()),expected_squared_helical_RMS=float(power.sum()),
        expected_ZA_RMS=float(expected_za),positive_helicity=meta['plus']['normalized_helicity'],
        negative_helicity=meta['minus']['normalized_helicity'],
        max_longitudinal_fraction=max(meta[v]['longitudinal_fraction'] for v in ('plus','minus')))
    assert abs(checks['positive_helicity']-1)<1e-12 and abs(checks['negative_helicity']+1)<1e-12
    assert checks['max_longitudinal_fraction']<1e-25
    atomic_json(out/'validation/report.json',checks)
    atomic_json(out/'analysis.json',dict(complete=True,config_sha256=digest(out/'config.json'),
        seed=cfg['seed'],plane_z=plane_z,tracked_points=int(track.sum()),slice_points=int(plane_mask.sum()),
        kmin=kmin,kmax=kmax,rows=rows,source_field_diagnostics=meta,
        helical_k_RMS=float(np.sqrt(np.sum(weights*k*k)))))
    print('Verified the saved fiducial fields, spectrum normalization and three particle maps',flush=True)


def read_data(out):
    out = Path(out)
    cfg = json.loads((out/'config.json').read_text())
    report = json.loads((out/'analysis.json').read_text())
    with np.load(out/'examples.npz') as z:
        data = {key:z[key] for key in z.files}
    return cfg,report,data


def save_figure(fig,out,name):
    for ext in ('png','pdf'):
        fig.savefig(Path(out)/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    return fig


def spectrum_figure(out):
    import matplotlib.pyplot as plt
    cfg,report,data = read_data(out)
    k = data['mode_k']
    unique,index = np.unique(k,return_index=True)
    fig,axes = plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    axes[0].loglog(unique,data['mode_power'][index],color='#2878b5',lw=2)
    axes[0].set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',
        ylabel=r'$A^2W_{\mathbf{k}}\ [(h^{-1}\mathrm{Mpc})^2]$',title='Displacement power per Fourier mode')
    edges = data['shell_edges']
    axes[1].stairs(100*data['shell_fraction'],edges,fill=True,alpha=.5,color='#2878b5')
    axes[1].set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel='Fraction of displacement variance [%]',
        title=r'Contribution of each shell to $A^2$')
    axes[1].set_xscale('log')
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle(r'$L=1000\ h^{-1}\mathrm{Mpc}$, $A=16\ h^{-1}\mathrm{Mpc}$; 17,076 active modes',fontsize=12)
    return save_figure(fig,out,'fiducial_displacement_power')


def helicity_figure(out):
    import matplotlib.pyplot as plt
    cfg,_,_ = read_data(out)
    L,A = cfg['field']['L'],cfg['A']
    z = np.linspace(0,L,25)
    fig = plt.figure(figsize=(10.5,4.7))
    for i,hand in enumerate((1,-1),1):
        vx,vy = A*np.cos(2*np.pi*z/L),-hand*A*np.sin(2*np.pi*z/L)
        ax = fig.add_subplot(1,2,i,projection='3d')
        ax.quiver(np.zeros_like(z),np.zeros_like(z),z,vx,vy,np.zeros_like(z),
                  length=1,color='#2878b5',alpha=.6,arrow_length_ratio=.12)
        ax.plot(vx,vy,z,color='#d87c30',lw=2)
        ax.set(title=rf'Helicity ${hand:+d}$',xlim=(-1.2*A,1.2*A),ylim=(-1.2*A,1.2*A),zlim=(0,L))
        ax.text2D(.21,-.07,r'$\Psi_y\ [h^{-1}\mathrm{Mpc}]$',transform=ax.transAxes,ha='center',clip_on=False)
        ax.text2D(.78,-.07,r'$\Psi_x\ [h^{-1}\mathrm{Mpc}]$',transform=ax.transAxes,ha='center',clip_on=False)
        ax.text2D(-.12,.5,r'$q_z\ [h^{-1}\mathrm{Mpc}]$',transform=ax.transAxes,rotation=90,va='center')
        ax.set_box_aspect((1,1,1.3));ax.view_init(elev=22,azim=30)
    fig.suptitle('One circularly polarized mode: wavelength 1000, displacement strength 16',fontsize=12)
    fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.82,wspace=.35)
    return save_figure(fig,out,'single_mode_helicity')


def slice_figure(out):
    import matplotlib.pyplot as plt
    cfg,report,data = read_data(out)
    L = cfg['field']['L']
    plane,slices = data['plane'],data['slices']
    axis = np.unique(plane[:,0])
    limit = float(abs(slices[:,:,2]).max())
    fig,axes = plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,eps,vec in zip(axes,data['epsilons'],slices):
        artist = ax.pcolormesh(axis,axis,vec[:,2].reshape(len(axis),len(axis)).T,
            shading='nearest',cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.85)
        ax.quiver(plane[:,0],plane[:,1],vec[:,0],vec[:,1],angles='xy',scale_units='xy',scale=1,
                  color='#252525',width=.0026)
        ax.set(xlim=(0,L),ylim=(0,L),aspect='equal',xlabel=r'$q_x\ [h^{-1}\mathrm{Mpc}]$',
               ylabel=r'$q_y\ [h^{-1}\mathrm{Mpc}]$',title=rf'$\epsilon={eps:+g}$')
    fig.colorbar(artist,ax=axes,label=r'$\Psi_{{\rm hel},z}\ [h^{-1}\mathrm{Mpc}]$',shrink=.82)
    fig.suptitle(f"Current spectrum; A=16; same seed {cfg['seed']}; plane q_z={report['plane_z']:g}",fontsize=12)
    return save_figure(fig,out,'fiducial_random_field_slices')


def particle_figure(out):
    import matplotlib.pyplot as plt
    cfg,report,data = read_data(out)
    baseline = data['za_points']
    colors = ['#2878b5','#806399','#d95335']
    fig,axes = plt.subplots(1,3,figsize=(12.5,4.3),layout='constrained')
    for ax,eps,final,move,color in zip(axes,data['epsilons'],data['final_points'],data['helical_motion'],colors):
        # The central patch has no global periodic crossings; vectors are true displacements.
        np.testing.assert_allclose(final,baseline+move,atol=1e-12,rtol=0)
        ax.scatter(baseline[:,0],baseline[:,1],s=11,color='.6',label='ZA-only positions')
        ax.quiver(baseline[:,0],baseline[:,1],move[:,0],move[:,1],angles='xy',scale_units='xy',scale=1,
                  color=color,alpha=.65,width=.0022)
        ax.scatter(final[:,0],final[:,1],s=11,color=color,label='ZA + helical positions')
        ax.set(xlim=(200,800),ylim=(200,800),aspect='equal',xlabel=r'$x\ [h^{-1}\mathrm{Mpc}]$',
               ylabel=r'$y\ [h^{-1}\mathrm{Mpc}]$',title=rf'$A=16$, $\epsilon={eps:+g}$')
        ax.legend(loc='lower left',fontsize=7.5)
    fig.suptitle(f"Same {report['tracked_points']} particles from the initial plane; projected onto x–y",fontsize=12)
    return save_figure(fig,out,'fiducial_particle_displacements')


def rms_table(out):
    _,report,_ = read_data(out)
    lines = [r'**Measured displacement RMS for this realization** (all 32,768 particles; units $h^{-1}\mathrm{Mpc}$).',
             '',r'| ε | Expected helical RMS $A$ | Measured helical RMS | Measured ZA + helical RMS |',
             '|---:|---:|---:|---:|']
    for row in report['rows']:
        lines.append(f"| {row['epsilon']:+g} | {row['expected_helical_RMS']:g} | "
                     f"{row['measured_helical_RMS']:.3f} | {row['measured_total_RMS']:.3f} |")
    lines.extend(['',r'$A$ sets the ensemble RMS; a single realization fluctuates. The displacement vector has zero ensemble mean.'])
    return '\n'.join(lines)


def notebook_template(out):
    import nbformat
    out = Path(out).resolve()
    cfg,report,_ = read_data(out)
    md,code = nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells = [md(r'''# PV displacement: the current fiducial model

**Box size $L=1000\ h^{-1}\mathrm{Mpc}$; displacement strength $A=16\ h^{-1}\mathrm{Mpc}$.**

Use a Gaussian initial-density mesh of $256^3$ cells and a cell-centred lattice of $32^3=32768$ particles. The examples below reuse **production seed 1000** and the current displacement power spectrum.

## 1. Constructing the displacement field

The initial Gaussian density has RMS $\sigma_\delta=0.2$ and toy spectrum
$$P_\delta(k)\propto e^{-(kR_\delta)^2},\qquad R_\delta=6.25\ h^{-1}\mathrm{Mpc}.$$
The helical displacement uses
$$H(k)=\frac{P_\delta(k)e^{-(kR_h)^2}}{k^4}
\propto\frac{e^{-2(k\,6.25\,h^{-1}\mathrm{Mpc})^2}}{k^4},\qquad R_h=6.25\ h^{-1}\mathrm{Mpc},$$
$$W_{\mathbf k}=\frac{H(k)}{\sum_{\mathbf k\,\mathrm{active}}H(k)},\qquad
P_{\Psi,\mathbf k}=A^2W_{\mathbf k},\qquad \sum_{\mathbf k}P_{\Psi,\mathbf k}=A^2.$$

The **17,076 active modes** cover $2\pi/L\le k\le\pi(32)/L$, or $0.006283\le k\le0.100531\ h\,\mathrm{Mpc}^{-1}$. Zero and unpaired mesh-Nyquist modes are excluded.

The left plot shows displacement variance **per Fourier mode**, in squared-length units. The right plot sums the contributions in each $k$ shell; the fractions add to 100%.
'''),code(f'''from pathlib import Path
import importlib.util
from IPython.display import display, Markdown
%matplotlib inline
import matplotlib.pyplot as plt
OUT = Path({str(out)!r})
spec = importlib.util.spec_from_file_location('fiducial_intro_report', Path({str(Path(__file__).resolve())!r}))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
fig = report.spectrum_figure(OUT)
display(fig)
plt.close(fig)
'''),md(r'''## 2. What does helicity mean?

Helical Fourier vectors satisfy $i\mathbf k\times\mathbf e_\pm=\pm k\mathbf e_\pm$. Construct independent Gaussian fields $\mathbf u_\pm$ with unit ensemble RMS, then combine them as
$$\boldsymbol\Psi_{\rm hel}=A\left[\sqrt{\frac{1+\epsilon}{2}}\,\mathbf u_+
+\sqrt{\frac{1-\epsilon}{2}}\,\mathbf u_-\right].$$
$\epsilon=+1$ or $-1$ selects one handedness; $\epsilon=0$ has equal expected power in both. In every case,
$$\sqrt{\mathbb E[\langle|\boldsymbol\Psi_{\rm hel}|^2\rangle]}=A.$$

**One-mode illustration:** $\boldsymbol\Psi_\pm=A(\cos kq_z,\mp\sin kq_z,0)$, with wavelength $L=1000$. The arrows rotate in opposite directions. This illustrates handedness; the random fields below contain the full spectrum from Section 1.
'''),code('fig = report.helicity_figure(OUT)\ndisplay(fig)\nplt.close(fig)'),
    md(r'''## 3. Random-field examples at $A=16$

The same underlying Gaussian draws are used in all three panels. The slice is at $q_z=PLANE_Z\ h^{-1}\mathrm{Mpc}$, a plane of the initial particle lattice. Arrows show $(\Psi_{{\rm hel},x},\Psi_{{\rm hel},y})$ at their true coordinate scale; colour shows $\Psi_{{\rm hel},z}$. All panels share the same colour scale.
'''.replace('PLANE_Z',f"{report['plane_z']:g}")),code('fig = report.slice_figure(OUT)\ndisplay(fig)\nplt.close(fig)\ndisplay(Markdown(report.rms_table(OUT)))'),
    md(r'''## 4. Adding the displacement to the initial conditions

Derive the Zel’dovich approximation (ZA) displacement from the Gaussian density,
$$\boldsymbol\Psi_{\rm ZA}(\mathbf k)=i\mathbf k\,\delta_{\mathbf k}/k^2,$$
then place particles using both fields evaluated at their **initial** lattice positions:
$$\mathbf x(\mathbf q)=\left[\mathbf q+\boldsymbol\Psi_{\rm ZA}(\mathbf q)+\boldsymbol\Psi_{\rm hel}(\mathbf q)\right]\bmod L.$$

The plots track the same 256 particles from a central patch of the initial plane. Gray points show their **ZA-only** positions; coloured points and arrows show the effect of adding the helical displacement at $A=16$. The view is a projection onto $x$–$y$.

The particle spacing is $31.25\ h^{-1}\mathrm{Mpc}$, so $A$ is **0.512 particle spacings**. These are displaced initial conditions; no subsequent gravitational evolution is applied.
'''),code('fig = report.particle_figure(OUT)\ndisplay(fig)\nplt.close(fig)')]
    link = f'saved/{out.name}'
    cells.append(md(f'[Catalogue chirality measurements](field_PVdisp_B.ipynb) · '
        f'[Density-power response](field_PVdisp_Pk.ipynb) · [Basic PV definitions](basic.ipynb)\n\n'
        f'[Configuration]({link}/config.json) · [Plots]({link}/figures/) · [Example arrays]({link}/examples.npz) · '
        f'[Displacement power]({link}/mode_power.csv) · [RMS table]({link}/realization_RMS.csv) · '
        f'[Validation]({link}/validation/report.json)'))
    for cell in cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden':True}
    return nbformat.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3 (project)',language='python'),
        parity_results=dict(kind='run',topic='intro_disp_PV',run=Path(cfg['source']).name,
                            format_version=6,source_runs=[cfg['source']],example_directory=str(out))))


def render(out,refresh=False):
    import nbformat
    from nbclient import NotebookClient
    out = Path(out).resolve()
    check_manifest(out)
    cfg = json.loads((out/'config.json').read_text())
    target = Path(cfg['target'])
    expected_previous = cfg['previous_notebook_sha256']
    if refresh:
        expected_previous = json.loads((out/'report_provenance.json').read_text())['notebook_sha256']
        assert digest(target) == expected_previous, 'Notebook changed outside the saved report'
        archive = out/'report_revision/before'
        archive.mkdir(parents=True,exist_ok=False)
        for path in [target,out/'report_provenance.json',out/'output_sha256.json']:
            shutil.copy2(path,archive/path.name)
        shutil.copytree(out/'figures',archive/'figures')
    nb = notebook_template(out)
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    nbformat.validate(nb)
    assert digest(target) == expected_previous, 'Notebook changed during rendering'
    nbformat.write(nb,out/target.name)
    shutil.copy2(out/target.name,target)
    atomic_json(out/'report_provenance.json',dict(notebook_sha256=digest(target),
        config_sha256=digest(out/'config.json'),source_sha256=digest(Path(__file__)),arrays_sha256=digest(out/'examples.npz')))
    files = [out/target.name,out/'examples.npz',out/'analysis.json',out/'report_provenance.json',
             *out.glob('*.csv'),*(out/'figures').glob('*'),out/'validation/report.json',out/'validation/pytest.xml']
    atomic_json(out/'output_sha256.json',{str(p.relative_to(out)):digest(p) for p in files})
    atomic_json(OUTPUT_ROOT/'intro_disp_PV/active_example.json',dict(directory=str(out),
        config_sha256=digest(out/'config.json'),examples_sha256=digest(out/'examples.npz'),
        report_source=str(Path(__file__).resolve()),report_source_sha256=digest(Path(__file__))))
    print(f'Saved and activated {target}',flush=True)


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
        render(args.out,refresh=True)


if __name__ == '__main__':
    main()
