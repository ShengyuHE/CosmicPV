"""Matched displacement-strength scan with a previously frozen shape selection."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt

from parity_study.tests.independent_ics import audit
from parity_study.initial_conditions import displace_mesh, spectrum_shape
from parity_study.mesh_cases import case_parameters
from parity_study.seeds import seed_ids
from parity_study.tests.selections import measure
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest, load_config
from parity_study.summarize import mean_error

AMPLITUDES = [0., .25, .5, 1., 2., 4., 8., 16.]


def physical_scales(field):
    """Exact discrete-mode moments on the active band, without full vector meshes."""
    L, N = field['L'], field['N']
    kmin = field.get('kmin', 2*np.pi/L)
    kmax = field.get('kmax', np.pi*min(N, field['particle_grid'])/L)
    axis = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    if field.get('exclude_nyquist', True) and N % 2 == 0:
        axis = np.delete(axis, N//2)
    axis = axis[abs(axis) <= kmax]
    k = np.linalg.norm(np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1), axis=-1)
    k = k[(k >= kmin) & (k <= kmax) & (k > 0)]
    shape = spectrum_shape(k, field['spectrum'], np.ones(k.shape, dtype=bool))
    wd = shape/shape.sum()
    wh = shape*np.exp(-(k*field['helical_smoothing'])**2)/k**4
    wh /= wh.sum()
    za_rms = field['sigma_delta']*np.sqrt(np.sum(wd/k**2))
    k_rms = np.sqrt(np.sum(wh*k**2))
    return dict(length_unit='h^-1 Mpc (comoving)', mesh_spacing=L/N,
                particle_spacing=L/field['particle_grid'], smoothing=field['helical_smoothing'],
                expected_za_rms=float(za_rms), helical_k_rms=float(k_rms),
                inverse_k_rms_length=float(1/k_rms), kmin=float(kmin), kmax=float(kmax),
                n_modes=len(k))


def prepare(source, selection_path, out, *, amplitudes=None, optimize=False):
    source = Path(source).resolve()
    check_manifest(source)
    config = load_config(source)
    amplitudes = list(AMPLITUDES if amplitudes is None else map(float, amplitudes))
    if (len(amplitudes)<3 or not np.isfinite(amplitudes).all() or amplitudes[0]!=0
            or np.any(np.diff(amplitudes)<=0) or config['null']['A'] not in amplitudes):
        raise ValueError('Amplitudes must start at zero, increase strictly, and contain null.A')
    previous = {}
    if config.get('purpose') == 'amplitude_scan':
        _,old,old_path=verify(source)
        old_analysis=old_path.parent/'analysis.json'
        if not json.loads(old_analysis.read_text())['complete'] or not set(old['amplitudes'])<=set(amplitudes):
            raise ValueError('Extension must include the completed previous amplitude grid')
        previous=dict(previous_amplitude_protocol=str(old_path),
                      previous_amplitude_protocol_sha256=digest(old_path),
                      previous_analysis=str(old_analysis),previous_analysis_sha256=digest(old_analysis))
    if config['field'].get('density_seed') is not None:
        raise ValueError('The amplitude scan uses independent Gaussian ICs')
    if not json.loads((source/'validation/independent_ic_ensemble.json').read_text())['passed']:
        raise ValueError('Source ensemble has not passed its independence audit')
    selection = json.loads(selection_path.read_text())
    if config['field'] != dict(selection['field'], density_seed=None):
        raise ValueError('Source field differs from the discovery model')
    cuts = [c for c in selection['cuts'] if c['id'] in ('all', selection['primary_selection'])]
    if (out/'protocol.json').exists():
        raise FileExistsError(out/'protocol.json')
    protocol = dict(experiment='matched_amplitude_scan', created_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(source), source_manifest_sha256=digest(source/'manifest.json'),
        selection_protocol=str(selection_path.resolve()), selection_protocol_sha256=digest(selection_path),
        cuts=cuts, primary_selection=selection['primary_selection'], expected_sign=selection['expected_sign'],
        field=config['field'], seeds=list(seed_ids(config, config['scan']['seeds'])),
        amplitudes=amplitudes, control_amplitude=config['null']['A'],
        length_unit='h^-1 Mpc (comoving)', physical_scales=physical_scales(config['field']),
        analysis=f'Paired Student-t across ICs at each A. {len(amplitudes)-1} nonzero amplitudes form the '
                 f'primary family (Bonferroni); {len(amplitudes)-2} matched contrasts against A=2 form a separate family.',
        zero_amplitude='One ZA-only catalogue per IC represents all three helicities at A=0; D=0 exactly.',
        interpretation='Exploratory response characterization on the saved 25 ICs: A=2 was already inspected '
                       'and the cut was selected at A=2. These data do not independently validate an optimal A.',
        physical_interpretation='Lengths are assigned comoving h^-1 Mpc by the user. The power spectrum '
                                'remains a controlled toy shape; no cosmological redshift or PV model calibrates A.',
        paper='https://arxiv.org/html/2406.06080v2',**previous)
    if optimize:
        protocol['optimization']=dict(bootstrap_replicates=10000,bootstrap_seed=20261121,
            magnitude='Maximize abs(mean paired D) over the tested nonzero A values.',
            detectability='Maximize abs(mean paired D)/SEM over the tested nonzero A values.',
            bootstrap='Resample IC indices jointly across amplitudes; recompute both rankings. '
                      'Selection frequencies describe sampling stability, not probabilities of a true optimum.',
            comparisons='Compare signed response expected_sign*D for every pair of nonzero amplitudes; '
                        'Bonferroni over all pairs protects comparisons chosen after ranking.',
            limitation='Adaptive extension requested after the first scan; descriptive optimum on reused ICs. '
                       'Fresh ICs and resolution tests are needed to validate a preferred A.')
    atomic_json(out/'protocol.json', protocol)
    return out/'protocol.json'


def verify(run):
    check_manifest(run)
    cfg = load_config(run)
    path = Path(cfg['amplitude_protocol']['path'])
    if digest(path) != cfg['amplitude_protocol']['sha256']:
        raise ValueError('Amplitude protocol changed')
    p = json.loads(path.read_text())
    selection_path = Path(p['selection_protocol'])
    if digest(selection_path) != p['selection_protocol_sha256']:
        raise ValueError('Original shape selection changed')
    parent = json.loads(selection_path.read_text())
    expected_cuts = [c for c in parent['cuts'] if c['id'] in ('all', parent['primary_selection'])]
    if p['cuts'] != expected_cuts or p['primary_selection'] != parent['primary_selection']:
        raise ValueError('Amplitude scan cannot change the frozen cut')
    source = Path(p['source_run'])
    if digest(source/'manifest.json') != p['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for key in ('previous_amplitude_protocol','previous_analysis'):
        if key in p and digest(Path(p[key])) != p[key+'_sha256']:
            raise ValueError(f'Previous amplitude study changed: {key}')
    if (cfg['purpose'] != 'amplitude_scan' or cfg['field'] != p['field']
            or cfg['field'] != load_config(source)['field']
            or list(seed_ids(cfg, cfg['scan']['seeds'])) != p['seeds']
            or cfg['null']['seeds'] != len(p['seeds'])
            or cfg['scan']['amplitudes'] != p['amplitudes'][1:]
            or cfg['null']['amplitudes'] != p['amplitudes'][1:]
            or cfg['null']['A'] != p['control_amplitude']
            or cfg['length_unit'] != p['length_unit']):
        raise ValueError('Run differs from the frozen amplitude scan')
    return cfg, p, path


def reuse(run):
    """Copy audited fields and matching A=2/ZA catalogues; never modify the source."""
    run = Path(run).resolve()
    cfg, p, _ = verify(run)
    source = Path(p['source_run'])
    check_manifest(source)
    if not json.loads((source/'validation/independent_ic_ensemble.json').read_text())['passed']:
        raise ValueError('Source independence audit failed')
    for rel in [Path('code/src/parity_study')/name for name in
                ('initial_conditions.py','catalogue.py','delaunay.py')] + [Path('code/bin/DIVE')]:
        if digest(run/rel) != digest(source/rel):
            raise ValueError(f'Field or catalogue construction changed: {rel}')
    copied = []
    def copy(rel):
        a, b = source/rel, run/rel
        sha = digest(a)
        b.parent.mkdir(parents=True, exist_ok=True)
        if b.exists():
            if digest(b) != sha:
                raise ValueError(f'Refusing to overwrite different cached input: {b}')
        else:
            shutil.copy2(a, b)
        if digest(b) != sha:
            raise IOError(f'Copy checksum failed: {b}')
        copied.append(dict(path=str(rel), sha256=sha))
    old_cases = case_parameters(load_config(source))
    new_cases = case_parameters(cfg)
    for seed in p['seeds']:
        rel = Path('fields')/f'main_{seed:04d}.npz'
        with np.load(source/rel) as z:
            arrays = {k:z[k] for k in ('q','za','plus','minus')}
            meta = json.loads(str(z['metadata_json']))
        if meta['field'] != cfg['field'] or meta['density_seed'] != seed or meta['helical_seed'] != seed:
            raise ValueError('Source fields do not match the planned IC')
        copy(rel)
        for name, (a, eps) in old_cases.items():
            if new_cases.get(name) != (a, eps):
                continue
            rel = Path('records')/name/f'seed_{seed:04d}.npz'
            with np.load(source/rel) as z:
                meta = json.loads(str(z['metadata_json']))
                params = meta['parameters']
                if (meta['seed'] != seed or params['A'] != a or params['eps'] != eps
                        or params['density_seed'] != seed or params['helical_seed'] != seed):
                    raise ValueError('Source catalogue metadata mismatch')
                expected, _ = displace_mesh(arrays, A=a, eps=eps, L=cfg['field']['L'])
                np.testing.assert_array_equal(z['catalogue'], expected)
            copy(rel)
    copy(Path('fields/validation_mesh_0042.npz'))
    atomic_json(run/'reused_inputs.json',dict(source=str(source),files=copied,
                catalogues=sum(item['path'].startswith('records/') for item in copied),
                source_manifest_sha256=digest(source/'manifest.json')))
    print(f'Reused {len(copied)} verified files; only missing amplitudes will be generated.',flush=True)


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def amplitude_ranking(paired, amplitudes, *, expected_sign=-1, replicates=10000, seed=20261121):
    """Rank a fixed grid, preserving IC pairing in bootstrap and all-pair contrasts."""
    values=np.asarray(paired,float)
    amplitudes=np.asarray(amplitudes,float)
    use=amplitudes>0
    values=values[use]; amplitudes=amplitudes[use]
    if values.ndim!=2 or values.shape[1]<2 or not np.isfinite(values).all():
        raise ValueError('Need finite paired measurements from at least two ICs')
    n=len(values[0]); mean=values.mean(axis=1); sem=values.std(axis=1,ddof=1)/np.sqrt(n)
    strength=abs(mean)
    snr=np.divide(strength,sem,out=np.zeros_like(strength),where=sem>0)
    best_mean=int(np.argmax(strength)); best_snr=int(np.argmax(snr))
    rng=np.random.default_rng(seed)
    winners_mean=np.zeros(len(values),dtype=int); winners_snr=np.zeros(len(values),dtype=int)
    valid_snr=0
    for start in range(0,replicates,500):
        indices=rng.integers(0,n,size=(min(500,replicates-start),n))
        draws=values[:,indices]
        means=abs(draws.mean(axis=2)); errors=draws.std(axis=2,ddof=1)/np.sqrt(n)
        winners_mean+=np.bincount(np.argmax(means,axis=0),minlength=len(values))
        valid=np.all(errors>0,axis=0)
        valid_snr+=int(valid.sum())
        winners_snr+=np.bincount(np.argmax(means[:,valid]/errors[:,valid],axis=0),minlength=len(values))
    family=len(values)*(len(values)-1)//2
    contrasts=[]
    for i,a in enumerate(amplitudes):
        for j in range(i+1,len(amplitudes)):
            stats=mean_error(expected_sign*(values[i]-values[j]))
            contrasts.append(dict(A=float(a),other_A=float(amplitudes[j]),**stats,
                p_all_pairs_bonferroni=min(1.,family*stats['p_two_sided']) if stats['p_two_sided'] is not None else None))
    scores=[dict(A=float(a),absolute_mean=float(strength[i]),snr=float(snr[i]),
                 bootstrap_best_magnitude=winners_mean[i]/replicates,
                 bootstrap_best_snr=winners_snr[i]/valid_snr if valid_snr else None)
            for i,a in enumerate(amplitudes)]
    # Rank identification uses the pre-specified response sign, rather than post-hoc absolute-value contrasts.
    peak=best_mean
    indistinguishable=[]
    for j,a in enumerate(amplitudes):
        if j==peak:
            indistinguishable.append(float(a)); continue
        stats=mean_error(expected_sign*(values[peak]-values[j]))
        corrected=min(1.,family*stats['p_two_sided']) if stats['p_two_sided'] is not None else None
        if stats['mean']<=0 or corrected is None or corrected>=.05:
            indistinguishable.append(float(a))
    return dict(best_magnitude_A=float(amplitudes[best_mean]),best_snr_A=float(amplitudes[best_snr]),
        magnitude_peak_at_upper_boundary=best_mean==len(amplitudes)-1,
        snr_peak_at_upper_boundary=best_snr==len(amplitudes)-1,
        peak_has_expected_sign=bool(np.sign(mean[peak])==expected_sign),
        not_distinguished_from_magnitude_peak=indistinguishable,
        bootstrap_replicates=replicates,bootstrap_valid_snr=valid_snr,bootstrap_seed=seed,
        pairwise_family_size=family,scores=scores,pairwise_contrasts=contrasts)


def analyze(run):
    run = Path(run).resolve()
    cfg, p, path = verify(run)
    if not json.loads((run/'validation/report.json').read_text()).get('passed'):
        raise ValueError('Construction validation did not pass')
    audit_result, za = audit(run, cfg, p['seeds'], p['cuts'])
    out = path.parent
    signals = np.empty((len(p['cuts']),len(p['amplitudes']),3,len(p['seeds'])))
    signals[:,0] = za[:,None,:]
    fractions = np.empty((len(p['cuts']),len(p['amplitudes'])))
    fractions[:,0] = np.nan
    rows, hashes = [], {}
    for ai,a in enumerate(p['amplitudes'][1:],1):
        result = measure(run,p['cuts'],p['seeds'],a)
        signals[:,ai] = result['signals']
        fractions[:,ai] = [row['retained_fraction'] for row in result['summaries']]
        rows.extend(dict(A=a,**row) for row in result['rows'])
        hashes.update(result['input_sha256'])
    for j,seed in enumerate(p['seeds']):
        rel = Path('records/null_za')/f'seed_{seed:04d}.npz'
        hashes[str(rel)] = digest(run/rel)
        with np.load(run/rel) as z:
            tets = z['tetrahedra']
        from parity_study.tests.selections import selection_mask
        for ci,cut in enumerate(p['cuts']):
            count = int(selection_mask(tets,cut,cfg['field']['L']/cfg['field']['particle_grid']).sum())
            # Store the physical ZA catalogue once; all helicities coincide at A=0.
            rows.append(dict(A=0.,selection=cut['id'],epsilon=0,seed=seed,S=float(za[ci,j]),
                contribution=float(za[ci,j]*count/len(tets)),selected=count,total=len(tets),fraction=count/len(tets)))
    for ci,cut in enumerate(p['cuts']):
        fractions[ci,0] = np.mean([r['fraction'] for r in rows if r['A']==0 and r['selection']==cut['id']])
    paired = signals[:,:,2]-signals[:,:,0]
    anchor = p['amplitudes'].index(p['control_amplitude'])
    summaries, contrasts, controls = [], [], []
    nonzero = len(p['amplitudes'])-1
    for ci,cut in enumerate(p['cuts']):
        for ai,a in enumerate(p['amplitudes']):
            stats = mean_error(paired[ci,ai])
            summaries.append(dict(selection=cut['id'],label=cut['label'],A=a,
                retained_fraction=float(fractions[ci,ai]),**stats,
                p_amplitude_bonferroni=min(1.,nonzero*stats['p_two_sided']) if stats['p_two_sided'] is not None else None))
            for ei,eps in enumerate((-1,0,1)):
                controls.append(dict(selection=cut['id'],A=a,epsilon=eps,**mean_error(signals[ci,ai,ei])))
            if ai not in (0,anchor):
                stats = mean_error(paired[ci,ai]-paired[ci,anchor])
                contrasts.append(dict(selection=cut['id'],A=a,reference_A=p['control_amplitude'],**stats,
                    p_contrast_bonferroni=min(1.,(nonzero-1)*stats['p_two_sided']) if stats['p_two_sided'] is not None else None))
    # Preserve the full cross-A covariance; matched amplitudes are correlated.
    covariance = np.array([np.cov(d,ddof=1)/len(p['seeds']) for d in paired])
    scale=p['physical_scales']
    scales=[dict(A=a,A_over_mesh=a/scale['mesh_spacing'],A_over_particle=a/scale['particle_spacing'],
                 A_over_ZA=a/scale['expected_za_rms'],A_over_Rh=a/scale['smoothing'],
                 rms_helical_gradient=a*scale['helical_k_rms']) for a in p['amplitudes']]
    selected = [r for r in summaries if r['selection']==p['primary_selection'] and r['A']>0]
    detected = [r for r in selected if r['p_amplitude_bonferroni'] is not None
                and r['p_amplitude_bonferroni']<.05 and np.sign(r['mean'])==p['expected_sign']]
    report = dict(run=str(run),protocol_sha256=digest(path),complete=True,primary_selection=p['primary_selection'],
        amplitudes=p['amplitudes'],seeds=p['seeds'],independent_ic_audit=audit_result,
        summaries=summaries,contrasts_against_A2=contrasts,helicity_means=controls,scales=scales,
        physical_scales=scale,first_detected_A=min(r['A'] for r in detected) if detected else None,
        interpretation=p['interpretation'])
    if 'optimization' in p:
        ci=next(i for i,c in enumerate(p['cuts']) if c['id']==p['primary_selection'])
        opt=p['optimization']
        report['optimization']=amplitude_ranking(paired[ci],p['amplitudes'],expected_sign=p['expected_sign'],
            replicates=opt['bootstrap_replicates'],seed=opt['bootstrap_seed'])
        write_csv(out/'amplitude_ranking.csv',report['optimization']['scores'])
        write_csv(out/'all_pair_contrasts.csv',report['optimization']['pairwise_contrasts'])
    atomic_npz(out/'amplitude_scan.npz',selection_ids=np.array([c['id'] for c in p['cuts']]),
               amplitudes=np.array(p['amplitudes']),seeds=np.array(p['seeds']),
               epsilons=np.array([-1,0,1]),signals=signals,paired=paired,mean_covariance=covariance,
               retained_fraction=fractions)
    for name,data in [('catalogues',rows),('summary',summaries),('contrasts',contrasts),('helicity_means',controls),('scales',scales)]:
        write_csv(out/f'{name}.csv',data)
    atomic_json(out/'input_sha256.json',hashes)
    atomic_json(out/'analysis.json',report)
    print(json.dumps(dict(complete=True,ICs=len(p['seeds']),catalogues=audit_result['checked_catalogues'],
                         first_detected_A=report['first_detected_A'],selected=selected),indent=2),flush=True)


def response_figure(report):
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True,
                          gridspec_kw={'width_ratios':[1.4,1]})
    extended='optimization' in report
    detail=max(64,2*report['optimization']['best_magnitude_A']) if extended else 2
    detail=min(detail,max(report['amplitudes']))
    for ax,maximum,title in zip(axes,(max(report['amplitudes']),detail),('Full scan','Peak detail' if extended else 'Small-amplitude detail')):
        for key,label,color in [('all','All tetrahedra','#777777'),(report['primary_selection'],'Frozen shape selection','#762a83')]:
            rows=[r for r in report['summaries'] if r['selection']==key and r['A']<=maximum]
            ax.errorbar([r['A'] for r in rows],[r['mean'] for r in rows],yerr=[r['sem'] for r in rows],
                        fmt='o-',capsize=4,label=label,color=color)
        ax.axhline(0,color='k',lw=.7)
        ax.set(xlabel=r'Displacement strength $A$ [$h^{-1}\mathrm{Mpc}$]',title=title)
        if extended and ax is axes[0]:
            ax.set_xscale('symlog',linthresh=1)
            ax.set_xlim(0,max(report['amplitudes'])*1.15)
    axes[0].set_ylabel(r'Matched catalogue difference $\overline{D}_B(A)$')
    if not extended: axes[1].set_xticks([0,.5,1,2])
    fig.suptitle(f"Amplitude response · {len(report['seeds'])} matched ICs · mean ± SEM")
    if not extended:
        ell=report['physical_scales']['particle_spacing']
        top=axes[0].secondary_xaxis('top',functions=(lambda a:a/ell,lambda r:r*ell))
        top.set_xlabel(r'$A/\ell_p$ (particle spacing)')
    axes[0].legend()
    return fig


def helicity_figure(report):
    fig,ax=plt.subplots(figsize=(8,4.3),constrained_layout=True)
    for eps,color in [(-1,'#2166ac'),(0,'#666666'),(1,'#b2182b')]:
        rows=[r for r in report['helicity_means'] if r['selection']==report['primary_selection'] and r['epsilon']==eps]
        ax.errorbar([r['A'] for r in rows],[r['mean'] for r in rows],yerr=[r['sem'] for r in rows],
                    fmt='o-',capsize=4,label=rf'$\epsilon={eps:+d}$',color=color)
    ax.axhline(0,color='k',lw=.7)
    if 'optimization' in report:
        ax.set_xscale('symlog',linthresh=1)
        ax.set_xlim(0,max(report['amplitudes'])*1.15)
    ax.set(xlabel=r'Displacement strength $A$ [$h^{-1}\mathrm{Mpc}$]',
           ylabel=r'Selected catalogue mean $\overline{S}_B(\epsilon; A)$',
           title=f"Frozen shape selection · {len(report['seeds'])} ICs · mean ± SEM")
    ax.legend()
    return fig


def ranking_figure(report):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    rows=report['optimization']['scores']
    for ax,key,title in zip(axes,('absolute_mean','snr'),(r'Signal magnitude $|\overline{D}_B|$',
            r'Detection score $|\overline{D}_B|/\mathrm{SEM}$')):
        ax.plot([r['A'] for r in rows],[r[key] for r in rows],'o-',color='#762a83')
        ax.set(xscale='log',xlabel=r'Displacement strength $A$ [$h^{-1}\mathrm{Mpc}$]',ylabel=title)
        best=max(rows,key=lambda r:r[key])
        ax.axvline(best['A'],color='#444444',ls='--',lw=1,label=rf"Best tested $A={best['A']:g}$")
        ax.legend()
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--amplitudes',type=float,nargs='+'); p.add_argument('--optimize',action='store_true')
    for stage in ('reuse','analyze'):
        p=sub.add_parser(stage); p.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare': print(prepare(args.source,args.selection,args.out,amplitudes=args.amplitudes,optimize=args.optimize))
    elif args.action=='reuse': reuse(args.run)
    else: analyze(args.run)


if __name__=='__main__':
    main()
