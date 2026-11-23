"""Run a validation or one restartable seed block from a frozen experiment."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .catalogue import unit_fields, displace, chiral_catalogue, evaluate_field, spectral_grid
from .delaunay import parity_stat, statistics
from .storage import atomic_json, atomic_npz, load_config, check_manifest
from .seeds import seed_start

KEYS = ['chi', 'chi_hat', 'sgn']


def measured(run, name, seed, cat, *, L, parameters, diagnostics=None, template=None, hands=None):
    path = run/'records'/name/f'seed_{seed:04d}.npz'
    if path.exists():
        with np.load(path) as saved:
            if saved['statistics'].shape != (3,):
                raise RuntimeError(f'Corrupt result: {path}')
        print(f'resume {name} seed={seed}', flush=True)
        return
    t0 = time.monotonic()
    data = parity_stat(cat, L=L, executable=run/'code/bin/DIVE')
    meta = dict(seed=seed, case=name, L=L, parameters=parameters,
                diagnostics=diagnostics or {}, n_tetrahedra=len(data),
                ambiguous_order_cells=int(np.count_nonzero(data[:,0] == 0)),
                volume_fraction=float(np.sum(abs(data[:,0])) / (6*L**3)),
                delaunay_seconds=time.monotonic()-t0)
    arrays = dict(catalogue=cat, statistics=statistics(data),
                  statistic_names=np.array(KEYS))
    if template is not None:
        lengths = np.sort(np.linalg.norm(template[1:]-template[0], axis=1))
        mol = np.all(np.isclose(data[:,2:5], lengths, rtol=1e-7, atol=1e-10), axis=1)
        meta['molecules'] = dict(n_identified=int(mol.sum()), n_injected=len(hands),
            injected_mean_handedness=float(hands.mean()),
            measured_mean=float(data[mol,1].mean()) if mol.any() else None,
            other_mean=float(data[~mol,1].mean()),
            molecule_contribution_to_all=float(data[mol,1].sum()/len(data)))
        arrays['injected_handedness'] = hands
        arrays['molecule_mask'] = mol
    if load_config(run)['save_tetrahedra']:
        arrays['tetrahedra'] = data
    arrays['metadata_json'] = np.array(json.dumps(meta, allow_nan=False))
    atomic_npz(path, **arrays)
    print(f"saved {name} seed={seed} Ntet={len(data)} S={arrays['statistics']} ({meta['delaunay_seconds']:.2f}s)", flush=True)


def fields(run, seed, field, label='main'):
    field = dict(field)
    field.setdefault('amplitude_convention', 'rms')
    path = run/'fields'/f'{label}_{seed:04d}.npz'
    if path.exists():
        with np.load(path) as z:
            meta = json.loads(str(z['metadata_json']))
            cached_field = dict(meta['field'])
            # Caches produced before RMS normalization used the notebook convention.
            cached_field.setdefault('amplitude_convention', 'notebook')
            if cached_field != field:
                raise ValueError(f'Field cache parameters differ: {path}. Use a new run for changed parameters.')
            return z['q'], z['plus'], z['minus'], meta
    t0 = time.monotonic()
    q, p, m, meta = unit_fields(**field, seed=seed,
                                nthreads=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    meta.update(generation_seconds=time.monotonic()-t0, seed=seed, field=field)
    atomic_npz(path, q=q, plus=p, minus=m, metadata_json=np.array(json.dumps(meta)))
    print(f'cached fields {label} seed={seed} ({meta["generation_seconds"]:.2f}s)', flush=True)
    return q, p, m, meta


def seed_block(run, task):
    config = load_config(run)
    if config['field'].get('model') == 'mesh_za_helical':
        return mesh_seed_block(run, task, config)
    maximum = max(config[x]['seeds'] for x in ('null', 'scan', 'positive'))
    block = config['scheduler']['seeds_per_task']
    start = task*block
    if not 0 <= start < maximum:
        raise ValueError('Task index outside configured seed range')
    field = dict(config['field'])
    # Configurations from existing frozen runs predate this explicit option.
    field.setdefault('amplitude_convention', 'notebook')
    convention = field['amplitude_convention']
    L = field['L']
    for seed in range(start, min(start+block, maximum)):
        if seed < max(config['null']['seeds'], config['scan']['seeds']):
            q, plus, minus, fm = fields(run, seed, field)
        if seed < config['null']['seeds']:
            measured(run, 'null_poisson', seed, q, L=L, parameters={'A':0, 'eps':0},
                     diagnostics={'displacement_rms':0., 'rms_over_spacing':0.})
            A = config['null']['A']
            cat, diag = displace(q, plus, minus, A=A, eps=0, L=L, amplitude_convention=convention)
            measured(run, 'null_symmetric', seed, cat, L=L, parameters={'A':A, 'eps':0}, diagnostics=diag)
            if config['null'].get('include_original_L1_A1'):
                # Retain the original null cell's N=64, L=1, A=1 and Nyquist convention.
                original = dict(L=1., N=64, Ngal=10000, exclude_nyquist=False,
                                tolerance=field['tolerance'], amplitude_convention='notebook')
                oq, op, om, _ = fields(run, seed, original, label='notebook_null')
                cat, diag = displace(oq, op, om, A=1., eps=0, L=1., amplitude_convention='notebook')
                measured(run, 'null_notebook_L1_A1', seed, cat, L=1., parameters={'A':1., 'eps':0}, diagnostics=diag)
        if seed < config['scan']['seeds']:
            for A in config['scan']['amplitudes']:
                for eps, suffix in ((1, 'plus'), (-1, 'minus')):
                    cat, diag = displace(q, plus, minus, A=A, eps=eps, L=L, amplitude_convention=convention)
                    measured(run, f'scan_A{A:g}_{suffix}', seed, cat, L=L,
                             parameters={'A':A, 'eps':eps}, diagnostics=diag)
        if seed < config['positive']['seeds']:
            positive = config['positive']
            for fraction in positive['fractions']:
                cat, hand, template = chiral_catalogue(Ncl=positive['Ncl'], size=positive['size'],
                                                       f_R=fraction, seed=seed, L=1.)
                measured(run, f'positive_f{fraction:g}', seed, cat, L=1.,
                         parameters={'f_R':fraction, 'size':positive['size'], 'Ncl':positive['Ncl']},
                         template=template, hands=hand)
        atomic_json(run/'completed'/f'seed_{seed:04d}.json', {'seed':seed, 'completed':True})


def validate(run):
    if load_config(run)['field'].get('model') == 'mesh_za_helical':
        return validate_mesh(run)
    kw = dict(L=100., N=16, Ngal=1000, seed=42, exclude_nyquist=False, tolerance=1e-9)
    q, p, m, _ = unit_fields(**kw)
    q0, p0, m0, _ = unit_fields(**kw, method='direct')
    error = float(max(abs(p-p0).max(), abs(m-m0).max()))
    if error > 1e-5:
        raise AssertionError(f'NUFFT/direct mismatch: {error}')
    # Notebook T1 parameters, with all catalogue and tetrahedron outputs retained.
    field = dict(L=1000., N=64, Ngal=10000, exclude_nyquist=False,
                 tolerance=1e-9, amplitude_convention='notebook')
    q, p, m, fm = fields(run, 42, field, label='validation')
    cat, diag = displace(q,p,m,A=.002,eps=1,L=1000.,amplitude_convention='notebook')
    transforms = {'reference':(cat,1), 'inversion':((-cat)%1000,-1),
        'swap':(cat[:,[1,0,2]],-1),
        'rotation90':(np.c_[(-cat[:,1])%1000,cat[:,0],cat[:,2]],1),
        'translation':((cat+1000*np.array([.3712,.118,.6543]))%1000,1)}
    report = {'nufft_direct_max_error':error, 'T1':{}, 'field':fm, 'displacement':diag}
    ref = None
    for name, (c, sign) in transforms.items():
        d = parity_stat(c, L=1000., executable=run/'code/bin/DIVE')
        atomic_npz(run/'validation'/f'T1_{name}.npz', catalogue=c, tetrahedra=d)
        if ref is None:
            ref = d
        count_ok = len(d) == len(ref)
        chirality_error = float(abs(d[:,1].sum()-sign*ref[:,1].sum()))
        volume = float(abs(d[:,0]).sum()/6/1000**3)
        passed = bool(count_ok and chirality_error < 1e-7 and abs(volume-1)<1e-10)
        report['T1'][name] = dict(n_tetrahedra=len(d), sum_chi_hat=float(d[:,1].sum()),
                                  absolute_error=chirality_error, volume_fraction=volume, passed=passed)
        if not passed:
            atomic_json(run/'validation/report.json',report)
            raise AssertionError(f'T1 failed for {name}')
    # Helicity checks use the paired spectrum with Nyquist planes removed.
    _, _, _, hd = unit_fields(L=1000., N=16, Ngal=1000, seed=5, exclude_nyquist=True)
    for sign, name in [(1, 'plus'), (-1, 'minus')]:
        assert abs(hd[name]['normalized_helicity']-sign)<1e-12
        assert hd[name]['longitudinal_fraction']<1e-28
    report['helicity'] = hd
    report['passed'] = True
    atomic_json(run/'validation/report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def cached_mesh_fields(run, seed, field, label='main'):
    from .initial_conditions import mesh_fields
    path = run/'fields'/f'{label}_{seed:04d}.npz'
    if path.exists():
        with np.load(path) as z:
            meta = json.loads(str(z['metadata_json']))
            if meta['field'] != field or meta['seed'] != seed:
                raise ValueError(f'Field cache parameters differ: {path}')
            return {k:z[k] for k in ('q', 'za', 'plus', 'minus', 'delta_mesh')}, meta
    started = time.monotonic()
    arrays, meta = mesh_fields(**field, seed=seed, full_output=False,
                              nthreads=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    meta.update(field=field, seed=seed, generation_seconds=time.monotonic()-started)
    atomic_npz(path, **arrays, metadata_json=np.array(json.dumps(meta, allow_nan=False)))
    print(f'cached mesh ICs {label} seed={seed}', flush=True)
    return arrays, meta


def mesh_seed_block(run, task, config):
    from .initial_conditions import displace_mesh
    if config['positive']['seeds'] or config['null'].get('include_original_L1_A1'):
        raise ValueError('Mesh IC runs use ZA and equal-helicity nulls; legacy controls belong to separate runs')
    maximum = max(config[x]['seeds'] for x in ('null', 'scan'))
    block = config['scheduler']['seeds_per_task']
    start = task*block
    if not 0 <= start < maximum:
        raise ValueError('Task index outside configured seed range')
    field = config['field']
    first = seed_start(config)
    for index in range(start, min(start+block, maximum)):
        seed = first + index
        arrays, field_meta = cached_mesh_fields(run, seed, field)
        cases = []
        if index < config['null']['seeds']:
            # A fixed density gives the same ZA-only catalogue for every seed.
            # Measure it once, not as repeated independent observations.
            if field.get('density_seed') is None or index == 0:
                cases.append(('null_za', 0., 0.))
            from .mesh_cases import symmetric_cases
            cases.extend((name, A, 0.) for name, A in symmetric_cases(config))
        if index < config['scan']['seeds']:
            cases += [(f'scan_A{A:g}_{suffix}', A, eps)
                      for A in config['scan']['amplitudes'] for eps,suffix in ((1.,'plus'),(-1.,'minus'))]
        for name, A, eps in cases:
            cat, diag = displace_mesh(arrays, A=A, eps=eps, L=field['L'])
            measured(run, name, seed, cat, L=field['L'],
                     parameters=dict(A=A, eps=eps, model='mesh_za_helical',
                                     sigma_delta=field['sigma_delta'],
                                     density_seed=field_meta.get('density_seed', seed),
                                     helical_seed=seed,
                                     fixed_density=field.get('density_seed') is not None),
                     diagnostics=diag)
        atomic_json(run/'completed'/f'seed_{seed:04d}.json', {'seed':seed, 'completed':True})


def validate_mesh(run):
    from .initial_conditions import displace_mesh, cic_density
    field = load_config(run)['field']
    arrays, meta = cached_mesh_fields(run, 42, field, label='validation_mesh')
    assert meta['za_density_max_error'] < 1e-12
    assert meta['za_curl_max'] < 1e-12
    for name,sign in [('plus',1),('minus',-1)]:
        assert abs(meta[name]['normalized_helicity']-sign) < 1e-12
        assert meta[name]['longitudinal_fraction'] < 1e-28
    L = field['L']
    A = load_config(run)['scan']['amplitudes'][0]
    cat, diag = displace_mesh(arrays, A=A, eps=1., L=L)
    report = dict(model='mesh_za_helical', field=meta, displacement=diag, T1={})
    reference = None
    for name,c,sign in [('reference',cat,1), ('inversion',(-cat)%L,-1),
                         ('rotation90',np.c_[(-cat[:,1])%L,cat[:,0],cat[:,2]],1),
                         ('translation',(cat+L*np.array([.3712,.118,.6543]))%L,1)]:
        d = parity_stat(c, L=L, executable=run/'code/bin/DIVE')
        atomic_npz(run/'validation'/f'T1_{name}.npz', catalogue=c, tetrahedra=d)
        if reference is None:
            reference = d
        error = float(abs(d[:,1].sum()-sign*reference[:,1].sum()))
        volume = float(abs(d[:,0]).sum()/(6*L**3))
        passed = bool(len(d)==len(reference) and error<1e-7 and abs(volume-1)<1e-10)
        report['T1'][name] = dict(n_tetrahedra=len(d), sum_chi_hat=float(d[:,1].sum()),
                                  absolute_error=error, volume_fraction=volume, passed=passed)
        if not passed:
            atomic_json(run/'validation/report.json', report)
            raise AssertionError(f'Mesh IC geometry failed: {name}')
    rho = cic_density(cat, L=L, N=field['N'])
    report['cic_mean_delta'] = float(rho.mean())
    assert abs(rho.mean()) < 1e-12
    report['passed'] = True
    atomic_json(run/'validation/report.json', report)
    print('Mesh IC spectral, geometry and CIC validation passed', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['validate', 'seed-block'])
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--task-index', type=int, default=0)
    a = p.parse_args()
    run = a.run.resolve()
    check_manifest(run)
    if a.action == 'validate':
        validate(run)
    else:
        report = json.loads((run/'validation/report.json').read_text())
        if not report.get('passed'):
            raise RuntimeError('Validation has not passed')
        seed_block(run, a.task_index)

if __name__ == '__main__':
    main()
