"""Small, cached displacement examples for intro_disp_PV.ipynb; no triangulation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from parity_study import catalogue
from parity_study.catalogue import spectral_grid, evaluate_field, field_diagnostics, unit_fields
from parity_study.storage import atomic_json, atomic_npz, digest
from parity_study.paths import OUTPUT_ROOT, expose_output


def build_examples(project, *, force=False):
    project = Path(project).resolve()
    dest = OUTPUT_ROOT/'intro_disp_PV'
    dest.mkdir(parents=True, exist_ok=True)
    expose_output(project, dest)
    fingerprints = {p.name: digest(p) for p in (Path(__file__), Path(catalogue.__file__))}
    report_path = dest/'examples.json'
    if not force and report_path.exists() and (dest/'examples.npz').exists():
        previous = json.loads(report_path.read_text())
        if previous.get('source_sha256') == fingerprints and previous.get('arrays_sha256') == digest(dest/'examples.npz'):
            print(f'Reusing saved examples: {dest}', flush=True)
            return
    params = dict(L=100., N=16, Ngal=2000, seed=13, kmin=2*np.pi/100,
                  kmax=4*2*np.pi/100, exclude_nyquist=True, amplitude_convention='rms')
    L, N = params['L'], params['N']
    q, plus, minus, meta = unit_fields(**params)
    # Reconstruct the exact small realization's coefficients to evaluate slices
    # and curl. Match unit_fields' RNG order, and check against its sampled field.
    rng = np.random.RandomState(params['seed'])
    np.testing.assert_array_equal(rng.uniform(0, L, q.shape), q)
    kvec, e1, e2, active, _ = spectral_grid(**{k:params[k] for k in
        ('L', 'N', 'Ngal', 'kmin', 'kmax', 'exclude_nyquist')})
    weight = active.astype(float)/active.sum()
    coeff = []
    for sign in (1, -1):
        z = rng.randn(N,N,N) + 1j*rng.randn(N,N,N)
        coeff.append((z*np.sqrt(weight))[...,None]*(e1+sign*1j*e2)/np.sqrt(2))
    errors = []
    for c, expected in zip(coeff, (plus, minus)):
        actual = evaluate_field(q[:32], c, L=L, method='direct')
        errors.append(float(abs(actual-expected[:32]).max()))
        np.testing.assert_allclose(actual, expected[:32], atol=1e-7, rtol=0)
    axis = np.linspace(0, L, 25, endpoint=False) + L/50
    x, y = np.meshgrid(axis, axis, indexing='xy')
    plane = np.c_[x.ravel(), y.ravel(), np.full(x.size,L/2)]
    epsilons = [-1., 0., 1.]
    slices, diagnostics = [], []
    for eps in epsilons:
        c = np.sqrt((1+eps)/2)*coeff[0]+np.sqrt((1-eps)/2)*coeff[1]
        slices.append(evaluate_field(plane,c,L=L))
        diagnostics.append(dict(eps=eps, **field_diagnostics(c,kvec)))
    normalization = []
    # Change both grid size and active mode count, retaining the production
    # default band prescription. These are field-variance checks, not PV tests.
    for grid in (8,16,24):
        for seed in range(24):
            _, p, m, info = unit_fields(L=L,N=grid,Ngal=1024,seed=seed,amplitude_convention='rms')
            for eps in epsilons:
                u = np.sqrt((1+eps)/2)*p + np.sqrt((1-eps)/2)*m
                normalization.append(dict(N=grid,seed=seed,eps=eps,n_modes=info['n_modes'],
                    mean_square=float(np.mean(np.sum(u*u,axis=1)))))
    atomic_npz(dest/'examples.npz', q=q, plus=plus, minus=minus,
               plane=plane, slices=np.array(slices), epsilons=np.array(epsilons))
    atomic_json(report_path, dict(
        created_at=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(timespec='seconds'),
        purpose='Small normalized field illustrations; no Delaunay measurements or PV ensemble.',
        parameters=params, field=meta, diagnostics=diagnostics,
        nufft_direct_max_errors=errors, normalization=normalization,
        normalization_parameters=dict(L=L,Ngal=1024,seeds=list(range(24)),grids=[8,16,24],
                                      epsilons=epsilons,amplitude_convention='rms',exclude_nyquist=True),
        source_sha256=fingerprints, arrays_sha256=digest(dest/'examples.npz')))
    print(f'Saved normalized field examples: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--force',action='store_true')
    args = parser.parse_args()
    build_examples(args.project,force=args.force)
