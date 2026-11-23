"""Gaussian mesh -> Zel'dovich + helical displacement -> lattice particles.

The construction follows Eqs. 1--8 of arXiv:2406.06080v2, with a selectable
input spectrum and our RMS-length normalization for the helical amplitude.
Fourier coefficients use the series convention f(x)=sum_k f_k exp(i k.x).
There is no gravity integration or helical contribution to initial velocities.
"""
import numpy as np

from .catalogue import spectral_grid, evaluate_field, field_diagnostics, displace

MODEL = 'mesh_za_helical'


def validate_field(field):
    if field.get('model') != MODEL or field.get('amplitude_convention') != 'rms':
        raise ValueError("Mesh ICs require model='mesh_za_helical', amplitude_convention='rms'")
    N, npart = field['N'], field['particle_grid']
    if int(N) != N or int(npart) != npart or min(N, npart) < 4:
        raise ValueError('N and particle_grid must be integers >=4')
    if field['Ngal'] != npart**3:
        raise ValueError('A regular particle load requires Ngal == particle_grid**3')
    if not field.get('exclude_nyquist', True):
        raise ValueError('Mesh ICs require paired modes: exclude_nyquist=True')
    density_seed = field.get('density_seed')
    if density_seed is not None and (isinstance(density_seed, (bool, np.bool_)) or
            not isinstance(density_seed, (int, np.integer)) or density_seed < 0):
        raise ValueError('density_seed must be a nonnegative integer or None')
    for key in ('L', 'sigma_delta', 'helical_smoothing'):
        value = field[key]
        if not np.isfinite(value) or value < 0 or (key == 'L' and value == 0):
            raise ValueError(f'Invalid {key}')
    kmax = field.get('kmax', np.pi*min(N, npart)/field['L'])
    if kmax is not None and kmax > np.pi*min(N, npart)/field['L']:
        raise ValueError('kmax must resolve both the field mesh and particle load')


def spectrum_shape(k, spec, active):
    """Positive shape; an overall input amplitude is replaced by sigma_delta.

    Tables are embedded as k/p arrays in the frozen JSON configuration, so
    changing or moving an external input file cannot change a submitted run.
    """
    shape = np.zeros_like(k)
    if spec['kind'] == 'power_law':
        index = float(spec.get('index', 0.))
        smoothing = float(spec.get('smoothing', 0.))
        if not np.isfinite([index, smoothing]).all() or smoothing < 0:
            raise ValueError('Require finite spectral index and smoothing >=0')
        shape[active] = (k[active]/k[active].min())**index * np.exp(-(k[active]*smoothing)**2)
    elif spec['kind'] == 'table':
        kt, pt = np.asarray(spec['k'], float), np.asarray(spec['p'], float)
        if (kt.ndim != 1 or kt.shape != pt.shape or len(kt) < 2 or
                not np.isfinite(kt).all() or not np.isfinite(pt).all() or
                np.any(kt <= 0) or np.any(pt <= 0) or np.any(np.diff(kt) <= 0)):
            raise ValueError('Power table needs increasing positive k and positive finite P(k)')
        if k[active].min() < kt[0] or k[active].max() > kt[-1]:
            raise ValueError('Power table must cover every active mode; no extrapolation')
        shape[active] = np.exp(np.interp(np.log(k[active]), np.log(kt), np.log(pt)))
    else:
        raise ValueError('spectrum.kind must be power_law or table')
    if not np.isfinite(shape).all() or shape.sum() <= 0:
        raise ValueError('Spectrum has no finite positive power on active modes')
    return shape


def _real_coefficients(coeff):
    neg = (-np.arange(len(coeff))) % len(coeff)
    return (coeff + coeff[np.ix_(neg, neg, neg)].conj())/2


def to_mesh(coeff):
    # norm='forward' removes the inverse transform's 1/N^3 factor.
    return np.fft.ifftn(coeff, axes=(0, 1, 2), norm='forward').real.copy()


def mesh_fields(*, model=MODEL, amplitude_convention='rms', L=100., N=16,
                particle_grid=16, Ngal=4096, sigma_delta=.2, helical_smoothing=6.25,
                spectrum=None, kmin=None, kmax=None, exclude_nyquist=True,
                seed=0, density_seed=None, nthreads=1, tolerance=1e-10, method='nufft',
                full_output=True):
    """Construct fields first, then evaluate them on a cell-centred lattice.

    Scalar and two helical fields have independent random streams. Changing
    amplitudes or particle_grid preserves the Fourier draws at fixed N/seed.
    density_seed=None preserves the original shared seed convention. Set a
    density_seed to fix density and ZA while seed varies the two helical draws.
    Returned plus/minus fields have ensemble RMS 1, not per-realization RMS 1.
    full_output=False retains only the density mesh and sampled particle fields;
    spectral checks still run, but redundant vector meshes are not materialized.
    """
    validate_field(dict(model=model, amplitude_convention=amplitude_convention,
                        L=L, N=N, particle_grid=particle_grid, Ngal=Ngal,
                        sigma_delta=sigma_delta, helical_smoothing=helical_smoothing,
                        kmax=kmax, exclude_nyquist=exclude_nyquist, density_seed=density_seed))
    kmax = np.pi*min(N, particle_grid)/L if kmax is None else kmax
    kvec, e1, e2, active, grid = spectral_grid(
        L=L, N=N, Ngal=Ngal, kmin=2*np.pi/L if kmin is None else kmin,
        kmax=kmax, exclude_nyquist=True)
    k2 = np.sum(kvec**2, axis=-1)
    k = np.sqrt(k2)
    shape = spectrum_shape(k, spectrum or dict(kind='power_law', index=0., smoothing=L/N), active)
    density_weights = shape/shape.sum()
    # Eq. 7 defines power for k^2 Phi_R. Since Psi_R = Phi_R e_R,
    # its displacement-power shape is P_delta(k) exp(-k^2 R_h^2)/k^4.
    hel_shape = np.zeros_like(shape)
    hel_shape[active] = shape[active]*np.exp(-k2[active]*helical_smoothing**2)/k2[active]**2
    if hel_shape.sum() <= 0:
        raise ValueError('Helical smoothing suppresses all supported modes')
    hel_weights = hel_shape/hel_shape.sum()
    streams = np.random.SeedSequence(seed).spawn(3)
    fixed_density = density_seed is not None
    scalar_seed = int(density_seed) if fixed_density else int(seed)
    if fixed_density:
        streams[0] = np.random.SeedSequence(scalar_seed).spawn(3)[0]
    rngs = [np.random.default_rng(s) for s in streams]
    white = rngs[0].normal(size=(N, N, N))
    delta_k = np.fft.fftn(white)/np.sqrt(N**3)*np.sqrt(density_weights)*sigma_delta
    invk2 = np.zeros_like(k2)
    invk2[active] = 1/k2[active]
    za_k = 1j*kvec*invk2[..., None]*delta_k[..., None]
    coeffs = []
    for rng, sign in zip(rngs[1:], (1, -1)):
        z = rng.normal(size=(N, N, N)) + 1j*rng.normal(size=(N, N, N))
        c = np.sqrt(hel_weights)[..., None]*z[..., None]*(e1 + sign*1j*e2)/np.sqrt(2)
        coeffs.append(_real_coefficients(c))
    plus_k, minus_k = coeffs
    # Particle placement happens only after the mesh fields are constructed.
    axis = (np.arange(particle_grid) + .5)*L/particle_grid
    q = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    values = [evaluate_field(q, c, L=L, method=method, nthreads=nthreads, tolerance=tolerance)
              for c in (za_k, plus_k, minus_k)]
    arrays = dict(q=q, za=values[0], plus=values[1], minus=values[2],
                  delta_mesh=to_mesh(delta_k))
    if full_output:
        arrays.update(delta_k=delta_k, za_k=za_k, plus_k=plus_k, minus_k=minus_k,
                      za_mesh=to_mesh(za_k), plus_mesh=to_mesh(plus_k),
                      minus_mesh=to_mesh(minus_k))
    recovered_delta = -1j*np.sum(kvec*za_k, axis=-1)
    meta = dict(model=MODEL, particle_load='cell_centred_lattice',
                amplitude_convention='rms', sigma_delta_expected=float(sigma_delta),
                sigma_delta_measured=float(np.sqrt(np.sum(abs(delta_k)**2))),
                expected_za_rms=float(sigma_delta*np.sqrt(np.sum(density_weights*invk2))),
                expected_helical_unit_rms=1., helical_spectrum='P_delta exp(-k^2 R_h^2) / k^4',
                za_density_max_error=float(np.max(abs(recovered_delta-delta_k))),
                za_curl_max=float(np.max(abs(1j*np.cross(kvec, za_k)))),
                plus=field_diagnostics(plus_k, kvec), minus=field_diagnostics(minus_k, kvec),
                density_spectrum=spectrum or dict(kind='power_law', index=0., smoothing=L/N),
                density_seed=scalar_seed, helical_seed=int(seed), fixed_density=fixed_density,
                rng='scalar: SeedSequence(density_seed).spawn(3)[0]; '
                    'helical: SeedSequence(seed).spawn(3)[1:3]', **grid)
    return arrays, meta


def displace_mesh(arrays, *, A, eps, L, response=1.):
    """x=q+Psi_ZA(q)+Psi_hel(q); both terms are evaluated at the same q.

    This is additive Lagrangian IC construction, not sequential evaluation of
    the helical field at already-Zel'dovich-displaced positions.
    """
    _, diag = displace(arrays['q'], arrays['plus'], arrays['minus'],
                       A=A, eps=eps, L=L, response=response, amplitude_convention='rms')
    helical = response*A*(np.sqrt((1+eps)/2)*arrays['plus'] + np.sqrt((1-eps)/2)*arrays['minus'])
    total = arrays['za'] + helical
    rms = float(np.sqrt(np.mean(np.sum(total**2, axis=1))))
    diag.update(model=MODEL, helical_rms=diag['displacement_rms'],
                helical_rms_over_spacing=diag['rms_over_spacing'],
                za_rms=float(np.sqrt(np.mean(np.sum(arrays['za']**2, axis=1)))),
                total_displacement_rms=rms,
                total_rms_over_spacing=rms/(L/len(total)**(1/3)))
    return (arrays['q']+total) % L, diag


def build_ic_catalogue(*, A=1., eps=1., response=1., **kwargs):
    arrays, _ = mesh_fields(**kwargs)
    return displace_mesh(arrays, A=A, eps=eps, L=kwargs.get('L', 100.), response=response)[0]


def cic_density(positions, *, L, N):
    """Deposit unit-mass particles on a periodic mesh; return delta=rho/rhobar-1."""
    u = np.asarray(positions)*N/L
    base = np.floor(u).astype(int)
    frac = u-base
    rho = np.zeros((N, N, N))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                offset = np.array([a, b, c])
                w = np.prod(np.where(offset, frac, 1-frac), axis=1)
                idx = (base+offset) % N
                np.add.at(rho, tuple(idx.T), w)
    return rho/(len(positions)/N**3)-1
