"""Helical displacement fields with an explicit amplitude convention.

The default 'rms' convention normalizes the power weights over active discrete
modes to sum to one and uses P_disp(k) = A**2 * P_normalized(k). Thus
E[|psi(x)|**2] = A**2 before the response factor. This is an ensemble RMS,
not a constraint imposed on each realization. The Fourier sum still has no
FFT normalization and is projected to its real part, as in the notebook.

The explicit 'notebook' convention preserves P_disp(k) = A * band_mask(k),
so E[|psi(x)|**2] = A * N_active_modes. RandomState preserves the notebook's
MT19937 draw order without mutating numpy's global random state.
"""
import numpy as np
import finufft
from scipy.spatial.transform import Rotation


def check_amplitude_convention(convention):
    if convention not in ('rms', 'notebook'):
        raise ValueError("amplitude_convention must be 'rms' or 'notebook'")


def spectral_grid(*, L, N, Ngal, kmin=None, kmax=None, exclude_nyquist=True):
    if L <= 0 or N < 4 or Ngal < 4:
        raise ValueError("Require L>0, N>=4, Ngal>=4")
    k1d = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kvec = np.stack(np.meshgrid(k1d, k1d, k1d, indexing="ij"), axis=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    khat = kvec / np.where(kmag > 0, kmag, 1)[..., None]
    e1 = np.cross([0., 0., 1.], khat)
    bad = np.linalg.norm(e1, axis=-1) < 1e-12
    e1[bad] = np.cross([1., 0., 0.], khat[bad])
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-30)
    e2 = np.cross(khat, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=-1, keepdims=True), 1e-30)
    ell = L / Ngal ** (1 / 3)
    kmin = 0.1 * 2 * np.pi / ell if kmin is None else float(kmin)
    kmax = np.pi * N / L if kmax is None else float(kmax)
    if not 0 <= kmin <= kmax <= np.pi * N / L:
        raise ValueError("Require 0 <= kmin <= kmax <= Nyquist")
    active = (kmag >= kmin) & (kmag <= kmax) & (kmag > 0)
    # For off-grid evaluation an even-grid Nyquist mode lacks its +k partner.
    if exclude_nyquist and N % 2 == 0:
        active[N // 2, :, :] = False
        active[:, N // 2, :] = False
        active[:, :, N // 2] = False
    if not active.any():
        raise ValueError("The selected band contains no Fourier modes")
    return kvec, e1, e2, active, dict(kmin=kmin, kmax=kmax,
                                    n_modes=int(active.sum()), mean_spacing=ell)


def evaluate_field(q, coefficients, *, L, method="nufft", nthreads=1, tolerance=1e-9):
    if method == "nufft":
        y = np.ascontiguousarray((2 * np.pi * q / L).T)
        f = np.ascontiguousarray(np.moveaxis(coefficients, -1, 0))
        return finufft.nufft3d2(*y, f, isign=1, eps=tolerance,
                               modeord=1, nthreads=nthreads).T.real.copy()
    if method != "direct":
        raise ValueError(f"Unknown field evaluation method: {method}")
    N = coefficients.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kvec = np.stack(np.meshgrid(k, k, k, indexing="ij"), axis=-1)
    active = np.any(coefficients != 0, axis=-1)
    out = np.zeros(q.shape, dtype=complex)
    for wave, amplitude in zip(kvec[active], coefficients[active]):
        out += np.exp(1j * (q @ wave))[:, None] * amplitude
    return out.real


def field_diagnostics(coefficients, kvec):
    N = coefficients.shape[0]
    neg = (-np.arange(N)) % N
    effective = (coefficients + coefficients[np.ix_(neg, neg, neg)].conj()) / 2
    power = np.sum(abs(effective)**2, axis=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    curl = 1j * np.cross(kvec, effective)
    norm = np.sum(kmag * power)
    helicity = np.sum((effective.conj() * curl).real) / norm if norm else 0.
    div_power = np.sum(abs(np.sum(kvec * effective, axis=-1))**2)
    denom = np.sum(kmag**2 * power)
    return dict(spectral_rms=float(np.sqrt(power.sum())),
                normalized_helicity=float(helicity),
                longitudinal_fraction=float(div_power / denom) if denom else 0.)


def unit_fields(*, L=1000., N=64, Ngal=10000, seed=42, kmin=None,
                kmax=None, exclude_nyquist=True, nthreads=1, tolerance=1e-9,
                method="nufft", amplitude_convention="rms"):
    """Return A=1 fields; their expected RMS is one in the 'rms' convention.

    For the top-hat spectrum each populated mode has weight 1/N_active_modes.
    Both helicity fields use the same weights; mixing them leaves total expected
    power unchanged. No realization-dependent normalization is applied.
    """
    check_amplitude_convention(amplitude_convention)
    rng = np.random.RandomState(seed)
    q = rng.uniform(0, L, (int(Ngal), 3))
    kvec, e1, e2, active, meta = spectral_grid(
        L=L, N=N, Ngal=Ngal, kmin=kmin, kmax=kmax,
        exclude_nyquist=exclude_nyquist)
    power = active.astype(float)
    if amplitude_convention == 'rms':
        power /= power.sum()
    meta.update(amplitude_convention=amplitude_convention,
                unit_power_sum=float(power.sum()),
                expected_unit_rms=float(np.sqrt(power.sum())))
    amplitude = np.sqrt(power)
    fields = []
    for name, sign in (("plus", 1), ("minus", -1)):
        z = rng.randn(N, N, N) + 1j * rng.randn(N, N, N)
        coeff = (z * amplitude)[..., None] * (e1 + sign * 1j * e2) / np.sqrt(2)
        meta[name] = field_diagnostics(coeff, kvec)
        fields.append(evaluate_field(q, coeff, L=L, method=method,
                                     nthreads=nthreads, tolerance=tolerance))
    return q, fields[0], fields[1], meta


def displace(q, plus, minus, *, A, eps, L, response=1., amplitude_convention="rms"):
    """Apply fields made with the same amplitude_convention.

    RMS refers to the unwrapped displacement, before modulo-L positions are
    formed. In the 'rms' convention the expected total RMS is abs(response)*A;
    the measured catalogue RMS fluctuates between realizations.
    """
    check_amplitude_convention(amplitude_convention)
    if not np.isfinite([A, eps, L, response]).all() or A < 0 or abs(eps) > 1 or L <= 0:
        raise ValueError("Require finite A>=0, |eps|<=1, L>0, response")
    # Scaling a normalized unit field by A implements P_disp = A**2 * P_k
    # while allowing the costly fields to be reused across amplitudes.
    scale = A if amplitude_convention == 'rms' else np.sqrt(A)
    disp = response * scale * (np.sqrt((1 + eps) / 2) * plus
                                    + np.sqrt((1 - eps) / 2) * minus)
    rms = float(np.sqrt(np.mean(np.sum(disp**2, axis=1))))
    return (q + disp) % L, dict(amplitude_convention=amplitude_convention,
                               expected_displacement_rms=float(abs(response)*A) if amplitude_convention == 'rms' else None,
                               displacement_rms=rms,
                               rms_over_spacing=rms / (L / len(q)**(1/3)))


def build_catalogue(*, A=1000., eps=1., response=1., amplitude_convention="rms", **kwargs):
    """Generate positions with expected displacement RMS abs(response)*A.

    Use amplitude_convention='notebook' to reproduce the old per-mode power A.
    """
    check_amplitude_convention(amplitude_convention)
    L = kwargs.get("L", 1000.)
    if A == 0:
        rng = np.random.RandomState(kwargs.get("seed", 42))
        return rng.uniform(0, L, (int(kwargs.get("Ngal", 10000)), 3))
    q, plus, minus, _ = unit_fields(**kwargs, amplitude_convention=amplitude_convention)
    return displace(q, plus, minus, A=A, eps=eps, L=L, response=response,
                    amplitude_convention=amplitude_convention)[0]


TEMPLATE = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1.5, 0.], [0., 0., 2.2]])


def chiral_catalogue(*, Ncl=2500, f_R=1., size=0.005, L=1., seed=0):
    if Ncl < 1 or not 0 <= f_R <= 1 or not 0 < size < L:
        raise ValueError("Invalid molecule configuration")
    rng = np.random.default_rng(seed)
    template = TEMPLATE - TEMPLATE.mean(0)
    template *= size / abs(template).max()
    hand = np.where(rng.random(Ncl) < f_R, 1., -1.)
    rotations = Rotation.random(Ncl, random_state=rng).as_matrix()
    shapes = hand[:, None, None] * template[None]
    points = rng.uniform(0, L, (Ncl, 1, 3)) + np.einsum("nij,nkj->nki", rotations, shapes)
    return points.reshape(-1, 3) % L, hand, template


def chi_hat_ref(tetrahedron):
    s = ((tetrahedron[:, None] - tetrahedron[None])**2).sum(-1).sum(1)
    if np.unique(s).size != 4:
        raise ValueError("Distance ordering is ambiguous for this tetrahedron")
    t = tetrahedron[np.argsort(s)]
    v = t[1:] - t[0]
    return np.linalg.det(v) / np.prod(np.linalg.norm(v, axis=1))
