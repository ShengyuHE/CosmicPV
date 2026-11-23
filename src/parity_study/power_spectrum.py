"""Periodic, equal-weight point-catalogue power from NUFFT Fourier sums."""
import finufft
import numpy as np


def point_modes(positions, *, L, n_modes=128, tolerance=1e-11, nthreads=1):
    """Return delta_k = sum_j exp(-i k.x_j)/N, with the DC mode removed.

    FFT ordering; no particle-to-mesh assignment or assignment-window correction.
    n_modes specifies the requested Fourier cube, not the IC mesh resolution.
    """
    x = np.asarray(positions, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Require finite, nonempty N by 3 positions')
    if not np.isfinite(L) or L <= 0 or n_modes < 4 or int(n_modes) != n_modes:
        raise ValueError('Require L > 0 and integer n_modes >= 4')
    angles = ((x/L + .5) % 1 - .5) * (2*np.pi)
    coeff = finufft.nufft3d1(
        *[np.ascontiguousarray(angles[:, i]) for i in range(3)],
        np.full(len(x), 1/len(x), dtype=np.complex128),
        (n_modes,)*3, eps=tolerance, isign=-1, modeord=1, nthreads=nthreads)
    coeff[0, 0, 0] = 0
    return coeff


def wavenumbers(n_modes, L):
    axis = 2*np.pi*np.fft.fftfreq(n_modes, d=L/n_modes)
    return np.sqrt(axis[:, None, None]**2 + axis[None, :, None]**2 + axis[None, None, :]**2)


def shell_average(values, k, edges):
    """Average all explicit modes, counting both conjugate partners.

    Empty shells are NaN. Mode counts are not independent-realization counts.
    """
    values, k, edges = np.asarray(values), np.asarray(k), np.asarray(edges)
    if values.shape != k.shape or edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError('Incompatible arrays or nonincreasing shell edges')
    index = np.searchsorted(edges, k.ravel(), side='right') - 1
    keep = (k.ravel() > 0) & (index >= 0) & (index < len(edges)-1)
    index = index[keep]
    counts = np.bincount(index, minlength=len(edges)-1)
    def average(a):
        sums = np.bincount(index, weights=a.ravel()[keep], minlength=len(edges)-1)
        return np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return average(values), average(k), counts


def catalogue_power(positions, *, L, n_modes=128, edges=None, tolerance=1e-11, nthreads=1):
    coeff = point_modes(positions, L=L, n_modes=n_modes, tolerance=tolerance, nthreads=nthreads)
    k = wavenumbers(n_modes, L)
    if edges is None:
        edges = (2*np.pi/L)*np.arange(.5, n_modes/2, 1.)
    edges = np.asarray(edges, float)
    if edges[0] <= 0 or edges[-1] >= np.pi*n_modes/L:
        raise ValueError('Use positive edges inside the Fourier cube, excluding Nyquist planes')
    power, centers, counts = shell_average(L**3*abs(coeff)**2, k, edges)
    return dict(k=centers, edges=edges, nmodes=counts, power_raw=power,
                poisson_reference=L**3/len(positions)), coeff


def density_mesh_modes(delta, *, n_modes=None):
    """Fourier-series coefficients of a periodic density-contrast mesh.

    Mesh samples are at x_j=j*L/N, as in initial_conditions.to_mesh. Return
    a requested Fourier cube by selecting signed modes, without downsampling
    the real-space mesh. Parseval diagnostics use the full original mesh.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or len(set(delta.shape)) != 1 or not np.isfinite(delta).all():
        raise ValueError('Require a finite cubic density-contrast mesh')
    N = delta.shape[0]
    n = N if n_modes is None else n_modes
    if int(n) != n or not 4 <= n <= N:
        raise ValueError('Requested Fourier cube must have 4 <= n_modes <= N')
    coeff = np.fft.fftn(delta, norm='forward')
    mean = float(coeff[0,0,0].real)
    coeff[0,0,0] = 0
    variance = float(np.var(delta))
    spectral_variance = float(np.sum(abs(coeff)**2))
    index = np.rint(np.fft.fftfreq(n)*n).astype(int) % N
    cropped = coeff[np.ix_(index,index,index)]
    return cropped, dict(mean=mean, variance=variance, spectral_variance=spectral_variance,
                         parseval_absolute_error=abs(variance-spectral_variance))
