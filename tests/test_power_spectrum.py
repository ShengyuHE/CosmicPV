import numpy as np
import pytest

from parity_study.power_spectrum import point_modes, catalogue_power, shell_average, wavenumbers, density_mesh_modes


def test_matches_direct_sum_including_negative_modes():
    L = 13.
    x = np.random.default_rng(12).uniform(-L, 2*L, (37, 3))
    c = point_modes(x, L=L, n_modes=10, tolerance=1e-12)
    for mode in [(1, 0, 0), (-2, 1, 3), (4, -3, -1), (-5, -5, 0)]:
        direct = np.exp(-2j*np.pi/L*(x@np.array(mode))).mean()
        np.testing.assert_allclose(c[tuple(v % 10 for v in mode)], direct, atol=1e-11)
    assert c[0, 0, 0] == 0


def test_reflection_translation_and_volume_units():
    L = 17.
    x = np.random.default_rng(13).uniform(0, L, (100, 3))
    a, _ = catalogue_power(x, L=L, n_modes=16)
    for y in [(-x) % L, (x + [2.3, -5.2, 1.8]) % L]:
        b, _ = catalogue_power(y, L=L, n_modes=16)
        np.testing.assert_allclose(a['power_raw'], b['power_raw'], rtol=1e-9)
    b, _ = catalogue_power(2*x, L=2*L, n_modes=16)
    np.testing.assert_allclose(b['power_raw'], 8*a['power_raw'], rtol=1e-9)
    np.testing.assert_allclose(b['k'], a['k']/2)


def test_lattice_and_known_single_mode_density():
    # Equal-weight points with a tiny longitudinal sinusoidal displacement.
    # Linear density amplitude is a, so each of the +/- x modes has P=V*a^2/4.
    L, n, a = 10., 12, 1e-4
    q = np.stack(np.meshgrid(*([(np.arange(n)+.5)*L/n]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    uniform, _ = catalogue_power(q, L=L, n_modes=12)
    assert uniform['power_raw'].max() < 1e-16
    x = q.copy()
    x[:, 0] -= a*L/(2*np.pi)*np.sin(2*np.pi*q[:, 0]/L)
    c = point_modes(x, L=L, n_modes=12, tolerance=1e-13)
    assert L**3*abs(c[1, 0, 0])**2 == pytest.approx(L**3*a*a/4, rel=1e-6)


def test_shell_accounting_and_poisson_normalization():
    k = wavenumbers(12, 100.)
    p, centers, counts = shell_average(np.ones_like(k), k, [0.5, 1.5, 2.5]*np.array(2*np.pi/100))
    assert counts[0] == 18  # six axial + twelve face-diagonal modes
    np.testing.assert_array_equal(p, 1)
    # Ensemble of uniform independent points has <P>=V/N.
    powers = []
    for seed in range(12):
        x = np.random.default_rng(seed).uniform(0, 100., (200, 3))
        result, _ = catalogue_power(x, L=100., n_modes=16)
        powers.append(np.average(result['power_raw'], weights=result['nmodes']))
    assert np.mean(powers) == pytest.approx(100.**3/200, rel=.06)


def test_density_mesh_normalization_and_signed_mode_crop():
    N = 24
    x,y,z = np.meshgrid(*([2*np.pi*np.arange(N)/N]*3), indexing='ij')
    delta = .2*np.cos(2*x-y+3*z) + .3*np.sin(7*x) + .04
    coeff, audit = density_mesh_modes(delta, n_modes=12)
    assert coeff[2,-1,3] == pytest.approx(.1, abs=1e-14)
    assert coeff[-2,1,-3] == pytest.approx(.1, abs=1e-14)
    assert coeff[0,0,0] == 0
    assert audit['mean'] == pytest.approx(.04)
    assert audit['variance'] == pytest.approx((.2**2+.3**2)/2)
    assert audit['parseval_absolute_error'] < 1e-14
    # The high-frequency component is omitted, rather than aliased into -5.
    assert abs(coeff[-5,0,0]) < 1e-14
    assert np.sum(abs(coeff)**2) == pytest.approx(.2**2/2)


def test_density_mesh_and_point_catalogue_share_linear_power_units():
    L, n, amplitude = 10., 24, 1e-4
    axis = np.arange(n)*L/n
    delta = amplitude*np.cos(2*np.pi*axis[:,None,None]/L)*np.ones((1,n,n))
    mesh, _ = density_mesh_modes(delta, n_modes=12)
    q = np.stack(np.meshgrid(*([(np.arange(n)+.5)*L/n]*3), indexing='ij'), axis=-1).reshape(-1,3)
    q[:,0] -= amplitude*L/(2*np.pi)*np.sin(2*np.pi*q[:,0]/L)
    points = point_modes(q,L=L,n_modes=12,tolerance=1e-13)
    assert L**3*abs(points[1,0,0])**2 == pytest.approx(L**3*abs(mesh[1,0,0])**2,rel=1e-6)
