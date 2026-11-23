import numpy as np
import pytest

from parity_study.initial_conditions import (
    mesh_fields, displace_mesh, cic_density, spectrum_shape)
from parity_study.catalogue import evaluate_field


def small_field(**overrides):
    return dict(model='mesh_za_helical', amplitude_convention='rms', L=100.,
                N=8, particle_grid=8, Ngal=512, sigma_delta=.2,
                helical_smoothing=4., spectrum=dict(kind='power_law', index=0., smoothing=4.),
                **overrides)


def test_mesh_spectral_identities_and_sampling():
    arrays, meta = mesh_fields(**small_field(), seed=3)
    assert meta['za_density_max_error'] < 1e-14
    assert meta['za_curl_max'] < 1e-14
    assert arrays['delta_mesh'].mean() == pytest.approx(0, abs=1e-15)
    assert np.mean(arrays['delta_mesh']**2) == pytest.approx(np.sum(abs(arrays['delta_k'])**2))
    for name, sign in [('plus',1),('minus',-1)]:
        assert meta[name]['normalized_helicity'] == pytest.approx(sign, abs=1e-13)
        assert meta[name]['longitudinal_fraction'] < 1e-28
        coeff = arrays[name+'_k']
        inverse = np.fft.ifftn(coeff, axes=(0,1,2), norm='forward')
        assert abs(inverse.imag).max() < 1e-14
        direct = evaluate_field(arrays['q'][:13], coeff, L=100., method='direct')
        np.testing.assert_allclose(arrays[name][:13], direct, atol=1e-8)
        nodes = np.stack(np.meshgrid(*([np.arange(8)*100/8]*3), indexing='ij'), axis=-1).reshape(-1,3)
        at_nodes = evaluate_field(nodes, coeff, L=100., method='nufft', tolerance=1e-12)
        np.testing.assert_allclose(at_nodes, arrays[name+'_mesh'].reshape(-1,3), atol=1e-10)


def test_additive_lagrangian_particle_map_and_zero_helicity_limit():
    arrays, _ = mesh_fields(**small_field(), seed=7)
    zero, diag = displace_mesh(arrays, A=0., eps=1., L=100.)
    np.testing.assert_allclose(zero, (arrays['q']+arrays['za']) % 100)
    assert diag['helical_rms'] == 0
    assert diag['za_rms'] > 0
    for eps in (-1., -.5, 0., .5, 1.):
        cat, diag = displace_mesh(arrays, A=3., eps=eps, L=100.)
        hel = 3*(np.sqrt((1+eps)/2)*arrays['plus'] + np.sqrt((1-eps)/2)*arrays['minus'])
        np.testing.assert_allclose(cat, (arrays['q']+arrays['za']+hel) % 100)
        assert diag['expected_displacement_rms'] == 3.
        assert diag['total_displacement_rms'] == pytest.approx(
            np.sqrt(np.mean(np.sum((arrays['za']+hel)**2, axis=1))))


@pytest.mark.parametrize('N', [8, 12])
def test_ensemble_power_normalization(N):
    powers = []
    config = small_field()
    config.update(N=N, particle_grid=N, Ngal=N**3)
    for seed in range(24):
        _, m = mesh_fields(**config, seed=seed)
        powers.append([m['sigma_delta_measured']**2/.2**2,
                       m['plus']['spectral_rms']**2, m['minus']['spectral_rms']**2])
    # Low-k-heavy helical spectra have large realization scatter; average the
    # ensemble, rather than forcing every saved field to the requested RMS.
    np.testing.assert_allclose(np.mean(powers, axis=0), 1., atol=.2, rtol=0)
    assert np.std(powers, axis=0).min() > 0


def test_separate_random_streams_and_particle_loading():
    config = small_field()
    a, _ = mesh_fields(**config, seed=5)
    config['sigma_delta'] = .4
    b, _ = mesh_fields(**config, seed=5)
    np.testing.assert_allclose(b['delta_k'], 2*a['delta_k'])
    np.testing.assert_array_equal(a['plus_k'], b['plus_k'])
    np.testing.assert_array_equal(a['minus_k'], b['minus_k'])
    config.update(particle_grid=10, Ngal=1000)
    c, _ = mesh_fields(**config, seed=5)
    np.testing.assert_array_equal(b['delta_k'], c['delta_k'])
    np.testing.assert_array_equal(b['plus_k'], c['plus_k'])
    assert len(c['q']) == 1000
    assert not np.array_equal(a['plus_k'], a['minus_k'])


def test_periodic_cic_mass_and_uniform_load():
    arrays, _ = mesh_fields(**small_field(), seed=0)
    np.testing.assert_allclose(cic_density(arrays['q'], L=100., N=8), 0, atol=1e-14)
    q = np.random.default_rng(9).uniform(-200, 300, (200,3))
    d = cic_density(q, L=100., N=8)
    assert d.mean() == pytest.approx(0., abs=1e-14)
    assert d.min() >= -1.
    np.testing.assert_allclose(d, cic_density(q % 100, L=100., N=8), atol=2e-14)


def test_fixed_density_seed_changes_only_helical_displacement():
    config = small_field()
    original, _ = mesh_fields(**config, seed=3)
    fixed, meta = mesh_fields(**config, seed=3, density_seed=3)
    other, _ = mesh_fields(**config, seed=4, density_seed=3)
    original_other, _ = mesh_fields(**config, seed=4)
    # Setting density_seed equal to the old seed preserves that realization.
    for key in ('delta_k', 'za_k', 'plus_k', 'minus_k', 'delta_mesh'):
        np.testing.assert_array_equal(original[key], fixed[key])
    for key in ('q', 'delta_k', 'za_k', 'delta_mesh'):
        np.testing.assert_array_equal(fixed[key], other[key])
    np.testing.assert_allclose(fixed['za'], other['za'], rtol=0, atol=1e-12)
    for key in ('plus_k', 'minus_k'):
        assert not np.array_equal(fixed[key], other[key])
        np.testing.assert_array_equal(other[key], original_other[key])
    assert meta['density_seed'] == meta['helical_seed'] == 3
    assert meta['fixed_density']
    # The undisplaced ZA catalogue remains fixed; finite helical maps change.
    np.testing.assert_allclose(
        displace_mesh(fixed, A=0, eps=0, L=100)[0],
        displace_mesh(other, A=0, eps=0, L=100)[0], rtol=0, atol=1e-12)
    assert not np.allclose(
        displace_mesh(fixed, A=2, eps=1, L=100)[0],
        displace_mesh(other, A=2, eps=1, L=100)[0])


@pytest.mark.parametrize('density_seed', [-1, 1.5, True])
def test_invalid_fixed_density_seed(density_seed):
    with pytest.raises(ValueError, match='density_seed'):
        mesh_fields(**small_field(), density_seed=density_seed)


def test_table_spectrum_and_invalid_particle_count():
    k = np.array([0., 1., 2., 4.])
    active = k>0
    p = spectrum_shape(k, dict(kind='table', k=[1.,4.], p=[1.,1/16]), active)
    np.testing.assert_allclose(p[active], k[active]**-2)
    with pytest.raises(ValueError, match='cover every active mode'):
        spectrum_shape(k, dict(kind='table', k=[1.,2.], p=[1.,.25]), active)
    config = small_field()
    config['Ngal'] = 500
    with pytest.raises(ValueError, match='Ngal == particle_grid'):
        mesh_fields(**config)


def test_mesh_cache_rejects_changed_density_parameters(tmp_path):
    from parity_study.run import cached_mesh_fields
    config = small_field()
    cached_mesh_fields(tmp_path, 1, config)
    config['sigma_delta'] = .4
    with pytest.raises(ValueError, match='Field cache parameters differ'):
        cached_mesh_fields(tmp_path, 1, config)


def test_compact_fields_and_cache_preserve_catalogues(tmp_path):
    from parity_study.run import cached_mesh_fields
    config = small_field()
    full, full_meta = mesh_fields(**config, seed=9)
    compact, compact_meta = mesh_fields(**config, seed=9, full_output=False)
    assert full_meta == compact_meta
    assert set(compact) == {'q', 'za', 'plus', 'minus', 'delta_mesh'}
    assert compact['delta_mesh'].flags.owndata
    for key in compact:
        np.testing.assert_array_equal(compact[key], full[key])
    for eps in (-1., 0., 1.):
        expected = displace_mesh(full, A=2., eps=eps, L=100.)[0]
        np.testing.assert_array_equal(displace_mesh(compact, A=2., eps=eps, L=100.)[0], expected)
    saved, _ = cached_mesh_fields(tmp_path, 9, config)
    reloaded, _ = cached_mesh_fields(tmp_path, 9, config)
    for key in compact:
        # The batch cache can use more FFT threads than the direct test calls.
        np.testing.assert_allclose(saved[key], compact[key], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(reloaded[key], saved[key])
