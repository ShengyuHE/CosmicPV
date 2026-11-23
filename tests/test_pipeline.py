import itertools
import json
from pathlib import Path
import finufft
import numpy as np
import pytest
from parity_study.catalogue import (unit_fields, build_catalogue, displace,
                                    chiral_catalogue, chi_hat_ref, spectral_grid, TEMPLATE)
from parity_study.delaunay import parity_stat, PROJECT


def test_nufft_matches_direct():
    kw = dict(L=100., N=12, Ngal=300, seed=7, tolerance=1e-11)
    q, p, m, meta = unit_fields(**kw)
    q0, p0, m0, _ = unit_fields(**kw, method="direct")
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_allclose(p, p0, atol=1e-7)
    np.testing.assert_allclose(m, m0, atol=1e-7)
    assert meta['plus']['normalized_helicity'] == pytest.approx(1, abs=1e-12)
    assert meta['minus']['normalized_helicity'] == pytest.approx(-1, abs=1e-12)
    assert meta['plus']['longitudinal_fraction'] < 1e-28
    assert meta['minus']['longitudinal_fraction'] < 1e-28


@pytest.mark.parametrize('eps', [-1., 0., .4, 1.])
def test_matches_notebook_when_nyquist_retained(eps):
    notebook = Path(__file__).resolve().parents[1] / 'tp4a-pv-delaunay/parity_injection.ipynb'
    if not notebook.exists():
        # Source snapshots retain the reference notebook at this path too.
        raise FileNotFoundError(notebook)
    cells = json.loads(notebook.read_text())['cells']
    source = next(''.join(c['source']) for c in cells
                  if c['cell_type'] == 'code' and 'def build_catalogue(' in ''.join(c['source']))
    scope = {'np': np, 'finufft': finufft}
    exec(compile(source, str(notebook), 'exec'), scope)
    kw = dict(L=100., N=8, Ngal=64, seed=13, A=.002, eps=eps)
    expected = scope['build_catalogue'](**kw, NUFFT=False)
    actual = build_catalogue(**kw, exclude_nyquist=False, method='direct', amplitude_convention='notebook')
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    # Changing convention preserves the catalogue at the converted physical amplitude.
    _, _, _, _, meta = spectral_grid(L=kw['L'], N=kw['N'], Ngal=kw['Ngal'], exclude_nyquist=False)
    normalized_kw = dict(kw, A=np.sqrt(kw['A'] * meta['n_modes']))
    normalized = build_catalogue(**normalized_kw, exclude_nyquist=False, method='direct')
    np.testing.assert_allclose(normalized, expected, rtol=0, atol=1e-12)


def test_amplitude_reuse_and_rng_isolation():
    np.random.seed(123)
    state = np.random.get_state()
    q, p, m, _ = unit_fields(L=100, N=8, Ngal=64, seed=3)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    _, d1 = displace(q, p, m, A=.001, eps=.2, L=100)
    _, d2 = displace(q, p, m, A=.004, eps=.2, L=100)
    assert d2['displacement_rms'] == pytest.approx(4*d1['displacement_rms'])
    assert d2['expected_displacement_rms'] == pytest.approx(.004)


@pytest.mark.parametrize('N, cutoff_fraction', [(8, 1.), (16, .5), (16, 1.)])
def test_ensemble_rms_is_A_across_mode_counts_and_helicities(N, cutoff_fraction):
    # Verify physical-space variance, including the real-part projection and
    # both helicity weights. Fixed seeds give a reproducible ensemble check.
    A, response, L = 6.2, 1.7, 100.
    powers = {eps: [] for eps in (-1, 0, 1)}
    for seed in range(16):
        q, plus, minus, meta = unit_fields(L=L, N=N, Ngal=2048, seed=seed,
                                           kmax=cutoff_fraction*np.pi*N/L)
        assert meta['unit_power_sum'] == pytest.approx(1)
        assert meta['expected_unit_rms'] == pytest.approx(1)
        for eps in powers:
            _, diag = displace(q, plus, minus, A=A, eps=eps, L=L, response=response)
            assert diag['expected_displacement_rms'] == pytest.approx(response*A)
            powers[eps].append((diag['displacement_rms']/(response*A))**2)
    for values in powers.values():
        # Ensemble tolerance allows finite field/mode sampling; a missing
        # normalization or factor-of-two error is far outside this interval.
        assert np.mean(values) == pytest.approx(1, abs=.08)
        assert np.std(values) > 0  # Individual realizations are not forced to unit RMS.


def test_old_cache_cannot_be_used_as_rms_field(tmp_path):
    from parity_study.run import fields
    from parity_study.storage import atomic_npz
    field = dict(L=100., N=8, Ngal=32, exclude_nyquist=True)
    cache = tmp_path/'fields/main_0000.npz'
    atomic_npz(cache, q=np.zeros((32,3)), plus=np.zeros((32,3)), minus=np.zeros((32,3)),
               metadata_json=np.array(json.dumps({'field':field})))
    with pytest.raises(ValueError, match='cache parameters differ'):
        fields(tmp_path, 0, {**field, 'amplitude_convention':'rms'})


def test_unknown_amplitude_convention_rejected():
    with pytest.raises(ValueError, match='amplitude_convention'):
        build_catalogue(A=0, amplitude_convention='unknown')


def test_template_ordering():
    for perm in itertools.permutations(range(4)):
        assert chi_hat_ref(TEMPLATE[list(perm)]) == pytest.approx(1)
    assert chi_hat_ref(-TEMPLATE) == pytest.approx(-1)
    assert chi_hat_ref(TEMPLATE[:, [1, 0, 2]]) == pytest.approx(-1)
    with pytest.raises(ValueError, match='ambiguous'):
        chi_hat_ref(np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1.]]))


def test_explicit_box_validation():
    q = np.random.default_rng(1).uniform(0, 1000, (20, 3))
    with pytest.raises(TypeError):
        parity_stat(q)
    with pytest.raises(ValueError, match='box size'):
        parity_stat(q, L=1)


def test_periodic_geometry_symmetries_and_scaling():
    q = np.random.default_rng(1).uniform(0, 1, (300, 3))
    ref = parity_stat(q, L=1)
    assert np.sum(abs(ref[:, 0]))/6 == pytest.approx(1, rel=1e-12)
    for transformed, sign in [((-q)%1, -1), (q[:, [1,0,2]], -1),
                              (np.c_[(-q[:,1])%1,q[:,0],q[:,2]], 1),
                              ((q+[.3712,.118,.6543])%1, 1)]:
        data = parity_stat(transformed, L=1)
        assert len(data) == len(ref)
        np.testing.assert_allclose(np.sort(sign*data[:,1]), np.sort(ref[:,1]), atol=1e-10)
    scaled = parity_stat(1000*q, L=1000)
    assert len(scaled) == len(ref)
    np.testing.assert_allclose(np.sort(scaled[:,0]/1e9),np.sort(ref[:,0]),atol=1e-12)
    np.testing.assert_allclose(np.sort(scaled[:,1]),np.sort(ref[:,1]),atol=1e-10)


def test_molecule_control():
    for fraction, expected in [(1., 1.), (0., -1.)]:
        q, hand, template = chiral_catalogue(Ncl=100, size=.002, f_R=fraction)
        data = parity_stat(q, L=1)
        lengths = np.sort(np.linalg.norm(template[1:]-template[0], axis=1))
        mol = np.all(np.isclose(data[:,2:], lengths, rtol=1e-7, atol=1e-10), axis=1)
        assert mol.sum() >= 95
        np.testing.assert_allclose(data[mol,1],expected,atol=1e-10)
