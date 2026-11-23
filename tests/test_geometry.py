import itertools

import numpy as np
import pytest

from parity_study.delaunay import parity_stat
from parity_study.geometry import edge_geometry, tetrahedron_geometry


def measure_vertices(points):
    edges = [np.linalg.norm(points[i]-points[j]) for i,j in itertools.combinations(range(4),2)]
    volume = abs(np.linalg.det(points[1:]-points[0]))/6
    return edge_geometry(volume, edges)


def test_quality_geometric_limits_and_vertex_invariance():
    regular = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1.]])
    assert measure_vertices(regular)['quality'] == pytest.approx(1)
    points = np.array([[0.,0.,0.],[2,0,0],[.3,1,0],[.1,.2,.7]])
    reference = measure_vertices(points)
    for perm in itertools.permutations(range(4)):
        g = measure_vertices(points[list(perm)])
        assert g['quality'] == pytest.approx(reference['quality'])
        assert g['edge_ratio'] == pytest.approx(reference['edge_ratio'])
    flat = points.copy()
    flat[-1,-1] = 1e-9
    assert measure_vertices(flat)['quality'] < 1e-5
    for scale in (1e-6, 1000.):
        g = measure_vertices(-scale*points+3*scale)
        assert g['quality'] == pytest.approx(reference['quality'])
        assert g['edge_ratio'] == pytest.approx(reference['edge_ratio'])
        assert g['edge_rms'] == pytest.approx(scale*reference['edge_rms'])


def test_geometry_requires_true_volume_and_six_edges():
    with pytest.raises(ValueError, match='12-column'):
        tetrahedron_geometry(np.ones((3,5)))
    with pytest.raises(ValueError, match='impossible'):
        edge_geometry(10., np.ones(6))
    # An exact tie has zero parity weight but a nonzero geometric volume.
    row = np.array([[0.,0.,1.,1.,1.,np.sqrt(2)/12,*np.ones(6)]])
    assert tetrahedron_geometry(row)['quality'][0] == pytest.approx(1)


def test_dive_optional_geometry_preserves_chirality_and_symmetries():
    points = np.random.default_rng(713).uniform(0,1,(130,3))
    basic = parity_stat(points,L=1)
    data = parity_stat(points,L=1,geometry=True)
    np.testing.assert_array_equal(data[:,:5], basic)
    ref = tetrahedron_geometry(data)
    assert ref['volume'].sum() == pytest.approx(1,abs=1e-12)
    for points2,L,sign in [((-points)%1,1,-1),(points[:,[1,0,2]],1,-1),
                            ((points+.123)%1,1,1),(1000*points,1000,1)]:
        transformed = parity_stat(points2,L=L,geometry=True)
        g = tetrahedron_geometry(transformed)
        for key in ('quality','edge_ratio'):
            np.testing.assert_allclose(np.sort(g[key]),np.sort(ref[key]),atol=1e-10)
        np.testing.assert_allclose(np.sort(g['edge_rms']/L),np.sort(ref['edge_rms']),atol=1e-10)
        keep = (g['quality'] >= .6)&(g['quality'] < .8)
        keep0 = (ref['quality'] >= .6)&(ref['quality'] < .8)
        assert keep.sum() == keep0.sum()
        assert transformed[keep,1].mean() == pytest.approx(sign*data[keep0,1].mean(),abs=1e-10)


def test_nested_thinning_is_matched_and_does_not_use_positions():
    from parity_study.tests.selection_geometry import thinning_indices
    first = thinning_indices(256,10,[1.,.5,.25])
    second = thinning_indices(256,10,[1.,.5,.25])
    assert [len(x) for x in first] == [256,128,64]
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
    assert set(first[2]) <= set(first[1]) <= set(first[0])
    assert not np.array_equal(first[1],thinning_indices(256,11,[1.,.5,.25])[1])
