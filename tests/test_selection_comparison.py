import numpy as np
import pytest

from parity_study.tests.selection_comparison import combine_disjoint
from parity_study.tests.selection_bins import paired_summary


def test_union_weights_tetrahedra_within_each_helicity_before_pairing():
    # Different populations in the two helicities rule out averaging their bin differences.
    plus, plus_n=combine_disjoint([[.2,.8],[.3,.5]],[[1,9],[5,5]])
    minus, minus_n=combine_disjoint([[.1,.4],[.1,.2]],[[9,1],[5,5]])
    np.testing.assert_allclose(plus,[.74,.4])
    np.testing.assert_allclose(minus,[.13,.15])
    np.testing.assert_array_equal(plus_n,[10,10])
    np.testing.assert_array_equal(minus_n,[10,10])
    delta,n,mean,sem=paired_summary(minus[:,None],plus[:,None])
    np.testing.assert_allclose(delta[:,0],[.61,.25])
    assert n[0]==2
    assert mean[0]==pytest.approx(.43)
    assert sem[0]==pytest.approx(.18)


def test_empty_component_does_not_drop_a_valid_union():
    mean,count=combine_disjoint([[np.nan,.4],[np.nan,np.nan]],[[0,5],[0,0]])
    np.testing.assert_allclose(mean,[.4,np.nan],equal_nan=True)
    np.testing.assert_array_equal(count,[5,0])
    with pytest.raises(ValueError): combine_disjoint([np.nan],[1])
    with pytest.raises(ValueError): combine_disjoint([.2],[-1])
    with pytest.raises(ValueError): combine_disjoint([.2],[1.5])
