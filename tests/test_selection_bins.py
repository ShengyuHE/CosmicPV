import numpy as np
import pytest

from parity_study.tests.selection_bins import binned_signal, paired_summary


def test_bin_edges_empty_bins_and_conditional_means():
    # Lower edges are included; the final upper edge is included too.
    shape = np.array([0., .1, .25, .75, 1.])
    chi = np.array([.2, .4, -.6, .8, 1.])
    mean, count = binned_signal(shape, chi, np.linspace(0, 1, 5))
    np.testing.assert_array_equal(count, [2, 1, 0, 2])
    np.testing.assert_allclose(mean, [.3, -.6, np.nan, .9], equal_nan=True)
    assert np.nansum(count*mean)/len(shape) == pytest.approx(chi.mean())


def test_sem_uses_matched_differences_and_missing_pairs():
    minus = np.array([[.1, .2, np.nan, .1], [.3, .4, np.nan, np.nan], [.5, np.nan, np.nan, np.nan]])
    plus = np.array([[.2, .3, .2, .4], [.4, .7, .2, .3], [.6, .8, .2, .3]])
    delta, n, mean, sem = paired_summary(minus, plus)
    np.testing.assert_array_equal(n, [3, 2, 0, 1])
    np.testing.assert_allclose(mean, [.1, .2, np.nan, .3], equal_nan=True)
    np.testing.assert_allclose(sem, [0., .1, np.nan, np.nan], atol=1e-15, equal_nan=True)
    assert np.isnan(delta[2, 1])


def test_realizations_have_equal_weight_despite_different_populations():
    minus, plus = [], []
    for count, signal in [(1, .1), (100, .3)]:
        minus.append(binned_signal(np.full(count, .5), np.zeros(count), [0, 1])[0])
        plus.append(binned_signal(np.full(count, .5), np.full(count, signal), [0, 1])[0])
    _, n, mean, sem = paired_summary(minus, plus)
    assert n[0] == 2
    assert mean[0] == pytest.approx(.2)
    assert sem[0] == pytest.approx(.1)


def test_invalid_geometry_values_are_not_silently_dropped():
    with pytest.raises(ValueError):
        binned_signal([1.01], [.2], [0, 1])
    with pytest.raises(ValueError):
        binned_signal([.5], [np.nan], [0, 1])


@pytest.mark.parametrize('n', [10, 50])
def test_report_uses_actual_ensemble_size_and_output_path(tmp_path, n):
    import json
    from parity_study.tests.selection_bins import build_notebook
    config = dict(seeds=list(range(1000, 1000+n)), source=str(tmp_path/'source'),
                  target=str(tmp_path/'selection.ipynb'), output_link=f'saved/selection_{n}/fine_bins')
    (tmp_path/'config.json').write_text(json.dumps(config))
    nb = build_notebook(tmp_path)
    text = '\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    assert f'{n} Gaussian initial conditions' in text
    assert f'1000–{999+n}' in text
    assert r'\frac1{n_B}\sum_{s\in\mathcal C_B}' in text
    assert r'\sqrt{n_B}' in text
    assert f'unlabelled bars use all {n} pairs' in text
    assert 'incomplete pairs are omitted from that bin' in text
    assert f'saved/selection_{n}/fine_bins/summary.csv' in text
    assert '__N_IC__' not in text
