import pytest

from parity_study.summarize import chi_hat_scan, expected_cases


def test_primary_scan_uses_only_normalized_chirality():
    # A strong auxiliary-statistic fluctuation must not choose the primary peak
    # or change the correction family.
    rows = [
        dict(A=1, statistic='chi', p_two_sided=1e-12, t=12),
        dict(A=1, statistic='chi_hat', p_two_sided=.01, t=2.7),
        dict(A=2, statistic='sgn', p_two_sided=1e-8, t=8),
        dict(A=2, statistic='chi_hat', p_two_sided=.4, t=-.85),
    ]
    scan = chi_hat_scan(rows, [1, 2])
    assert scan['comparisons'] == scan['available_comparisons'] == 2
    assert scan['peak']['statistic'] == 'chi_hat'
    assert scan['peak']['A'] == 1
    assert scan['bonferroni_p'] == pytest.approx(.02)
    assert [r['p_bonferroni'] for r in scan['rows']] == pytest.approx([.02, .8])
    assert all('p_bonferroni' not in r for r in rows)


def test_partial_scan_keeps_planned_amplitude_correction():
    # Missing amplitudes and singleton seed estimates cannot shrink the family.
    rows = [dict(A=1, statistic='chi_hat', p_two_sided=.02, t=2.4),
            dict(A=2, statistic='chi_hat', p_two_sided=None, t=None)]
    scan = chi_hat_scan(rows, [1, 2, 3])
    assert scan['comparisons'] == 3
    assert scan['available_comparisons'] == 1
    assert scan['bonferroni_p'] == pytest.approx(.06)
    assert scan['rows'][1]['p_bonferroni'] is None
    empty = chi_hat_scan([], [1, 2, 3])
    assert empty['bonferroni_p'] is None
    assert empty['peak'] is None


def test_fixed_ic_counts_za_control_once():
    config = dict(field=dict(model='mesh_za_helical', density_seed=0),
                  null=dict(seeds=10, A=2), scan=dict(seeds=10, amplitudes=[.5, 2]),
                  positive=dict(seeds=0))
    cases = expected_cases(config)
    assert cases['null_za'] == 1
    assert cases['null_symmetric'] == 10
    assert cases['scan_A2_plus'] == cases['scan_A2_minus'] == 10
    assert sum(cases.values()) == 51
    del config['field']['density_seed']
    assert expected_cases(config)['null_za'] == 10
