import json
import os
from pathlib import Path

import numpy as np
import pytest

from parity_study.tests.amplitude_scan import AMPLITUDES, amplitude_ranking, analyze, physical_scales, prepare, verify
from parity_study.initial_conditions import mesh_fields
from parity_study.mesh_cases import symmetric_cases
from parity_study.run import mesh_seed_block
from parity_study.tests.selections import measure
from parity_study.storage import atomic_json, digest
from parity_study.summarize import expected_cases


def small_field():
    return dict(model='mesh_za_helical',amplitude_convention='rms',L=100.,N=8,
                particle_grid=8,Ngal=512,sigma_delta=.2,helical_smoothing=4.,
                spectrum=dict(kind='power_law',index=0.,smoothing=4.),
                density_seed=None,exclude_nyquist=True,tolerance=1e-10)


def test_physical_scales_match_generated_spectral_normalization():
    field=small_field()
    scale=physical_scales(field)
    _,meta=mesh_fields(**field,seed=3)
    assert scale['n_modes']==meta['n_modes']
    assert scale['expected_za_rms']==pytest.approx(meta['expected_za_rms'],rel=1e-14)
    assert scale['mesh_spacing']==scale['particle_spacing']==12.5
    assert scale['kmin']<=scale['helical_k_rms']<=scale['kmax']


@pytest.mark.parametrize('amplitudes', [[.5,.5,2],[.5,1],[0,2]])
def test_invalid_control_amplitudes_rejected(amplitudes):
    with pytest.raises(ValueError,match='unique positive'):
        symmetric_cases(dict(null=dict(A=2,amplitudes=amplitudes)))


def test_magnitude_and_detectability_can_choose_different_peaks():
    noise=np.linspace(-1,1,25)
    values=np.array([noise*0,-.03+.05*noise,-.08+.003*noise,-.12+.04*noise,-.01+.05*noise[::-1]])
    result=amplitude_ranking(values,[0,1,2,4,8],replicates=1000,seed=7)
    assert result['best_magnitude_A']==4
    assert result['best_snr_A']==2
    assert not result['magnitude_peak_at_upper_boundary']
    assert not result['snr_peak_at_upper_boundary']
    assert result['not_distinguished_from_magnitude_peak']==[4.]
    assert sum(r['bootstrap_best_magnitude'] for r in result['scores'])==pytest.approx(1.)
    assert sum(r['bootstrap_best_snr'] for r in result['scores'])==pytest.approx(1.)
    assert result['pairwise_family_size']==6
    # Pairing cancels shared IC scatter; it must not use the independent-sample SEM.
    row=next(r for r in result['pairwise_contrasts'] if r['A']==2 and r['other_A']==4)
    assert row['sem']==pytest.approx(np.std(values[2]-values[3],ddof=1)/5)
    for row in result['pairwise_contrasts']:
        assert row['p_all_pairs_bonferroni']==pytest.approx(min(1,6*row['p_two_sided']))
    assert amplitude_ranking(values,[0,1,2,4,8],replicates=1000,seed=7)==result


def test_boundary_peak_and_joint_bootstrap():
    noise=np.linspace(-.01,.01,25)
    values=np.array([noise*0,-.03+noise,-.08+2*noise,-.15+3*noise])
    result=amplitude_ranking(values,[0,1,2,4],replicates=100,seed=1)
    assert result['magnitude_peak_at_upper_boundary']
    assert result['snr_peak_at_upper_boundary']
    assert result['scores'][-1]['bootstrap_best_magnitude']==1


def test_matched_amplitude_pipeline_and_frozen_cut(tmp_path):
    executable=Path(os.environ.get('DIVE_EXECUTABLE',Path(__file__).resolve().parents[1]/'build/dive/DIVE'))
    if not executable.exists():
        pytest.skip('DIVE required for the small integration test')
    source=tmp_path/'source'
    old=dict(field=small_field(),null=dict(seeds=2,A=2.),scan=dict(seeds=2,amplitudes=[2.]),
             positive=dict(seeds=0),seed_start=100,scheduler=dict(seeds_per_task=1),save_tetrahedra=True)
    atomic_json(source/'config.json',old)
    atomic_json(source/'manifest.json',dict(sha256={}))
    atomic_json(source/'validation/independent_ic_ensemble.json',dict(passed=True))
    parent=tmp_path/'selection.json'
    cuts=[dict(id='all',label='All',variable='all',low=None,high=None),
          dict(id='shape_2',label='Shape third 2',variable='shape',low=.690511141,high=.7567675935)]
    atomic_json(parent,dict(field=dict(old['field'],density_seed=0),cuts=cuts,
                           primary_selection='shape_2',expected_sign=-1))
    protocol=prepare(source,parent,tmp_path/'out',optimize=True)
    config=dict(old,purpose='amplitude_scan',length_unit='h^-1 Mpc (comoving)',
                scan=dict(seeds=2,amplitudes=AMPLITUDES[1:]),
                null=dict(seeds=2,A=2.,amplitudes=AMPLITUDES[1:]),
                amplitude_protocol=dict(path=str(protocol),sha256=digest(protocol)))
    run=tmp_path/'run'
    atomic_json(run/'config.json',config)
    atomic_json(run/'manifest.json',dict(sha256={}))
    atomic_json(run/'validation/report.json',dict(passed=True))
    (run/'code/bin').mkdir(parents=True)
    (run/'code/bin/DIVE').symlink_to(executable)
    assert sum(expected_cases(config).values())==44
    for task in range(2): mesh_seed_block(run,task,config)
    analyze(run)
    result=json.loads((tmp_path/'out/analysis.json').read_text())
    assert result['independent_ic_audit']['checked_catalogues']==44
    assert 'optimization' in result
    assert result['optimization']['pairwise_family_size']==21
    with np.load(tmp_path/'out/amplitude_scan.npz') as z:
        assert z['signals'].shape==(2,8,3,2)
        np.testing.assert_array_equal(z['paired'][:,0],0)
        np.testing.assert_array_equal(z['signals'][:,0,0],z['signals'][:,0,2])
        np.testing.assert_allclose(np.diag(z['mean_covariance'][1]),np.var(z['paired'][1],axis=1,ddof=1)/2)
        at_half=measure(run,cuts,[100,101],.5)
        np.testing.assert_array_equal(z['signals'][:,2],at_half['signals'])
    assert len(result['contrasts_against_A2'])==12
    for row in result['summaries']:
        if row['p_two_sided'] is not None:
            assert row['p_amplitude_bonferroni']==pytest.approx(min(1,7*row['p_two_sided']))
    changed=json.loads(protocol.read_text())
    changed['cuts'][1]['low']+=.001
    atomic_json(protocol,changed)
    config['amplitude_protocol']['sha256']=digest(protocol)
    atomic_json(run/'config.json',config)
    with pytest.raises(ValueError,match='cannot change the frozen cut'):
        verify(run)
