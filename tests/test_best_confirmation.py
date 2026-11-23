import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.stats import norm

from parity_study.tests.best_confirmation import (CONSTRUCTION_FILES, PURPOSE, analyze,
    distribution_figure, distribution_summary, prepare, verify)
from parity_study.run import mesh_seed_block
from parity_study.storage import atomic_json, digest


def experiment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    executable = Path(os.environ.get('DIVE_EXECUTABLE',root/'build/dive/DIVE'))
    if not executable.exists():
        pytest.skip('DIVE required')
    source = tmp_path/'source'
    for rel in CONSTRUCTION_FILES:
        dest = source/'code'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(executable if rel=='bin/DIVE' else root/rel,dest)
    atomic_json(source/'manifest.json',dict(sha256={}))
    field = dict(model='mesh_za_helical',amplitude_convention='rms',L=100.,N=8,
                 particle_grid=8,Ngal=512,sigma_delta=.2,helical_smoothing=4.,
                 spectrum=dict(kind='power_law',index=0.,smoothing=4.),
                 density_seed=None,exclude_nyquist=True,tolerance=1e-10)
    cuts = [dict(id='all',label='All',variable='all',low=None,high=None),
            dict(id='shape_2',label='Shape third 2',variable='shape',low=.690511141,high=.756767594)]
    selection = tmp_path/'selection.json'
    atomic_json(selection,dict(discovery_seeds=list(range(10)),confirmation_seeds=list(range(10,35))))
    previous = dict(selection_protocol=str(selection),selection_protocol_sha256=digest(selection),
                    field=field,cuts=cuts,primary_selection='shape_2',expected_sign=-1,
                    seeds=list(range(100,125)),length_unit='h^-1 Mpc (comoving)')
    previous_path = tmp_path/'scan/protocol.json'
    atomic_json(previous_path,previous)
    atomic_json(previous_path.parent/'analysis.json',dict(complete=True,
                protocol_sha256=digest(previous_path),optimization=dict(best_snr_A=16.)))
    monkeypatch.setattr('parity_study.tests.amplitude_scan.verify',lambda _: ({},previous,previous_path))
    path = prepare(source,tmp_path/'out',first=1000,count=3)
    p = json.loads(path.read_text())
    cfg = dict(purpose=PURPOSE,field=field,seed_start=1000,
               scan=dict(seeds=3,amplitudes=[16.]),null=dict(seeds=3,A=16.),
               positive=dict(seeds=0),save_tetrahedra=True,scheduler=dict(seeds_per_task=1),
               confirmation_protocol=dict(path=str(path),sha256=digest(path)))
    run = tmp_path/'run'
    shutil.copytree(source/'code',run/'code')
    atomic_json(run/'manifest.json',dict(sha256={}))
    atomic_json(run/'config.json',cfg)
    atomic_json(run/'validation/report.json',dict(passed=True))
    return run,cfg,p,path


def test_fresh_confirmation_rejects_overlap_and_changed_configuration(tmp_path,monkeypatch):
    run,cfg,p,path = experiment(tmp_path,monkeypatch)
    assert verify(run)[1]['seeds']==[1000,1001,1002]
    for change in ('old_seeds','amplitude','fixed_density'):
        bad = copy.deepcopy(cfg)
        if change=='old_seeds': bad['seed_start']=100
        if change=='amplitude': bad['scan']['amplitudes']=[24.]
        if change=='fixed_density': bad['field']['density_seed']=0
        atomic_json(run/'config.json',bad)
        with pytest.raises(ValueError): verify(run)
    atomic_json(run/'config.json',cfg)
    p['cuts'][1]['low']+=.001
    atomic_json(path,p)
    cfg['confirmation_protocol']['sha256']=digest(path)
    atomic_json(run/'config.json',cfg)
    with pytest.raises(ValueError,match='frozen selection'): verify(run)


def test_gaussian_curve_uses_catalogue_scatter_not_sem():
    values = np.array([[-.05,0,.05,.10],[.05,.10,.15,.20],[.15,.20,.25,.30]])
    signals = values[None,:,:]
    rows = distribution_summary(signals,[dict(id='selected')])
    assert rows[0]['sd']==pytest.approx(np.std(values[0],ddof=1))
    assert rows[0]['sem']==pytest.approx(rows[0]['sd']/2)
    fig = distribution_figure(dict(A=16),signals,0)
    curve = fig.axes[0].lines[0]
    np.testing.assert_allclose(curve.get_ydata(),norm.pdf(curve.get_xdata(),rows[0]['mean'],rows[0]['sd']))
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_fresh_confirmation_audit_and_paired_statistics(tmp_path,monkeypatch):
    run,cfg,p,path = experiment(tmp_path,monkeypatch)
    for task in range(3): mesh_seed_block(run,task,cfg)
    report = analyze(run)
    assert report['catalogue_count']==12
    assert report['independent_ic_audit']['checked_catalogues']==12
    with np.load(path.parent/'confirmation.npz') as z:
        assert z['signals'].shape==(2,3,3)
        d = z['signals'][1,2]-z['signals'][1,0]
        np.testing.assert_array_equal(z['paired'][1],d)
        assert report['primary']['mean']==pytest.approx(d.mean())
        assert report['primary']['sem']==pytest.approx(d.std(ddof=1)/np.sqrt(3))
    assert len(report['distributions'])==6
