import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from parity_study.tests.independent_ics import prepare, verify, audit
from parity_study.run import mesh_seed_block
from parity_study.tests.selections import measure
from parity_study.storage import atomic_json, atomic_npz, digest


def experiment(tmp_path):
    field = dict(model='mesh_za_helical', amplitude_convention='rms', L=100., N=8,
                 particle_grid=8, Ngal=512, sigma_delta=.2, helical_smoothing=4.,
                 spectrum=dict(kind='power_law', index=0., smoothing=4.),
                 density_seed=0, exclude_nyquist=True, tolerance=1e-10)
    cuts = [dict(id='all', label='All tetrahedra', variable='all', low=None, high=None),
            dict(id='shape_2', label='Shape third 2', variable='shape', low=.690511141, high=.7567675935)]
    parent = dict(field=field, discovery_seeds=list(range(10)), confirmation_seeds=list(range(10,20)),
                  A=2., cuts=cuts, primary_selection='shape_2', expected_sign=-1)
    parent_path = tmp_path/'parent.json'
    atomic_json(parent_path, parent)
    reference = tmp_path/'reference.json'
    atomic_json(reference, dict(extension=dict(added_seeds=list(range(20,35)))))
    path = prepare(parent_path, reference, tmp_path/'analysis', first=100, count=2)
    protocol = json.loads(path.read_text())
    config = dict(field=protocol['field'], seed_start=100, purpose='independent_ic_confirmation',
                  null=dict(seeds=2,A=2.), scan=dict(seeds=2,amplitudes=[2.]),
                  positive=dict(seeds=0), scheduler=dict(seeds_per_task=1), save_tetrahedra=True,
                  selection_protocol=dict(path=str(path),sha256=digest(path)))
    return config, protocol, path


def test_independent_protocol_freezes_cut_and_density_model(tmp_path):
    config, protocol, path = experiment(tmp_path)
    assert verify(config, protocol, path) == [100,101]
    for key, value in [('density_seed', 0), ('sigma_delta', .4)]:
        bad = copy.deepcopy(config)
        bad['field'][key] = value
        with pytest.raises(ValueError, match='Only the fixed-density'):
            verify(bad, protocol, path)
    changed = copy.deepcopy(protocol)
    changed['cuts'][1]['low'] += .001
    atomic_json(path, changed)
    config['selection_protocol']['sha256'] = digest(path)
    with pytest.raises(ValueError, match='frozen selection'):
        verify(config, changed, path)


def test_independent_protocol_rejects_old_seeds_and_mutations(tmp_path):
    config, protocol, path = experiment(tmp_path)
    bad = copy.deepcopy(config)
    bad['seed_start'] = 20
    with pytest.raises(ValueError, match='overlap'):
        verify(bad, protocol, path)
    changed = copy.deepcopy(protocol)
    changed['seeds'] = [100,101,102]
    atomic_json(path, changed)
    with pytest.raises(ValueError, match='changed after submission'):
        verify(config, changed, path)


def test_independent_ics_matched_catalogues_and_measurement(tmp_path):
    executable = Path(os.environ.get('DIVE_EXECUTABLE',
                      Path(__file__).resolve().parents[1]/'build/dive/DIVE'))
    if not executable.exists():
        pytest.skip('DIVE executable required for the small integration test')
    config, protocol, path = experiment(tmp_path)
    run = tmp_path/'run'
    (run/'code/bin').mkdir(parents=True)
    (run/'code/bin/DIVE').symlink_to(executable)
    atomic_json(run/'config.json', config)
    for task in range(2):
        mesh_seed_block(run, task, config)
    report, za = audit(run, config, [100,101], protocol['cuts'])
    assert report['checked_catalogues'] == 8
    assert all(count == 2 for count in report['distinct_fields'].values())
    assert za.shape == (2,2)
    result = measure(run, protocol['cuts'], [100,101], 2.)
    assert result['signals'].shape == (2,3,2)
    np.testing.assert_array_equal(result['paired'], result['signals'][:,2]-result['signals'][:,0])
    # Reusing an incorrect density seed must be rejected even when file names match.
    record = run/'records/scan_A2_plus/seed_0101.npz'
    with np.load(record) as z:
        arrays = {k:z[k] for k in z.files}
    metadata = json.loads(str(arrays['metadata_json']))
    metadata['parameters']['density_seed'] = 100
    arrays['metadata_json'] = np.array(json.dumps(metadata))
    atomic_npz(record, **arrays)
    with pytest.raises(ValueError, match='Mismatched catalogue'):
        measure(run, protocol['cuts'], [100,101], 2.)
    with pytest.raises(ValueError, match='Mismatched catalogue'):
        audit(run, config, [100,101], protocol['cuts'])
