import copy
import json
import numpy as np
import pytest

from parity_study.tests.selections import selection_mask, candidate_cuts, verify_confirmation
from parity_study.seeds import seed_ids
from parity_study.storage import digest
from parity_study.run import mesh_seed_block


def test_all_selection_masks_preserve_reflection_and_boundaries():
    rng = np.random.default_rng(18)
    tets = np.column_stack((rng.normal(size=80), rng.uniform(-1, 1, 80),
                            np.sort(rng.uniform(1, 3, (80, 3)), axis=1)))
    tets[:6, 1] = [.45, -.45, .50, -.50, .55, -.55]
    reflected = tets.copy()
    reflected[:, :2] *= -1
    cuts = candidate_cuts(tets, 1)
    for cut in cuts:
        mask = selection_mask(tets, cut, 1)
        mirrored = selection_mask(reflected, cut, 1)
        np.testing.assert_array_equal(mask, mirrored)
        if mask.any():
            assert tets[mask, 1].mean() == -reflected[mirrored, 1].mean()
    band = next(c for c in cuts if c['id'] == 'abs_0.50_0.55')
    np.testing.assert_array_equal(selection_mask(tets, band, 1)[:6],
                                  [False, False, True, True, False, False])


def test_confirmation_rejects_reused_seeds_and_changed_protocol(tmp_path):
    protocol = dict(discovery_seeds=list(range(10)), confirmation_seeds=list(range(10, 20)),
                    field=dict(model='mesh_za_helical'), A=2)
    path = tmp_path/'protocol.json'
    path.write_text(json.dumps(protocol))
    config = dict(field=protocol['field'], seed_start=10, scan=dict(seeds=10, amplitudes=[2]),
                  null=dict(seeds=10), selection_protocol=dict(sha256=digest(path)))
    assert verify_confirmation(config, protocol, path) == list(range(10, 20))
    overlapping = copy.deepcopy(config)
    overlapping['seed_start'] = 5
    with pytest.raises(ValueError, match='overlap'):
        verify_confirmation(overlapping, protocol, path)
    path.write_text(json.dumps(dict(protocol, A=3)))
    with pytest.raises(ValueError, match='changed'):
        verify_confirmation(config, protocol, path)


def test_nonzero_seed_start_routes_real_random_seeds(monkeypatch, tmp_path):
    import parity_study.run as runner
    import parity_study.initial_conditions as ic
    config = dict(field=dict(model='mesh_za_helical', density_seed=0, L=1000, sigma_delta=.2),
                  seed_start=10, null=dict(seeds=2, A=2), scan=dict(seeds=2, amplitudes=[2]),
                  positive=dict(seeds=0), scheduler=dict(seeds_per_task=1))
    calls, random_seeds = [], []
    def fields(run, seed, field):
        random_seeds.append(seed)
        return {}, dict(density_seed=0)
    monkeypatch.setattr(runner, 'cached_mesh_fields', fields)
    monkeypatch.setattr(ic, 'displace_mesh', lambda *a, **kw: (None, {}))
    monkeypatch.setattr(runner, 'measured', lambda run, name, seed, cat, **kw:
                        calls.append((name, seed, kw['parameters'])))
    mesh_seed_block(tmp_path, 0, config)
    mesh_seed_block(tmp_path, 1, config)
    assert random_seeds == [10, 11]
    assert [(name, seed) for name, seed, p in calls if name == 'null_za'] == [('null_za', 10)]
    assert len(calls) == 7
    assert all(seed == p['helical_seed'] and p['density_seed'] == 0 for _, seed, p in calls)
    assert list(seed_ids(config, 2)) == [10, 11]


def test_extension_preserves_cut_and_separates_added_seeds(tmp_path):
    parent = dict(discovery_seeds=list(range(10)), confirmation_seeds=list(range(10,20)),
                  field=dict(model='mesh_za_helical'), A=2, cuts=[dict(id='shape_2',low=.69,high=.76)],
                  primary_selection='shape_2', expected_sign=-1)
    parent_path = tmp_path/'original.json'
    parent_path.write_text(json.dumps(parent))
    extension = dict(parent_protocol=str(parent_path),parent_protocol_sha256=digest(parent_path),
                     previous_seeds=list(range(10,20)),added_seeds=list(range(20,35)))
    protocol = dict(parent,confirmation_seeds=list(range(10,35)),extension=extension)
    path = tmp_path/'extended.json'
    path.write_text(json.dumps(protocol))
    config = dict(field=parent['field'],seed_start=10,scan=dict(seeds=25,amplitudes=[2]),
                  null=dict(seeds=25),selection_protocol=dict(sha256=digest(path)))
    assert verify_confirmation(config,protocol,path) == list(range(10,35))
    changed = copy.deepcopy(protocol)
    changed['cuts'][0]['low'] = .70
    path.write_text(json.dumps(changed))
    config['selection_protocol']['sha256'] = digest(path)
    with pytest.raises(ValueError,match='cannot change cuts'):
        verify_confirmation(config,changed,path)


@pytest.mark.parametrize('start', [-1, True, 1.5])
def test_bad_seed_start_rejected(start):
    with pytest.raises(ValueError, match='nonnegative integer'):
        seed_ids(dict(field=dict(model='mesh_za_helical'), seed_start=start), 10)
