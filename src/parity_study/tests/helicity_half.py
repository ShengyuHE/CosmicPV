"""Fresh matched epsilon +/-0.5 catalogues with a preselected Q interval."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np

from parity_study.delaunay import parity_stat
from parity_study.geometry import tetrahedron_geometry
from parity_study.initial_conditions import displace_mesh, mesh_fields, validate_field
from parity_study.paths import expose_output
from parity_study.run import cached_mesh_fields
from parity_study.storage import atomic_json, atomic_npz, check_manifest, digest
from parity_study.summarize import mean_error
from parity_study.tests.selection_geometry import cut_mask

CASES = ('eps_m0p5', 'eps_p0p5')
EPSILONS = (-.5, .5)
FIELD_KEYS = ('delta_mesh', 'za', 'plus', 'minus')


def prepare(project, source, out, first=1050, count=50):
    project, source, out = (Path(p).resolve() for p in (project, source, out))
    old = json.loads((source/'config.json').read_text())
    reference = json.loads((source/'analysis.json').read_text())
    assert reference['complete'] and reference['config_sha256'] == digest(source/'config.json')
    expected = json.loads((source/'output_sha256.json').read_text())
    for name in ('analysis.json', 'catalogue_signals.npz'):
        assert digest(source/name) == expected[name]
    original = Path(old['confirmation'])
    original_protocol = json.loads((original/'protocol.json').read_text())
    field = original_protocol['field']
    validate_field(field)
    assert field['density_seed'] is None
    seeds = list(range(first, first+count))
    if count < 2 or first <= max(old['seeds']):
        raise ValueError('Require at least two fresh seeds beyond the previous ensemble')
    assert old['selection'] == dict(id='Q_wide', variable='Q', low=.4, high=.8, upper_inclusive=False)
    original_run = Path(json.loads((original/'analysis.json').read_text())['run'])
    for filename in ('initial_conditions.py', 'catalogue.py'):
        assert digest(project/'src/parity_study'/filename) == digest(original_run/'code/src/parity_study'/filename)
    out.mkdir(parents=True, exist_ok=False)
    cuts = [dict(id='all', variable='all', low=None, high=None),
            dict(id='reference_ratio', variable='reference_ratio',
                 low=.6905111410000623, high=.7567675934849439),
            dict(id='Q_wide', variable='quality', low=.4, high=.8, upper_inclusive=False)]
    protocol = dict(experiment='fresh_half_helicity_Q_selection',
        created=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),
        field=field, A=original_protocol['A'], length_unit='h^-1 Mpc (comoving)',
        seeds=seeds, epsilons=list(EPSILONS), cases=list(CASES), cuts=cuts, primary_selection='Q_wide',
        expected_sign=-1, source_Q_report=str(source), source_seeds=old['seeds'],
        source_sha256={name: digest(source/name) for name in ('config.json','analysis.json','catalogue_signals.npz')},
        sampling='Independent Gaussian density and helical pair per seed; both helicities share density, ZA, lattice and the same helical draws within a seed.',
        mixture='Psi_hel = A [sqrt((1+epsilon)/2) u_plus + sqrt((1-epsilon)/2) u_minus]; fixed ensemble RMS A.',
        primary_difference='D_s,0.5 = S_s(+0.5)-S_s(-0.5), using 0.4 <= Q < 0.8 without an r cut.',
        uncertainty='Sample standard deviation of paired differences divided by sqrt(n_pairs).',
        stopping_rule=f'Analyze all {count} planned ICs; no adaptation of A, Q cut, epsilon or sample size.',
        scope='New independent ICs for the intermediate helicities. The older epsilon=-1,0,+1 histograms use a separate ensemble and are descriptive references.',
        scheduler=dict(partition='public-cpu', account='pepe', cpus=4, memory='24G',
                       time='00:10:00', concurrency=8))
    atomic_json(out/'protocol.json', protocol)
    # Keep a conventional field config for shared cache utilities.
    atomic_json(out/'config.json', dict(field=field, purpose=protocol['experiment']))
    for folder in ('src', 'tests', 'slurm'):
        shutil.copytree(project/folder, out/'code'/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    (out/'code/bin').mkdir()
    shutil.copy2(project/'build/dive/DIVE', out/'code/bin/DIVE')
    shutil.copy2(project/'logs/patches/dive-parity.patch', out/'code/dive-parity.patch')
    shutil.copy2(project/'logs/vendor/sources.json', out/'code/dive_sources.json')
    shutil.copy2(project/'pyproject.toml', out/'code/pyproject.toml')
    (out/'before').mkdir()
    target = project/'results/field_PVdisp_B.ipynb'
    shutil.copy2(target, out/'before'/target.name)
    atomic_json(out/'notebook_input.json', dict(target=str(target), sha256=digest(target)))
    (out/'logs').mkdir()
    paths = [out/'protocol.json', out/'config.json', out/'notebook_input.json',
             *sorted((out/'code').rglob('*'))]
    atomic_json(out/'manifest.json', dict(project=str(project), sha256={
        str(p.relative_to(out)): digest(p) for p in paths if p.is_file()}))
    expose_output(project, out)


def selection_rows(tets, protocol):
    geometry = tetrahedron_geometry(tets)
    volume_error = float(abs(geometry['volume'].sum()/protocol['field']['L']**3 - 1))
    if volume_error > 1e-10:
        raise ValueError('Periodic tetrahedron volumes do not cover the box')
    rows = []
    for cut in protocol['cuts']:
        keep = cut_mask(tets, geometry, cut)
        if not keep.any():
            raise ValueError(f"Empty selection: {cut['id']}")
        rows.append(dict(selection=cut['id'], S=float(tets[keep,1].mean()),
                         selected=int(keep.sum()), total=len(tets), retained_fraction=float(keep.mean())))
    return rows, volume_error


def validate(out):
    out = Path(out)
    check_manifest(out)
    p = json.loads((out/'protocol.json').read_text())
    small = dict(p['field'], L=100., N=12, particle_grid=8, Ngal=512)
    arrays, metadata = mesh_fields(**small, seed=731, full_output=False)
    assert metadata['za_density_max_error'] < 1e-12 and metadata['za_curl_max'] < 1e-12
    details = []
    for eps in EPSILONS:
        cat, diag = displace_mesh(arrays, A=p['A'], eps=eps, L=100.)
        expected = (arrays['q'] + (arrays['za'] + p['A']*(
            np.sqrt((1+eps)/2)*arrays['plus'] + np.sqrt((1-eps)/2)*arrays['minus']))) % 100.
        np.testing.assert_array_equal(cat, expected)
        assert diag['expected_displacement_rms'] == p['A']
        original = parity_stat(cat, L=100., executable=out/'code/bin/DIVE', geometry=True)
        inverted = parity_stat((-cat)%100., L=100., executable=out/'code/bin/DIVE', geometry=True)
        reference, err = selection_rows(original, dict(p, field=small))
        reflected, err2 = selection_rows(inverted, dict(p, field=small))
        for left, right in zip(reference, reflected):
            assert left['selected'] == right['selected']
            np.testing.assert_allclose(left['S'], -right['S'], atol=1e-11)
        details.append(dict(epsilon=eps, exact_particle_map=True, selected_parity_passed=True,
                            max_volume_error=max(err,err2)))
    atomic_json(out/'validation/report.json', dict(passed=True, checks=details,
                 protocol_sha256=digest(out/'protocol.json')))
    print('Intermediate-helicity particle maps and selected parity checks passed', flush=True)


def measure(out, task):
    out = Path(out)
    check_manifest(out)
    p = json.loads((out/'protocol.json').read_text())
    validation = json.loads((out/'validation/report.json').read_text())
    assert validation['passed'] and validation['protocol_sha256'] == digest(out/'protocol.json')
    if not 0 <= task < len(p['seeds']):
        raise ValueError('Task outside planned seeds')
    seed = p['seeds'][task]
    completed = out/'completed'/f'seed_{seed:04d}.json'
    if completed.exists():
        record = json.loads(completed.read_text())
        assert record['protocol_sha256'] == digest(out/'protocol.json')
        assert all(digest(out/name) == sha for name,sha in record['file_sha256'].items())
        print(f'Verified completed seed {seed}', flush=True)
        return
    started = time.monotonic()
    arrays, meta = cached_mesh_fields(out, seed, p['field'])
    assert meta['density_seed'] == meta['helical_seed'] == seed and not meta['fixed_density']
    assert meta['za_density_max_error'] < 1e-12 and meta['za_curl_max'] < 1e-12
    fingerprints = {key: hashlib.sha256(np.ascontiguousarray(arrays[key]).view(np.uint8)).hexdigest()
                    for key in (*FIELD_KEYS, 'q')}
    cache = out/'fields'/f'main_{seed:04d}.npz'
    hashes = {str(cache.relative_to(out)): digest(cache)}
    rows, errors = [], []
    for case, eps in zip(p['cases'], p['epsilons']):
        cat, diagnostics = displace_mesh(arrays, A=p['A'], eps=eps, L=p['field']['L'])
        name = f'records/{case}/seed_{seed:04d}.npz'
        path = out/name
        if path.exists():
            with np.load(path) as z:
                saved_meta = json.loads(str(z['metadata_json']))
                assert saved_meta['protocol_sha256'] == digest(out/'protocol.json')
                assert saved_meta['seed'] == seed and saved_meta['epsilon'] == eps
                assert saved_meta['field_sha256'] == hashes[str(cache.relative_to(out))]
                np.testing.assert_array_equal(z['catalogue'], cat)
                tets = z['tetrahedra']
        else:
            tets = parity_stat(cat, L=p['field']['L'], executable=out/'code/bin/DIVE', geometry=True)
            atomic_npz(path, catalogue=cat, tetrahedra=tets,
                metadata_json=np.array(json.dumps(dict(seed=seed, case=case, epsilon=eps,
                    A=p['A'], field=p['field'], protocol_sha256=digest(out/'protocol.json'),
                    field_sha256=hashes[str(cache.relative_to(out))], diagnostics=diagnostics,
                    columns=['chi','chi_hat','ref_lmin','ref_lmid','ref_lmax','volume',
                             'edge_1','edge_2','edge_3','edge_4','edge_5','edge_6']))))
        measured, error = selection_rows(tets, p)
        rows.extend(dict(seed=seed, epsilon=eps, case=case, **r) for r in measured)
        errors.append(error)
        hashes[name] = digest(path)
        print(f'seed={seed} epsilon={eps:+g} tetrahedra={len(tets)}', flush=True)
    atomic_json(completed, dict(seed=seed, protocol_sha256=digest(out/'protocol.json'),
        rows=rows, file_sha256=hashes, field_fingerprints=fingerprints,
        max_volume_error=max(errors), elapsed_seconds=time.monotonic()-started))


def summarize(out):
    out = Path(out)
    check_manifest(out)
    p = json.loads((out/'protocol.json').read_text())
    seeds, cuts = p['seeds'], p['cuts']
    assert not set(seeds).intersection(p['source_seeds'])
    signals = np.full((len(cuts),2,len(seeds)), np.nan)
    counts = np.zeros(signals.shape, dtype=np.int64)
    totals = np.zeros((2,len(seeds)), dtype=np.int64)
    rows, output_hashes = [], {}
    fingerprints = {key: set() for key in (*FIELD_KEYS, 'q')}
    for case in p['cases']:
        assert {v.name for v in (out/'records'/case).glob('*.npz')} == {f'seed_{s:04d}.npz' for s in seeds}
    for si, seed in enumerate(seeds):
        path = out/'completed'/f'seed_{seed:04d}.json'
        done = json.loads(path.read_text())
        assert done['protocol_sha256'] == digest(out/'protocol.json')
        assert done['max_volume_error'] < 1e-10
        assert all(digest(out/name)==sha for name,sha in done['file_sha256'].items())
        output_hashes.update(done['file_sha256'])
        output_hashes[str(path.relative_to(out))] = digest(path)
        for key in fingerprints: fingerprints[key].add(done['field_fingerprints'][key])
        with np.load(out/'fields'/f'main_{seed:04d}.npz') as z:
            arrays = {key:z[key] for key in ('q','za','plus','minus')}
            meta = json.loads(str(z['metadata_json']))
        assert meta['field']==p['field'] and meta['density_seed']==meta['helical_seed']==seed
        for ei,(case,eps) in enumerate(zip(p['cases'],p['epsilons'])):
            with np.load(out/'records'/case/f'seed_{seed:04d}.npz') as z:
                positions = (arrays['q']+(arrays['za']+p['A']*(
                    np.sqrt((1+eps)/2)*arrays['plus']+np.sqrt((1-eps)/2)*arrays['minus'])))%p['field']['L']
                np.testing.assert_array_equal(z['catalogue'], positions)
                measured, _ = selection_rows(z['tetrahedra'], p)
            for ci, row in enumerate(measured):
                saved = next(r for r in done['rows'] if r['case']==case and r['selection']==row['selection'])
                assert all(saved[k]==v for k,v in row.items())
                signals[ci,ei,si], counts[ci,ei,si] = row['S'], row['selected']
                totals[ei,si] = row['total']
                rows.append(saved)
    assert all(len(fingerprints[k])==len(seeds) for k in FIELD_KEYS)
    assert len(fingerprints['q'])==1
    assert np.isfinite(signals).all() and np.all(counts>0)
    paired = signals[:,1]-signals[:,0]
    distributions = [dict(selection=cut['id'], epsilon=eps, **mean_error(signals[ci,ei]),
        mean_selected=float(counts[ci,ei].mean()), retained_fraction=float((counts[ci,ei]/totals[ei]).mean()))
        for ci,cut in enumerate(cuts) for ei,eps in enumerate(p['epsilons'])]
    differences = [dict(selection=cut['id'], **mean_error(paired[ci])) for ci,cut in enumerate(cuts)]
    audit = dict(passed=True, independent_initial_conditions=len(seeds), checked_catalogues=2*len(seeds),
        exact_matched_particle_maps=True, distinct_fields={k:len(v) for k,v in fingerprints.items()},
        no_overlap_with_previous_seeds=True)
    atomic_json(out/'validation/ensemble.json', audit)
    dest = out/'summary'
    atomic_npz(dest/'signals.npz', signals=signals, counts=counts, totals=totals, paired=paired,
               seeds=np.array(seeds), epsilons=np.array(p['epsilons']), selections=np.array([c['id'] for c in cuts]))
    for name,data in [('catalogue_signals',rows),('distribution_summary',distributions),('paired_summary',differences)]:
        with (dest/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    primary = next(r for r in differences if r['selection']==p['primary_selection'])
    atomic_json(dest/'analysis.json', dict(complete=True,protocol_sha256=digest(out/'protocol.json'),
        A=p['A'],seeds=seeds,epsilons=p['epsilons'],primary_selection=p['primary_selection'],
        primary=primary,distributions=distributions,summaries=differences,audit=audit,scope=p['scope']))
    for path in [*dest.glob('*'), out/'validation/ensemble.json',out/'validation/report.json']:
        output_hashes[str(path.relative_to(out))] = digest(path)
    atomic_json(out/'output_sha256.json', output_hashes)
    print(json.dumps(primary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','validate','measure','summarize'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--task',type=int)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.project,args.source,args.out)
    elif args.action=='measure': measure(args.out,args.task)
    else: globals()[args.action](args.out)


if __name__=='__main__': main()
