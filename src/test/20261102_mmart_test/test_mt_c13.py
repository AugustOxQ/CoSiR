"""Quick data-free checks of the independent MMArt C13 implementation.

All filesystem operations in compare tests are replaced in memory: no pytest
temporary directories, caches, or writes outside the authorized C13 outputs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_c13 as c

COMPARE_ROOT = c.SMOKE_OUT / 'unit_round'
COMPARE_OUT = c.SMOKE_OUT / 'unit_compare'


def mock_compare_integrity(monkeypatch):
    """Claim/layout tests exercise integrity helpers separately below."""
    monkeypatch.setattr(c, 'compare_matched_identity', lambda *args: ({}, {}))
    monkeypatch.setattr(c, 'compare_code_provenance', lambda doc, commit: dict(
        derive_commit=commit, committed_source_sha256=doc.get('source_sha256'),
        derived_source_sha256=doc.get('source_sha256'), current_source_sha256='current',
        error=None, matches=True))


def record_report(failed=0):
    return {'failed': failed, 'code_provenance': {'matches': True}}


def cells():
    return [dict(zip(c.IDENTITY, k), lo=.02, r=.1 if k[0] == 'SC4' else None,
                 n_draws=2000 if k[3] == 'clu' else 10000)
            for e in c.ENCODERS for k in sorted(c.expected_ids(e))]


def test_sha_order_and_subset_join():
    keys = [f'style/{i}.jpg' for i in range(50)]
    expected = sorted(keys, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())
    assert c.hashed_order(reversed(keys)) == expected
    assert c.keys_sha(expected) == hashlib.sha256('\n'.join(expected).encode()).hexdigest()
    assert c.join_rows(['z', 'x', 'y'], ['y', 'z']).tolist() == [2, 0]


@pytest.mark.parametrize('stored,wanted', [(['x', 'x'], ['x']), (['x'], ['y']), (['x'], ['x', 'x'])])
def test_bad_key_joins(stored, wanted):
    with pytest.raises(AssertionError):
        c.join_rows(stored, wanted)


def test_cut_filter_order_inclusive_threshold_and_md5_tie():
    # The near/exact rows share the survivors' md5: removal must precede dedup.
    keys = ['exact', 'near', 'same-a', 'same-b', 'far', 'keep']
    index = pd.DataFrame(dict(key=keys, split=['new'] * 6,
                              md5=['shared', 'shared', 'shared', 'shared', 'far', 'keep'],
                              exact_dup_trainval=[True, False, False, False, False, False]))
    near = pd.DataFrame(dict(key=list(reversed(keys)), max_cos=[.1, .85, .3, .2, .95, .99]))
    result = c.cut(index, near, .95)
    winner = c.hashed_order(['same-a', 'same-b'])[0]
    assert result['keys'] == c.hashed_order([winner, 'far', 'keep'])
    assert result['dropped'] == dict(exact=1, near=1, md5=1)
    assert c.cut(index, near, .85)['keys'] == c.hashed_order([winner, 'keep'])
    assert result['sha256'] == c.keys_sha(result['keys'])


def test_cut_joins_nearcopy_by_key_and_rejects_missing_rows():
    index = pd.DataFrame(dict(key=['a', 'b'], split=['new', 'new'], md5=['a', 'b'], exact_dup_trainval=[False, False]))
    near = pd.DataFrame(dict(key=['b', 'a'], max_cos=[.96, .2]))
    assert c.cut(index, near, .95)['keys'] == ['a']
    with pytest.raises((AssertionError, KeyError)):
        c.cut(index, near.iloc[:1], .95)


def test_twice_normalisation_retains_float32_rounding():
    raw = np.random.default_rng(7).normal(size=(20, 17)).astype(np.float16)
    x = raw.astype(np.float32)
    once = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    twice = once / np.linalg.norm(once, axis=1, keepdims=True)
    assert np.array_equal(c.store_retrieval(raw), twice)
    assert twice.dtype == np.float32
    assert np.any(once != twice), 'fixture must expose the second division'
    with pytest.raises(AssertionError):
        c.unit(np.zeros((2, 3), np.float32))


def test_vector_uses_distinct_clustering_and_retrieval_conventions(monkeypatch):
    raw = np.random.default_rng(19).normal(size=(8, 11)).astype(np.float16)
    store = c.unit(raw, floor=True)
    work = object.__new__(c.Computation)
    monkeypatch.setattr(work, 'features', lambda *args: store)
    monkeypatch.setattr(work, 'exported', lambda *args: raw.astype(np.float32))
    assert np.array_equal(work.vector('siglip2', 'frozen', 'val', 'emotion'), store)
    assert np.array_equal(work.vector('qwen', 'teacher', 'val', 'emotion', retrieval=True), c.store_retrieval(raw))
    assert np.array_equal(work.vector('siglip2', 'cstar', 'val', 'emotion', 0), raw.astype(np.float32))
    assert np.array_equal(work.vector('siglip2', 'cstar', 'val', 'emotion', 0, retrieval=True), c.unit(raw))


@pytest.mark.parametrize('lo,r,want', [(1e-15, None, 'WIN'), (0., None, 'not WIN'),
    (-.1, None, 'not WIN'), (1e-15, .1, 'WIN'), (0., .1, 'LEVEL'),
    (-.099999, .1, 'LEVEL'), (-.1, .1, 'BELOW'), (-.100001, .1, 'BELOW'), (0., 0., 'BELOW')])
def test_strict_outcome_thresholds(lo, r, want):
    assert c.cell_outcome(lo, r) == want


@pytest.mark.parametrize('sc,want', [('SC1', 'fails'), ('SC2', 'fails'), ('SC3', 'qualified')])
def test_verdict_conjunctions(sc, want):
    rows = cells()
    assert c.decisions(rows)['verdicts'] == dict.fromkeys(c.ENCODERS, 'holds')
    next(x for x in rows if x['encoder'] == 'siglip2' and x['sc'] == sc)['lo'] = 0.
    assert c.decisions(rows)['verdicts']['siglip2'] == want
    assert c.decisions(rows)['verdicts']['siglip1'] == 'holds'


def test_live_missing_cells_and_duplicate_identities_fail():
    rows = cells()
    with pytest.raises(AssertionError, match='missing live'):
        c.decisions(rows[1:])
    with pytest.raises(AssertionError, match='duplicate'):
        c.decisions(rows + [deepcopy(rows[0])])
    missing_enc = rows[0]['encoder']
    assert c.decisions(rows[1:], partial=True)['verdicts'][missing_enc] == 'pending'


@pytest.mark.parametrize('gate,key,all_enc', [('i_nearcopy', None, True), ('iii', None, True),
    ('i_default', 'qwen', True), ('i_default', 'siglip2', False), ('ii', 'siglip2', False)])
def test_d17_global_and_encoder_voids(gate, key, all_enc):
    status = c.good_status()
    if key:
        status['gates'][gate][key] = 'failed'
    else:
        status['gates'][gate] = 'failed'
    result = c.decisions(cells(), status)
    affected = set(c.ENCODERS) if all_enc else {'siglip2'}
    for e in c.ENCODERS:
        assert result['verdicts'][e] == ('inconclusive' if e in affected else 'holds')
    assert all(x['status'] == 'inconclusive' for x in result['cells'] if x['encoder'] in affected)
    # A missing numerical path is excused only by an explicit failed gate.
    c.decisions([x for x in cells() if x['encoder'] not in affected], status)


@pytest.mark.parametrize('gate', ['i_teacher', 'v'])
def test_d17_teacher_aspect_only(gate):
    status = c.good_status()
    status['gates'][gate]['emotion'] = 'failed'
    result = c.decisions(cells(), status)
    assert all(v == 'holds' for v in result['verdicts'].values())
    bad = [x for x in result['cells'] if x['status'] == 'inconclusive']
    assert len(bad) == 8 and all(x['aspect'] == 'emotion' and x['sc'] == 'SC4' for x in bad)


def test_d17_mapper_void_preserves_sc4_and_plain_cells():
    status = c.good_status()
    status['gates']['iv']['siglip2'] = 'failed'
    result = c.decisions(cells(), status)
    assert result['verdicts']['siglip2'] == 'inconclusive'
    assert all(x['sc'] == 'SC3' for x in result['cells'] if x['status'] == 'inconclusive')
    assert all(x['claim'] != 'inconclusive' for x in result['sc4']['siglip2'])


@pytest.mark.parametrize('key', ['ranges', 'picks'])
def test_d17_stage1_encoder_path(key):
    status = c.good_status()
    status['stage1'][key]['siglip2'] = 'failed'
    result = c.decisions(cells(), status)
    assert result['verdicts']['siglip2'] == 'inconclusive'
    bad = [x for x in result['cells'] if x['status'] == 'inconclusive']
    assert len(bad) == 9 and all(x['encoder'] == 'siglip2' and x['sc'] in ('SC2', 'SC4') for x in bad)


def test_d17_readout_replay_and_teacher_pick_dependencies():
    status = c.good_status()
    status['gates']['vi'] = 'failed'
    result = c.decisions(cells(), status)
    assert all(v == 'inconclusive' for v in result['verdicts'].values())
    assert all(x['task'] == 'ret' for x in result['cells'] if x['status'] == 'inconclusive')
    status = c.good_status()
    status['stage1']['teacher_pick'] = 'failed'
    result = c.decisions(cells(), status)
    assert all(v == 'holds' for v in result['verdicts'].values())
    assert sum(x['status'] == 'inconclusive' for x in result['cells']) == 12
    status = c.good_status()
    status['stage1']['ranges']['qwen'] = 'failed'
    assert c.voids(status, 'siglip2', 'SC4', 'genre', 'ret')
    assert not c.voids(status, 'siglip2', 'SC4', 'genre', 'clu')
    status['stage1']['ranges'] = 'failed'
    assert all(v == 'inconclusive' for v in c.decisions(cells(), status)['verdicts'].values())


def test_status_missing_or_pending_is_not_silently_ok():
    status = c.good_status()
    status['gates']['iii'] = 'pending'
    with pytest.raises(AssertionError):
        c.decisions(cells(), status)


@pytest.mark.parametrize('enc,a,task', [('qwen', 'emotion', 'clu'), ('qwen', 'genre', 'ret'),
                                      ('siglip2', 'style', 'ret')])
def test_d7_claims(enc, a, task):
    assert c.sc4_claim(enc, a, task, 'WIN') == 'better than'
    assert c.sc4_claim(enc, a, task, 'LEVEL') == 'level with or better than'
    assert c.sc4_claim(enc, a, task, 'BELOW') == 'no claim'
    assert c.sc4_claim(enc, a, task, 'WIN', void=True) == 'inconclusive'


def test_non_d7_win_does_not_license_better_than():
    assert c.sc4_claim('qwen', 'style', 'ret', 'WIN') == 'level with or better than'
    assert c.sc4_claim('siglip1', 'emotion', 'ret', 'WIN') == 'level with or better than'


def test_comparison_absolute_tolerance_and_exact_categories():
    result = c.Comparison()
    result.add('numeric/a', 0., 1e-6, True)
    result.add('numeric/b', 0., 1.00001e-6, True)
    result.add('numeric/missing', .1, None, True)
    result.add('category/pick', 'center+k10', 'none+k0')
    result.add('category/bool', True, 1)
    result.add('numeric/nan', .1, float('nan'), True)
    assert result.report()['failed'] == 5
    assert result.items[0]['pass_'] is True
    result.tree('table', {'a': .2}, {'a': .2, 'unexpected': 1})
    assert result.report()['failed'] == 6


def full_doc():
    ranges = {e: {task: {a: dict(r=.01, per_seed={str(s): .1 + s * .0025 for s in c.SEEDS}) for a in c.ASPECTS}
                  for task in ('clu', 'ret')} for e in c.ENCODERS}
    picks = {}
    for key in ['teacher'] + [f'{e}:{arm}' for e in c.ENCODERS for arm in ('cstar', 'frozen')]:
        width = 768 if key == 'clip_l14:frozen' else 1152 if key in ('siglip1:frozen', 'siglip2:frozen') else 2048
        picks[key] = {'table': {c.rt_readout.name(ro): {} for ro in c.rt_readout.menu(width)}}
    return dict(complete=True, unavailable=[], sections=list(c.FULL_SECTIONS), n_clu=2000, n_ret=10000, kmeans_threads=8,
                cutsets={'test': {}, 'far': {}}, cells=cells(), ranges=ranges, picks=picks)


def test_record_full_coverage_draw_counts_and_failures():
    doc = full_doc()
    assert c.full_scope(doc)
    assert c.record_refusal(doc, record_report(), cells()) is None
    assert 'failed' in c.record_refusal(doc, record_report(1), cells())
    reduced = deepcopy(cells())
    reduced[0]['n_draws'] = 40
    assert 'draws' in c.record_refusal(doc, record_report(), reduced)
    for field, value in [('sections', ['cut']), ('n_clu', 40), ('n_ret', 100), ('complete', False), ('cells', cells()[1:])]:
        changed = {**doc, field: value}
        assert 'subset' in c.record_refusal(changed, {'failed': 0}, cells())
    doc['picks']['teacher']['table'].pop('none+k0')
    assert not c.full_scope(doc)


@pytest.mark.parametrize('threads', [1, 8])
def test_kmeans_limits_blas_and_openmp_for_every_state(monkeypatch, threads):
    calls = []
    before = {(r['filepath'], r['user_api']): r['num_threads'] for r in threadpool_info()}
    class Model:
        def __init__(self, **params):
            self.params = params

        def fit(self, x):
            libraries = threadpool_info()
            assert {'blas', 'openmp'} <= {r['user_api'] for r in libraries}
            assert all(r['num_threads'] == threads for r in libraries if r['user_api'] in ('blas', 'openmp'))
            assert x.dtype == np.float32
            calls.append(self.params)
            self.labels_ = np.arange(len(x)) % 20
            return self
    monkeypatch.setattr(c, 'MiniBatchKMeans', Model)
    x = c.unit(np.random.default_rng(6).normal(size=(40, 8)))
    result = c.kmeans_partitions(x, threads)
    assert result.shape == (3, 40) and result.dtype == np.int32
    assert calls == [dict(n_clusters=20, random_state=s, n_init=3, batch_size=4096) for s in (0, 1, 2)]
    assert before == {(r['filepath'], r['user_api']): r['num_threads'] for r in threadpool_info()}


def test_partition_cache_is_bound_to_thread_count(monkeypatch):
    work = object.__new__(c.Computation)
    seen = []
    monkeypatch.setattr(work, 'array', lambda name, inputs, fn: seen.append((name, inputs)) or fn())
    monkeypatch.setattr(c, 'kmeans_partitions', lambda x, n: np.full((3, len(x)), n, np.int32))
    x = c.unit(np.ones((40, 8), np.float32))
    for n in (1, 8):
        work.kmeans_threads = n
        assert (work.parts(x, [str(i) for i in range(40)]) == n).all()
    assert seen[0][1][-1] == {'kmeans_threads': 1}
    assert seen[1][1][-1] == {'kmeans_threads': 8}


@pytest.mark.parametrize('value', [1, None, True])
def test_record_refuses_wrong_or_missing_kmeans_threads(value):
    doc = full_doc()
    if value is None:
        doc.pop('kmeans_threads')
    else:
        doc['kmeans_threads'] = value
    assert 'kmeans_threads 8' in c.record_refusal(doc, {'failed': 0}, cells())


@pytest.mark.parametrize('option,want', [([], 8), (['--kmeans-threads', '1'], 1)])
def test_cli_kmeans_thread_default_and_sensitivity_option(monkeypatch, option, want):
    called = []
    monkeypatch.setattr(sys, 'argv', ['mt_c13.py', 'derive', '--sections', 'cut', *option])
    monkeypatch.setattr(c, 'derive', lambda *args, **kwargs: called.append(kwargs))
    assert c.main() == 0
    assert called == [{'kmeans_threads': want}]


def test_compare_refuses_subset_without_writing_record(monkeypatch):
    doc = dict(root=str(COMPARE_ROOT), complete=True, source_sha256='source', unavailable=[],
               inputs={}, cutsets={}, ranges={}, picks={}, cells=[], sections=['cut'],
               scope='subset:cut', n_clu=40, n_ret=100)
    reports = []
    monkeypatch.setattr(c, 'source_identity', lambda: 'source')
    monkeypatch.setattr(c, 'read_json', lambda _: doc)
    monkeypatch.setattr(Path, 'exists', lambda _: False)
    monkeypatch.setattr(c, 'write_json', lambda p, v: reports.append(v))
    monkeypatch.setattr(Path, 'write_text', lambda *args: pytest.fail('record must not be written'))
    mock_compare_integrity(monkeypatch)
    with pytest.raises(ValueError, match='subset'):
        c.compare(root=COMPARE_ROOT, records=COMPARE_ROOT / 'records', out=COMPARE_OUT, write_record=True)
    assert reports and reports[0]['record'].startswith('refused:')


def comparison_fixture(monkeypatch, rows=None, status=None):
    rows = [x for x in cells() if x['encoder'] == 'siglip2'] if rows is None else rows
    status = c.good_status() if status is None else status
    decided = c.decisions(rows, status, encoders=('siglip2',))
    outcome_rows = [{**x, 'status': 'decided' if x['status'] == 'ok' else x['status']} for x in decided['cells']]
    claims = [{**x, 'status': 'decided' if x['status'] == 'ok' else x['status'],
               'void': ['failed gate'] if x['void'] else [],
               'claim': None if x['claim'] == 'no claim' else x['claim']} for x in decided['sc4']['siglip2']]
    kept = c.hashed_order(['a', 'b'])
    cut = dict(threshold=.95, n=2, n_new=2, dropped=dict(exact=0, near=0, md5=0),
               sha256=c.keys_sha(kept), keys=kept)
    doc = dict(root=str(COMPARE_ROOT), complete=True, source_sha256='derive-source', unavailable=[],
               inputs={}, cutsets={'test': cut}, ranges={}, picks={}, cells=rows,
               sections=['cut', 'cells:siglip2'], scope='subset:cut,cells:siglip2', n_clu=2000, n_ret=10000, kmeans_threads=8)
    out = dict(status=status, cells=outcome_rows, verdicts={'siglip2': {'verdict': decided['verdicts']['siglip2']}},
               sc4={'siglip2': claims})
    result_bytes = {COMPARE_ROOT / 'stage2/cells_clu.json': lambda: json.dumps([x for x in rows if x['task'] == 'clu']).encode(),
                    COMPARE_ROOT / 'stage2/cells_ret.json': lambda: json.dumps([x for x in rows if x['task'] == 'ret']).encode(),
                    COMPARE_ROOT / 'stage2/outcomes.json': lambda: json.dumps(out).encode()}
    record_cut = {k: v for k, v in cut.items() if k != 'keys'}
    record_cut['sha256'] = c.keys_sha(sorted(kept))
    values = {COMPARE_OUT / 'derived.json': doc, COMPARE_ROOT / 'records/status.json': status,
              COMPARE_ROOT / 'records/cutsets.json': {'test': record_cut}}
    monkeypatch.setattr(c, 'source_identity', lambda: 'compare-source')
    monkeypatch.setattr(c, 'read_json', lambda p: values[Path(p)])
    monkeypatch.setattr(Path, 'exists', lambda p: p in values or p in result_bytes)
    monkeypatch.setattr(Path, 'read_bytes', lambda p: result_bytes[p]())
    monkeypatch.setattr(c, 'write_json', lambda *args: None)
    mock_compare_integrity(monkeypatch)
    return doc, out


def compare_fixture():
    return c.compare(root=COMPARE_ROOT, records=COMPARE_ROOT / 'records', out=COMPARE_OUT)


def test_compare_cut_digest_and_decided_status_layouts(monkeypatch):
    comparison_fixture(monkeypatch)
    result = compare_fixture()
    assert result['pass'] and result['failed'] == 0
    assert result['derive_source_sha256'] == 'derive-source'
    assert result['kmeans_threads'] == 8


def test_compare_below_null_and_voided_sc4_claims(monkeypatch):
    rows = [x for x in cells() if x['encoder'] == 'siglip2']
    next(x for x in rows if x['sc'] == 'SC4' and x['aspect'] == 'emotion' and x['task'] == 'clu')['lo'] = -.1
    status = c.good_status()
    status['gates']['i_teacher']['style'] = 'failed'
    _, out = comparison_fixture(monkeypatch, rows, status)
    below = next(x for x in out['sc4']['siglip2'] if x['aspect'] == 'emotion' and x['task'] == 'clu')
    voided = next(x for x in out['sc4']['siglip2'] if x['aspect'] == 'style' and x['task'] == 'clu')
    assert below['outcome'] == 'BELOW' and below['claim'] is None
    assert voided['status'] == 'inconclusive' and voided['void'] and voided['claim'] == 'inconclusive'
    assert compare_fixture()['pass']
    voided['claim'] = None
    result = compare_fixture()
    failures = [x for x in result['items'] if not x['pass']]
    assert len(failures) == 1 and failures[0]['name'].endswith('style.clu.claim')
    voided['claim'] = 'inconclusive'
    below.pop('claim')
    assert not compare_fixture()['pass'], 'missing claim must not be accepted as null'


def test_compare_rejects_duplicate_sc4_identities(monkeypatch):
    _, out = comparison_fixture(monkeypatch)
    out['sc4']['siglip2'].append(deepcopy(out['sc4']['siglip2'][0]))
    result = compare_fixture()
    assert not result['pass']
    assert any(x['name'].endswith('/unique_identities') and not x['pass'] for x in result['items'])


@pytest.mark.parametrize('claim', ['WIN', 'BELOW', 'inconclusive', 'better than', 'level with or better than', '', False, None])
def test_only_no_claim_maps_to_null(claim):
    assert c.compare_claim_value('no claim') is None
    assert c.compare_claim_value(claim) is not None


def matched_side(enc, seed, digest):
    suffix = ('percept/core/stage2/k40' if enc == 'qwen' else 'matched/' + c.MATCHED[enc])
    return dict(encoder=enc, seed=seed, view=c.DEFAULT_VIEW[enc],
                mapper='/frozen/root/' + suffix + f'/P3__s{seed}.npz', sha256=digest)


@pytest.mark.parametrize('enc,seed', [(e, s) for e in c.ENCODERS for s in c.SEEDS])
def test_matched_identity_all_twenty_exports(enc, seed):
    digest = 'a' * 64
    expected, actual = c.compare_matched_identity(enc, seed, matched_side(enc, seed, digest), digest, digest)
    assert expected == actual


@pytest.mark.parametrize('field,value', [('encoder', 'qwen'), ('seed', 2), ('seed', True),
    ('view', 'qwen_default'), ('mapper', '/matched/M8/P3__s1.npz'),
    ('mapper', '/matched/M2/P3__s0.npz'), ('sha256', 'b' * 64), ('input', None),
    ('input', 'b' * 64), ('npz', None)])
def test_matched_identity_rejects_misbound_sidecar_or_bytes(field, value):
    digest = 'a' * 64
    side, npz, input_hash = matched_side('siglip2', 1, digest), digest, digest
    if field == 'input':
        input_hash = value
    elif field == 'npz':
        npz = value
    else:
        side[field] = value
    expected, actual = c.compare_matched_identity('siglip2', 1, side, npz, input_hash)
    assert expected != actual


def test_code_provenance_matches_git_file_and_records_current_hash(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    data = b'committed source\n'
    digest = hashlib.sha256(data).hexdigest()
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=data, stderr=b'')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(c, 'file_sha', lambda _: 'current-file-hash')
    composite = hashlib.sha256((c.frozen_identity_with(data) + digest + 'current-file-hash').encode()).hexdigest()
    stale = hashlib.sha256((c.frozen_identity_with(Path(c.__file__).read_bytes()) + digest
                            + 'current-file-hash').encode()).hexdigest()
    assert not c.compare_code_provenance({'source_sha256': stale}, 'f093fb7')['matches']   # frozen part from disk
    result = c.compare_code_provenance({'source_sha256': composite}, 'f093fb7')
    assert result['matches'] and result['committed_source_sha256'] == composite
    assert not c.compare_code_provenance({'source_sha256': digest}, 'f093fb7')['matches']   # the bare file hash is not it
    assert result['current_source_sha256'] == 'current-file-hash'
    assert commands[0][-1] == 'f093fb7:src/test/20261102_mmart_test/mt_c13.py'
    assert not c.compare_code_provenance({'source_sha256': 'different'}, 'f093fb7')['matches']
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b'', stderr=b'bad commit'))
    result = c.compare_code_provenance({'source_sha256': composite}, 'missing')
    assert not result['matches'] and result['error'] == 'bad commit'


def test_frozen_identity_with_current_bytes_equals_the_frozen_identity():
    current = Path(c.__file__).read_bytes()
    assert c.frozen_identity_with(current) == c.frozen.frozen_identity.__wrapped__()
    assert c.frozen_identity_with(current + b'#') != c.frozen.frozen_identity.__wrapped__()


def test_record_refuses_failed_or_missing_code_identity():
    for report in ({'failed': 0}, {'failed': 1, 'code_provenance': {'matches': False}}):
        assert 'derive-commit' in c.record_refusal(full_doc(), report, cells())


@pytest.mark.parametrize('option,want', [([], 'f093fb7'), (['--derive-commit', 'abc1234'], 'abc1234')])
def test_compare_cli_derive_commit(monkeypatch, option, want):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['mt_c13.py', 'compare', *option])
    monkeypatch.setattr(c, 'compare', lambda *args, **kwargs: calls.append(kwargs) or {'pass': True})
    assert c.main() == 0
    assert calls == [{'derive_commit': want}]


def test_seedwise_crossfit_matches_frozen_arm_table(monkeypatch):
    rng = np.random.default_rng(42)
    xs = {('emotion', s): c.unit(rng.normal(size=(48, 8)).astype(np.float32)) for s in c.SEEDS}
    labels = {'emotion': np.arange(48, dtype=np.int64) % 3}
    ros = c.rt_readout.menu(8)
    pieces = []
    for s in c.SEEDS:
        rows = c.rt_readout.crossfit_all({('emotion', None): xs['emotion', s]}, labels, ros, workers=1)
        pieces.append([rows[ro]['emotion'] for ro in ros])
    means = np.mean(pieces, axis=0)
    expected, _ = c.ac_select.arm_table(xs, labels, {'emotion': .01}, workers=1)
    assert all(means[i] == expected[c.rt_readout.name(ro)]['map']['emotion'] for i, ro in enumerate(ros))


def test_output_guard():
    assert c.check_output(c.SMOKE_OUT / 'tests') == c.SMOKE_OUT / 'tests'
    with pytest.raises(AssertionError):
        c.check_output('/tmp/c13-test')
