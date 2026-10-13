"""The per-read agreement file carries what the apply-rule step (ticket 19) reads: phase, smoke, pass_file,
held_pass_sha256, rederived_counts, next to ticket 25's own agreement/compared/failed/items. Small synthetic passes."""
import copy
import hashlib
import json
import sys
import numpy as np
import pytest
import rd9_compare as C
import rd9_inputs
import rd9_stats as S

ARMS = ('aff_fused', 'aff_cf', 'cosine', 'rca', 'B', 'B0')
METRICS = ('r1', 'gain', 'other', 'swap', 'strict')


def make_arrays(a28=False):    # a28 True: level 1 passes; a28 'nogo': level 1 fails (P4 count is high)
    n = 40
    rng = np.random.default_rng(3)
    arrays = {'cl': np.repeat(np.arange(10), 4), 'pair_index': np.tile(np.arange(4), 10), 'seed_index': np.zeros(n, int)}
    for arm in ARMS:
        for m in METRICS:
            arrays[f'{arm}__{m}'] = np.zeros(n)
    arrays['aff_fused__r1'] = rng.integers(0, 3, n) / 4
    arrays['aff_fused__gain'] = rng.integers(0, 3, n) / 4
    for arm, hi in (('cosine', 2), ('rca', 3), ('B', 4), ('B0', 4 if a28 == 'nogo' else 1 if a28 else 3), ('aff_cf', 3)):    # other counts per check
        arrays[f'{arm}__r1'] = rng.integers(0, hi, n) / 4
    arrays['rca__gain'] = rng.integers(0, 2, n) / 4
    return arrays


def make_stats(a28=False):
    stats = S.pass_record(make_arrays(a28), a28=bool(a28))
    stats.update(seeds=[55, 56, 57] if a28 else [52, 53, 54], episodes_sha256={'x': 'a' * 64})
    return stats


def write(path, value):
    path.write_text(json.dumps(rd9_inputs.jsonable(value)))
    return path


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(rd9_inputs, 'RD', tmp_path)   # outputs are allowed only below RD
    return tmp_path


def stage(work, mode='held', a28=False, read='H6', name='held_pass_H6.json', tweak=None):
    stats = make_stats(a28)
    record = dict(read=read, backbone='clip_b32', mode='read', stats=stats)
    if a28:
        record['episode_records'] = {str(s): {pair: {'set_aside': []} for pair in
                                    ('emotion__style', 'emotion__genre', 'style__genre')} for s in stats['seeds']}
    own = write(work / 'own.json', record)
    arrays = make_arrays(a28)
    np.savez(work / 'own.npz', **arrays)
    np.savez(work / 'rep.npz', **arrays)
    reported = copy.deepcopy(stats)
    if a28:
        reported.update(set_aside=[], flagged_languages=['Emakhuwa', 'IsiNdebele'],
                        near_duplicates=dict(cos_099=3, cos_098=7))
        write(work / 'language_flags_a28.json', dict(flagged=['Emakhuwa', 'IsiNdebele'], thresholds={},
                                                   english_median_words=10, languages={}))
        write(work / 'near_duplicates_a28.json', dict(cos_099=3, cos_098=7, train_side=912,
                                                    read_paintings=746, pairs_099=[]))
    if mode is not None: reported['mode'] = mode
    if tweak: tweak(reported)
    rep = write(work / name, reported)
    return stats, own, rep


def write_pins(work, flags=None, name='pins_a28.json'):
    """A pins file naming the flags file the way pins_a28.json does (a path under results/ and its SHA-256)."""
    flags = flags or work / 'language_flags_a28.json'
    digest = hashlib.sha256(flags.read_bytes()).hexdigest() if flags.exists() else '0' * 64
    return write(work / name, dict(files={'results/language_flags_a28.json': dict(
        path=str(work / 'results/language_flags_a28.json'), sha256=digest)}))


def run(work, own, rep, monkeypatch, out='agr.json', extra=(), input_paths=True, pins=True):
    # The pins file is built at call time: several tests rewrite the flags file after stage().
    if pins and '--a28-pins' not in extra and (work / 'language_flags_a28.json').exists():
        extra = [*extra, '--a28-pins', str(write_pins(work))]
    argv = ['rd9_compare.py', '--derived', str(own), '--derived-arrays', str(work / 'own.npz'), '--reported', str(rep),
            '--reported-arrays', str(work / 'rep.npz'), '--out', str(work / out)]
    if input_paths:
        argv += ['--language-flags', str(work / 'language_flags_a28.json'),
                 '--near-duplicates', str(work / 'near_duplicates_a28.json')]
    argv += list(extra)
    monkeypatch.setattr(sys, 'argv', argv)
    C.main()
    return json.loads((work / out).read_text())


def test_schema_and_values(work, monkeypatch):
    stats, own, rep = stage(work)
    got = run(work, own, rep, monkeypatch)
    assert set(got) == {'agreement', 'compared', 'failed', 'items', 'independent_identity', 'read', 'backbone',
                        'phase', 'smoke', 'pass_file', 'held_pass_sha256', 'rederived_counts'}
    assert got['agreement'] is True and got['failed'] == 0 and got['compared'] == len(got['items']) >= 1
    assert got['phase'] == 2 and type(got['phase']) is int
    assert got['smoke'] is False
    assert got['read'] == 'H6'
    assert got['pass_file'] == 'held_pass_H6.json'
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()
    assert list(got['rederived_counts']) == ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
    assert got['rederived_counts'] == {j: stats['checks'][j]['n'] for j in S.CHECK_IDS}
    assert all(type(v) is int for v in got['rederived_counts'].values())


def test_smoke_comes_from_the_pass_files_mode(work, monkeypatch):
    _, own, rep = stage(work, mode='smoke')
    assert run(work, own, rep, monkeypatch)['smoke'] is True


def test_two_level_read_has_the_same_keys(work, monkeypatch):
    stats, own, rep = stage(work, a28=True, read='H8', name='held_pass_H8_fix1.json')
    got = run(work, own, rep, monkeypatch)
    assert got['pass_file'] == 'held_pass_H8_fix1.json' and got['read'] == 'H8'
    assert got['rederived_counts'] == {j: stats['checks'][j]['n'] for j in S.CHECK_IDS}


def test_hash_is_of_the_file_bytes(work, monkeypatch):
    _, own, rep = stage(work)
    rep.write_bytes(rep.read_bytes() + b'\n  ')           # the same JSON, other bytes
    got = run(work, own, rep, monkeypatch)
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()


def test_counts_are_the_independent_passes_not_the_reported(work, monkeypatch):
    stats, own, rep = stage(work, tweak=lambda r: r['checks']['P3'].update(n=r['checks']['P3']['n'] + 1))
    with pytest.raises(SystemExit) as e:
        run(work, own, rep, monkeypatch)
    assert e.value.code == 1
    got = json.loads((work / 'agr.json').read_text())       # a disagreement is still written, with provenance
    assert got['agreement'] is False and got['failed'] >= 1
    assert got['rederived_counts']['P3'] == stats['checks']['P3']['n']
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()


@pytest.mark.parametrize('mode', [None, 'held ', 'Held', 'dev', True, 1])
def test_missing_or_unknown_mode_crashes(work, monkeypatch, mode):
    _, own, rep = stage(work, mode=None, tweak=None if mode is None else (lambda r: r.update(mode=mode)))
    with pytest.raises(ValueError, match='mode'):
        run(work, own, rep, monkeypatch)
    assert not (work / 'agr.json').exists()


def test_mode_inside_a_stats_wrapper_is_read(work, monkeypatch):
    stats, own, rep = stage(work, mode=None)
    write(rep, dict(stats=dict(json.loads(rep.read_text()), mode='smoke')))
    assert run(work, own, rep, monkeypatch)['smoke'] is True


def test_missing_reported_inputs_crash(work, monkeypatch):
    _, own, rep = stage(work)
    with pytest.raises(AssertionError, match='runner scalars AND arrays'):
        monkeypatch.setattr(sys, 'argv', ['x', '--derived', str(own), '--derived-arrays', str(work / 'own.npz'),
                                          '--out', str(work / 'agr.json')])
        C.main()
    with pytest.raises(FileNotFoundError):
        run(work, own, work / 'no_such_pass.json', monkeypatch)
    assert not (work / 'agr.json').exists()


@pytest.mark.parametrize('bad', ['drop', 'float', 'bool', 'extra'])
def test_derived_counts_must_be_seven_integers(work, monkeypatch, bad):
    stats, own, rep = stage(work)
    record = json.loads(own.read_text())
    n = record['stats']['checks']
    if bad == 'drop': del n['P7']
    elif bad == 'float': n['P2']['n'] = 3.0
    elif bad == 'bool': n['P2']['n'] = True
    else: n['P8'] = dict(n['P1'])
    write(own, record)
    with pytest.raises((ValueError, KeyError, AssertionError)):
        run(work, own, rep, monkeypatch)
    assert not (work / 'agr.json').exists()


TWO_LEVEL = dict(a28=True, read='H8', name='held_pass_H8.json')
NO_GO = dict(a28='nogo', read='H8', name='held_pass_H8.json')


def test_two_level_no_go_reaches_agreement(work, monkeypatch):
    stats, own, rep = stage(work, **NO_GO)
    assert stats['decision']['outcome'] == 'NO-GO'                 # level 1 fails
    assert stats['holm_order']['level1'] == ['P6', 'P4'] and sorted(stats['holm_order']['level2']) == ['P1', 'P2', 'P3', 'P5', 'P7']
    for j in ('P1', 'P2', 'P3', 'P5', 'P7'):                       # level 2 exists although level 1 failed, as the runner writes it
        c = stats['checks'][j]
        assert c['holm_k'] in range(1, 6) and c['level_two_sided'] >= 0.95 and len(c['ci_holm']) == 2
        assert isinstance(c['passes'], bool) and isinstance(c['near_boundary'], bool)
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True and got['failed'] == 0
    keys = {i['key'] for i in got['items']}
    assert 'holm_order.level1.0' in keys and 'holm_order.level2.4' in keys and 'checks.P1.holm_k' in keys


def test_two_level_pass_still_agrees(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    assert run(work, own, rep, monkeypatch)['agreement'] is True


def test_runner_level_two_order_is_read_as_a_dict(work, monkeypatch):
    _, own, rep = stage(work, **NO_GO, tweak=lambda r: r['holm_order'].update(level2=r['holm_order']['level2'][::-1]))
    with pytest.raises(SystemExit):
        run(work, own, rep, monkeypatch)
    assert not json.loads((work / 'agr.json').read_text())['agreement']


def test_missing_key_is_a_difference(work, monkeypatch):
    _, own, rep = stage(work, **NO_GO, tweak=lambda r: r['checks']['P1'].pop('holm_k'))
    with pytest.raises(SystemExit):
        run(work, own, rep, monkeypatch)
    got = json.loads((work / 'agr.json').read_text())
    assert got['agreement'] is False
    assert [i['key'] for i in got['items'] if not i['equal']] == ['checks.P1.holm_k']


def test_none_on_one_side_only_is_a_difference(work, monkeypatch):
    _, own, rep = stage(work, **NO_GO, tweak=lambda r: r['checks']['P1'].update(holm_k=None))
    with pytest.raises(SystemExit):
        run(work, own, rep, monkeypatch)
    got = json.loads((work / 'agr.json').read_text())
    assert [i['key'] for i in got['items'] if not i['equal']] == ['checks.P1.holm_k']


@pytest.mark.parametrize('ours,theirs,equal', [(None, None, True), (None, 3, False), (3, None, False)])
def test_value_none_rule(ours, theirs, equal):
    c = C.Comparison()
    c.value('k', ours, theirs)
    assert c.items[0]['equal'] is equal


def test_value_missing_key_is_a_difference_even_when_ours_is_none():
    c = C.Comparison()
    c.value('o', {'a': None, 'b': 1}, {'b': 1})
    assert {i['key']: i['equal'] for i in c.items} == {'o.a': False, 'o.b': True}


def set_aside_example(own, rep):
    record = json.loads(own.read_text())
    record['episode_records']['55']['emotion__style']['set_aside'] = [
        dict(language='Emakhuwa', reason='values-subset', seed=55003),
        dict(language='IsiNdebele', reason='failed-draw', seed=55008)]
    record['episode_records']['56']['style__genre']['set_aside'] = [
        dict(language='Emakhuwa', reason='failed-draw', seed=56003)]
    write(own, record)
    reported = json.loads(rep.read_text())
    reported['set_aside'] = [dict(language='Emakhuwa', pair='style__genre', seed=56),
                             dict(language='IsiNdebele', pair='emotion__style', seed=55),
                             dict(language='Emakhuwa', pair='emotion__style', seed=55)]
    write(rep, reported)


def disagreement(work, own, rep, monkeypatch):
    with pytest.raises(SystemExit) as error:
        run(work, own, rep, monkeypatch)
    assert error.value.code == 1
    got = json.loads((work / 'agr.json').read_text())
    assert got['agreement'] is False
    return [i for i in got['items'] if not i['equal']]


def test_set_aside_compares_language_pair_and_read_seed_as_a_set(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    set_aside_example(own, rep)
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True
    assert any(i['key'].startswith('set_aside') for i in got['items'])


@pytest.mark.parametrize('field,value', [('language', 'IsiXhosa'), ('pair', 'emotion__genre'), ('seed', 57)])
def test_set_aside_difference_fails_agreement(work, monkeypatch, field, value):
    _, own, rep = stage(work, **TWO_LEVEL)
    set_aside_example(own, rep)
    reported = json.loads(rep.read_text())
    reported['set_aside'][0][field] = value
    write(rep, reported)
    failed = disagreement(work, own, rep, monkeypatch)
    assert all(i['key'].startswith('set_aside') for i in failed)


@pytest.mark.parametrize('missing', ['reported', 'records', 'seed', 'pair', 'derived_set_aside',
                                    'language', 'reported_pair', 'reported_seed'])
def test_missing_set_aside_fails_even_when_empty(work, monkeypatch, missing):
    _, own, rep = stage(work, **TWO_LEVEL)
    record, reported = json.loads(own.read_text()), json.loads(rep.read_text())
    if missing == 'reported': del reported['set_aside']
    elif missing == 'records': del record['episode_records']
    elif missing == 'seed': del record['episode_records']['56']
    elif missing == 'pair': del record['episode_records']['56']['emotion__genre']
    elif missing == 'derived_set_aside': del record['episode_records']['56']['emotion__genre']['set_aside']
    else:
        entry = dict(language='Emakhuwa', pair='emotion__style', seed=55)
        del entry[{'language': 'language', 'reported_pair': 'pair', 'reported_seed': 'seed'}[missing]]
        reported['set_aside'] = [entry]
    write(own, record)
    write(rep, reported)
    assert all(i['key'].startswith('set_aside') for i in disagreement(work, own, rep, monkeypatch))


@pytest.mark.parametrize('read', ['H6', 'H7'])
def test_artelingo_reads_skip_a28_claim_fields(work, monkeypatch, read):
    _, own, rep = stage(work, read=read, name=f'held_pass_{read}.json')
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True
    assert not any(i['key'].startswith(('set_aside', 'flagged_languages', 'near_duplicates')) for i in got['items'])


def test_h8_selection_replay_skips_a28_claim_fields(work, monkeypatch):
    _, own, rep = stage(work, read='H8', name='seed42_pass_counts.json')
    record = json.loads(own.read_text())
    record['mode'] = 'selection'
    write(own, record)
    assert run(work, own, rep, monkeypatch)['agreement'] is True


@pytest.mark.parametrize('ours,theirs', [(C.MISSING, []), (C.MISSING, C.MISSING), ([], C.MISSING)])
def test_missing_either_side_is_a_serializable_difference(ours, theirs):
    c = C.Comparison()
    c.value('claim', ours, theirs)
    assert c.record()['agreement'] is False
    json.dumps(c.record())


@pytest.mark.parametrize('side', ['reported', 'input', 'both'])
def test_flagged_languages_compare_as_sorted_lists(work, monkeypatch, side):
    _, own, rep = stage(work, **TWO_LEVEL,
                        tweak=(lambda r: r.update(flagged_languages=['IsiNdebele', 'Emakhuwa']))
                        if side in ('reported', 'both') else None)
    flags = json.loads((work / 'language_flags_a28.json').read_text())
    if side in ('input', 'both'): flags['flagged'].reverse()
    write(work / 'language_flags_a28.json', flags)
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True
    assert any(i['key'].startswith('flagged_languages') for i in got['items'])


@pytest.mark.parametrize('side', ['reported', 'input'])
def test_flagged_language_difference_fails_agreement(work, monkeypatch, side):
    _, own, rep = stage(work, **TWO_LEVEL)
    path, key = (rep, 'flagged_languages') if side == 'reported' else (work / 'language_flags_a28.json', 'flagged')
    record = json.loads(path.read_text())
    record[key] = ['Emakhuwa']
    write(path, record)
    assert all(i['key'].startswith('flagged_languages') for i in disagreement(work, own, rep, monkeypatch))


@pytest.mark.parametrize('side', ['reported', 'input', 'both'])
def test_missing_flagged_languages_fails_agreement(work, monkeypatch, side):
    _, own, rep = stage(work, **TWO_LEVEL)
    for which, path, key in [('reported', rep, 'flagged_languages'), ('input', work / 'language_flags_a28.json', 'flagged')]:
        if side in (which, 'both'):
            record = json.loads(path.read_text())
            del record[key]
            write(path, record)
    assert all(i['key'].startswith('flagged_languages') for i in disagreement(work, own, rep, monkeypatch))


def test_no_flagged_languages_is_an_explicit_agreement(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL, tweak=lambda r: r.update(flagged_languages=[]))
    write(work / 'language_flags_a28.json', dict(flagged=[]))
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True
    assert any(i['key'] == 'flagged_languages' and i['equal'] for i in got['items'])


def test_near_duplicate_counts_are_compared(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    got = run(work, own, rep, monkeypatch)
    assert got['agreement'] is True
    assert {i['key']: i['ours'] for i in got['items'] if i['key'].startswith('near_duplicates')} == {
        'near_duplicates.cos_099': 3, 'near_duplicates.cos_098': 7}


@pytest.mark.parametrize('field', ['cos_099', 'cos_098'])
@pytest.mark.parametrize('side', ['reported', 'input'])
def test_near_duplicate_difference_fails_agreement(work, monkeypatch, field, side):
    _, own, rep = stage(work, **TWO_LEVEL)
    path = rep if side == 'reported' else work / 'near_duplicates_a28.json'
    record = json.loads(path.read_text())
    counts = record['near_duplicates'] if side == 'reported' else record
    counts[field] += 1
    write(path, record)
    assert [i['key'] for i in disagreement(work, own, rep, monkeypatch)] == ['near_duplicates.' + field]


@pytest.mark.parametrize('field', ['cos_099', 'cos_098'])
@pytest.mark.parametrize('side', ['reported', 'input', 'both'])
def test_missing_near_duplicate_count_fails_agreement(work, monkeypatch, field, side):
    _, own, rep = stage(work, **TWO_LEVEL)
    for which, path in [('reported', rep), ('input', work / 'near_duplicates_a28.json')]:
        if side in (which, 'both'):
            record = json.loads(path.read_text())
            counts = record['near_duplicates'] if which == 'reported' else record
            del counts[field]
            write(path, record)
    assert [i['key'] for i in disagreement(work, own, rep, monkeypatch)] == ['near_duplicates.' + field]


def test_missing_near_duplicates_object_fails_agreement(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL, tweak=lambda r: r.pop('near_duplicates'))
    assert all(i['key'].startswith('near_duplicates') for i in disagreement(work, own, rep, monkeypatch))


@pytest.mark.parametrize('field,value', [('train_side', 911), ('read_paintings', 745),
                                        ('train_side', None), ('read_paintings', None)])
def test_near_duplicate_population_is_asserted(work, monkeypatch, field, value):
    _, own, rep = stage(work, **TWO_LEVEL)
    path = work / 'near_duplicates_a28.json'
    record = json.loads(path.read_text())
    if value is None: del record[field]
    else: record[field] = value
    write(path, record)
    with pytest.raises(AssertionError, match=field):
        run(work, own, rep, monkeypatch)
    assert not (work / 'agr.json').exists()


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
def test_claim_inputs_default_to_the_round_results(work, monkeypatch, read):
    _, own, rep = stage(work, a28=True, read=read, name=f'held_pass_{read}.json')
    results = work / 'results'
    results.mkdir()
    for name in ('language_flags_a28.json', 'near_duplicates_a28.json'):
        write(results / name, json.loads((work / name).read_text()))
    monkeypatch.setattr(C, 'HERE', work)
    assert run(work, own, rep, monkeypatch, input_paths=False)['agreement'] is True


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
@pytest.mark.parametrize('field', ['set_aside', 'flagged_languages', 'near_duplicates'])
def test_every_a28_read_requires_each_claim_field(work, monkeypatch, read, field):
    _, own, rep = stage(work, a28=True, read=read, name=f'held_pass_{read}.json', tweak=lambda r: r.pop(field))
    assert all(i['key'].startswith(field) for i in disagreement(work, own, rep, monkeypatch))


def test_empty_derived_set_aside_against_a_reported_entry_is_a_difference(work, monkeypatch):
    # Real data: the builders most likely set nothing aside, so the live comparison is [] against the runner's list.
    _, own, rep = stage(work, **TWO_LEVEL,
                        tweak=lambda r: r.update(set_aside=[dict(language='IsiXhosa', pair='emotion__genre', seed=56)]))
    failed = disagreement(work, own, rep, monkeypatch)
    assert [i['key'] for i in failed] == ['set_aside']
    assert failed[0]['reason'] == 'sequence length/type'


def test_reported_empty_set_aside_against_a_derived_entry_is_a_difference(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    record = json.loads(own.read_text())
    record['episode_records']['56']['emotion__genre']['set_aside'] = [dict(language='IsiXhosa', reason='failed-draw', seed=56004)]
    write(own, record)
    assert [i['key'] for i in disagreement(work, own, rep, monkeypatch)] == ['set_aside']


def test_empty_flagged_list_in_the_input_against_a_reported_language_is_a_difference(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL, tweak=lambda r: r.update(flagged_languages=['Emakhuwa']))
    write(work / 'language_flags_a28.json', dict(flagged=[]))
    failed = disagreement(work, own, rep, monkeypatch)
    assert [i['key'] for i in failed] == ['flagged_languages']
    assert failed[0]['reason'] == 'sequence length/type'


def test_reported_empty_flagged_list_against_an_input_language_is_a_difference(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL, tweak=lambda r: r.update(flagged_languages=[]))
    assert [i['key'] for i in disagreement(work, own, rep, monkeypatch)] == ['flagged_languages']


def test_agreement_records_the_sha256_of_both_claim_inputs(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    got = run(work, own, rep, monkeypatch)
    for field, name in (('language_flags_sha256', 'language_flags_a28.json'),
                        ('near_duplicates_sha256', 'near_duplicates_a28.json')):
        assert got[field] == hashlib.sha256((work / name).read_bytes()).hexdigest()


@pytest.mark.parametrize('read', ['H6', 'H7'])
def test_reads_without_claim_fields_record_no_claim_input_sha(work, monkeypatch, read):
    _, own, rep = stage(work, read=read, name=f'held_pass_{read}.json')
    got = run(work, own, rep, monkeypatch, input_paths=False)
    assert 'language_flags_sha256' not in got and 'near_duplicates_sha256' not in got


def test_flags_file_must_equal_its_pins_entry(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    pins = write_pins(work)
    # a stale copy: same shape, different bytes; the runner and the comparer would both read it and agree
    write(work / 'language_flags_a28.json', dict(flagged=['Emakhuwa', 'IsiNdebele'], thresholds={'stale': 1}))
    with pytest.raises(AssertionError, match='language_flags_a28.json'):
        run(work, own, rep, monkeypatch, extra=['--a28-pins', str(pins)])
    assert not (work / 'agr.json').exists()


def test_flags_file_without_a_pins_entry_is_refused(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    pins = write(work / 'pins_other.json', dict(files={'results/other.json': dict(path='/x/other.json', sha256='0' * 64)}))
    with pytest.raises(AssertionError, match='language_flags_a28.json'):
        run(work, own, rep, monkeypatch, extra=['--a28-pins', str(pins)])
    assert not (work / 'agr.json').exists()


def test_missing_pins_file_crashes(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    with pytest.raises(FileNotFoundError):
        run(work, own, rep, monkeypatch, extra=['--a28-pins', str(work / 'nope.json')])
    assert not (work / 'agr.json').exists()


def test_near_duplicates_file_is_not_pinned_only_recorded(work, monkeypatch):
    _, own, rep = stage(work, **TWO_LEVEL)
    pins = json.loads(write_pins(work).read_text())
    assert not any('near_duplicates' in key for key in pins['files'])
    assert run(work, own, rep, monkeypatch)['agreement'] is True


def test_pinned_bytes_are_the_bytes_that_are_parsed(work):
    path = write(work / 'x.json', dict(a=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert rd9_inputs.read_pinned(path, {'results/x.json': digest}) == path.read_bytes()
    with pytest.raises(AssertionError, match='x.json'):
        rd9_inputs.read_pinned(path, {'results/x.json': '1' * 64})
    with pytest.raises(AssertionError, match='x.json'):
        rd9_inputs.read_pinned(path, {'results/y.json': digest})
    with pytest.raises(AssertionError, match='x.json'):
        rd9_inputs.read_pinned(path, {'a/x.json': digest, 'b/x.json': digest})
