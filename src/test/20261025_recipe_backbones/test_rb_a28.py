"""Tests of rb_a28 on small fixtures; the real-data checks run with RB_SLOW=1."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import rb_paths  # noqa: F401  (sys.path)
import rb_a28 as a28

slow = pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='real files: set RB_SLOW=1')

EMO = ['a', 'b', 'c', 'something else']       # fixture names; code = sorted position, "something else" last here
STYLE = ['s0', 's1']
GENRE = ['g0', 'g1', 'g2']


def fixture(rows):
    """rows: (painting, language, emotion, style, genre, held) per caption, in CSV order, all in the read set."""
    csv = pd.DataFrame(rows, columns=['painting', 'language', 'emotion', 'art_style', 'genre', 'held'])
    read = pd.DataFrame(dict(row_id=np.arange(len(csv)), painting=csv.painting, language=csv.language,
                             emotion=0, style=0, genre=0, held=csv.held, dropped=False))   # parquet codes are junk
    return read, csv.drop(columns='held')


def labels(rows, names):
    read, csv = fixture(rows)
    return a28.build_labels(a28.join_rows(read, csv), names, EMO, STYLE, GENRE)


def test_emotion_is_the_plurality_with_other_merged_into_something_else_and_ties_to_the_smallest_code():
    rows = [('p1', 'en', 'b', 's0', 'g0', True)] * 2 + [('p1', 'fr', 'c', 's0', 'g0', True)] \
        + [('p2', 'en', 'other', 's1', 'g2', False)] * 2 + [('p2', 'fr', 'a', 's1', 'g2', False)] \
        + [('p3', 'en', 'c', 's1', 'g1', False), ('p3', 'fr', 'a', 's1', 'g1', False)] \
        + [('p4', 'en', 'something else', 's0', 'g1', True), ('p4', 'fr', 'other', 's0', 'g1', True),
           ('p4', 'de', 'b', 's0', 'g1', True)]
    out = labels(rows, ['p1', 'p2', 'p3', 'p4'])
    # p1 b; p2 other -> something else (code 3) beats a; p3 tie a/c -> a (0); p4 something else x2 beats b
    assert out['emotion'].tolist() == [1, 3, 0, 3]
    assert out['ties'].tolist() == [False, False, True, False]
    assert out['style'].tolist() == [0, 1, 1, 0] and out['genre'].tolist() == [0, 2, 1, 1]
    assert out['held'].tolist() == [True, False, False, True]
    assert all(out[k].dtype == np.int64 for k in ('emotion', 'style', 'genre'))
    assert out['n_ties'] == 1 and out['n_unique'] == 3 and out['n_captions'].tolist() == [3, 3, 2, 3]


def test_no_label_is_masked_and_unknown_names_fail():
    rows = [('p1', 'en', 'a', 's0', 'g0', True), ('p2', 'en', 'b', 's1', 'g1', True)]
    out = labels(rows, ['p1', 'p2'])
    assert all((out[k] >= 0).all() for k in ('emotion', 'style', 'genre'))
    for column, value in (('emotion', 'joy'), ('art_style', 'cubist'), ('genre', 'nude_painting')):
        read, csv = fixture(rows)
        csv.loc[1, column] = value
        with pytest.raises(AssertionError):
            a28.build_labels(a28.join_rows(read, csv), ['p1', 'p2'], EMO, STYLE, GENRE)


def test_style_and_genre_must_be_constant_per_painting():
    for column in ('art_style', 'genre'):
        read, csv = fixture([('p1', 'en', 'a', 's0', 'g0', True), ('p1', 'fr', 'a', 's0', 'g0', True)])
        csv.loc[1, column] = 's1' if column == 'art_style' else 'g1'
        with pytest.raises(AssertionError, match='constant'):
            a28.build_labels(a28.join_rows(read, csv), ['p1'], EMO, STYLE, GENRE)


def test_the_join_is_by_row_id_not_by_position_and_checks_painting_and_language():
    rows = [('p1', 'en', 'a', 's0', 'g0', True), ('p2', 'fr', 'b', 's1', 'g1', False),
            ('p3', 'de', 'c', 's0', 'g2', True)]
    read, csv = fixture(rows)
    read = read.iloc[[2, 0]].reset_index(drop=True)       # the read set keeps rows 2 and 0 only, in that order
    joined = a28.join_rows(read, csv)
    assert joined.painting.tolist() == ['p3', 'p1'] and joined.emotion.tolist() == ['c', 'a']
    bad = read.copy()
    bad.loc[0, 'painting'] = 'p1'
    with pytest.raises(AssertionError, match='painting'):
        a28.join_rows(bad, csv)
    bad = read.copy()
    bad.loc[0, 'language'] = 'en'
    with pytest.raises(AssertionError, match='language'):
        a28.join_rows(bad, csv)
    bad = read.copy()
    bad.loc[0, 'row_id'] = 99
    with pytest.raises(AssertionError):
        a28.join_rows(bad, csv)
    bad = read.copy()
    bad.loc[1, 'row_id'] = 2
    with pytest.raises(AssertionError):
        a28.join_rows(bad, csv)


def test_held_must_be_constant_per_painting():
    read, csv = fixture([('p1', 'en', 'a', 's0', 'g0', True), ('p1', 'fr', 'a', 's0', 'g0', True)])
    read.loc[1, 'held'] = False
    with pytest.raises(AssertionError, match='constant'):
        a28.build_labels(a28.join_rows(read, csv), ['p1'], EMO, STYLE, GENRE)


def test_the_painting_list_must_be_the_read_sets_paintings():
    rows = [('p1', 'en', 'a', 's0', 'g0', True), ('p2', 'en', 'b', 's1', 'g1', True)]
    read, csv = fixture(rows)
    joined = a28.join_rows(read, csv)
    for names in (['p1'], ['p2', 'p1'], ['p1', 'p2', 'p3']):
        with pytest.raises(AssertionError):
            a28.build_labels(joined, names, EMO, STYLE, GENRE)


def test_culture_labels_are_the_plurality_of_one_languages_captions_on_paintings_it_has():
    rows = [('p1', 'en', 'a', 's0', 'g0', True), ('p1', 'en', 'a', 's0', 'g0', True), ('p1', 'en', 'b', 's0', 'g0', True),
            ('p1', 'fr', 'b', 's0', 'g0', True),
            ('p2', 'en', 'c', 's1', 'g1', True), ('p2', 'en', 'b', 's1', 'g1', True),   # tie b/c -> b (1)
            ('p3', 'fr', 'other', 's1', 'g1', True), ('p3', 'fr', 'c', 's1', 'g1', True), ('p3', 'fr', 'c', 's1', 'g1', True)]
    read, csv = fixture(rows)
    joined = a28.join_rows(read, csv)
    pooled = a28.build_labels(joined, ['p1', 'p2', 'p3'], EMO, STYLE, GENRE)
    cul = a28.culture_labels(joined, ['p1', 'p2', 'p3'], EMO)
    assert sorted(cul) == ['en', 'fr']
    assert cul['en']['idx'].tolist() == [0, 1] and cul['en']['emotion'].tolist() == [0, 1]
    assert cul['fr']['idx'].tolist() == [0, 2] and cul['fr']['emotion'].tolist() == [1, 2]
    assert cul['en']['n_captions'].tolist() == [3, 2] and cul['fr']['n_captions'].tolist() == [1, 3]
    assert cul['en']['ties'].tolist() == [False, True]
    # agreement with the pooled label: share of its paintings whose language label equals the pooled one
    assert pooled['emotion'].tolist() == [0, 1, 2]
    assert cul['en']['agree'] == pytest.approx(1.0)       # en p1 a == a, p2 b == b (the pooled tie also goes to b)
    assert cul['fr']['agree'] == pytest.approx(0.5)       # fr p1 b != a, p3 c == c


# --- the reading -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('d, lower, expected', [
    (0.05, 0.01, 'A-HOLDS'), (0.0001, 0.00001, 'A-HOLDS'), (0.05, 0.0, 'A-QUAL'), (0.05, -0.01, 'A-QUAL'),
    (1e-9, -1e-9, 'A-QUAL'), (0.0, -0.1, 'A-FAILS'), (-0.02, -0.05, 'A-FAILS'), (0.0, 0.0, 'A-FAILS')])
def test_outcomes_at_the_thresholds(d, lower, expected):
    assert a28.outcome(d, lower) == expected


def test_outcome_rejects_nonfinite_values():
    for d, lower in ((float('nan'), 0.1), (0.1, float('inf'))):
        with pytest.raises(AssertionError):
            a28.outcome(d, lower)


# --- reading outputs ---------------------------------------------------------------------------------------------

N = 40
SHA = 'LIST746'


def unit32(n, w, seed):
    x = np.random.default_rng(seed).normal(size=(n, w)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def sidecar(path, arm, seed, aspect, scale='sample', list_sha256=SHA, override=None):
    """The index's sidecar, written from the saved array (independent of rb_a28)."""
    import hashlib
    with np.load(path) as z:
        key, = z.files
        x = z[key]
    meta = dict(list_sha256=list_sha256, rows=int(x.shape[0]), width=int(x.shape[1]), dtype=str(x.dtype),
                sha256=hashlib.sha256(x.tobytes()).hexdigest(), scale=scale, arm=arm, seed=seed, aspect=aspect)
    meta.update(override or {})
    path.with_suffix('.json').write_text(__import__('json').dumps(meta))


def put(outputs, arm, seed, x, aspect='emotion', scale='sample', **kw):
    """Write one a28 output where rb_a28 expects it; returns the path."""
    if arm[0] == 'K':
        path = outputs / f'part2/{scale}/a28/{arm}/{aspect}__s{seed}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, img=x.astype(np.float16))
        sidecar(path, arm, seed, aspect, scale, **kw)
    else:
        path = outputs / f'part2/{scale}/a28/{arm}/{"P2" if arm == "B6" else "P3"}__s{seed}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, prob=x.astype(np.float32))
        sidecar(path, arm, seed, 'percept', scale, **kw)
    return path


def prob(seed):
    return np.random.default_rng(seed).random((N, 40)).astype(np.float32)


@pytest.mark.parametrize('arm', ['K1', 'K2', 'B1', 'B6', 'M2'])
def test_outputs_are_read_with_their_own_sidecar_check(tmp_path, arm):
    x = unit32(N, 2048, 1) if arm[0] == 'K' else prob(1)
    path = put(tmp_path, arm, 2, x)
    aspect = 'emotion' if arm[0] == 'K' else 'percept'
    got = a28.read_output(path, arm, 'sample', 2, aspect, SHA, N)
    np.testing.assert_array_equal(got, x.astype(np.float32))
    assert got.dtype == np.float32
    for key, value in (('list_sha256', 'OTHER'), ('rows', N + 1), ('width', 39), ('dtype', 'float64'),
                       ('sha256', '0' * 64), ('scale', 'full'), ('arm', 'B2'), ('seed', 3), ('aspect', 'style')):
        put(tmp_path, arm, 2, x, override={key: value})
        with pytest.raises(AssertionError, match=key):
            a28.read_output(path, arm, 'sample', 2, aspect, SHA, N)
    path.with_suffix('.json').unlink()
    with pytest.raises(AssertionError, match='sidecar'):
        a28.read_output(path, arm, 'sample', 2, aspect, SHA, N)


def test_a_ct_style_held_sidecar_does_not_pass_for_b1(tmp_path):
    path = put(tmp_path, 'B1', 0, prob(1))
    path.with_suffix('.json').write_text('{"held_sha256": "%s", "rows": %d}' % (SHA, N))
    with pytest.raises(AssertionError):
        a28.read_output(path, 'B1', 'sample', 0, 'percept', SHA, N)


def test_value_checks_rows_range_norm_and_finiteness(tmp_path):
    for arm, bad in (('K2', unit32(N, 2048, 1) * 2), ('B1', prob(1) * 2), ('B6', prob(1)[:, :39])):
        path = put(tmp_path, arm, 0, bad)
        with pytest.raises(AssertionError):
            a28.read_output(path, arm, 'sample', 0, 'emotion' if arm[0] == 'K' else 'percept', SHA, N)
    nan = prob(1)
    nan[2, 2] = np.nan
    for arm in ('B1', 'B6', 'M2'):                      # no arm of stage 5 may drop out: non-finite fails
        path = put(tmp_path, arm, 0, nan)
        with pytest.raises(AssertionError):
            a28.read_output(path, arm, 'sample', 0, 'percept', SHA, N)
    path = put(tmp_path, 'K2', 0, unit32(N, 2048, 1))
    with pytest.raises(AssertionError):
        a28.read_output(path, 'K2', 'sample', 0, 'emotion', SHA, N + 1)
    with pytest.raises(AssertionError, match='keys|files'):    # a percept file under a K arm
        np.savez(path, prob=prob(1))
        a28.read_output(path, 'K2', 'sample', 0, 'emotion', SHA, N)


# --- the verdict and the reported files, on a synthetic 40-painting list -------------------------------------------

def make_setup():
    names = [f'p{i:02d}' for i in range(N)]
    emo = np.arange(N) % 4                                    # 4 classes of 10 paintings
    rows = []
    for i, name in enumerate(names):
        style, genre = f's{i % 2}', f'g{i % 3}'
        rows.append((name, 'en', EMO[emo[i]], style, genre, i < 12))
        rows.append((name, 'en', EMO[emo[i]], style, genre, i < 12))
        rows.append((name, 'fr', EMO[(emo[i] + (i % 5 == 0)) % 4], style, genre, i < 12))
        if i % 4:
            rows.append((name, 'de', EMO[emo[i]], style, genre, i < 12))
    read, csv = fixture(rows)
    joined = a28.join_rows(read, csv)
    return a28.make_setup(joined, names, SHA, fake_images, fake_images, EMO, STYLE, GENRE)


def fake_images(key):
    x = np.random.default_rng(len(key)).normal(size=(N, 64)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def aligned_k(labels, seed):
    centers = unit32(int(labels.max()) + 1, 2048, 100 + seed).astype(np.float32)
    return centers[labels]


def aligned_prob(labels):
    p = np.full((len(labels), 40), 0.1, dtype=np.float32)
    p[np.arange(len(labels)), labels * 10] = 0.9
    return p


def write_all(outputs, setup, model_good, seeds=(0, 1), scale='sample', model='K2'):
    """Model aspects either aligned with the label (good) or random; B1/B6/matched the other way round."""
    for seed in seeds:
        for aspect in ('emotion', 'style', 'genre'):
            y = setup['labels'][aspect]
            put(outputs, model, seed, aligned_k(y, seed) if model_good else unit32(N, 2048, seed + 7),
                aspect=aspect, scale=scale)
        for arm in ('B1', 'B6', a28.MATCHED[model]):
            y = setup['labels']['emotion']
            put(outputs, arm, seed, np.random.default_rng(seed + 3).random((N, 40)).astype(np.float32)
                if model_good else aligned_prob(y), scale=scale)


def run(tmp_path, setup, models=('K2',), seeds=(0, 1), **kw):
    return a28.run_verdict(list(models), tmp_path, setup, workers=1, n_resamples=20, seeds=seeds, **kw)


def test_verdict_a_holds_when_the_model_beats_every_entry_and_a_fails_when_one_entry_beats_it(tmp_path):
    setup = make_setup()
    write_all(tmp_path, setup, True)
    rec = run(tmp_path, setup)
    block = rec['blocks']['K2']
    assert block['outcome_id'] == 'A-HOLDS' and block['d'] > 0.5 and block['interval'][0] > 0
    assert block['arms'] == ['K2', 'B1', 'B6', 'M2'] and block['scale'] == 'sample'
    assert sorted(block['A_e']) == ['B1-own', 'B1-r20', 'B6-own', 'B6-r20', 'M2-own', 'M2-r20']
    assert len(block['bootstrap_d']) == 20 and block['bootstrap']['n_rows'] == N
    assert rec['stage'] == 5 and rec['list_sha256'] == SHA
    out2 = tmp_path / 'b'
    write_all(out2, setup, False, model='K3')
    rec2 = run(out2, setup, models=('K3',))
    assert rec2['blocks']['K3']['outcome_id'] == 'A-FAILS' and rec2['blocks']['K3']['d'] < -0.5


def test_k1_and_k4_have_b1_as_their_only_matched_arm(tmp_path):
    setup = make_setup()
    for model in ('K1', 'K4'):
        write_all(tmp_path, setup, True, model=model)
        write_all(tmp_path, setup, True, model='K2')       # an unused matched arm file must not matter
    rec = run(tmp_path, setup, models=('K1', 'K4'))
    for model in ('K1', 'K4'):
        assert rec['blocks'][model]['arms'] == [model, 'B1', 'B6']
        assert sorted(rec['blocks'][model]['A_e']) == ['B1-own', 'B1-r20', 'B6-own', 'B6-r20']


def test_a_model_with_a_bad_or_missing_output_gets_no_outcome_and_the_others_keep_theirs(tmp_path):
    setup = make_setup()
    write_all(tmp_path, setup, True, model='K2')
    write_all(tmp_path, setup, True, model='K3')
    (tmp_path / 'part2/sample/a28/K3/genre__s1.npz').unlink()
    rec = run(tmp_path, setup, models=('K2', 'K3'))
    assert rec['blocks']['K2']['outcome_id'] == 'A-HOLDS'
    assert rec['blocks']['K3']['outcome_id'] is None and 'genre__s1' in rec['blocks']['K3']['v_failure']


def test_the_model_scale_is_full_only_when_stage_4_gave_it_an_outcome(tmp_path):
    assert a28.model_scale('K2', tmp_path) == 'sample'
    v4 = tmp_path / 'part2/verdict_stage4.json'
    v4.parent.mkdir(parents=True)
    v4.write_text(json.dumps(dict(blocks=dict(K2=dict(outcome_id='S2-GO'), K3=dict(outcome_id=None)))))
    assert a28.model_scale('K2', tmp_path) == 'full'
    assert a28.model_scale('K3', tmp_path) == 'sample' and a28.model_scale('K5', tmp_path) == 'sample'
    assert a28.model_scale('K1', tmp_path) == 'sample'
    (tmp_path / 'part2/full/a28/K1').mkdir(parents=True)
    assert a28.model_scale('K1', tmp_path) == 'full'


def test_a_changed_verdict_block_is_refused_without_a_reason(tmp_path):
    setup = make_setup()
    write_all(tmp_path, setup, True)
    run(tmp_path, setup)
    run(tmp_path, setup)                                     # the same content again is fine
    for seed in (0, 1):
        put(tmp_path, 'B1', seed, aligned_prob(setup['labels']['emotion']))
    with pytest.raises(AssertionError, match='refusing'):
        run(tmp_path, setup)


def test_reported_files_need_the_verdict_first_and_then_cover_the_reported_items(tmp_path):
    setup = make_setup()
    write_all(tmp_path, setup, True)
    with pytest.raises(AssertionError, match='verdict'):
        a28.run_reported(['K2'], tmp_path, setup, workers=1, n_resamples=20, seeds=(0, 1))
    assert not (tmp_path / 'part2/reported_stage5.json').exists()
    run(tmp_path, setup)
    rec = a28.run_reported(['K2'], tmp_path, setup, workers=1, n_resamples=20, seeds=(0, 1))
    assert json.loads((tmp_path / 'part2/reported_stage5.json').read_text())['blocks'].keys() == {'K2'}
    block = rec['blocks']['K2']
    assert block['verdict_outcome'] == 'A-HOLDS'
    assert set(block['subsets']) == {'held_248', 'outside_498'}      # sizes 12 and 28 here
    assert (block['subsets']['held_248']['n'], block['subsets']['outside_498']['n']) == (12, 28)
    assert set(block['aspects']) == {'emotion', 'style', 'genre'}
    for aspect, item in block['aspects'].items():
        assert {'vs_a', 'vs_b', 'vs_max_ab', 'vs_Fk'} <= set(item), aspect
        assert item['vs_Fk']['A_e'].keys() == {'F-r20'}
        assert item['vs_a']['A_e'].keys() == {'B1-r20', 'B1-own', 'B6-r20', 'B6-own'}
        assert item['vs_b']['A_e'].keys() == {'M2-r20', 'M2-own'}
    emotion = block['aspects']['emotion']
    assert {'vs_F1', 'vs_T'} <= set(emotion) and 'vs_T' not in block['aspects']['style']
    assert emotion['vs_max_ab']['d'] == pytest.approx(block['verdict_d'])    # same entries, same draws as the verdict
    cul = block['culture']
    assert sorted(cul['languages']) == ['de', 'en', 'fr']
    en = cul['languages']['en']
    assert en['n_paintings'] == N and en['agree_pooled'] == pytest.approx(1.0) and en['ami'] > 0.8
    assert cul['languages']['de']['n_paintings'] == 30
    assert cul['languages']['fr']['agree_pooled'] == pytest.approx(0.8) and cul['languages']['fr']['ami'] < en['ami']


def test_k4_reports_with_b1_as_its_matched_arm_and_no_separate_max(tmp_path):
    setup = make_setup()
    write_all(tmp_path, setup, True, model='K4')
    run(tmp_path, setup, models=('K4',))
    block = a28.run_reported(['K4'], tmp_path, setup, workers=1, n_resamples=20, seeds=(0, 1))['blocks']['K4']
    emotion = block['aspects']['emotion']
    assert 'vs_max_ab' not in emotion and 'vs_F1' not in emotion          # K4's Fk is F1
    assert emotion['vs_b']['A_e'].keys() == {'B1-r20', 'B1-own'}
    assert emotion['vs_a']['d'] == pytest.approx(block['verdict_d'])


# --- real data (RB_SLOW=1 ...) ---------------------------------------------------------------------------------------

@slow
def test_real_labels_729_unique_17_ties_and_the_248_498_split():
    import rb_encoders
    setup = a28.real_setup()
    labels = setup['labels']
    assert (labels['n_unique'], labels['n_ties']) == (729, 17)
    assert setup['names'] == rb_encoders.a28_names() and setup['sha256'] == rb_encoders.a28_sha256()
    assert (int(labels['held'].sum()), int((~labels['held']).sum())) == (248, 498)
    assert len(setup['culture']) == 27 and labels['n_captions'].sum() == 74569
    assert set(labels['emotion']) <= set(range(9)) and set(labels['style']) <= set(range(27))
    assert set(labels['genre']) == {0, 1, 2, 3, 4, 6, 7, 8, 9}             # our nine; "nude painting" (5) is absent
    # the read set's own style and genre codes are ours, a cross-check of the name join (not used for labels)
    import pandas as pd
    frame = pd.read_parquet(rb_encoders.READ_SET)
    first = frame.drop_duplicates('painting').set_index('painting').loc[setup['names']]
    assert (first['style'].to_numpy() == labels['style']).all() and (first['genre'].to_numpy() == labels['genre']).all()
    # the codes by name equal sorted names of the public CSV and the repo's genre list
    csv = pd.read_csv(a28.CSV)
    assert a28.STYLE_NAMES == sorted(csv.art_style.unique()) and set(csv.genre) <= set(a28.GENRE_NAMES)
    from src.data.wikiart_genre import GENRE_NAMES
    assert a28.GENRE_NAMES == GENRE_NAMES
    assert a28.EMOTION_NAMES == sorted(set(csv.emotion.replace({'other': 'something else'})))
