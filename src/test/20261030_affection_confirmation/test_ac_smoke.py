"""Smoke mode (ticket 13): first-N keys per split, smoke views and exports in a scratch folder, key-list checks.

Synthetic annotations and files only (no GPU, no real data). With SMOKE = 0 nothing here changes the full-mode
behaviour; test_ac_data.py keeps covering that.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ac_paths
import ac_data
import ac_export
import ac_extract
import rs_vh
import rt_crl
import rt_data
from test_ac_data import annotations, basis, write_json

FI = 'you_aaai_affective'
SMOKE = 2
ROUND = Path(__file__).resolve().parent


@pytest.fixture
def smoke_round(tmp_path, monkeypatch):
    """Val 4 keys (3 non-FI), test 3 keys (2 non-FI); FI sorts last; SMOKE = 2; outputs and views in tmp_path."""
    path = tmp_path / 'annotations.csv'
    annotations(path, [
        ('a', '1.jpg', 'val', 'awe'), ('a', '1.jpg', 'val', 'amusement'), ('a', '2.jpg', 'val', 'sadness'),
        ('b', '3.jpg', 'val', 'fear'), (FI, 'f1.jpg', 'val', 'awe'),
        ('a', '4.jpg', 'test', 'anger'), ('b', '5.jpg', 'test', 'awe'), (FI, 'f2.jpg', 'test', 'fear'),
        ('a', 't.jpg', 'train', 'awe'), ('x', 'ign.jpg', 'ignored', 'fear')])
    monkeypatch.setattr(ac_paths, 'ANNOTATIONS', path)
    monkeypatch.setattr(ac_paths, 'N', {'val': 4, 'test': 3})
    out = tmp_path / 'outputs'
    monkeypatch.setattr(ac_paths, 'OUTPUTS', out)
    monkeypatch.setattr(ac_paths, 'STORE', tmp_path / 'store')
    monkeypatch.setattr(ac_paths, 'VIEWS', out / 'smoke_views' / 'qwen3vl_emb_2b_aspect_prompts')
    monkeypatch.setattr(ac_paths, 'CLIP_VIEW', out / 'smoke_views' / 'clip_l14_valtest')
    monkeypatch.setattr(ac_paths, 'SMOKE', SMOKE)
    monkeypatch.setattr(rt_data, 'CSTAR_ROOT', tmp_path / 'r10_cstar')
    monkeypatch.setattr(rt_data.rt_paths, 'PERCEPT_OUT', tmp_path / 'r11_percept')
    return tmp_path


def truncated():
    return ac_data.keys('val') + ac_data.keys('test')


def full_keys(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(ac_paths, 'SMOKE', 0)
        return ac_data.keys('val'), ac_data.keys('test')


# ---- keys and labels --------------------------------------------------------------------------------------------

def test_smoke_truncates_every_per_split_accessor_after_the_full_asserts(smoke_round, monkeypatch):
    full = {}
    with monkeypatch.context() as patch:
        patch.setattr(ac_paths, 'SMOKE', 0)
        for split in ac_paths.SPLITS:
            full[split] = (ac_data.keys(split), ac_data.labels(split), ac_data.ties(split),
                           ac_data.emotion_dist(split), ac_data.sources(split))
    assert full['val'][0] == ['a/1.jpg', 'a/2.jpg', 'b/3.jpg', f'{FI}/f1.jpg']   # FI sorts last
    for split in ac_paths.SPLITS:
        k, lab, tie, dist, src = full[split]
        assert ac_data.keys(split) == k[:SMOKE]
        np.testing.assert_array_equal(ac_data.labels(split), lab[:SMOKE])
        np.testing.assert_array_equal(ac_data.ties(split), tie[:SMOKE])
        np.testing.assert_array_equal(ac_data.emotion_dist(split), dist[:SMOKE])
        assert ac_data.sources(split) == src[:SMOKE]
        assert ac_data.labels(split).dtype == np.int64 and ac_data.emotion_dist(split).dtype == np.float64
        counts = ac_data.label_counts(split)
        assert counts['n'] == SMOKE and sum(counts['classes']) == SMOKE
    assert ac_data.ties('val').tolist() == [True, False]
    assert ac_data.label_counts('val')['ties'] == 1


def test_smoke_value_is_read_at_call_time_and_nothing_stale_is_served(smoke_round, monkeypatch):
    assert len(ac_data.keys('val')) == SMOKE
    monkeypatch.setattr(ac_paths, 'SMOKE', 3)
    assert ac_data.keys('val') == ['a/1.jpg', 'a/2.jpg', 'b/3.jpg']
    assert ac_data.labels('val').shape == (3,) and ac_data.emotion_dist('val').shape == (3, 9)
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    assert len(ac_data.keys('val')) == 4 and ac_data.labels('val').shape == (4,)
    assert len(ac_data.keys('test')) == 3


def test_smoke_full_split_asserts_still_run_before_truncation(smoke_round, monkeypatch):
    monkeypatch.setattr(ac_paths, 'N', {'val': 5, 'test': 3})        # the annotation file has 4 val keys
    with pytest.raises(AssertionError, match='split key count'):
        ac_data.keys('val')


def test_smoke_larger_than_the_non_fi_count_is_refused(smoke_round, monkeypatch):
    monkeypatch.setattr(ac_paths, 'SMOKE', 3)        # val has 3 non-FI keys, test only 2
    assert len(ac_data.keys('val')) == 3
    with pytest.raises(AssertionError, match='non-FI'):
        ac_data.keys('test')
    monkeypatch.setattr(ac_paths, 'SMOKE', 4)
    with pytest.raises(AssertionError, match='non-FI'):
        ac_data.keys('val')


def test_smoke_prefix_must_not_contain_fi(smoke_round, monkeypatch):
    # If a non-FI source sorted after FI, the first N keys would include FI images that are not in the store.
    monkeypatch.setattr(ac_paths, 'FI_SOURCE', 'a')
    with pytest.raises(AssertionError, match='FI key among'):
        ac_data.keys('val')


# ---- store, views, exports --------------------------------------------------------------------------------------

def write_store(stored):
    path = ac_paths.STORE
    write_json(path / 'index.json', {
        'dataset': 'affection_local', 'backbone': 'qwen3vl_emb_2b', 'smoke': False, 'dtype': 'float16', 'dim': 2048,
        'n_images': len(stored), 'images': [dict(zip(('source', 'image_name'), k.split('/'))) for k in stored]})
    x = basis(len(stored), 2048)
    np.save(path / 'img.npy', x)
    return x


def test_plain_store_joins_the_truncated_keys_by_key_not_position(smoke_round):
    # Shuffled store with train and a non-requested val key; smoke rows are still found by key.
    stored = ['b/5.jpg', 'a/t.jpg', 'b/3.jpg', 'a/2.jpg', 'a/4.jpg', 'a/1.jpg']
    x = write_store(stored)
    np.testing.assert_array_equal(ac_data.embed('plain', 'val'), x[[5, 3]].astype(np.float32))
    np.testing.assert_array_equal(ac_data.embed('plain', 'test'), x[[4, 0]].astype(np.float32))
    assert ac_data.embed('plain', 'val').shape == (SMOKE, 2048)


def smoke_view(arm, keys=None, smoke=True, mutate=lambda side: None):
    is_clip = arm == 'clip'
    path = ac_paths.CLIP_VIEW if is_clip else ac_paths.VIEWS / ac_paths.WORDINGS[arm.removeprefix('teacher_')]
    stored = truncated() if keys is None else keys
    x = basis(len(stored), 768 if is_clip else 2048)
    instruction = None if is_clip else (ac_paths.PHOTO_INSTRUCTION if arm == 'teacher_photo' else
                                        rt_data.ct2_data.itp_data.PROMPTS['emotion_focus'])
    side = {'instruction': instruction,
            'model_id': rt_crl.CLIP_ID if is_clip else 'Qwen/Qwen3-VL-Embedding-2B',
            'model_revision': rt_crl.CLIP_REVISION if is_clip else rt_data.ct2_data.QWEN_REVISION,
            'dtype': 'float16', 'shape': list(x.shape), 'smoke': smoke,
            'gate': {'ok': True, 'n_nonfinite': 0, 'n_bad_norm': 0, 'max_norm_dev': 0.0},
            'keys_sha256': rs_vh.names_sha(stored), 'img_sha256': rs_vh.array_sha(x)}
    mutate(side)
    write_json(path / 'keys.json', stored)
    write_json(path / 'meta.json', side)
    np.save(path / 'img.npy', x)
    return path, x


def test_views_in_smoke_return_the_first_n_rows_per_split(smoke_round):
    assert ac_paths.VIEWS.parent == ac_paths.OUTPUTS / 'smoke_views'
    for arm in ('teacher_artwork', 'teacher_photo', 'clip'):
        _, x = smoke_view(arm)
        np.testing.assert_array_equal(ac_data.embed(arm, 'val'), x[:2].astype(np.float32))
        np.testing.assert_array_equal(ac_data.embed(arm, 'test'), x[2:].astype(np.float32))


@pytest.mark.parametrize('case', ['full_keys', 'other_keys', 'shuffled', 'flag_false', 'extra_key', 'wrong_hash'])
def test_view_in_smoke_refuses_a_key_list_or_flag_that_is_not_the_truncated_one(smoke_round, monkeypatch, case):
    val, test = full_keys(monkeypatch)
    keys = truncated()
    kwargs = {}
    if case == 'full_keys':
        keys = val + test
    elif case == 'other_keys':
        keys = ['a/1.jpg', 'b/3.jpg', 'a/4.jpg', 'b/5.jpg']          # real keys, but not the first two of val
    elif case == 'shuffled':
        keys = [keys[1], keys[0], keys[2], keys[3]]
    elif case == 'extra_key':
        keys = keys + ['a/t.jpg']
    elif case == 'flag_false':
        kwargs['smoke'] = False                                        # the right keys, a full-run flag
    elif case == 'wrong_hash':
        kwargs['mutate'] = lambda side: side.update(keys_sha256='wrong')
    smoke_view('teacher_artwork', keys, **kwargs)
    with pytest.raises(AssertionError):
        ac_data.embed('teacher_artwork', 'val')


def test_smoke_view_is_refused_with_smoke_zero_and_full_view_with_smoke_positive(smoke_round, monkeypatch):
    val, test = full_keys(monkeypatch)
    smoke_view('teacher_artwork')                                  # right keys for SMOKE = 2, flagged smoke
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    with pytest.raises(AssertionError):
        ac_data.embed('teacher_artwork', 'val')
    # A complete view (full keys, smoke false) is fine with SMOKE = 0 and refused with SMOKE > 0.
    smoke_view('teacher_photo', val + test, smoke=False)
    assert ac_data.embed('teacher_photo', 'val').shape == (4, 2048)
    monkeypatch.setattr(ac_paths, 'SMOKE', SMOKE)
    with pytest.raises(AssertionError):
        ac_data.embed('teacher_photo', 'val')


def smoke_export(arm, split, seed=0, head='emotion', keys=None, rows=None, smoke=True, mutate=lambda side: None):
    percept = arm == 'percept'
    path = ac_paths.OUTPUTS / 'exports' / arm / split / f'{"P3" if percept else head}__s{seed}.npz'
    stored = ac_data.keys(split) if keys is None else keys
    x = basis(len(stored) if rows is None else rows, 40 if percept else 2048, np.float32 if percept else np.float16)
    if percept:
        x[:, 5] = .5
    weights = rs_vh.names_sha([arm, head, str(seed)])
    reference = (rt_data.rt_paths.PERCEPT_OUT / f'held/percept/core__k40/P3__s{seed}.json' if percept else
                 rt_data.CSTAR_ROOT / f'held/k11=40/{head}__s{seed}.json')
    write_json(reference, {'weights_sha256': weights})
    side = {'keys_sha256': rs_vh.names_sha(stored), 'array_sha256': rs_vh.array_sha(x), 'weights_sha256': weights,
            'split': split, 'seed': seed, 'smoke': smoke}
    side.update({'device': 'cuda'} if percept else {'aspect': head})
    mutate(side)
    write_json(path.with_suffix('.json'), side)
    np.savez(path, **{'prob' if percept else 'img': x})
    return x


@pytest.mark.parametrize('split', ['val', 'test'])
@pytest.mark.parametrize('arm,seed,head', [('cstar', 0, 'emotion'), ('cstar', 3, 'style'), ('percept', 1, 'emotion')])
def test_exports_in_smoke_join_the_truncated_split(smoke_round, split, arm, seed, head):
    x = smoke_export(arm, split, seed, head)
    actual = ac_data.embed(arm, split, seed, head=head)
    assert actual.shape == (SMOKE, x.shape[1])
    if arm == 'percept':
        assert ac_data.percept_prob(split, seed).tobytes() == x.tobytes()


@pytest.mark.parametrize('arm', ['cstar', 'percept'])
@pytest.mark.parametrize('case', ['full_keys', 'other_keys', 'shuffled', 'flag_false'])
def test_export_in_smoke_refuses_a_key_list_or_flag_that_is_not_the_truncated_one(smoke_round, monkeypatch, arm, case):
    val, _ = full_keys(monkeypatch)
    keys, kwargs = ac_data.keys('val'), {}
    if case == 'full_keys':
        keys = val
    elif case == 'other_keys':
        keys = ['a/1.jpg', 'b/3.jpg']
    elif case == 'shuffled':
        keys = keys[::-1]                       # same set, other order: the digest differs, so it is refused
    else:
        kwargs['smoke'] = False
    smoke_export(arm, 'val', keys=keys, **kwargs)
    with pytest.raises(AssertionError):
        ac_data.embed(arm, 'val', seed=0)
    if arm == 'percept':
        with pytest.raises(AssertionError):
            ac_data.percept_prob('val', 0)


@pytest.mark.parametrize('arm', ['cstar', 'percept'])
def test_smoke_export_is_refused_with_smoke_zero_and_full_export_with_smoke_positive(smoke_round, monkeypatch, arm):
    val, test = full_keys(monkeypatch)
    smoke_export(arm, 'val')                                     # smoke file for SMOKE = 2
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    with pytest.raises(AssertionError):
        ac_data.embed(arm, 'val', seed=0)
    smoke_export(arm, 'val', keys=val, smoke=False)              # full file, rewritten in place
    assert ac_data.embed(arm, 'val', seed=0).shape[0] == 4
    monkeypatch.setattr(ac_paths, 'SMOKE', SMOKE)
    with pytest.raises(AssertionError):
        ac_data.embed(arm, 'val', seed=0)


def test_smoke_flag_alone_decides_even_for_identical_keys(smoke_round, monkeypatch):
    # Make the full test split equal its truncated list (two non-FI keys), so only the smoke flag can tell them apart.
    monkeypatch.setattr(ac_paths, 'N', {'val': 4, 'test': 2})
    path = ac_paths.ANNOTATIONS
    path.write_text(path.read_text().replace(f'{FI},f2.jpg,test,fear', f'{FI},f2.jpg,train,fear'))
    monkeypatch.setattr(ac_paths, 'SMOKE', 2)
    smoke_export('cstar', 'test')
    assert ac_data.embed('cstar', 'test', 0).shape == (2, 2048)
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    assert ac_data.keys('test') == ['a/4.jpg', 'b/5.jpg']
    with pytest.raises(AssertionError, match='export identity'):
        ac_data.embed('cstar', 'test', 0)


def test_crl_arm_in_smoke_uses_the_truncated_image_rows(smoke_round, monkeypatch):
    monkeypatch.setattr(ac_paths, 'R11_CRL_LISTS_FILE', smoke_round / 'lists.json')
    write_json(ac_paths.R11_CRL_LISTS_FILE, {'aspects': {a: {'values': ['joy', 'fear', 'calm'], 'count': 3}
                                                       for a in ac_paths.ASPECTS}})
    write_store(['b/5.jpg', 'a/t.jpg', 'b/3.jpg', 'a/2.jpg', 'a/4.jpg', 'a/1.jpg'])
    smoke_view('clip')
    path = ac_paths.OUTPUTS / 'crl' / 'clip_painting_emotion.npy'
    text = np.zeros((3, 768), dtype=np.float32)
    text[0, :2], text[1, 1], text[2, 2] = (2**-.5, 2**-.5), 1, 1
    texts = [f'a painting with the emotion of {v}' for v in ('joy', 'fear', 'calm')]
    write_json(path.with_suffix('.json'), {
        'texts': texts, 'encoder': 'clip', 'model_id': rt_crl.CLIP_ID, 'model_revision': rt_crl.CLIP_REVISION,
        'instruction': None, 'template': 'a painting with the {aspect} of {value}'})
    np.save(path, text)
    out = ac_data.embed('crl_clip_painting', 'val')
    assert out.shape == (SMOKE, 3) and out.dtype == np.float32


# ---- smoke_rows and the ArtELingo arm ----------------------------------------------------------------------------

def test_smoke_rows_is_identity_without_smoke_and_takes_the_first_rows_with_it(monkeypatch):
    x = np.arange(20).reshape(10, 2)
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    assert ac_data.smoke_rows(x) is x
    monkeypatch.setattr(ac_paths, 'SMOKE', 3)
    np.testing.assert_array_equal(ac_data.smoke_rows(x), x[:3])
    assert ac_data.smoke_rows(list('abcdef')) == ['a', 'b', 'c']
    monkeypatch.setattr(ac_paths, 'SMOKE', 11)
    with pytest.raises(AssertionError):
        ac_data.smoke_rows(x)


def test_artelingo_qwen_crl_in_smoke_is_the_first_rows_of_the_full_arm(smoke_round, monkeypatch):
    lists = smoke_round / 'lists.json'
    write_json(lists, {'aspects': {a: {'values': ['joy', 'fear', 'calm'], 'count': 3} for a in ac_paths.ASPECTS}})
    monkeypatch.setattr(ac_paths, 'R11_CRL_LISTS_FILE', lists)
    path = ac_paths.OUTPUTS / 'crl' / 'qwen_painting_emotion.npy'
    text = np.zeros((3, 2048), dtype=np.float32)
    text[0, :4], text[1, :4], text[2, 3] = (.5, .5, .5, .5), (-.5, .5, -.5, .5), 1
    write_json(path.with_suffix('.json'), {
        'texts': [f'a painting with the emotion of {v}' for v in ('joy', 'fear', 'calm')], 'encoder': 'qwen',
        'model_id': 'Qwen/Qwen3-VL-Embedding-2B', 'model_revision': rt_data.ct2_data.QWEN_REVISION,
        'instruction': "Represent the user's input.", 'template': 'a painting with the {aspect} of {value}'})
    np.save(path, text)
    names = ['p1', 'p2', 'p3', 'p4']
    img = basis(4, 2048, np.float32)
    monkeypatch.setattr(rt_data, 'names', lambda part: names)
    monkeypatch.setattr(rt_data, 'embed', lambda arm, part, aspect, seed=None: img)
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    full = ac_data.artelingo_crl_qwen('val', 'emotion')
    assert full.shape == (4, 3)
    monkeypatch.setattr(ac_paths, 'SMOKE', 2)
    small = ac_data.artelingo_crl_qwen('val', 'emotion')
    assert small.shape == (2, 3) and small.tobytes() == full[:2].tobytes()


@pytest.mark.real_data
def test_r11_artelingo_names_are_sorted_like_the_vh_smoke_order():
    # ac_vh's smoke takes names[:N] of the sorted ArtELingo names per split; r11's rt_data.names is that order.
    spec = importlib.util.spec_from_file_location('independent_vh', ROUND / 'ac_vh.py')
    vh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vh)
    frozen = vh.Data.frozen_art_names.func(None)         # a cached_property that reads no instance state
    for part in ('val', 'held'):
        names = rt_data.names(part)
        assert names == sorted(names) and len(names) == (6152 if part == 'val' else 12281)
        assert names == frozen[part], f'rt_data.names({part!r}) differs from ac_vh frozen_art_names'


# ---- ac_paths ---------------------------------------------------------------------------------------------------

def run_import(env_extra, drop=('AC_OUTPUTS', 'AC_SMOKE')):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROUND))
    code = ('import ac_paths as P; '
            'print(P.SMOKE, P.OUTPUTS, P.VIEWS, P.CLIP_VIEW, P.STORE)')
    return subprocess.run([sys.executable, '-c', code], env=env, cwd=str(ROUND), capture_output=True, text=True)


def test_smoke_with_the_default_outputs_fails_at_import():
    done = run_import({'AC_SMOKE': '5'})
    assert done.returncode != 0 and 'AssertionError' in done.stderr and 'AC_OUTPUTS' in done.stderr
    explicit = run_import({'AC_SMOKE': '5', 'AC_OUTPUTS': '/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/'})
    assert explicit.returncode != 0 and 'AssertionError' in explicit.stderr


def test_smoke_paths_point_under_the_scratch_outputs_and_the_store_stays_real(tmp_path):
    done = run_import({'AC_SMOKE': '5', 'AC_OUTPUTS': str(tmp_path)})
    assert done.returncode == 0, done.stderr
    smoke, outputs, views, clip, store = done.stdout.split()
    assert (smoke, outputs) == ('5', str(tmp_path))
    assert views == str(tmp_path / 'smoke_views' / 'qwen3vl_emb_2b_aspect_prompts')
    assert clip == str(tmp_path / 'smoke_views' / 'clip_l14_valtest')
    assert store == '/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b'


def test_without_smoke_the_default_paths_are_unchanged(tmp_path):
    done = run_import({})
    assert done.returncode == 0, done.stderr
    smoke, outputs, views, clip, store = done.stdout.split()
    assert smoke == '0' and outputs == '/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs'
    assert views == '/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts'
    assert clip == '/data/SSD2/pre_extract/affection_local/clip_l14_valtest'
    assert run_import({'AC_SMOKE': '0'}).returncode == 0                  # explicit zero is full mode
    assert run_import({'AC_SMOKE': '-1', 'AC_OUTPUTS': str(tmp_path)}).returncode != 0


# ---- the smoke inputs written by ac_extract and ac_export match the truncated keys ---------------------------------

def test_export_requested_keys_equal_the_truncated_keys_with_or_without_ac_smoke(smoke_round, monkeypatch):
    for split in ac_paths.SPLITS:
        # AC_SMOKE = N and --smoke N: the same first N non-FI keys that ac_data.keys returns.
        assert ac_export._requested(split, SMOKE) == ac_data.keys(split)
        assert rs_vh.names_sha(ac_export._requested(split, SMOKE)) == rs_vh.names_sha(ac_data.keys(split))
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)                           # --smoke N alone: same list
    for split in ac_paths.SPLITS:
        assert ac_export._requested(split, SMOKE) == full_keys(monkeypatch)[ac_paths.SPLITS.index(split)][:SMOKE]
    # sidecars of this list are accepted by the loader in smoke mode
    monkeypatch.setattr(ac_paths, 'SMOKE', SMOKE)
    for split in ac_paths.SPLITS:
        smoke_export('cstar', split, keys=ac_export._requested(split, SMOKE))
        assert ac_data.embed('cstar', split, 0).shape == (SMOKE, 2048)


def test_export_refuses_a_smoke_n_that_differs_from_ac_smoke_and_a_full_run_in_smoke_mode(smoke_round):
    with pytest.raises(AssertionError, match='AC_SMOKE'):
        ac_export._requested('val', 3)
    with pytest.raises(AssertionError, match='AC_SMOKE'):
        ac_export._requested('val', None)       # would write truncated keys into a sidecar marked smoke=false


def test_extract_smoke_flag_and_canonical_folders_under_ac_smoke(smoke_round, monkeypatch):
    keys = truncated()
    assert keys == ac_extract.valtest_keys()
    # with AC_SMOKE the extraction is a smoke extraction even though its keys equal valtest_keys()
    assert ac_extract.is_smoke_run(None, keys) is True
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    assert ac_extract.is_smoke_run(None, keys) is True                  # not the full list (full mode)
    assert ac_extract.is_smoke_run(None, ac_extract.valtest_keys()) is False
    assert ac_extract.is_smoke_run(7, ac_extract.valtest_keys()) is True
    # the smoke views folder is never a canonical folder; the real AFF folders are, in both modes
    for smoke in (SMOKE, 0):
        monkeypatch.setattr(ac_paths, 'SMOKE', smoke)
        assert not ac_extract.is_canonical(ac_paths.OUTPUTS / 'smoke_views' / 'qwen3vl_emb_2b_aspect_prompts' / 'emotion_focus')
        assert ac_extract.is_canonical(ac_paths.AFF / 'qwen3vl_emb_2b_aspect_prompts' / 'emotion_focus')
        assert ac_extract.is_canonical(ac_paths.AFF / 'clip_l14_valtest')


def test_extract_in_ac_smoke_must_be_flagged_smoke(smoke_round):
    with pytest.raises(AssertionError, match='smoke'):
        ac_extract.extract('emotion_focus', truncated(), ac_paths.VIEWS / 'emotion_focus', smoke=False)


def test_empty_ac_smoke_means_no_smoke():
    done = run_import({'AC_SMOKE': ''})
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[0] == '0'


def test_export_impressions_and_reinfer_refuse_a_smoke_n_that_differs_from_ac_smoke(smoke_round):
    for call in (ac_export.export_impressions, ac_export.reinfer):
        for bad in (3, None):
            with pytest.raises(AssertionError, match='AC_SMOKE'):
                call(smoke=bad)
    with pytest.raises(AssertionError, match='AC_SMOKE'):       # the CLI: --smoke 3 against AC_SMOKE = 2
        ac_export.main(['impressions', '--smoke', '3'])
    with pytest.raises(AssertionError, match='AC_SMOKE'):
        ac_export.main(['reinfer'])
