"""Ticket 03 (CPU only): key and row selection, text lookup by csv_row, pending/seal logic, and that a tiny view sealed
by pf_extract (fake encoder output) is accepted by pf_data's reader. No model, no GPU."""
import json

import numpy as np
import pytest

import pf_paths
import pf_data
import pf_extract
import rs_vh
from test_pf_data import World, unit16, write_teacher

IDENTITY = {'model_id': 'Qwen/Qwen3-VL-Embedding-2B', 'model_revision': pf_data.rt_data.ct2_data.QWEN_REVISION}


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    monkeypatch.setattr(pf_paths, 'N_TEXTS', {'train': 8, 'val': 4})
    return w


def test_image_keys_are_the_sorted_train_keys_and_smoke_takes_the_first_n(world, monkeypatch):
    assert pf_extract.image_keys() == ['a/1.jpg', 'a/2.jpg', 'b/1.jpg']
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    assert pf_extract.image_keys() == ['a/1.jpg', 'a/2.jpg']


def test_text_selection_is_train_rows_then_val_rows_in_text_rows_order(world, monkeypatch):
    csv_rows, keys = pf_extract.text_selection()
    tr, va = pf_data.text_rows('train'), pf_data.text_rows('val')
    assert csv_rows == tr['csv_row'].tolist() + va['csv_row'].tolist() and len(csv_rows) == 12
    assert keys == tr['key'] + va['key']
    assert set(keys[:8]) == {'a/1.jpg', 'a/2.jpg', 'b/1.jpg'} and set(keys[8:]) == {'c/v1.jpg', 'c/v2.jpg'}
    monkeypatch.setattr(pf_paths, 'SMOKE', 1)                    # smoke: one train photo, all val rows
    csv_rows, keys = pf_extract.text_selection()
    assert set(keys[:4]) == {'a/1.jpg'} and len(keys) == 4 + 4


def test_text_lookup_is_by_csv_row_not_by_position():
    column = [f'text {i}' for i in range(20)]
    assert pf_extract.lookup_texts(column, [7, 3, 19]) == ['text 7', 'text 3', 'text 19']
    with pytest.raises(AssertionError):
        pf_extract.lookup_texts(column, [3, 20])
    with pytest.raises(AssertionError):
        pf_extract.lookup_texts(column, [-1])
    with pytest.raises(AssertionError, match='blank'):
        pf_extract.lookup_texts(['ok', '  ', 'x'], [0, 1])
    with pytest.raises(AssertionError, match='blank'):
        pf_extract.lookup_texts(['ok', 'x', None], [0, 2])


def test_utterances_reads_only_the_utterance_column(tmp_path, monkeypatch):
    path = tmp_path / 'a.csv'
    path.write_text('workerid,image_name,dataset,utterance,emotion,split\nu,i.jpg,d,"hello, world",awe,train\n'
                    'u,j.jpg,d,second,fear,val\n')
    monkeypatch.setattr(pf_paths, 'ANNOTATIONS', path)
    assert pf_extract.utterances([1, 0]) == ['second', 'hello, world']


def test_pending_target_is_refused_and_a_failed_seal_leaves_the_pending_folder(world, tmp_path):
    out = tmp_path / 'v' / 'emotion_focus_train'
    pending = pf_extract.pending_of(out)
    assert pending.name == 'emotion_focus_train.pending'
    pending.mkdir(parents=True)
    with pytest.raises(AssertionError, match='pending'):
        pf_extract.pending_of(out)
    bad = np.full((3, 2048), 0.5, dtype=np.float16)              # norm far from 1
    with pytest.raises(AssertionError, match='gate'):
        pf_extract.seal_images(pending, out, ['a/1.jpg', 'a/2.jpg', 'b/1.jpg'], bad, IDENTITY, False, 1.0)
    assert pending.exists() and not out.exists()
    (pending / 'keys.json').unlink()
    pending.rmdir()
    out.mkdir()
    with pytest.raises(AssertionError, match='existing'):
        pf_extract.pending_of(out)


def test_smoke_flag_must_match_pf_smoke(world, monkeypatch):
    assert pf_extract.check_smoke_mode(False) is False
    with pytest.raises(AssertionError):
        pf_extract.check_smoke_mode(True)
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    with pytest.raises(AssertionError):
        pf_extract.check_smoke_mode(False)
    assert pf_extract.check_smoke_mode(True) is True


def seal_both(world):
    """Seal tiny views as the extraction would (random unit rows stand in for the encoder)."""
    keys = pf_extract.image_keys()
    x = unit16(world.rng, len(keys))
    pending = pf_extract.pending_of(world.train_view)
    pending.mkdir(parents=True)
    np.save(pending / 'img.npy', x)
    pf_extract.seal_images(pending, world.train_view, keys, np.load(pending / 'img.npy'), IDENTITY, False, 2.0)
    csv_rows, row_keys = pf_extract.text_selection()
    t = unit16(world.rng, len(csv_rows))
    pending = pf_extract.pending_of(world.txt_view)
    pending.mkdir(parents=True)
    np.save(pending / 'txt.npy', t)
    pf_extract.seal_texts(pending, world.txt_view, csv_rows, row_keys, np.load(pending / 'txt.npy'), IDENTITY,
                          False, 3.0)
    return x, t


def test_views_sealed_by_pf_extract_are_accepted_by_the_pf_data_reader(world):
    x, t = seal_both(world)
    assert not world.train_view.with_name('emotion_focus_train.pending').exists()
    meta = json.loads((world.txt_view / 'meta.json').read_text())
    assert meta['view'] == 'emotion_focus_txt' and meta['smoke'] is False and meta['gate']['ok'] is True
    rows = json.loads((world.txt_view / 'rows.json').read_text())
    assert meta['rows_sha256'] == rs_vh.names_sha([f'{c}\t{k}' for c, k in zip(rows['csv_row'], rows['key'])])
    write_teacher(world)
    keys = ['c/v1.jpg', 'b/1.jpg', 'a/1.jpg']
    got = pf_data.photo_arrays(keys)                           # reads both views through every reader check
    assert got['E'].shape == (3, 2048) and got['T'].shape == (3, 2048)
    stored, y = pf_data._train_view()
    assert stored == ['a/1.jpg', 'a/2.jpg', 'b/1.jpg'] and np.array_equal(y, x)


def test_check_writes_views_json_and_catches_a_changed_row(world, tmp_path, monkeypatch):
    monkeypatch.setattr(pf_paths, 'OUTPUTS', tmp_path / 'out')
    seal_both(world)
    report = pf_extract.check()
    assert report['ok'] and report['images']['rows'] == 3 and report['texts']['rows'] == 12
    assert json.loads((tmp_path / 'out' / 'stage0a' / 'views.json').read_text())['ok'] is True
    t = np.load(world.txt_view / 'txt.npy')
    t[0] = t[1]
    np.save(world.txt_view / 'txt.npy', t)
    with pytest.raises(AssertionError, match='hash'):
        pf_extract.check()


def test_a_smoke_view_is_marked_smoke_and_refused_by_a_full_reader(world, monkeypatch):
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    keys = pf_extract.image_keys()
    pending = pf_extract.pending_of(world.train_view)
    pending.mkdir(parents=True)
    x = unit16(world.rng, 2)
    np.save(pending / 'img.npy', x)
    pf_extract.seal_images(pending, world.train_view, keys, x, IDENTITY, True, 1.0)
    assert pf_data._train_view()[0] == keys
    monkeypatch.setattr(pf_paths, 'SMOKE', 0)
    with pytest.raises(AssertionError):
        pf_data._train_view()
