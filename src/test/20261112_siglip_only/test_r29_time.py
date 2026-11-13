"""Table 8 plumbing (CPU only; nothing here loads a model or touches the GPU): K* from the committed pick record, and the
texts the SigLIP 1 text pass is timed on (plain, or prefixed exactly as ticket 02's store builds them)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_time_inputs as T
import r29_extract as X

CAPTIONS = ['A calm lake at dusk.', 'THE Sky is Red', 'sad', 'Two figures; one is crying!']


def write_pick(path, arm, form, **drop):
    doc = dict(kstar=arm, kstar_form=form, p1_form='plain', flag=None)
    path.write_text(json.dumps({k: v for k, v in doc.items() if k not in drop}))
    return path


def committed(monkeypatch, ok=True):
    """Stand in for the git check (tested in r29_paths); records what it was asked about."""
    seen = []
    def fake(path, why):
        seen.append(Path(path))
        if not ok:
            raise RuntimeError(f'{path} is not committed unchanged in HEAD: {why}')
    monkeypatch.setattr(T.P, 'require_committed', fake)
    return seen


@pytest.mark.parametrize('arm,form', [('siglip1', 'plain'), ('siglip1_prefix', 'prefixed')])
def test_read_kstar_returns_arm_and_form_of_a_committed_pick(tmp_path, monkeypatch, arm, form):
    seen = committed(monkeypatch)
    path = write_pick(tmp_path / 'prefix_pick.json', arm, form)
    assert T.read_kstar(path) == (arm, form)
    assert seen == [path]


def test_read_kstar_refuses_an_absent_or_uncommitted_pick(tmp_path):
    with pytest.raises(RuntimeError, match='prefix_pick.json'):           # absent: the real order-rule check
        T.read_kstar(tmp_path / 'prefix_pick.json')


def test_read_kstar_refuses_a_pick_not_committed_unchanged(tmp_path, monkeypatch):
    committed(monkeypatch, ok=False)
    path = write_pick(tmp_path / 'prefix_pick.json', 'siglip1', 'plain')
    with pytest.raises(RuntimeError, match='not committed'):
        T.read_kstar(path)


def test_read_kstar_with_the_real_git_check_refuses_an_untracked_pick(tmp_path):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    path = write_pick(tmp_path / 'prefix_pick.json', 'siglip1', 'plain')
    with pytest.raises(RuntimeError, match='not committed'):
        T.read_kstar(path)


def test_read_kstar_asserts_keys_and_consistent_form(tmp_path, monkeypatch):
    committed(monkeypatch)
    for key in ('kstar', 'kstar_form', 'p1_form', 'flag'):
        with pytest.raises(AssertionError):
            T.read_kstar(write_pick(tmp_path / 'prefix_pick.json', 'siglip1', 'plain', **{key: 1}))
    for arm, form in (('siglip1', 'prefixed'), ('siglip1_prefix', 'plain'), ('qwen', 'plain')):
        with pytest.raises(AssertionError):
            T.read_kstar(write_pick(tmp_path / 'prefix_pick.json', arm, form))


def test_plain_texts_are_the_captions():
    assert T.caption_texts(CAPTIONS, 'plain') == CAPTIONS


def test_prefixed_texts_are_prefix_plus_caption_once_before_lowercasing():
    texts = T.caption_texts(CAPTIONS, 'prefixed')
    assert texts == [T.P.PREFIX + c for c in CAPTIONS]
    assert texts[1] == 'Represent the caption: THE Sky is Red'            # case kept: the encoder lowercases, not this function
    assert all(t.count(T.P.PREFIX) == 1 for t in texts)


def test_prefixed_texts_equal_the_input_of_the_prefixed_store_build(monkeypatch):
    seen = []
    enc = SimpleNamespace(name=X.extractor.SIGLIP1, dim=3,
                          encode_texts=lambda texts, bs: seen.append(list(texts)) or np.zeros((len(texts), 3), np.float32))
    monkeypatch.setattr(X.extractor, 'count_truncated', lambda e, t: 0)
    X._encode_chunk(enc, CAPTIONS, T.P.PREFIX, 2)
    assert seen == [T.caption_texts(CAPTIONS, 'prefixed')]
    seen.clear()
    X._encode_chunk(enc, CAPTIONS, '', 2)
    assert seen == [T.caption_texts(CAPTIONS, 'plain')]


def test_unknown_form_refused():
    with pytest.raises(AssertionError):
        T.caption_texts(CAPTIONS, 'prefix')
