"""CPU coverage for decision logic and the two extra validation joins."""
from pathlib import Path

import numpy as np

import tx_data as d
import tx_extras as e
import tx_summary as s


def test_rule_uses_both_comparators_and_requires_nine_cells(monkeypatch):
    records = []
    values = dict(cstar=0.8, cstar_free=0.25, plain=0.2, teacher=0.3)
    for row, value in values.items():
        for a, direction in s.CELLS:
            for readout, delta in [('r0', -0.01), ('center', 0)]:
                records.append(dict(row=row, aspect=a, entry=a, direction=direction,
                                    readout=readout, map=value + delta))
    picks = s.pick_labels(records)
    assert all(r['readout'] == 'center' for r in picks.values())
    pid = np.asarray([0, 0, 1, 2], dtype=np.int64)
    monkeypatch.setattr(s, 'perquery', lambda out, rec: (np.full(4, rec['map']), pid))
    cstar = s.rule(Path('.'), picks, 'cstar', 3)
    free = s.rule(Path('.'), picks, 'cstar_free', 3)
    assert cstar['count'] == 9 and cstar['outcome'] == 1
    assert free['count'] == 0 and free['outcome'] == 2
    assert s.rule(Path('.'), picks, 'cstar', 3, smoke=True)['outcome'] is None
    del picks[('teacher', 'style', 't2i')]
    incomplete = s.rule(Path('.'), picks, 'cstar', 3)
    assert not incomplete['complete'] and incomplete['outcome'] is None


def test_semart_validation_pairs_cpu():
    for name in ('plain', 'siglip2_so400m'):
        text, image, pid, names = e.semart_rows(name, smoke=3)
        assert len(image) == len(names) == 3
        assert len(text) == len(pid)
        np.testing.assert_array_equal(np.unique(pid), np.arange(3))
        np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1, atol=2e-6)


def test_mmart_only_validation_join_and_own_labels():
    index, texts = d.mmart_texts()
    labels, names = e.mmart_labels(index)
    assert len(index) == 5769 and set(texts) == set(d.MMART_COLUMNS)
    assert all(len(v) == len(index) for v in texts.values())
    for aspect, y in labels.items():
        assert y.dtype == np.int64 and y.shape == (5769,)
        assert ((y >= -1) & (y < len(names[aspect]))).all()


def test_teacher_parity_and_instructions_with_fake_encoder(monkeypatch):
    import tx_extract as x
    from pav_prompts import PROMPTS
    c = dict(text=np.asarray(['text'] * 300), sample_id=np.arange(300))
    vectors = np.tile(np.asarray([[1, 0]], dtype=np.float32), (300, 1))
    seen, gates = [], []

    class Encoder:
        instruction = None

        def encode_texts(self, texts, batch_size):
            seen.append((self.instruction, len(texts), batch_size))
            return vectors[:len(texts)]

    monkeypatch.setattr(x, 'completed', lambda *args: False)
    monkeypatch.setattr(x, 'load', lambda *args: Encoder())
    monkeypatch.setattr(x.data, 'val_captions', lambda: c)
    monkeypatch.setattr(x.data, 'text_rows', lambda *args: vectors)
    monkeypatch.setattr(x, 'write_json', lambda path, value: gates.append(value))
    saved = []
    monkeypatch.setattr(x, 'encode_store', lambda *args, **kwargs: saved.append((args, kwargs)))
    x.teacher(Path('.'), 'cpu', 2)
    assert seen == [(PROMPTS['emotion_focus'], 256, 128), (PROMPTS['default'], 256, 128)]
    assert len(saved) == 2 and [r[0][3] for r in saved] == ['style_word', 'genre_word']
    assert gates[0]['default']['mean_cosine'] == 1
