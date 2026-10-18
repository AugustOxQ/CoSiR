"""CPU checks of token pooling, ordered publication, and interrupted extraction."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pc_data
import pc_roberta as roberta


class Encoded(dict):
    def to(self, device):
        return self


def test_pooling_includes_special_tokens_and_ignores_padding():
    def tokenizer(texts, **kwargs):
        assert kwargs == dict(padding=True, truncation=True, max_length=64, return_tensors='pt')
        return Encoded(attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))

    class Model:
        training = False

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            values = torch.tensor([[2., 4., 6., 100.], [1., 3., 5., 7.]])
            return SimpleNamespace(last_hidden_state=values[:, :, None].expand(-1, -1, 768))

    x = roberta.encode(['short', 'long'], tokenizer, Model(), 'cpu')
    np.testing.assert_array_equal(x, np.full((2, 768), 4., dtype=np.float32))


def fake_encoder(monkeypatch, s, outdir, fail=False):
    # Nonsequential real IDs make accidental sorting and offset-based IDs visible.
    texts = [f'caption {sample_id}' for sample_id in s['sample_ids']]
    monkeypatch.setattr(pc_data, 'caption_texts', lambda sample: texts)
    monkeypatch.setattr(roberta, 'load_encoder', lambda device: (None, None))

    def encode(batch, tokenizer, model, device):
        assert not (outdir / 'done.json').exists()
        assert not (outdir / 'txt.npy').exists()
        if fail:
            raise RuntimeError('interrupted encoder')
        ids = [int(text.split()[-1]) for text in batch]
        return np.broadcast_to(np.array(ids, dtype=np.float32)[:, None], (len(ids), 768))

    monkeypatch.setattr(roberta, 'encode', encode)


def test_full_store_satisfies_reader_and_preserves_caption_order(tmp_path, monkeypatch):
    s = pc_data.load_frozen()
    outdir = tmp_path / 'store'
    fake_encoder(monkeypatch, s, outdir)
    done = roberta.extract(s, outdir, device='cpu')
    assert set(path.name for path in outdir.iterdir()) == {'txt.npy', 'ids.npz', 'done.json'}
    assert json.loads((outdir / 'done.json').read_text()) == done
    assert done['count'] == 99999 and done['sample_sha256'] == pc_data.fingerprint(s)
    assert done['snapshot_revision'] == roberta.REVISION
    monkeypatch.setattr(pc_data, 'ROBERTA', outdir)
    x = pc_data.caption_features(s, 'roberta')
    np.testing.assert_array_equal(x[:, 0], s['sample_ids'].astype(np.float32))
    np.testing.assert_array_equal(x[:, -1], s['sample_ids'].astype(np.float32))
    with pytest.raises(AssertionError, match='finished store'):
        roberta.extract(s, outdir, device='cpu')


def test_interrupted_write_never_publishes_done(tmp_path, monkeypatch):
    s = pc_data.load_frozen()
    outdir = tmp_path / 'interrupted'
    fake_encoder(monkeypatch, s, outdir, fail=True)
    with pytest.raises(RuntimeError, match='interrupted encoder'):
        roberta.extract(s, outdir, device='cpu')
    assert not (outdir / 'done.json').exists()
    assert not (outdir / 'txt.npy').exists()
    with pytest.raises(AssertionError, match='new or empty'):
        roberta.extract(s, outdir, device='cpu')


def test_smoke_checks_stored_rows_before_publication(tmp_path, monkeypatch):
    s = pc_data.load_frozen()
    outdir = tmp_path / 'smoke'
    fake_encoder(monkeypatch, s, outdir)

    def check(folder, texts, device):
        assert not (folder / 'done.json').exists()
        assert texts == [f'caption {sample_id}' for sample_id in s['sample_ids'][:3]]
        x = np.load(folder / 'txt.pending.npy')
        np.testing.assert_array_equal(x[:, 0], s['sample_ids'][:3])
        raise AssertionError('row comparison failed')

    monkeypatch.setattr(roberta, 'smoke_check', check)
    with pytest.raises(AssertionError, match='row comparison failed'):
        roberta.extract(s, outdir, device='cpu', smoke=True, count=3)
    assert not (outdir / 'done.json').exists()
