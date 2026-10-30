"""CPU tests of ac_extract: keys, path resolution on real keys, sealing, refusal to overwrite (fake encoder)."""
import json

import numpy as np
import pytest

import ac_extract as X
import ac_paths as P
import rs_vh

pytestmark = pytest.mark.real_data


class FakeEnc:
    def encode_images(self, ims, batch_size=32):
        rng = np.random.default_rng(len(ims))
        v = rng.normal(size=(len(ims), X.WIDTH['emotion_focus']))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope='module')
def real_keys():
    keys = X.valtest_keys()
    fi = [k for k in keys if k.startswith(P.FI_SOURCE + '/')]
    coco = [k for k in keys if k.startswith('coco_affective/')]
    assert len(keys) == 12752 and len(set(keys)) == 12752
    return coco[:2] + [k for k in keys if 'val2014' in k][:1] + fi[:2] + fi[-1:]


def test_check_keys():
    with pytest.raises(AssertionError):
        X.check_keys(['a/b', 'a/b'])
    with pytest.raises(AssertionError):
        X.check_keys(['nokey'])
    assert X.parse_keys('["a/b", "c/d"]') == ['a/b', 'c/d']


def test_resolve_real_keys(real_keys):
    spec, rows = X.spec_for(real_keys)
    assert len(rows) == len(real_keys)
    for k, r in zip(real_keys, rows):
        im = spec.load_image(int(r))
        assert im.mode == 'RGB' and min(im.size) > 10
        im.close()
    with pytest.raises(AssertionError):
        X.spec_for(['coco_affective/COCO_train2014_999999999999.jpg'])


def test_smoke_keys_cover_fi():
    s = X.smoke_keys(64)
    assert len(s) == len(set(s)) == 64 and any(k.startswith(P.FI_SOURCE) for k in s)


def test_parity_keys_b():
    k = X.parity_keys_b()
    assert len(k) == 200 and sum(x.startswith(P.FI_SOURCE) for x in k) == 100
    import ac_data
    assert set(k) <= set(ac_data.keys('val'))


def test_parity_report():
    a = np.eye(3, dtype=np.float32)
    assert X.parity_report(a, a)['ok']
    r = X.parity_report(a, np.roll(a, 1, axis=1))
    assert not r['ok'] and r['min_cosine'] == 0


def test_extract_seal_and_refuse(real_keys, tmp_path, monkeypatch):
    monkeypatch.setattr(X, 'load_encoder', lambda view, device='cuda': (
        FakeEnc(), dict(model_id='fake', model_revision='0' * 40)))
    out = tmp_path / 'emotion_focus'
    meta = X.extract('emotion_focus', real_keys, out, smoke=True)
    assert not out.with_name('emotion_focus.pending').exists()
    assert sorted(p.name for p in out.iterdir()) == ['img.npy', 'keys.json', 'meta.json']
    x = np.load(out / 'img.npy')
    assert x.dtype == np.float16 and x.shape == (len(real_keys), 2048)
    assert json.loads((out / 'keys.json').read_text()) == real_keys
    m = json.loads((out / 'meta.json').read_text())
    assert m == json.loads(json.dumps(meta)) and m['smoke'] is True and m['gate']['ok']
    assert m['keys_sha256'] == rs_vh.names_sha(real_keys) and m['img_sha256'] == rs_vh.array_sha(x)
    assert m['instruction'].startswith('Represent the emotion that this artwork')
    before = (out / 'img.npy').read_bytes()
    with pytest.raises(AssertionError, match='not overwritten'):
        X.extract('emotion_focus', real_keys, out, smoke=True)
    assert (out / 'img.npy').read_bytes() == before
    pend = tmp_path / 'p.pending'
    pend.mkdir()
    with pytest.raises(AssertionError, match='pending'):
        X.extract('emotion_focus', real_keys, tmp_path / 'p', smoke=True)


def test_canonical_and_gate_guards(real_keys, tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match='smoke'):
        X.extract('emotion_focus_photo', real_keys, P.VIEWS / 'emotion_focus_photo', smoke=True)
    with pytest.raises(AssertionError, match='valtest'):
        X.extract('clip_l14', real_keys, P.CLIP_VIEW, smoke=False)

    class Bad(FakeEnc):
        def encode_images(self, ims, batch_size=32):
            return super().encode_images(ims) * 1.5
    monkeypatch.setattr(X, 'load_encoder', lambda view, device='cuda': (Bad(), dict(model_id='f', model_revision='0')))
    # encode_rows asserts shape only; the sealing gate must reject non-unit rows and leave only .pending
    with pytest.raises(AssertionError, match='gate'):
        X.extract('emotion_focus', real_keys, tmp_path / 'bad', smoke=True)
    assert not (tmp_path / 'bad').exists()


def test_instructions():
    assert X.instruction_for('emotion_focus_photo') == P.PHOTO_INSTRUCTION
    assert X.instruction_for('clip_l14') is None
    assert X.instruction_for('default') == "Represent the user's input."
