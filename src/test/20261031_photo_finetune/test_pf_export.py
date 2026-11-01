"""Ticket 06: the export loader's three refusals on a tiny fake run (CPU); gate (iii) (slow, real data, controller)."""
import json

import numpy as np
import pytest
import torch

import pf_data
import pf_export
import pf_train
import rs_vh

KEYS = {'val': [f'src/v{i}.jpg' for i in range(5)], 'test': [f'src/t{i}.jpg' for i in range(4)]}
RUN = 'sweep/row1'


@pytest.fixture()
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(pf_data, 'keys', lambda split: KEYS[split])
    _, view = pf_train.build_emotion(torch.device('cpu'), seed=0)
    state = {k: v.clone() for k, v in view.state_dict().items()}
    folder = tmp_path / 'runs' / RUN
    folder.mkdir(parents=True)
    torch.save(state, folder / 'emotion.pt')
    sha = rs_vh.file_sha(folder / 'emotion.pt')
    (folder / 'run.json').write_text(json.dumps(dict(status='ok', weights_sha256=sha)))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 2048)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    img = pf_export.infer(state, x)
    assert img.dtype == np.float16 and img.shape == (5, 2048)
    pf_export._write(RUN, 'val', KEYS['val'], img, sha, tmp_path)
    return tmp_path, folder, img, sha


def test_loader_accepts_and_embeds(run):
    out, _, img, _ = run
    assert np.array_equal(pf_export.stored(RUN, 'val', outputs=out), img)
    e = pf_export.embed(RUN, 'val', outputs=out)
    assert e.dtype == np.float32 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)


def test_refuses_other_weights(run):
    out, folder, _, sha = run
    side_path = out / 'exports' / RUN / 'val/emotion.json'
    side = json.loads(side_path.read_text())
    side['weights_sha256'] = '1' * 64                          # export made by another checkpoint
    side_path.write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='weights hash'):
        pf_export.stored(RUN, 'val', outputs=out)
    side['weights_sha256'] = sha
    side_path.write_text(json.dumps(side))
    (folder / 'run.json').write_text(json.dumps(dict(status='ok', weights_sha256='0' * 64)))
    with pytest.raises(AssertionError, match='differs from run.json'):   # run.json disagrees with the checkpoint
        pf_export.stored(RUN, 'val', outputs=out)


def test_refuses_wrong_key_list(run, monkeypatch):
    out = run[0]
    monkeypatch.setattr(pf_data, 'keys', lambda split: list(reversed(KEYS[split])))
    with pytest.raises(AssertionError, match='keys hash'):
        pf_export.stored(RUN, 'val', outputs=out)


def test_refuses_wrong_array_hash(run):
    out = run[0]
    side_path = out / 'exports' / RUN / 'val/emotion.json'
    side = json.loads(side_path.read_text())
    side['array_sha256'] = '2' * 64
    side_path.write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='array hash'):
        pf_export.stored(RUN, 'val', outputs=out)


def test_publish_refuses_conflict(run):
    out, _, img, sha = run
    with pytest.raises(FileExistsError):
        pf_export._write(RUN, 'val', KEYS['val'], img[::-1].copy(), sha, out)


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_iii(tmp_path):
    result = pf_export.gate_iii(outputs=tmp_path)
    assert len(result['max_abs_diff']) == 10 and result['ok'], result['max_abs_diff']
