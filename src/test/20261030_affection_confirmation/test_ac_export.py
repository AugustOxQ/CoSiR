"""ac_export on small subsets (CPU only). The PercepT parts need the GPU and are run by the controller."""
import json

import numpy as np
import pytest

import ac_paths
import ac_data
import ac_export
import rs_vh


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(ac_paths, 'OUTPUTS', tmp_path)
    return tmp_path


@pytest.mark.real_data
def test_cstar_export_matches_reinference_and_loader_checks(out):
    requested = ac_export._requested('val', 16)
    assert len(requested) == 16 and all(not k.startswith(ac_paths.FI_SOURCE + '/') for k in requested)
    x = ac_export.default_images(requested)
    assert x.dtype == np.float32 and abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3
    img, sha = ac_export.cstar_infer(x, 'emotion', 0)
    side = dict(keys_sha256=rs_vh.names_sha(requested), array_sha256=rs_vh.array_sha(img), weights_sha256=sha,
                split='val', seed=0, aspect='emotion', rows=16, smoke=False)
    path = ac_export._publish(out / 'exports/cstar/val/emotion__s0.npz', 'img', img, side)
    # ac_data's own reader accepts it (identity, frozen weights hash, array hash, unit norm)
    got = ac_data._export('cstar', 'val', 0, 'emotion', requested)
    np.testing.assert_array_equal(got, img)
    assert img.dtype == np.float16 and img.shape == (16, 2048)
    # a second publish with the same content is kept; different content is refused
    ac_export._publish(path, 'img', img, side)
    with pytest.raises(FileExistsError):
        ac_export._publish(path, 'img', img[::-1].copy(), side)


@pytest.mark.real_data
def test_full_smoke_export_writes_all_heads(out):
    paths = ac_export.export_cstar(('val',), smoke=8)
    assert len(paths) == 15
    side = json.loads(paths[0].with_suffix('.json').read_text())
    assert side['smoke'] is True and side['split'] == 'val' and side['aspect'] == 'emotion'


@pytest.mark.real_data
def test_impressions_export(out):
    path = ac_export.export_impressions(smoke=12)
    side = json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:
        img = z['img']
    assert img.shape == (12, 2048) and img.dtype == np.float16 and len(side['keys']) == 12
    assert side['keys_sha256'] == rs_vh.names_sha(side['keys']) and side['weights_sha256']
    assert np.abs(np.linalg.norm(img.astype(np.float32), axis=1) - 1).max() < 1e-3


@pytest.mark.real_data
def test_reinfer_small_subset_passes(out):
    assert ac_export.reinfer(smoke=48) == 0
    doc = json.loads((out / 'stage0a/cstar_reinfer.json').read_text())
    assert doc['ok'] and len(doc['items']) == 15


def test_percept_device_must_match_training_device():
    with pytest.raises(AssertionError, match='training device'):
        ac_export.percept_infer(0, np.zeros((2, 2048), np.float32), 'cpu')
