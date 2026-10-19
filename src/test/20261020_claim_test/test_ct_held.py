"""CPU parity checks for the shared painting identities, labels and inputs."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import ct_held

for folder in ('20261018_purified_buddy_pilot', '20261017_image_transfer_pilot',
               '20261018_percept_core'):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / folder))
from pbp_graphs import label_vectors
from itp_data import default_images
from pc_stage2 import image_inputs
from pc_data import patch_tokens


HELD_SHA256 = 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')


@pytest.fixture(scope='module')
def sample():
    with np.load(SAMPLE, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def test_held_names_reproduce_the_grouped_split():
    names = ct_held.held_names()
    assert isinstance(names, list) and len(names) == len(set(names)) == 12281
    assert names == sorted(names)
    assert names[0] == 'a.y.-jackson_early-spring-quebec-1923'
    assert names[-1] == 'zinaida-serebriakova_young-girl'
    assert ct_held.names_sha256(names) == HELD_SHA256
    assert hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest() == HELD_SHA256


def test_development_labels_equal_the_existing_scorer(sample):
    subset = sample['subset_pid']
    labels = ct_held.painting_labels(sample['all_painting_ids'][subset].tolist())
    expected = label_vectors(sample, subset)
    assert set(labels) == {'emotion', 'style', 'genre'}
    for aspect in expected:
        assert labels[aspect].dtype == np.int64 and labels[aspect].shape == (5000,)
        np.testing.assert_array_equal(labels[aspect], expected[aspect])


def test_held_list_round_trip_is_idempotent(tmp_path):
    path = tmp_path / 'held' / 'held_paintings.json'
    ct_held.write_held_list(path)
    names, sha256 = ct_held.read_held_list(path)
    assert names == ct_held.held_names() and sha256 == HELD_SHA256
    assert json.loads(path.read_text()) == dict(names=names, sha256=sha256)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    ct_held.write_held_list(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize('fault', ['hash', 'order', 'count', 'different'])
def test_held_list_rejects_corruption_and_refuses_replacement(tmp_path, fault):
    path = tmp_path / 'held_paintings.json'
    names = ct_held.held_names()
    if fault == 'order':
        names = names[::-1]
    elif fault == 'count':
        names = names[:-1]
    elif fault == 'different':
        names[-1] = 'zz-different-painting'
    sha256 = ct_held.names_sha256(names) if fault != 'hash' else 'wrong hash'
    path.write_text(json.dumps(dict(names=names, sha256=sha256)))
    before = path.read_bytes()
    if fault != 'different':
        with pytest.raises(AssertionError):
            ct_held.read_held_list(path)
    with pytest.raises(AssertionError):
        ct_held.write_held_list(path)
    assert path.read_bytes() == before


def test_qwen_default_images_equal_development_rows_bitwise(sample):
    expected = default_images(sample)[0][sample['subset_pid']]
    actual = ct_held.qwen_default_images(sample['subset_painting_ids'].tolist())
    assert actual.shape == (5000, 2048) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_clip_global_images_equal_stage2_inputs_bitwise(sample):
    expected = image_inputs(sample, 'P2', {'eval': sample['subset_pid']}, 'cpu')['eval'].numpy()
    actual = ct_held.clip_global_images(sample['subset_painting_ids'].tolist())
    assert actual.shape == (5000, 512) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_clip_patches_equal_development_rows_bitwise(sample):
    expected = patch_tokens(sample, sample['subset_pid'])
    actual = ct_held.clip_patches(sample['subset_painting_ids'].tolist())
    assert actual.shape == (5000, 50, 512) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('loader,tail', [('qwen_default_images', (2048,)),
                                       ('clip_global_images', (512,)),
                                       ('clip_patches', (50, 512))])
def test_every_held_painting_has_finite_input_rows(loader, tail):
    names = ct_held.held_names()
    x = getattr(ct_held, loader)(names)
    assert x.shape == (12281, *tail) and x.dtype == np.float32
    for start in range(0, len(x), 256):
        assert np.isfinite(x[start:start + 256]).all()
    if len(tail) == 1:
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, rtol=0, atol=2e-7)


def test_held_label_counts_equal_independent_annotation_tallies():
    report = json.loads(Path(__file__).with_name('ct_held_counts.json').read_text())
    assert report['rows'] == 12281 and report['names_sha256'] == HELD_SHA256
    labels = ct_held.painting_labels(ct_held.held_names())
    for aspect, values in labels.items():
        expected = report['aspects'][aspect]
        assert values.shape == (12281,) and values.dtype == np.int64
        assert int((values < 0).sum()) == expected['masked']
        assert np.bincount(values[values >= 0]).tolist() == [c['count'] for c in expected['classes']]
        assert sum(c['count'] for c in expected['classes']) + expected['masked'] == 12281
    assert report['aspects']['genre']['masked'] == 2332


@pytest.mark.parametrize('loader', ['painting_labels', 'qwen_default_images',
                                   'clip_global_images', 'clip_patches'])
def test_requested_order_and_duplicates_are_preserved(loader, sample):
    names = sample['subset_painting_ids'][:3].tolist()
    load = getattr(ct_held, loader)
    original = load(names)
    repeated = load([names[2], names[0], names[2], names[1]])
    if loader == 'painting_labels':
        for aspect in original:
            np.testing.assert_array_equal(repeated[aspect], original[aspect][[2, 0, 2, 1]])
    else:
        np.testing.assert_array_equal(repeated.view(np.uint32), original[[2, 0, 2, 1]].view(np.uint32))


def test_style_must_be_constant_per_painting(tmp_path, monkeypatch, sample):
    records = json.loads(ct_held.ANNOTATIONS.read_text())
    name = str(sample['subset_painting_ids'][0])
    rows = [i for i, record in enumerate(records) if record['painting'] == name]
    assert len(rows) > 1
    records[rows[-1]]['art_style'] = next(record['art_style'] for record in records
                                         if record['art_style'] != records[rows[0]]['art_style'])
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(records))
    monkeypatch.setattr(ct_held, 'ANNOTATIONS', path)
    with pytest.raises(AssertionError, match='style must be constant'):
        ct_held.painting_labels([name])
