"""Part 1 reported readings (ticket 04): one end-to-end test on small synthetic exports, plus the pick gate."""
import hashlib
import json

import numpy as np
import pytest

import rb_paths  # noqa: F401  (sys.path)
import rb_ami
import rb_part1_report as rep
import rb_pick
import rb_rows
import test_rb_pick as tp

N, N_OLD = tp.N, 120
KEYS = rep.REFERENCE_KEYS


def unit16(rng, y, centres, signal, n):
    x = signal * centres[np.where(y >= 0, y, 0)] + rng.standard_normal((n, 2048))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def add_files(tmp_path, labels, old_labels):
    """Old-5,000 and epoch-4 exports and the trainer record fields the report reads, next to the pick's P exports."""
    rng = np.random.default_rng(9)
    centres = {a: rng.standard_normal((tp.CLASSES[a], 2048)) for a in tp.CLASSES}
    for recipe in rb_rows.ORDER:
        folder = tmp_path / 'part1' / 'train' / recipe
        for a in rb_pick.ASPECTS:
            path = folder / f'{a}__s0.json'
            record = json.loads(path.read_text())
            files = record['files']
            for label, tag, y, n in (('old5000', '', old_labels[a], N_OLD), ('P', '__e4', labels[a], N),
                                     ('old5000', '__e4', old_labels[a], N_OLD)):
                img, cap = unit16(rng, y, centres[a], 0.05, n), unit16(rng, y, centres[a], 0.05, n)
                name = f'{a}__s0__{label}{tag}.npz'
                np.savez(folder / name, img=img, cap=cap)
                files[name] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tp.sha(v))
                               for k, v in (('img', img), ('cap', cap))}
            record.update(best_epoch=7, epochs_run=12, snapshot_epoch=4, caption_scored=recipe not in rb_rows.IMAGE_ONLY)
            path.write_text(json.dumps(record))


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('report')
    signal = {r: tp.flat(0.05) for r in rb_rows.ORDER}
    labels, p_sha = tp.setup(tmp_path, signal)
    rb_pick.run(tmp_path, labels, n_draws=10, workers=2, n_rows=N)
    old_labels = tp.synthetic_labels(seed=4)
    old_labels = {a: v[:N_OLD].copy() for a, v in old_labels.items()}
    add_files(tmp_path, labels, old_labels)
    return tmp_path, labels, old_labels


def refs(seed=1):
    rng = np.random.default_rng(seed)
    def make():
        x = rng.standard_normal((N, 2048)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    return {k: make for k in KEYS}


def test_report_end_to_end_values_files_and_gate(tree, tmp_path):
    outputs, labels, old_labels = tree
    pick_path = outputs / 'part1' / 'pick.json'
    pick_before = pick_path.read_bytes()
    record = rep.run(outputs, labels, old_labels, refs(), n_rows=N, n_old=N_OLD)
    assert pick_path.read_bytes() == pick_before                       # never written
    pick = json.loads(pick_path.read_text())
    assert record['recipes'] == list(rb_rows.ORDER) and set(record['references_p']) == set(KEYS)
    # values: direct computation for one recipe and aspect
    old = np.load(outputs / 'part1' / 'train' / 'R6-3' / 'genre__s0__old5000.npz')['img']
    y = old_labels['genre']
    direct = rb_ami.score(y, rb_ami.partitions(old), np.flatnonzero(y >= 0))
    assert record['old5000']['R6-3']['genre'] == pytest.approx(direct, abs=1e-12)
    assert record['p_image_pick']['R7-5']['style'] == pytest.approx(pick['recipes']['R7-5']['style']['A'], abs=1e-9)
    cap = np.load(outputs / 'part1' / 'train' / 'R0-3' / 'emotion__s0__P.npz')['cap']
    y = labels['emotion']
    assert record['p_caption']['R0-3']['emotion'] == pytest.approx(
        rb_ami.score(y, rb_ami.partitions(cap), np.flatnonzero(y >= 0)), abs=1e-12)
    assert record['p_caption']['R7-5']['style'] == 'not scored' and record['p_caption']['R7-3']['genre'] == 'not scored'
    for by in record['p_leiden'].values():
        assert all(len(v['seeds']) == 3 and np.isfinite(v['mean']) for v in by.values())
    assert record['best_epochs']['R0-5']['emotion'] == dict(best_epoch=7, epochs_run=12, snapshot_epoch=4)
    assert all(np.isfinite(v) for by in record['p_e4'].values() for v in by.values())
    # files: json and md, 4 decimals, rerun identical, other results refused
    folder = outputs / 'part1'
    md = (folder / 'reported.md').read_text()
    assert json.loads((folder / 'reported.json').read_text()) == record
    assert f"{record['old5000']['R0-5']['emotion']:.4f}" in md and 'not scored' in md and 'Leiden' in md
    assert rep.run(outputs, labels, old_labels, refs(), n_rows=N, n_old=N_OLD) == record
    with pytest.raises(AssertionError, match='other results'):
        rep.run(outputs, labels, old_labels, refs(seed=2), n_rows=N, n_old=N_OLD)
    # wrong labels (not the pick's) are refused before any reading
    wrong = dict(labels, emotion=np.roll(labels['emotion'], 1))
    with pytest.raises(AssertionError, match="P labels differ"):
        rep.run(outputs, wrong, old_labels, refs(), n_rows=N, n_old=N_OLD)


def test_no_pick_no_report(tree, tmp_path):
    _, labels, old_labels = tree
    with pytest.raises(AssertionError, match='pick.json missing'):
        rep.run(tmp_path, labels, old_labels, refs(), n_rows=N, n_old=N_OLD)
    assert not (tmp_path / 'part1').exists()
