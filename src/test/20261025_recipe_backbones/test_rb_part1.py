"""Part 1 trainer on a tiny fake cohort (CPU): P and epoch-4 exports appear and leave the old-5,000 arrays alone."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

import rb_paths
import rb_part1
import rb_rows
import rb_stage0
from ab_paths import ms_train

N_SAMPLE, N_FAKE_P = 19889, 600


def unit(rng, n):
    x = rng.standard_normal((n, 2048)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pairs(rng, n_nodes, m):
    code = np.unique(rng.integers(0, n_nodes * n_nodes, size=4 * m))
    a, b = code // n_nodes, code % n_nodes
    keep = a < b
    p = np.stack([a[keep], b[keep]], axis=1)[:m].astype(np.int64)
    return p, rng.uniform(0.2, 1.0, size=len(p)).astype(np.float32)


@pytest.fixture(scope='module')
def cohort():
    rng = np.random.default_rng(5)
    subset = np.sort(rng.permutation(N_SAMPLE)[:5000]).astype(np.int64)
    s = dict(first=np.arange(N_SAMPLE), subset_pid=subset, sample_ids=np.arange(3),
             pid=np.arange(3), all_painting_ids=np.array([str(i) for i in range(N_SAMPLE)]),
             subset_image_indices=np.arange(3))
    fit, dev = ms_train.disjoint_split(s)
    x = dict(img=unit(rng, N_SAMPLE), cap=unit(rng, N_SAMPLE))
    graphs, manifests = {}, {}
    for aspect, row in (('emotion', 'Gres'), ('style', 'Fres05'), ('genre', 'R0o')):
        fp, fw = pairs(rng, len(fit), 1100)
        dp, dw = pairs(rng, len(dev), 300)
        graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev, rows={row: dict(fit=(fp, fw), dev=(dp, dw))})
        manifests[aspect] = dict(identity=dict(method='fake'), archives={aspect: 'x'},
                                 counts={aspect: dict(fit={row: len(fp)}, dev={row: len(dp)})})
    teacher = unit(rng, N_SAMPLE)
    p = dict(img=unit(rng, N_FAKE_P), cap=unit(rng, N_FAKE_P), sha256='f' * 64,
             names=[str(i) for i in range(N_FAKE_P)], counts=dict(train=500, val=100))
    return s, x, graphs, manifests, teacher, p


def run(cohort, tmp, name, with_p, max_epochs=4):
    s, x, graphs, manifests, teacher, p = cohort
    recipe = rb_part1.resolved(name)
    aspects = list(rb_part1.ASPECTS)
    with patch.object(rb_part1, 'N_P', N_FAKE_P), \
            patch.object(ms_train.ags_trainimg, 'aspect_images', lambda s_, key: teacher):
        torch.manual_seed(0)
        return rb_part1.train_cohort(s, {a: x for a in aspects}, graphs, manifests, recipe, name, aspects,
                                     torch.device('cpu'), tmp, p=p if with_p else None,
                                     max_epochs=max_epochs, snapshot_epoch=4)


def load(tmp, name, fname):
    with np.load(tmp / 'part1' / 'train' / name / fname) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def r0(cohort, tmp_path_factory):
    on, off = tmp_path_factory.mktemp('on'), tmp_path_factory.mktemp('off')
    run(cohort, on, 'R0-5', True)
    run(cohort, off, 'R0-5', False)
    return on, off


def test_p_and_epoch4_exports_have_the_right_shapes(r0):
    on, _ = r0
    for aspect in rb_part1.ASPECTS:
        for label, n in (('old5000', 5000), ('P', N_FAKE_P)):
            for tag in ('', '__e4'):
                z = load(on, 'R0-5', f'{aspect}__s0__{label}{tag}.npz')
                assert set(z) == {'img', 'cap'}
                assert all(v.dtype == np.float16 and v.shape == (n, 2048) for v in z.values())
        record = json.loads((on / 'part1/train/R0-5' / f'{aspect}__s0.json').read_text())
        assert record['p_exports'] and record['p_sha256'] == 'f' * 64 and record['snapshot_epoch'] == 4
        assert record['epochs_run'] == 4 and 1 <= record['best_epoch'] <= 4 and record['caption_scored']
        assert len(record['files']) == 4 and len(record['state_files']) == 2
        state = torch.load(on / 'part1/train/R0-5' / f'{aspect}__s0.pt', weights_only=True)
        assert all(k.startswith(('img.', 'cap.')) for k in state)


def test_epoch4_differs_from_best_unless_best_is_4(r0):
    on, _ = r0
    for aspect in rb_part1.ASPECTS:
        record = json.loads((on / 'part1/train/R0-5' / f'{aspect}__s0.json').read_text())
        best, e4 = (load(on, 'R0-5', f'{aspect}__s0__old5000{t}.npz')['img'] for t in ('', '__e4'))
        assert np.array_equal(best, e4) == (record['best_epoch'] == 4)


def test_p_exports_leave_old5000_arrays_and_best_epochs_unchanged(r0):
    on, off = r0
    for aspect in rb_part1.ASPECTS:
        for tag in ('', '__e4'):
            a = load(on, 'R0-5', f'{aspect}__s0__old5000{tag}.npz')
            b = load(off, 'R0-5', f'{aspect}__s0__old5000{tag}.npz')
            assert all(np.array_equal(a[k], b[k]) for k in a)
        ms = f'{aspect}__{ {"emotion": "Gres", "style": "Fres05", "genre": "R0o"}[aspect] }__s0.npz'
        a, b = load(on, 'R0-5', ms), load(off, 'R0-5', ms)
        assert all(np.array_equal(a[k], b[k]) for k in a)
        ja, jb = (json.loads((d / 'part1/train/R0-5' / f'{aspect}__s0.json').read_text()) for d in r0)
        assert ja['best_epoch'] == jb['best_epoch'] and ja['history'][0]['dev_loss'] == jb['history'][0]['dev_loss']
        assert not (off / 'part1/train/R0-5' / f'{aspect}__s0__P.npz').exists()


def test_add_p_from_saved_states_equals_in_process_p(cohort, r0):
    on, off = r0
    s, x, graphs, manifests, teacher, p = cohort
    with patch.object(rb_part1, 'N_P', N_FAKE_P):
        rb_part1.add_p(rb_part1.resolved('R0-5'), 'R0-5', list(rb_part1.ASPECTS), 'cpu', off, p)
    for aspect in rb_part1.ASPECTS:
        for tag in ('', '__e4'):
            a = load(on, 'R0-5', f'{aspect}__s0__P{tag}.npz')
            b = load(off, 'R0-5', f'{aspect}__s0__P{tag}.npz')
            assert all(np.array_equal(a[k], b[k]) for k in a)
        assert json.loads((off / 'part1/train/R0-5' / f'{aspect}__s0.json').read_text())['p_exports']


def test_image_only_recipe_exports_caption_outputs_not_scored(cohort, tmp_path):
    run(cohort, tmp_path, 'R7-3', True)
    for aspect in rb_part1.ASPECTS:
        record = json.loads((tmp_path / 'part1/train/R7-3' / f'{aspect}__s0.json').read_text())
        assert record['caption_scored'] is False and record['caption_side'] == 'not scored'
        z = load(tmp_path, 'R7-3', f'{aspect}__s0__P.npz')
        assert z['cap'].shape == (N_FAKE_P, 2048)


def test_stage0_gate(tmp_path):
    with pytest.raises(AssertionError):
        rb_part1.require_stage0(tmp_path, False)
    code = rb_part1.code_hashes()['all']
    path = tmp_path / 'stage0.json'
    for record, smoke, ok in (({'pass': True, 'smoke': False}, False, True),
                              ({'pass': False, 'smoke': False}, False, False),
                              ({'pass': False, 'smoke': True}, False, False),
                              ({'pass': True, 'smoke': False}, True, False),
                              ({'pass': False, 'smoke': True, 'items': [1]}, True, True)):
        path.write_text(json.dumps(dict(record, code_sha256=code, items=record.get('items', []))))
        if ok:
            rb_part1.require_stage0(tmp_path, smoke)
        else:
            with pytest.raises(AssertionError):
                rb_part1.require_stage0(tmp_path, smoke)
    path.write_text(json.dumps({'pass': True, 'smoke': False, 'code_sha256': 'x', 'items': []}))
    with pytest.raises(AssertionError):
        rb_part1.require_stage0(tmp_path, False)


def test_check_p_rejects_bad_rows():
    good = np.ones((N_FAKE_P, 2048), np.float16) / np.float16(np.sqrt(2048))
    with patch.object(rb_part1, 'N_P', N_FAKE_P):
        rb_part1.check_p(good, 'good')
        for bad in (good[:-1], good * np.float16(1.2), np.where(np.arange(N_FAKE_P)[:, None] == 3, np.nan, good).astype(np.float16)):
            with pytest.raises(AssertionError):
                rb_part1.check_p(bad, 'bad')


def test_stage0_compare(r0, tmp_path):
    on, _ = r0
    out = tmp_path / 'out'
    ab = tmp_path / 'ab'
    for recipe, old in rb_rows.REGRESSION.items():
        (out / 'part1/train' / recipe).mkdir(parents=True)
        (ab / 'train' / old).mkdir(parents=True)
        for aspect, graph in rb_stage0.GRAPHS.items():
            z = load(on, 'R0-5', f'{aspect}__s0__old5000.npz')
            np.savez(out / 'part1/train' / recipe / f'{aspect}__s0__old5000.npz', **z)
            (out / 'part1/train' / recipe / f'{aspect}__s0.json').write_text(json.dumps({'best_epoch': 3}))
            np.savez(ab / 'train' / old / f'{aspect}__{graph}__s0.npz', **z)
            (ab / 'train' / old / f'{aspect}__{graph}__s0.json').write_text(json.dumps({'best_epoch': 3}))
    items = rb_stage0.compare(out, ab)
    assert len(items) == 9 and all('reason' not in i for i in items)
    f = ab / 'train/R6/style__Fres05__s0.npz'
    z = dict(np.load(f))
    z['img'] = z['img'].copy()
    z['img'][0, 0] += np.float16(0.01)
    np.savez(f, **z)
    (ab / 'train/R7/genre__R0o__s0.json').write_text(json.dumps({'best_epoch': 4}))
    bad = {(i['recipe'], i['aspect']) for i in rb_stage0.compare(out, ab) if 'reason' in i}
    assert bad == {('R6-3', 'style'), ('R7-3', 'genre')}


def test_hooks_leave_the_frozen_trainer_bitwise_alone(cohort, r0, tmp_path):
    """The same cohort through plain `ab_train.run_cohort` (no hooks) gives the hooked run's trainer files."""
    import ab_train
    s, x, graphs, manifests, teacher, p = cohort
    recipe = rb_part1.resolved('R0-5')
    aspects = list(rb_part1.ASPECTS)
    with patch.object(ms_train.ags_trainimg, 'aspect_images', lambda s_, key: teacher):
        torch.manual_seed(0)
        ab_train.run_cohort(s, {a: x for a in aspects}, graphs, manifests, recipe, 'R0-5', aspects, 0,
                            torch.device('cpu'), tmp_path, False, 4)
    for aspect, graph in rb_stage0.GRAPHS.items():
        with np.load(tmp_path / 'R0-5' / f'{aspect}__{graph}__s0.npz') as plain:
            hooked = load(r0[0], 'R0-5', f'{aspect}__{graph}__s0.npz')
            assert all(np.array_equal(plain[k], hooked[k]) for k in plain.files)
