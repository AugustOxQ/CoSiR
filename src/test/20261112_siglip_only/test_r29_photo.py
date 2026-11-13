"""Tests of r29_photo (ticket 12): the Affection SigLIP 1 caption store and its gate (viii), the caption mean D (D16),
the guards (pass rule, K* keys, start checkpoint widths), and the photo trainer against r14's (gate (vii) trainer part)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_photo as PH  # noqa: E402
from test_r29_extract import encoder  # noqa: E402  (the fake SigLIP text encoder of ticket 02's tests)

BP, P, X = PH.BP, PH.P, PH.X
PREFIX = P.PREFIX


@pytest.fixture
def work():
    root = P.REPO / '.scratch/r29-t12-tests'
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ store and gate (viii)
def reference(extra_rows=0):
    images = [dict(source='coco', image_name='a.jpg', splits=['train'], n_texts=2),
              dict(source='flickr', image_name='b.jpg', splits=['val'], n_texts=3)]
    texts = [dict(img=0, csv_row=10, emotion='x', workerid='w', split='train'),
             dict(img=1, csv_row=11, emotion='x', workerid='w', split='val'),
             dict(img=0, csv_row=12, emotion='x', workerid='w', split='ignored'),
             dict(img=1, csv_row=13, emotion='x', workerid='w', split='val'),
             dict(img=1, csv_row=15, emotion='x', workerid='w', split='val')]
    return dict(dataset='affection_local', backbone='qwen3vl_emb_2b', n_images=2, n_texts=5, images=images,
                texts=texts, extra=dict(sources=['coco', 'flickr']))


def write_csv(path, utterances=None):
    rows = ['workerid,image_name,dataset,utterance,emotion,split']
    names = {10: ('a.jpg', 'coco', 'train'), 11: ('b.jpg', 'flickr', 'val'), 12: ('a.jpg', 'coco', 'ignored'),
             13: ('b.jpg', 'flickr', 'val'), 15: ('b.jpg', 'flickr', 'val')}
    for r in range(16):
        name, source, split = names.get(r, ('z.jpg', 'other', 'train'))
        text = (utterances or {}).get(r, f'Text number {r} LOUD')
        rows.append(f'w,{name},{source},{text},x,{split}')
    Path(path).write_text('\n'.join(rows) + '\n')


def test_affection_texts_join_by_csv_row_and_check_photo_and_split(work):
    ref, csv = reference(), work / 'a.csv'
    write_csv(csv)
    assert PH.affection_texts(ref, csv) == [f'Text number {r} LOUD' for r in (10, 11, 12, 13, 15)]
    bad = json.loads(json.dumps(ref))
    bad['texts'][1]['img'] = 0                       # the row's photo is another photo
    with pytest.raises(AssertionError, match='photo'):
        PH.affection_texts(bad, csv)
    bad = json.loads(json.dumps(ref))
    bad['texts'][0]['split'] = 'val'
    with pytest.raises(AssertionError, match='split'):
        PH.affection_texts(bad, csv)


def test_affection_texts_refuse_empty_text(work):
    write_csv(work / 'a.csv', {11: ' '})
    with pytest.raises(AssertionError):
        PH.affection_texts(reference(), work / 'a.csv')


@pytest.mark.parametrize('form', ['plain', 'prefixed'])
def test_build_store_keeps_the_references_rows_and_prefix(work, form):
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv')
    texts = PH.affection_texts(ref, work / 'a.csv')
    check = PH.build_store(ref, texts, enc, work / 'store', form=form)
    index = json.loads((work / 'store/index.json').read_text())
    assert check['n_texts'] == 5 and index['texts'] == ref['texts'] and index['images'] == ref['images']
    assert index['prefix'] == ('' if form == 'plain' else PREFIX.lower()) and index['text_only'] is True
    sent = enc.p.tokenizer.calls[0][0]
    assert all(t.startswith('represent the caption: ') == (form == 'prefixed') for t in sent)
    assert PH.X.check_store(work / 'store', dict(ref, texts=ref['texts'][::-1]),
                            expected_prefix=None if form == 'plain' else PREFIX, scheme='affection')['ok']


def test_store_check_joins_by_csv_row_and_photo(work):
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv')
    PH.build_store(ref, PH.affection_texts(ref, work / 'a.csv'), enc, work / 'store', form='plain')
    wrong = json.loads(json.dumps(ref))
    wrong['texts'][1]['img'] = 0
    with pytest.raises(AssertionError, match='mismatch'):
        X.check_store(work / 'store', wrong, scheme='affection')
    missing = json.loads(json.dumps(ref))
    missing['texts'][-1]['csv_row'] = 99
    with pytest.raises(AssertionError, match='coverage'):
        X.check_store(work / 'store', missing, scheme='affection')


def test_gate_viii_records_counts_truncation_and_refuses_after_three_tries(work):
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv', {12: 'x ' * 70})      # one text over 64 tokens: cut, counted, row kept (D14)
    PH.build_store(ref, PH.affection_texts(ref, work / 'a.csv'), enc, work / 'store', form='plain')
    rec = PH.gate_viii(work / 'store', ref, form='plain', record_folder=work / 'gates')
    assert rec['gate'] == 'viii_affection' and rec['status'] == 'pass' and rec['tries'] == 1
    d = rec['details']
    assert d['n_texts'] == d['n_texts_reference'] == 5 and d['photo_counts_equal'] is True
    assert d['n_truncated'] == 1 and d['truncation_flag'] and d['truncation_limit'] == 64
    assert d['form'] == 'plain' and d['prefix'] == '' and d['index_sha256'] == PH.store_hash(work / 'store')
    PH.gate_viii(work / 'store', ref, form='plain', record_folder=work / 'gates')
    PH.gate_viii(work / 'store', ref, form='plain', record_folder=work / 'gates')
    with pytest.raises(AssertionError, match='three tries'):
        PH.gate_viii(work / 'store', ref, form='plain', record_folder=work / 'gates')


def test_gate_viii_fails_and_records_a_store_of_another_form(work):
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv')
    PH.build_store(ref, PH.affection_texts(ref, work / 'a.csv'), enc, work / 'store', form='plain')
    with pytest.raises(AssertionError, match='prefix'):
        PH.gate_viii(work / 'store', ref, form='prefixed', record_folder=work / 'gates')
    rec = json.loads((work / 'gates/viii_affection.json').read_text())
    assert rec['status'] == 'fail' and 'prefix' in rec['details']['error']


# ------------------------------------------------------------------------------------------------ the caption mean (D16)
def make_store(work, rows, csv_rows, name='st'):
    """A text store folder with the given per-text float16 rows for the given csv_rows (its own order)."""
    folder = work / name
    folder.mkdir()
    np.save(folder / 'txt.npy', np.asarray(rows, dtype=np.float16))
    index = dict(texts=[dict(csv_row=c, img=0, split='train') for c in csv_rows])
    (folder / 'index.json').write_text(json.dumps(index))
    return folder


def test_caption_means_join_by_csv_row_use_eligible_rows_only_and_unit_norm(work, monkeypatch):
    # photo a: rows 10 and 12 (12 is ignored: not eligible); photo b: rows 11, 13
    monkeypatch.setattr(PH.PD, '_eligible_rows', lambda keys: dict(
        key=['b', 'a', 'b'], csv_row=np.array([13, 10, 11]), store_row=np.array([0, 0, 0])))
    rows = {10: [1., 0, 0], 11: [0, 2., 0], 12: [0, 0, 9.], 13: [0, 0, 2.]}
    order = [13, 12, 11, 10]                           # the store's own order differs from the default index's
    store = make_store(work, [rows[c] for c in order], order)
    got = PH.caption_means(['a', 'b'], store, width=3)
    expect_b = np.array([0, 2., 2.]) / 2
    expect = np.stack([[1., 0, 0], expect_b / np.linalg.norm(expect_b)]).astype(np.float32)
    assert got.dtype == np.float32 and got.shape == (2, 3)
    np.testing.assert_allclose(got, expect, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1, atol=1e-6)


def test_caption_means_refuse_a_missing_row_and_a_wrong_width(work, monkeypatch):
    monkeypatch.setattr(PH.PD, '_eligible_rows', lambda keys: dict(
        key=['a', 'a'], csv_row=np.array([10, 11]), store_row=np.array([0, 1])))
    store = make_store(work, [[1., 0, 0]], [10])
    with pytest.raises(AssertionError, match='csv_row'):
        PH.caption_means(['a'], store, width=3)
    store = make_store(work, [[1., 0, 0], [0, 1., 0]], [10, 11], name='st2')
    with pytest.raises(AssertionError, match='width'):
        PH.caption_means(['a'], store, width=4)


# ---------------------------------------------------------------------------------------------------------- guards
def test_read_prefix_pick_asserts_keys_and_consistency(work):
    path = work / 'pick.json'
    good = dict(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='plain', flag=None)
    path.write_text(json.dumps(good))
    assert PH.read_prefix_pick(path) == good
    for bad in (dict(good, kstar='siglip1'), dict(good, kstar='qwen'), {k: v for k, v in good.items() if k != 'p1_form'},
                dict(good, kstar_form='both')):
        path.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            PH.read_prefix_pick(path)


@pytest.mark.parametrize('doc', [None, dict(status='fail'), dict(status='parked'), dict()])
def test_train_refuses_unless_the_pass_rule_says_pass(work, doc):
    path = work / 'pass_rule.json'
    if doc is not None:
        path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='pass'):
        PH.require_pass(path)
    path.write_text(json.dumps(dict(status='pass')))
    PH.require_pass(path)


def test_train_command_refuses_before_reading_anything_without_pass(work, monkeypatch):
    monkeypatch.setattr(PH, 'PASS_RULE', work / 'none.json')
    monkeypatch.setattr(PH, 'read_prefix_pick', lambda *a: pytest.fail('read before the pass check'))
    with pytest.raises(AssertionError, match='pass'):
        PH.main(['train', '--device', 'cpu'])


# -------------------------------------------------------------------------------------------------- the trainer
def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def part(n, width, cap_width, seed):
    rng = np.random.default_rng(seed)
    return dict(keys=[f'k{i}' for i in range(n)], I=unit(rng.normal(size=(n, width))),
                D=unit(rng.normal(size=(n, cap_width))), T=unit(rng.normal(size=(n, 2048))))


def pairs(n, count, seed):
    rng = np.random.default_rng(seed)
    p = {(int(a), int(b)) for a, b in (sorted(rng.choice(n, 2, replace=False)) for _ in range(count))}
    return np.array(sorted(p), dtype=np.int64)


SETTINGS = dict(start='random', lr=0.001, k=80, correction=True, teacher_weight=0.25, epochs=1)
CPU = torch.device('cpu')


def tiny(cap_width):
    return part(40, 1152, cap_width, 1), part(14, 1152, cap_width, 2), (pairs(40, 120, 3), pairs(14, 30, 4))


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and tuple(a) == tuple(b)


def test_qwen_caption_equals_r14s_start_state_loss_gradients_and_trained_state():
    fit, val, graph = tiny(2048)
    out = PH.compare_trainers(fit, val, graph, SETTINGS, 1152, CPU, seed=100, cstar_root=None)
    assert out == dict(start_state=True, loss=True, gradients=True, trained_state=True, history=True)


def test_compare_trainers_detects_a_changed_trainer(monkeypatch):
    fit, val, graph = tiny(2048)
    real, loss = PH.caption_patches, BP.ab_train.switched_head_loss

    def patched(*a, **k):                       # r14's side has run by now: only the new side sees a scaled loss
        monkeypatch.setattr(BP.ab_train, 'switched_head_loss', lambda *x, **y: loss(*x, **y) * 1.001)
        return real(*a, **k)

    monkeypatch.setattr(PH, 'caption_patches', patched)
    out = PH.compare_trainers(fit, val, graph, SETTINGS, 1152, CPU, seed=100, cstar_root=None)
    assert out['start_state'] and not out['loss'] and not out['gradients'] and not out['history']


def test_width_1152_changes_only_the_caption_trunks_first_layer():
    state = lambda cap: {k: v.shape for k, v in PH.build_state(1152, cap, 100).items()}   # noqa: E731
    a, b = state(2048), state(1152)
    assert tuple(a) == tuple(b) == BP.pf_train.CKPT_KEYS
    assert [k for k in a if a[k] != b[k]] == ['cap.trunk.0.weight']
    assert tuple(b['cap.trunk.0.weight'])[1] == 1152 and tuple(a['cap.trunk.0.weight'])[1] == 2048


def test_siglip_caption_trains_one_epoch_and_a_wrong_caption_width_is_refused():
    fit, val, graph = tiny(1152)
    with PH.caption_patches('siglip1'):
        state, epoch, dev, hist = BP.fit_photo('t', SETTINGS, 1152, 0, 100, fit, val, graph, CPU)
        assert state['cap.trunk.0.weight'].shape[1] == 1152 and np.isfinite(dev) and epoch == 1
        wrong = tiny(2048)[0]
        with pytest.raises(AssertionError):
            BP.fit_photo('t', SETTINGS, 1152, 0, 100, wrong, val, graph, CPU)


def test_patches_unwind_after_success_and_failure():
    before = (BP.build_emotion, BP._tensors, BP.start_root, BP.train_run, BP.pf_train.checked_load)
    with PH.caption_patches('siglip1'):
        assert BP.build_emotion is not before[0]
    with pytest.raises(RuntimeError):
        with PH.caption_patches('siglip1'):
            raise RuntimeError
    assert (BP.build_emotion, BP._tensors, BP.start_root, BP.train_run, BP.pf_train.checked_load) == before


def test_qwen_caption_installs_the_model_and_tensor_patches_but_no_start_patch():
    before = (BP.start_root, BP.pf_train.checked_load)
    with PH.caption_patches('qwen'):
        assert BP.build_emotion.__module__ == 'r29_photo' and BP._tensors.__module__ == 'r29_photo'
        assert (BP.start_root, BP.pf_train.checked_load) == before       # gate (vii) starts from K0's checkpoint


def test_gate_vii_side_b_runs_the_patched_build_and_tensor_check_at_2048(monkeypatch):
    fit, val, graph = tiny(2048)
    seen = []
    real = PH.M.build_model
    monkeypatch.setattr(PH.M, 'build_model', lambda *a, **k: (seen.append(k['cap_width']), real(*a, **k))[1])
    out = PH.compare_trainers(fit, val, graph, SETTINGS, 1152, CPU, seed=100, cstar_root=None)
    assert out == dict(start_state=True, loss=True, gradients=True, trained_state=True, history=True)
    assert seen and set(seen) == {2048}                                  # F1's build ran (r14's side never calls it)


def test_gate_vii_fails_when_f1s_model_build_differs_at_2048(monkeypatch):
    fit, val, graph = tiny(2048)
    real = PH.M.build_model

    def build(*a, **k):
        models, views = real(*a, **k)
        with torch.no_grad():
            models['cap'].trunk[0].weight.add_(1e-3)
        return models, views

    monkeypatch.setattr(PH.M, 'build_model', build)
    out = PH.compare_trainers(fit, val, graph, SETTINGS, 1152, CPU, seed=100, cstar_root=None)
    assert not out['start_state'] and not out['loss'] and not out['trained_state']


def test_gate_vii_fails_when_f1s_tensor_check_changes_the_caption_input(monkeypatch):
    fit, val, graph = tiny(2048)
    real = PH._tensors

    def tensors(cap_width):
        inner = real(cap_width)

        def f(part, device, width):
            out = inner(part, device, width)
            out['cap'] = out['cap'] * 1.001
            return out
        return f

    monkeypatch.setattr(PH, '_tensors', tensors)
    out = PH.compare_trainers(fit, val, graph, SETTINGS, 1152, CPU, seed=100, cstar_root=None)
    assert out['start_state'] and not out['loss'] and not out['gradients']


# ---------------------------------------------------------------------------- start checkpoint (sidecar widths)
def save_start(folder, seed, *, cap_width=1152, cap_source='siglip1', store='hash-siglip1', width=1152):
    """A start checkpoint of `cap_width` with a sidecar as ticket 03 writes (rb block with the cap fields)."""
    folder.mkdir(parents=True, exist_ok=True)
    state = PH.build_state(width, cap_width, 5)
    path = folder / f'emotion__s{seed}.pt'
    torch.save(state, path)
    side = dict(weights_sha256=PH.rs_vh.file_sha(path), config=dict(rb=dict(
        width=width, cap_width=cap_width, cap_source=cap_source, cap_store_sha256=store)))
    path.with_suffix('.json').write_text(json.dumps(side))
    return folder


@pytest.fixture
def hashes(monkeypatch):
    monkeypatch.setattr(PH.C, 'store_hash', lambda source: f'hash-{source}')


def load_start(folder, source='siglip1'):
    with PH.caption_patches(source):
        _, view = BP.build_emotion(CPU, 1152, seed=1)
        return BP.pf_train.checked_load(view, 0, folder)


def test_start_checkpoint_of_the_right_source_and_width_loads(work, hashes):
    assert len(load_start(save_start(work / 'w', 0))) == 64


@pytest.mark.parametrize('change,message', [
    (dict(cap_width=2048), 'cap_width'), (dict(cap_source='siglip1_prefix'), 'cap_source'),
    (dict(store='other'), 'cap_store_sha256'), (dict(width=2048), 'width')])
def test_start_checkpoint_sidecar_mismatch_is_refused(work, hashes, change, message):
    folder = save_start(work / 'w', 0)
    side = json.loads((folder / 'emotion__s0.json').read_text())
    side['config']['rb'].update({{'store': 'cap_store_sha256'}.get(k, k): v for k, v in change.items()})
    (folder / 'emotion__s0.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match=message):
        load_start(folder)


def test_start_checkpoint_without_cap_fields_is_refused(work, hashes):
    folder = save_start(work / 'w', 0)
    side = json.loads((folder / 'emotion__s0.json').read_text())
    del side['config']['rb']['cap_width']
    (folder / 'emotion__s0.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='cap_width'):
        load_start(folder)


def test_start_root_is_the_kstar_arm_folder(work):
    with PH.caption_patches('siglip1_prefix'):
        assert BP.start_root('siglip1', work) == work / 'weights' / 'siglip1_prefix'
        assert BP.start_root('siglip1', None) == P.OUT / 'cstar' / 'siglip1' / 'weights' / 'siglip1_prefix'


# ------------------------------------------------------------------- train through r14's loop, records and layout
class FakeGraphs:
    def __init__(self, graph):
        self.graph = graph

    def get(self, k, correction):
        assert (k, correction) == (80, True)
        return dict(fit_pairs=self.graph[0], dev_pairs=self.graph[1], manifest_sha256='g')


def test_train_writes_r14s_layout_with_caption_fields_from_kstars_start(work, hashes, monkeypatch):
    monkeypatch.setattr(PH, 'PASS_RULE', work / 'pass_rule.json')
    (work / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    monkeypatch.setattr(P, 'SMOKE_OUT', work)
    root = work / 'run'
    save_start(root / 'cstar/siglip1/weights/siglip1', 0)
    fit, val, graph = tiny(1152)
    cap = dict(cap_store_sha256='store-hash', cap_sha256='arrays')
    store = good_store(work, smoke=True)
    out = PH.train_arm('siglip1', [0], CPU, root, data=(fit, val, fit, val), graphs=FakeGraphs(graph), max_epochs=1,
                       cap_info=cap, smoke=True, store=store, gate_folder=work / 'gates')
    run = json.loads((root / 'photo/siglip1/s0/run.json').read_text())
    assert (root / 'photo/siglip1/s0/emotion.pt').is_file() and run['status'] == 'ok'
    assert run['train_seed'] == 100 and run['cstar_seed'] == 0 and run['encoder_width'] == 1152
    assert run['settings'] == dict(BP.PICK) and run['row'] == 'ref'
    assert (run['cap_source'], run['cap_width'], run['cap_store_sha256'], run['cap_sha256'], run['kstar_form']) == \
        ('siglip1', 1152, 'store-hash', 'arrays', 'plain')
    assert run['cstar_weights_sha256'] and out[('ref', 0)].state['cap.trunk.0.weight'].shape[1] == 1152
    state = torch.load(root / 'photo/siglip1/s0/emotion.pt', weights_only=True)
    assert state['cap.trunk.0.weight'].shape[1] == 1152


def test_gate_vii_record_shape(work):
    rec = PH.write_gate_vii(dict(start_state=True, loss=True, gradients=True, trained_state=True, history=True),
                            dict(n_fit=3), folder=work)
    on_disk = json.loads((work / 'vii_trainer.json').read_text())
    assert on_disk == rec and set(rec) == {'gate', 'status', 'tries', 'details', 'written'}
    assert rec['gate'] == 'vii_trainer' and rec['status'] == 'pass' and rec['tries'] == 1
    rec = PH.write_gate_vii(dict(start_state=True, loss=False, gradients=True, trained_state=True, history=True),
                            dict(n_fit=3), folder=work)
    assert rec['status'] == 'fail' and rec['tries'] == 2


# ------------------------------------------------------------------ the gate status rule (pass / fail / parked)
def test_gate_vii_writer_follows_the_status_rule_and_counts_each_try_once(work):
    good = dict(start_state=True, loss=True, gradients=True, trained_state=True, history=True)
    bad = dict(good, loss=False)
    seen = [PH.write_gate_vii(r, dict(n=1), folder=work) for r in (bad, bad, bad)]
    assert [(r['status'], r['tries']) for r in seen] == [('fail', 1), ('fail', 2), ('parked', 3)]
    with pytest.raises(AssertionError, match='three tries'):
        PH.write_gate_vii(good, dict(n=1), folder=work)
    assert json.loads((work / 'vii_trainer.json').read_text())['tries'] == 3          # the refused try left no trace
    assert PH.write_gate_vii(good, dict(n=1), folder=work / 'other')['status'] == 'pass'
    assert PH.write_gate_vii(good, dict(n=1), folder=work / 'other')['status'] == 'pass'      # a pass again: try 2
    assert PH.write_gate_vii({}, dict(n=1), folder=work / 'third')['status'] == 'fail'        # no checks is no pass


def test_gate_viii_writer_follows_the_status_rule_and_counts_each_try_once(work):
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv')
    PH.build_store(ref, PH.affection_texts(ref, work / 'a.csv'), enc, work / 'store', form='plain')
    for expect in (('fail', 1), ('fail', 2), ('parked', 3)):
        with pytest.raises(AssertionError, match='prefix'):
            PH.gate_viii(work / 'store', ref, form='prefixed', record_folder=work / 'gates')
        rec = json.loads((work / 'gates/viii_affection.json').read_text())
        assert (rec['status'], rec['tries']) == expect
    with pytest.raises(AssertionError, match='three tries'):
        PH.gate_viii(work / 'store', ref, form='plain', record_folder=work / 'gates')
    assert json.loads((work / 'gates/viii_affection.json').read_text())['tries'] == 3


def test_patched_build_seeds_the_model_as_the_unpatched_build_does():
    with PH.caption_patches('siglip1'):
        _, view = BP.build_emotion(CPU, 1152, seed=7)
        got = {k: v.detach().clone() for k, v in view.state_dict().items()}
    assert same(got, PH.build_state(1152, 1152, 7)) and not same(got, PH.build_state(1152, 1152, 8))


# ------------------------------------------------- train and export refuse (ruling 1, 3, 4; the pass rule; the smoke root)
def good_store(work, form='plain', smoke=False):
    """A store of the reference built by the fake encoder, with a passing `viii_affection.json` in work/gates."""
    ref, enc = reference(), encoder()
    write_csv(work / 'a.csv')
    PH.build_store(ref, PH.affection_texts(ref, work / 'a.csv'), enc, work / 'store', form=form, smoke=smoke)
    PH.gate_viii(work / 'store', ref, form=form, record_folder=work / 'gates', allow_smoke=smoke)
    return work / 'store'


def test_require_store_accepts_a_store_that_passed_gate_viii_in_kstars_form(work):
    for kstar, form in (('siglip1', 'plain'), ('siglip1_prefix', 'prefixed')):
        sub = work / kstar
        sub.mkdir()
        index = PH.require_store(kstar, good_store(sub, form), sub / 'gates')
        assert index['backbone'] == X.extractor.SIGLIP1


def rewrite(folder, backbone):
    """Make store and records consistent with a changed backbone, so that only the backbone check can refuse."""
    index = json.loads((folder / 'index.json').read_text())
    index['backbone'] = backbone
    (folder / 'index.json').write_text(json.dumps(index))
    progress = json.loads((folder / 'progress.json').read_text())
    progress['index_sha256'] = X._sha(folder / 'index.json')
    (folder / 'progress.json').write_text(json.dumps(progress))


@pytest.mark.parametrize('what,message', [
    ('no record', 'not run'), ('fail record', 'not pass'), ('parked record', 'not pass'), ('index changed', 'hash'), ('record of another store', 'hash'),
    ('smoke store', 'smoke'), ('other form', 'prefix'), ('siglip2', 'SigLIP 1'), ('record of the other form', 'form')])
def test_require_store_refuses(work, what, message):
    smoke = what == 'smoke store'
    store = good_store(work, smoke=smoke)
    gates, kstar = work / 'gates', 'siglip1'
    record = json.loads((gates / 'viii_affection.json').read_text())
    if what == 'no record':
        (gates / 'viii_affection.json').unlink()
    elif what in ('fail record', 'parked record'):
        record['status'] = what.split()[0]
        (gates / 'viii_affection.json').write_text(json.dumps(record))
    elif what == 'index changed':
        index = json.loads((store / 'index.json').read_text())
        index['extra'] = 'edited'
        (store / 'index.json').write_text(json.dumps(index))
    elif what == 'record of another store':       # the store is intact; its record holds another index hash
        record['details']['index_sha256'] = '0' * 64
        (gates / 'viii_affection.json').write_text(json.dumps(record))
    elif what == 'other form':
        kstar = 'siglip1_prefix'
    elif what == 'siglip2':
        rewrite(store, X.extractor.SIGLIP2)
        record['details']['index_sha256'] = X._sha(store / 'index.json')
        (gates / 'viii_affection.json').write_text(json.dumps(record))
    elif what == 'record of the other form':
        record['details']['form'] = 'prefixed'
        (gates / 'viii_affection.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError, match=message):
        PH.require_store(kstar, store, gates)


def stage2(monkeypatch, work, kstar='siglip1'):
    """pass_rule says pass; the pick names `kstar`; gate (viii) records under work/gates; BP.train must not be reached."""
    (work / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    (work / 'pick.json').write_text(json.dumps(dict(kstar=kstar, kstar_form=PH.FORMS[kstar], p1_form='plain', flag=None)))
    monkeypatch.setattr(PH, 'PASS_RULE', work / 'pass_rule.json')
    monkeypatch.setattr(PH, 'PICK_FILE', work / 'pick.json')
    monkeypatch.setattr(BP, 'train', lambda *a, **k: pytest.fail('trained without a valid store'))


def test_train_arm_refuses_without_a_passed_gate_viii_before_loading_data(work, monkeypatch):
    stage2(monkeypatch, work)
    monkeypatch.setattr(PH, 'load_data', lambda *a, **k: pytest.fail('data loaded'))
    with pytest.raises(AssertionError, match='not run'):
        PH.train_arm('siglip1', [0], CPU, P.OUT / 'photo_t12_unused_root', store=work / 'store', gate_folder=work / 'gates')


def test_train_arm_refuses_a_store_that_does_not_match_its_record(work, monkeypatch):
    stage2(monkeypatch, work)
    store = good_store(work)
    index = json.loads((store / 'index.json').read_text())
    index['extra'] = 'edited'
    (store / 'index.json').write_text(json.dumps(index))
    with pytest.raises(AssertionError, match='hash'):
        PH.train_arm('siglip1', [0], CPU, P.OUT / 'photo_t12_unused_root', store=store, gate_folder=work / 'gates')


def test_train_arm_refuses_a_kstar_other_than_the_pick_and_a_missing_pass_rule(work, monkeypatch):
    stage2(monkeypatch, work, kstar='siglip1')
    store = good_store(work)
    with pytest.raises(AssertionError, match='prefix_pick'):
        PH.train_arm('siglip1_prefix', [0], CPU, P.OUT / 'photo_t12_unused_root', store=store, gate_folder=work / 'gates')
    (work / 'pass_rule.json').write_text(json.dumps(dict(status='fail')))
    with pytest.raises(AssertionError, match='pass'):
        PH.train_arm('siglip1', [0], CPU, P.OUT / 'photo_t12_unused_root', store=store, gate_folder=work / 'gates')


def test_a_smoke_train_also_needs_a_passed_gate_viii_on_its_store(work, monkeypatch):
    monkeypatch.setattr(P, 'SMOKE_OUT', work)
    monkeypatch.setattr(BP, 'train', lambda *a, **k: pytest.fail('trained without a valid store'))
    with pytest.raises(AssertionError, match='not run'):
        PH.train_arm('siglip1', [0], CPU, work / 'run', smoke=True, store=work / 'store', gate_folder=work / 'gates')


def test_train_arm_reuses_a_run_only_of_the_same_source_form_and_store(work, monkeypatch):
    monkeypatch.setattr(P, 'SMOKE_OUT', work)
    store = good_store(work, smoke=True)
    reached = []
    monkeypatch.setattr(BP, 'train', lambda *a, **k: reached.append(1) or 'ok')
    fit, val, graph = tiny(1152)
    cap = dict(cap_store_sha256='store-hash', cap_sha256='arrays')
    root = work / 'run'
    folder = root / 'photo/siglip1/s0'
    folder.mkdir(parents=True)
    good = dict(cap_source='siglip1', kstar_form='plain', cap_store_sha256='store-hash')

    def go():
        return PH.train_arm('siglip1', [0], CPU, root, data=(fit, val, fit, val), graphs=FakeGraphs(graph), cap_info=cap,
                            smoke=True, store=store, gate_folder=work / 'gates')

    for change in (dict(cap_source='siglip1_prefix'), dict(kstar_form='prefixed'), dict(cap_store_sha256='other'),
                   dict(cap_source=None)):
        (folder / 'run.json').write_text(json.dumps(dict(good, **change)))
        with pytest.raises(AssertionError, match='move it aside'):
            go()
    no_fields = dict(good)
    del no_fields['kstar_form']                                         # an old run, or one that crashed before its fields
    (folder / 'run.json').write_text(json.dumps(no_fields))
    with pytest.raises(AssertionError, match='move it aside'):
        go()
    assert not reached
    (folder / 'run.json').write_text(json.dumps(good))
    assert go() == 'ok' and reached == [1]


def test_smoke_roots_outside_the_smoke_folder_are_refused(work, monkeypatch):
    smoke_dir = work / 'smoke'
    smoke_dir.mkdir()
    monkeypatch.setattr(P, 'SMOKE_OUT', smoke_dir)
    store = good_store(work, smoke=True)
    monkeypatch.setattr(PH.PD, 'fit_keys', lambda: (_ for _ in ()).throw(RuntimeError('smoke went on')))
    with pytest.raises(AssertionError, match='smoke outputs'):
        PH.train_arm('siglip1', [0], CPU, work / 'elsewhere', smoke=True, store=store, gate_folder=work / 'gates')
    with pytest.raises(AssertionError, match='smoke outputs'):
        PH.smoke('siglip1', 'cpu', root=work / 'elsewhere')
    assert not (work / 'elsewhere').exists()


def test_caption_means_refuse_a_duplicate_csv_row_in_the_store(work, monkeypatch):
    monkeypatch.setattr(PH.PD, '_eligible_rows', lambda keys: dict(key=['a'], csv_row=np.array([10]), store_row=np.array([0])))
    store = make_store(work, [[1., 0, 0], [0, 1., 0]], [10, 10])
    with pytest.raises(AssertionError, match='duplicate csv_row'):
        PH.caption_means(['a'], store, width=3)


def test_caption_means_sum_in_float64(work, monkeypatch):
    n = 101                       # 32768 + 100 times 2^-12 in one column: lost one by one in a float32 sum
    csv = list(range(10, 10 + n))
    monkeypatch.setattr(PH.PD, '_eligible_rows', lambda keys: dict(key=['a'] * n, csv_row=np.array(csv),
                                                                   store_row=np.arange(n)))
    rows = np.array([[32768.0, 100.0]] + [[2.0 ** -12, 100.0]] * 100, dtype=np.float16)
    store = make_store(work, rows, csv)
    got = PH.caption_means(['a'], store, width=2)
    mean = rows.astype(np.float64).sum(axis=0) / n
    expect = (mean / np.linalg.norm(mean)).astype(np.float32)
    np.testing.assert_allclose(got[0], expect, rtol=0, atol=2e-7)
    wrong = rows.astype(np.float32).cumsum(axis=0)[-1] / n               # what a float32 sum would give
    assert abs(wrong[0] - mean[0]) > 1e-4


def test_export_refuses_without_the_pass_rule_before_exporting(work, monkeypatch):
    monkeypatch.setattr(PH, 'PASS_RULE', work / 'none.json')
    monkeypatch.setattr(BP, 'export_arm', lambda *a, **k: pytest.fail('exported without the pass rule'))
    for dataset in ('affection', 'emoset'):
        with pytest.raises(AssertionError, match='pass'):
            PH.export_arm('siglip1', [0], ['val'], P.OUT, dataset)


def test_export_dataset_goes_to_r14s_photo_export_and_defaults_to_affection(work, monkeypatch):
    (work / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    (work / 'pick.json').write_text(json.dumps(dict(kstar='siglip1', kstar_form='plain', p1_form='plain', flag=None)))
    monkeypatch.setattr(PH, 'PASS_RULE', work / 'pass_rule.json')
    monkeypatch.setattr(PH, 'PICK_FILE', work / 'pick.json')
    calls = []

    def fake(dataset, arm, splits, seeds, *, root=None, cstar_root=None):
        calls.append((dataset, arm, splits, seeds, BP.build_emotion.__module__))   # the caption patches are on
        return [f'{dataset}/{arm}']

    monkeypatch.setattr(BP, 'export_arm', fake)
    PH.main(['export', '--seeds', '0', '--splits', 'test'])
    PH.main(['export', '--dataset', 'emoset', '--seeds', '0', '1', '--splits', 'val', 'test'])
    assert calls == [('affection', 'photo_siglip1', ['test'], [0], 'r29_photo'),
                     ('emoset', 'photo_siglip1', ['val', 'test'], [0, 1], 'r29_photo')]
    with pytest.raises(AssertionError):
        PH.export_arm('siglip1', [0], ['val'], P.OUT, 'artelingo')
