"""r29 caption-width trainer and exports (ticket 03; spec §3.2 K1, K2, E2, E3, §3.5 gate (i)).

r14's c* trainer (`bs_cstar.train`: `rg_full.train_cohort` -> `rb_student` -> `ct2_d2.run`) with ONE input changed: the
caption tower reads the painting means of a caption source of `r29_paths.CAPTION_SOURCES`. Nothing in r14 or r28 is
edited: the changes are `unittest.mock.patch`es that unwind on success or failure (as `r28_trainer`). Where each sits:

  * the tower input: `ct2_d2.run` is wrapped; it receives `inputs[aspect]['cap']` (Qwen's default caption mean, 2048,
    which `rg_full.train_cohort` still asserts) and hands the loop the source's means instead. Qwen3's graph spaces
    (`D`, `I`, `E`, ...), the stored graphs, the teacher (`ct2_data.instructed_images`), the content correction and the
    loss targets are read before and elsewhere, from Qwen3 at 2048, and are not touched. The default caption mean
    has two uses in the CT: the tower input (`ct2_d2.py` line 286, `inputs['cap']`, replaced here) and the graph space
    `D` (`ct2_d2.py` line 287; the stored graphs' identity check and graph building; kept as is);
  * the model: `rb_student.build_model` becomes r29_model's factory for a caption width other than 2048 (`rg_full`
    patches `ms_train.build_model` with `partial(rb_student.build_model, width=..)` at call time). At 2048 (source
    `qwen`) nothing is patched: r14's own objects;
  * the arm name is the source (`bs_cstar.row_arm`): files under `<root>/{weights,train,val,held}/<source>/`;
  * the `rb` block of every weights sidecar (`rb_student._rb_block`) and the run record (`bs_cstar._record`) get
    `cap_source`, `cap_width`, `cap_store_sha256`, `cap_sha256` and this code's hash.

Files (r14's layout) under the root: `weights/<source>/<aspect>__s<seed>.pt|json`, `train/<source>/run__s<seed>.json`
and `train/<source>/R6-5/...` (the validation export from the training loop), and from `export`: `val/<source>/` (6,152
validation paintings, image and caption side) and `held/<source>/` (12,281 held paintings, image side).

CLI (CUBLAS_WORKSPACE_CONFIG=:4096:8 is set when unset):
  r29_trainer train  --encoder E --cap SOURCE [--seeds ..] [--root DIR] [--device cuda] [--smoke [--max-epochs N]]
  r29_trainer export --encoder E --cap SOURCE --kind val held [--seeds ..] [--root DIR] [--out-root DIR] [--device D]
  r29_trainer gate-i-a --run r14|r14_b|new [--device cuda]      (gate (i)(a): trains seed 0 into OUT/gate_i_a/<run>)
  r29_trainer gate-i-a --compare                                (writes records/gates/i_a_weights.json)
  r29_trainer gate-i-c [--seeds 0] [--val-device D] [--held-device D]   (gate (i)(c) export part: records/gates/i_c.json)
"""
import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402
P.use_older_rounds()
import r29_captions as C  # noqa: E402
import r29_model as M  # noqa: E402
import r28_trainer  # noqa: E402  (imports r14's bs_cstar with r10 from this checkout and puts the older rounds on sys.path)
import ct2_d2  # noqa: E402
import ct2_data  # noqa: E402
P.check_older_imports()  # every older-round module loaded so far is the main checkout's copy

B = r28_trainer.B
ASPECTS = tuple(B.ASPECTS)
ENCODERS = ('siglip1', 'siglip2', 'clip_l14')          # bs_paths' keys; qwen's image is K0's
STAGE2_ENCODERS = ('siglip2', 'clip_l14')              # E2, E3: only after a pass (D3)
CAP_FIELDS = ('cap_source', 'cap_width', 'cap_store_sha256')
CODE_FILES = ('r29_trainer.py', 'r29_model.py', 'r29_captions.py')
BATCH = B.BATCH                                        # inference batch of every export (r14's: 512)
GATE_I_A = ('r14', 'r14_b', 'new')                     # the gate (i)(a) runs and their arms
GATE_I_A_ARM = {'r14': B.STAR_ARM, 'r14_b': B.STAR_ARM, 'new': 'qwen'}
HELD_STORED = lambda root, arm, a, s: Path(root) / 'held' / arm / f'{a}__s{s}.npz'   # noqa: E731
CAPTION_ARRAY_WIDTH = P.OUTPUT_WIDTH                   # Qwen3's caption mean, as `rg_full.train_cohort` asserts it


def code_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in CODE_FILES}


# ------------------------------------------------------------------------------------------------------- guards
def check_root(root, smoke=False):
    """r14's hygiene (nothing inside another round's outputs or the stores), plus: a smoke only under SMOKE_OUT and a
    real run only under OUT and never under SMOKE_OUT."""
    root = B.check_root(root, smoke)
    smoke_root = P.SMOKE_OUT.resolve()
    if smoke:
        assert root.is_relative_to(smoke_root), f'smoke outputs must be under {smoke_root}: {root}'
    else:
        assert root.is_relative_to(P.OUT.resolve()) and not root.is_relative_to(smoke_root), \
            f'outputs must be under {P.OUT} (not its smoke folder): {root}'
    return root


def require_pass(enc):
    """D3: E2 and E3 (stage 2) only after `records/pass_rule.json` says pass."""
    if enc in STAGE2_ENCODERS:
        path = P.RECORDS / 'pass_rule.json'
        assert path.is_file() and json.loads(path.read_text()).get('status') == 'pass', \
            f'{enc} is a stage 2 encoder: needs {path} with status "pass" (D3)'


def check_args(enc, source):
    assert enc in ENCODERS, f'encoder {enc!r}: not one of {ENCODERS}'
    assert source in P.CAPTION_SOURCES, f'caption source {source!r}: not one of {sorted(P.CAPTION_SOURCES)}'
    assert source in P.ENCODER_CAPTION_SOURCES[enc], f'caption source {source!r} does not match encoder {enc!r}'
    require_pass(enc)
    return P.CAPTION_SOURCES[source][1]


def default_root(enc, smoke):
    return (P.SMOKE_OUT if smoke else P.OUT) / 'cstar' / enc


# ------------------------------------------------------------------------------------------------ training patches
class _State:
    """What the patches share: the source's painting means (None for `qwen`: the CT's own array stays) and its hash."""
    cap = None
    cap_sha256 = None


@contextmanager
def caption_patches(source, width):
    """The patches of this module's training for `source` at caption input width `width` (module doc)."""
    state = _State()
    base_load, base_rb, base_run, base_record = B.load_data, B.rb_student._rb_block, ct2_d2.run, B._record

    def load_data(ctx):
        s, inputs, graphs, manifests, images = base_load(ctx)
        qwen = inputs[ASPECTS[0]]['cap']
        assert all(inputs[a]['cap'] is qwen for a in ASPECTS)
        assert qwen.shape == (len(s['first']), CAPTION_ARRAY_WIDTH) and qwen.dtype == np.float32
        if source == 'qwen':
            state.cap, state.cap_sha256 = None, B.rb_student.sha_array(qwen)
        else:
            cap = C.painting_means(source, s['all_painting_ids'])
            assert cap.shape == (len(s['first']), width) and cap.dtype == np.float32 and np.isfinite(cap).all()
            state.cap, state.cap_sha256 = cap, B.rb_student.sha_array(cap)
        return s, inputs, graphs, manifests, images

    def rb_block(*args, **kwargs):
        rb = base_rb(*args, **kwargs)
        assert state.cap_sha256 is not None, 'the caption input is read before the rb block'
        rb.update(cap_source=source, cap_width=width, cap_store_sha256=C.store_hash(source),
                  cap_sha256=state.cap_sha256, r29_code_sha256=code_hashes())
        return rb

    def run(s, inputs, *args, **kwargs):
        if state.cap is not None:
            assert state.cap.shape == (len(s['first']), width)
            assert all(x['cap'].shape == (len(s['first']), CAPTION_ARRAY_WIDTH) for x in inputs.values())
            inputs = {a: dict(x, cap=state.cap) for a, x in inputs.items()}     # the tower input only
        # The frozen make_config called by base_run writes config.data_sha256.cap / input_cap for Qwen3's graph
        # space D. The caption tower's actual input is identified by config.rb.cap_sha256 above.
        return base_run(s, inputs, *args, **kwargs)

    def record(*args, **kwargs):
        record = base_record(*args, **kwargs)
        record.update(cap_source=source, cap_width=width, cap_store_sha256=C.store_hash(source),
                      cap_sha256=state.cap_sha256, r29_code_sha256=code_hashes())
        return record

    cap_width = width

    def build_model(row, aspects, device, *, width):
        return M.build_model(row, aspects, device, width=width, cap_width=cap_width)

    with ExitStack() as stack:
        stack.enter_context(patch.object(B, 'row_arm', lambda row: source))
        stack.enter_context(patch.object(B, 'load_data', load_data))
        stack.enter_context(patch.object(B.rb_student, '_rb_block', rb_block))
        stack.enter_context(patch.object(ct2_d2, 'run', run))
        stack.enter_context(patch.object(B, '_record', record))
        if width != P.OUTPUT_WIDTH:         # at 2048 r14's own model objects
            stack.enter_context(patch.object(B.rb_student, 'build_model', build_model))
        yield state


def train(enc, source, seeds, device, root, *, smoke=False, max_epochs=None):
    """r14's c* trainer on encoder `enc` with the caption tower reading `source`. Returns {seed: Run}. A non-finite
    loss raises (`CstarNonFiniteLoss`, or the loop's own assertion); the rerun rule (D21) is the caller's."""
    width = check_args(enc, source)
    root, seeds = check_root(root, smoke), list(seeds)
    for seed in seeds:
        path = B.record_path(root, source, seed)
        if path.exists():
            assert json.loads(path.read_text()).get('r29_code_sha256') == code_hashes(), \
                f'{path}: r29 training code changed'
    with caption_patches(source, width):
        return B.train(enc, B.REF, seeds, device, root, keep='all', smoke=smoke, max_epochs=max_epochs, graph_out=root)


# ----------------------------------------------------------------------------------------------------- exports
def _sidecar(weights):
    """`r29_model.read_sidecar` with one addition: r14's K0 sidecars (under `K0_ROOT`) have no caption fields; their
    caption source is `qwen` at 2048 by construction (r14 trained them so). Any other sidecar must carry the fields."""
    weights = Path(weights)
    config = json.loads(weights.with_suffix('.json').read_text())['config']
    rb = dict(config['rb'])
    if not any(k in rb for k in (*CAP_FIELDS, 'cap_sha256')):
        assert weights.resolve().is_relative_to(P.K0_ROOT.resolve()), f'{weights}: sidecar lacks the caption fields'
        rb.update(cap_source='qwen', cap_width=P.OUTPUT_WIDTH, cap_store_sha256=C.store_hash('qwen'))
        config = dict(config, rb=rb)
    missing = [k for k in M.SIDECAR_KEYS if k not in rb]
    assert not missing, f'{weights}: sidecar rb block lacks {missing}'
    assert rb['cap_source'] in P.CAPTION_SOURCES, f'unknown caption source {rb["cap_source"]!r}'
    assert rb['cap_width'] == P.CAPTION_SOURCES[rb['cap_source']][1], f'{weights}: cap_width is not the source\'s'
    return config, rb


def load_view(weights, device):
    _, rb = _sidecar(weights)
    with patch.object(M, 'read_sidecar', _sidecar):
        return M.load_view(weights, device, expect_store_sha256=C.store_hash(rb['cap_source']))


def infer_sides(weights, inputs, device):
    """One saved best view on `inputs` {side: float32 [n, in]} as `bs_cstar._infer_sides` runs it (batches of 512,
    float16), at the widths the sidecar records."""
    device = torch.device(device)
    view = load_view(weights, device)
    out = {}
    with torch.no_grad():
        for side, x in inputs.items():
            x = np.asarray(x)
            assert x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), side
            y = np.empty((len(x), B.rb_student.OUT), dtype=np.float16)
            for a in range(0, len(x), BATCH):
                xx = torch.from_numpy(np.ascontiguousarray(x[a:a + BATCH])).to(device)
                y[a:a + len(xx)] = view[side](xx).cpu().numpy().astype(np.float16)
            out[side] = y
    return out


def held_inputs(backbone):
    names, names_sha = B.rb_student.held_names()
    return names, names_sha, dict(img=B.rb_encoders.full_images(backbone, names))


def validation_inputs(source, backbone):
    """(names, sha of names, {img, cap}) of the 6,152 validation paintings in the trainer's dev order: the encoder's
    default image and the source's painting means (`qwen`: the CT's default caption mean, `B._validation_inputs`')."""
    s = ct2_data.split2('full')
    names = [str(n) for n in s['all_painting_ids'][s['dev']]]
    return names, B._names_sha(names), dict(img=B.rb_encoders.full_images(backbone, names),
                                              cap=C.painting_means(source, names))


def _assert_training_validation(arrays, root, arm, aspect, seed):
    path = B.paths_of(root, arm, aspect, seed)[1]
    with np.load(path, allow_pickle=False) as stored:
        assert set(arrays) == set(stored.files) == {'img', 'cap'}, f'{path}: training-time validation sides differ'
        for side, x in arrays.items():
            y = stored[side]
            assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(), \
                f'{path}: {side} differs from training-time validation'


def export(enc, source, seeds, root, *, kinds=('val', 'held'), device=None, out_root=None, smoke=False, arm=None):
    """The ArtELingo validation (`val`: image and caption side) and held (image side) exports of stored checkpoints, at
    the widths each sidecar records, in r14's format and file names under `out_root/<kind>/<arm>/`. `arm` is the
    weights' arm folder (default: the source; K0's is `k11=40`). `device`: the one the weights were trained on unless
    given. Validation into the weights root requires the run record's training device and arrays bit-equal to the
    training-time validation file. Existing exports require matching metadata and current per-file code hashes.
    Returns {(aspect, seed): {side: array}} of the last kind."""
    width = check_args(enc, source)
    seeds, arm = list(seeds), arm or source
    assert seeds and len(set(seeds)) == len(seeds) and all(s in B.SEEDS for s in seeds)
    root = Path(root).resolve()
    out_root = check_root(out_root if out_root is not None else root, smoke)
    result = {}
    with patch.object(B, 'row_arm', lambda row: arm), B.training_context(enc, B.REF, export=True) as ctx:
        for kind in kinds:
            assert kind in ('val', 'held'), kind
            if kind == 'held':
                names, names_sha, inputs = held_inputs(ctx.backbone)
            else:
                names, names_sha, inputs = validation_inputs(source, ctx.backbone)
                assert inputs['cap'].shape == (len(names), width)
            assert inputs['img'].shape == (len(names), ctx.width)
            for seed in seeds:
                own_validation = kind == 'val' and out_root == root
                if own_validation:
                    run_device = torch.device(json.loads(B.record_path(root, arm, seed).read_text())['device'])
                for aspect in ASPECTS:
                    weights, config, weights_sha = B._check_weights(root, arm, aspect, seed, ctx.backbone, smoke)
                    _, rb = _sidecar(weights)
                    assert (rb['cap_source'], rb['cap_width'], rb['cap_store_sha256']) == (
                        source, width, C.store_hash(source)), f'{weights}: trained on another caption source or store'
                    recorded = torch.device(config['device'])
                    used = recorded if device is None else torch.device(device)
                    if own_validation:
                        assert recorded == run_device and used == run_device, \
                            f'{weights}: validation in the weights root requires recorded training device {run_device}'
                    path = out_root / kind / arm / f'{aspect}__s{seed}.npz'
                    meta = dict(kind=kind, arm=arm, scale=B.rg_full.SCALE, aspect=aspect, seed=seed,
                                encoder=ctx.backbone, rows=len(names), width=B.rb_student.OUT, dtype='float16',
                                list_sha256=names_sha, weights_sha256=weights_sha,
                                image_sha256=B.rb_student.sha_array(inputs['img']), device=str(recorded),
                                export_device=str(used), recipe=B.rg_full.RECIPE, switches=B.V.switches(arm),
                                cap_source=source, cap_width=width, cap_store_sha256=C.store_hash(source),
                                r29_code_sha256=code_hashes())
                    if kind == 'val':
                        meta['cap_sha256'] = B.rb_student.sha_array(inputs['cap'])
                    existing = path.exists()
                    if existing:
                        arrays, info = B.rb_student.read_outputs(path)
                        assert info.get('r29_code_sha256') == meta['r29_code_sha256'], \
                            f'{path}: r29 export code changed or code hash missing; use a fresh --out-root'
                        assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                    else:
                        arrays = infer_sides(weights, inputs, used)
                        for side, y in arrays.items():
                            problem = B.rg_full.vc3_problem(y, n_rows=len(names))
                            assert problem is None, f'{path}: {side}: {problem}'
                    if own_validation:
                        _assert_training_validation(arrays, root, arm, aspect, seed)
                    if not existing:
                        extra = dict(vc3_attempts=1, vc3_problems=[]) if kind == 'held' else {}
                        B.rb_student.write_outputs(path, arrays, dict(meta, sha256=B.rb_student.sha_array(arrays['img']), **extra))
                        arrays, info = B.rb_student.read_outputs(path)
                    result[aspect, seed] = arrays
    return result


# ---------------------------------------------------------------------------------------------------- gate (i)
def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_weights(root_a, arm_a, root_b, arm_b, seed=0):
    """Per head: file sha256 of each side's weights, whether the files are byte-equal, whether every tensor of the two
    state dicts is bit-equal, and both sidecars' best epoch and best dev loss. `equal`: all heads tensor-equal."""
    heads = {}
    for aspect in ASPECTS:
        pa, pb = (B.rb_student.weights_path(r, arm, aspect, seed) for r, arm in ((root_a, arm_a), (root_b, arm_b)))
        sa, sb = (torch.load(p, map_location='cpu', weights_only=True) for p in (pa, pb))
        ia, ib = (json.loads(p.with_suffix('.json').read_text()) for p in (pa, pb))
        tensors = sa.keys() == sb.keys() and all(
            sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)
        heads[aspect] = dict(sha256=[file_sha(pa), file_sha(pb)], bytes_equal=file_sha(pa) == file_sha(pb),
                             tensors_equal=bool(tensors), best_epoch=[ia['best_epoch'], ib['best_epoch']],
                             best_dev_loss=[ia['best_dev_loss'], ib['best_dev_loss']],
                             recorded_sha_ok=[ia['weights_sha256'] == file_sha(pa), ib['weights_sha256'] == file_sha(pb)])
    return dict(equal=all(h['tensors_equal'] for h in heads.values()), heads=heads)


def gate_i_a_root(name):
    return P.OUT / 'gate_i_a' / name


def gate_i_a_train(name, device):
    """Gate (i)(a): seed 0 on SigLIP 1 into OUT/gate_i_a/<name>. `r14`, `r14_b`: r14's trainer unpatched (its CLI's
    `train` call); `new`: this module with source qwen."""
    assert name in GATE_I_A, name
    root = gate_i_a_root(name)
    if name == 'new':
        return train('siglip1', 'qwen', [0], device, root)
    root = check_root(root)
    return B.train('siglip1', B.REF, [0], device, root, keep='all', graph_out=root)


def gate_i_a_compare(write=True):
    """The weights comparison of gate (i)(a) -> records/gates/i_a_weights.json. status `pass` when the new trainer's
    weights equal r14's; else `fail`: the verdict with the CUDA fallback (r14's own two runs) is ticket 05's, and
    `details` carries both comparisons when `r14_b` exists."""
    pairs = {'new_vs_r14': ('new', 'r14')}
    if (gate_i_a_root('r14_b') / 'weights').exists():
        pairs['r14_vs_r14_b'] = ('r14', 'r14_b')
    details = {}
    for key, (a, b) in pairs.items():
        details[key] = compare_weights(gate_i_a_root(a), GATE_I_A_ARM[a], gate_i_a_root(b), GATE_I_A_ARM[b])
        details[key]['runs'] = [a, b]
    records = {}
    for name in GATE_I_A:
        path = B.record_path(gate_i_a_root(name), GATE_I_A_ARM[name], 0)
        if path.exists():
            r = json.loads(path.read_text())
            records[name] = {k: r.get(k) for k in ('device', 'host', 'seconds', 'best_epoch', 'code_commit',
                                                   'code_dirty', 'r29_code_sha256', 'cap_source')}
    result = dict(status='pass' if details['new_vs_r14']['equal'] else 'fail',
                  details=dict(seed=0, encoder='siglip1', comparisons=details, run_records=records,
                               note='status is weights equality only; the CUDA fallback verdict is ticket 05\'s'))
    if write:
        C.write_gate_record(P.GATES, result, gate='i_a_weights')
    return result


def compare_exports(new_root, stored_root, arm, seeds, kinds=('val', 'held'), val_layout='train'):
    """New exports against stored ones, bit for bit: held `held/<arm>/` image; validation (image and caption) r14's
    training-time file `train/<arm>/R6-5/<aspect>__<graph row>__s<seed>.npz` (`val_layout` 'train': r14's readers read
    this one) or an export's `val/<arm>/<aspect>__s<seed>.npz` (`val_layout` 'val')."""
    items = {}
    for seed in seeds:
        for aspect in ASPECTS:
            for kind in kinds:
                ours = Path(new_root) / kind / arm / f'{aspect}__s{seed}.npz'
                if kind == 'held':
                    stored = HELD_STORED(stored_root, arm, aspect, seed)
                elif val_layout == 'train':
                    stored = B.paths_of(stored_root, arm, aspect, seed)[1]
                else:
                    stored = Path(stored_root) / 'val' / arm / f'{aspect}__s{seed}.npz'
                with np.load(ours, allow_pickle=False) as x, np.load(stored, allow_pickle=False) as y:
                    for side in (('img',) if kind == 'held' else ('img', 'cap')):
                        u, v = x[side], y[side]
                        ok = u.shape == v.shape and u.dtype == v.dtype
                        differ = int((u != v).sum()) if ok else None
                        items[f'{kind}/{aspect}__s{seed}/{side}'] = dict(
                            equal=bool(ok and differ == 0), differing_values=differ, shape=list(u.shape),
                            max_abs_diff=float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if ok else None)
    return dict(equal=all(i['equal'] for i in items.values()), items=items)


def stored_devices(arm, seed=0, root=None):
    """The device each stored K0 file was made on: validation = the training device (the weights' config), held =
    r14's export device (its sidecar's `export_device`)."""
    root = Path(root or P.K0_ROOT)
    val = json.loads(B.rb_student.weights_path(root, arm, ASPECTS[0], seed).with_suffix('.json').read_text())['config']['device']
    held = json.loads(HELD_STORED(root, arm, ASPECTS[0], seed).with_suffix('.json').read_text())['export_device']
    return dict(val=val, held=held)


def gate_i_c(seeds=(0,), val_device=None, held_device=None, out_root=None, r14_code=True, write=True):
    """Gate (i)(c), export part: from r14's SigLIP 1 weights (K0), this module's export writes the validation and held
    exports; they must equal r14's stored ones bit for bit. Devices default to those the stored files were made on.
    `r14_code`: r14's own export functions run on the same devices into `<out>/r14_code`, and the two exports are
    compared too (code against code, free of the hardware). Each invocation reserves a fresh try folder, including
    `write=False` checks; only recorded invocations increment the gate record's tries. r28's valpick cells are ticket
    05's part."""
    arm, seeds = P.K0_ARM, list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in B.SEEDS for s in seeds)
    automatic_root = out_root is None
    if automatic_root:
        attempt = 1
        while (P.OUT / 'gate_i_c' / f'try{attempt}').exists():
            attempt += 1
        out_root = P.OUT / 'gate_i_c' / f'try{attempt}'
    out_root = check_root(out_root)
    assert not out_root.exists() or (out_root.is_dir() and not any(out_root.iterdir())), \
        f'{out_root}: non-empty gate output; use a fresh --out-root for each try'
    out_root.mkdir(parents=True, exist_ok=not automatic_root)
    devices = stored_devices(arm, seeds[0])
    devices = dict(val=val_device or devices['val'], held=held_device or devices['held'])
    for kind in ('val', 'held'):
        export('siglip1', 'qwen', seeds, P.K0_ROOT, kinds=(kind,), device=devices[kind], out_root=out_root, arm=arm)
    details = dict(seeds=seeds, arm=arm, devices=devices, stored_root=str(P.K0_ROOT), out_root=str(out_root),
                   r29_code_sha256=code_hashes(),
                   vs_stored=compare_exports(out_root, P.K0_ROOT, arm, seeds))
    ok = details['vs_stored']['equal']
    if r14_code:
        ref_root = out_root / 'r14_code'
        B.export_validation('siglip1', B.REF, seeds, P.K0_ROOT, device=devices['val'], out_root=ref_root)
        B.export_held('siglip1', B.REF, seeds, P.K0_ROOT, device=devices['held'], out_root=ref_root)
        details['vs_r14_export_code'] = compare_exports(out_root, ref_root, arm, seeds, val_layout='val')
    if write:
        C.write_gate_record(P.GATES, dict(status='pass' if ok else 'fail', details=details), gate='i_c')
    return dict(status='pass' if ok else 'fail', details=details)


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # r14 asserts it for CUDA runs
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--encoder', required=True, choices=ENCODERS)
        p.add_argument('--cap', required=True, choices=sorted(P.CAPTION_SOURCES), help='caption source = the arm name')
        p.add_argument('--seeds', type=int, nargs='+', choices=B.SEEDS, default=list(B.SEEDS))
        p.add_argument('--root', type=Path, default=None, help='default: OUT/cstar/<encoder> (smoke: SMOKE_OUT/cstar/..)')
        p.add_argument('--smoke', action='store_true', help='2 epochs, 5000 fit pairs; only under SMOKE_OUT')
        if name == 'train':
            p.add_argument('--device', default='cuda')
            p.add_argument('--max-epochs', type=int, default=None, help='with --smoke only, at most 2')
        else:
            p.add_argument('--kind', nargs='+', choices=('val', 'held'), default=['val', 'held'])
            p.add_argument('--device', default=None, help='default: the device the weights were trained on')
            p.add_argument('--out-root', type=Path, default=None)
    p = sub.add_parser('gate-i-a')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', choices=GATE_I_A)
    group.add_argument('--compare', action='store_true')
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('gate-i-c')
    p.add_argument('--seeds', type=int, nargs='+', choices=B.SEEDS, default=[0])
    p.add_argument('--val-device', default=None)
    p.add_argument('--held-device', default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--no-r14-code', action='store_true')
    p.add_argument('--no-record', action='store_true', help='do not write records/gates/i_c.json (code checks)')
    args = parser.parse_args(argv)

    if args.command == 'train':
        root = args.root or default_root(args.encoder, args.smoke)
        runs = train(args.encoder, args.cap, args.seeds, args.device, root, smoke=args.smoke, max_epochs=args.max_epochs)
        for seed, run in runs.items():
            print(f'R29 train {args.encoder} {args.cap} s{seed} {run.record["status"]} '
                  f'seconds={run.record["seconds"]:.2f} best_epoch={run.record["best_epoch"]}', flush=True)
    elif args.command == 'export':
        root = args.root or default_root(args.encoder, args.smoke)
        export(args.encoder, args.cap, args.seeds, root, kinds=args.kind, device=args.device,
               out_root=args.out_root, smoke=args.smoke)
        print(f'R29 export {args.encoder} {args.cap}: {args.kind}', flush=True)
    elif args.command == 'gate-i-a':
        if args.compare:
            result = gate_i_a_compare()
            print(f'gate i_a weights: {result["status"]}', flush=True)
            return 0 if result['status'] == 'pass' else 1
        gate_i_a_train(args.run, args.device)
        print(f'R29 gate-i-a {args.run}: trained', flush=True)
    else:
        result = gate_i_c(args.seeds, args.val_device, args.held_device, args.out_root,
                          r14_code=not args.no_r14_code, write=not args.no_record)
        items = result['details']['vs_stored']['items']
        print(f'gate i_c: {result["status"]}; stored-equal {sum(i["equal"] for i in items.values())}/{len(items)}',
              flush=True)
        return 0 if result['status'] == 'pass' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
