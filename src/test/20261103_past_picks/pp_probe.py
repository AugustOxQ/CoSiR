"""r19 cost probe (spec §3.4 "Cost", SC15, gate (viii), D8, D22): time per image of c* and of the teacher, on the local 3090.

  pp_probe.py passes --enc X [--keys N] [--resume]   encoder passes of X (Qwen: default plus the three instructed passes
                                                     and the teacher's end to end); model time three repeats, end to
                                                     end one repeat; peak memory, parameters, Qwen vision tower share
  pp_probe.py heads  [--enc X ...] [--keys N] [--standin]   the three seed-0 trunk-and-head networks of X on the stored
                                                     default embeddings, three repeats
  pp_probe.py cells  [--enc X ...]                   OUT/stage3/cells_cost.json (index form), OUT/stage3/cost_table.json
                                                     and gate (viii) in OUT/stage0/gate_viii.json

What is timed (spec §3.4 Cost). Keys: the first 2,000 of `mt_data.keys('test')` (r15's test cut in sha256 order); batch 32
for every pass; 3 warm-up batches (the first three batches, run again afterwards as timed batches) run before every repeat
and are in no timing or sum; wall time around each batch with `torch.cuda.synchronize()` before and after.
  model time  images decoded to RGB at file size beforehand and held in memory (the images are never closed); timed is
              `enc.encode_images(batch, batch_size=32)` as the extractors call it: the encoder's own resize (Qwen: the
              extractor's pre-resize to at most 512 image tokens), the rest of its processor, transfer, forward, pooling,
              the copy of the embedding back to the host. Three repeats.
  end to end  one repeat, images from the files with the MMArt prep's loader (`bs_extract.encode_rows`: one prefetch thread
              decodes the next batch while the GPU works, `_open_rgb`). A batch's seconds are the gap between completions
              (the first batch also pays its own decode). The teacher's three instructed passes share one decode: one
              loop, three `encode_images` calls per batch. The files are in the page cache by then (the model-time images
              were read from them); the number is the prep's loop with warm files.
  networks    the three seed-0 networks of X's c* (r10's for Qwen, r14's and this round's for the others), fp32, each as
              `bs_cstar._infer_sides` runs it; timed per batch of 32 stored default embeddings: host-to-device copy once,
              three forwards, the three outputs copied back. Three repeats.
  per repeat  c* = X default pass (r) + networks (r); the teacher = the sum of its three passes (r).
Compute processes on the GPU (`nvidia-smi --query-compute-apps`) are polled before and after every repeat; a repeat with a
process other than this one is discarded (its batch file goes to OUT/probe/discarded/), rerun, and logged under
`repeats_discarded` in OUT/probe/passes.json. Raw files, enough to rebuild SC15 (`build_cells`): OUT/probe/batches/
<pass>__r<k>.json (model time, networks), <pass>__e2e.json (end to end) with per-batch seconds and images, and
OUT/probe/passes.json (what each file is, discards, peak memory, parameters, environment).
Passes: `<encoder>` (default image; Qwen: `qwen`), `qwen_emotion_focus`, `qwen_style_word`, `qwen_genre_word`,
`heads_<encoder>`, and `teacher` (end to end only). Smoke: `PP_SMOKE=1` with `PP_OUTPUTS`, any `--keys` other than 2,000.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pp_paths as PP

ENCODERS = PP.ENCODERS
TEACHER_VIEWS = tuple(PP.TEACHER_VIEW[a] for a in PP.ASPECTS)           # emotion_focus, style_word, genre_word
TEACHER_PASSES = tuple(f'qwen_{v}' for v in TEACHER_VIEWS)
TEACHER_E2E = 'teacher'
WHOLE_SET = 16384                                                       # MMArt test paintings (whole-set figure)


# ------------------------------------------------------------------------------------------------ pass names
def default_pass(enc):
    assert enc in ENCODERS, enc
    return enc


def heads_pass(enc):
    assert enc in ENCODERS, enc
    return f'heads_{enc}'


def passes_of(enc):
    """The encoder passes a `passes --enc` job times (the teacher's end to end comes on top, Qwen only)."""
    return [default_pass(enc)] + (list(TEACHER_PASSES) if enc == 'qwen' else [])


# ------------------------------------------------------------------------------------------------ the arithmetic
def batch_sizes(n, batch):
    assert n > 0 and batch > 0
    return [min(batch, n - i) for i in range(0, n, batch)]


def time_batches(run, batches, *, warmup, sync, clock):
    """Seconds per batch. `warmup` warm-up runs come first (the first batches, cycled when there are fewer), are
    synchronised and are not timed; then every batch is timed with a synchronisation before and after."""
    assert len(batches) > 0 and warmup >= 0
    for i in range(warmup):
        run(batches[i % len(batches)])
    sync()
    seconds = []
    for batch in batches:
        sync()
        t0 = clock()
        run(batch)
        sync()
        seconds.append(clock() - t0)
    return seconds


def batch_record(name, kind, repeat, batch_n, batch_seconds, **extra):
    """One raw timing file: per-batch images and seconds, with the totals they give. ms per image is the total seconds
    over the total images (not a mean of batch rates)."""
    assert len(batch_n) == len(batch_seconds) > 0, 'one timing per batch'
    assert all(isinstance(n, int) and n > 0 for n in batch_n), batch_n
    assert all(isinstance(s, float) and math.isfinite(s) and s >= 0 for s in batch_seconds), 'timings must be finite, >= 0'
    total = float(sum(batch_seconds))
    n = int(sum(batch_n))
    assert total > 0
    return {'pass': name, 'kind': kind, 'repeat': repeat, 'n_images': n, 'batch_n': list(batch_n),
            'batch_seconds': [float(s) for s in batch_seconds], 'total_seconds': total,
            'ms_per_image': 1000.0 * total / n, 'images_per_second': n / total, **extra}


def model_repeat(name, encode, items, repeat, batch_size, warmup, sync, clock, kind='model', **extra):
    """One repeat of a pass over `items` (decoded images, or stored embeddings), batches of `batch_size`."""
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    seconds = time_batches(encode, batches, warmup=warmup, sync=sync, clock=clock)
    return batch_record(name, kind, repeat, [len(b) for b in batches], seconds, batch_size=batch_size,
                        warmup_batches=warmup, **extra)


def end_to_end(run, batches, load, *, sync, clock):
    """Seconds per batch of a loop whose images come from `load(batch)` on one prefetch thread (the next batch loads while
    `run` works, as `bs_extract.encode_rows`). Seconds are the gaps between completions, the first from the loop's start.
    Images are closed after use. `run(images)` must return only when its work is done."""
    seconds = []
    with ThreadPoolExecutor(1, thread_name_prefix='probe-load') as pool:
        fut = pool.submit(load, batches[0])
        sync()
        last = clock()
        for i in range(len(batches)):
            ims = fut.result()
            if i + 1 < len(batches):
                fut = pool.submit(load, batches[i + 1])
            try:
                run(ims)
            finally:
                for im in ims:
                    getattr(im, 'close', lambda: None)()
            sync()
            now = clock()
            seconds.append(now - last)
            last = now
    return seconds


# ------------------------------------------------------------------------------------------------ polling
_APP = re.compile(r'^\s*(\d+)\s*,\s*(\d+)\s*MiB\s*$')


def parse_compute_apps(text):
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` -> [(pid, MiB)]."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _APP.match(line)
        assert m, f'unexpected nvidia-smi line: {line!r}'
        rows.append((int(m.group(1)), int(m.group(2))))
    return rows


def foreign(rows, own_pid):
    return [(pid, mib) for pid, mib in rows if pid != own_pid]


def poll_gpu():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout
    return parse_compute_apps(out)


def run_repeat(name, repeat, measure, poll, own_pid, max_attempts=5, wait=0.0, sleep=time.sleep):
    """measure() once between two polls. A repeat with another compute process before or after it is discarded and run
    again. Returns (record with `attempt`, discard log). The log keeps each discarded record."""
    discarded = []
    for attempt in range(1, max_attempts + 1):
        before = foreign(poll(), own_pid)
        record = measure()
        after = foreign(poll(), own_pid)
        if not before and not after:
            record['attempt'] = attempt
            return record, discarded
        discarded.append({'pass': name, 'repeat': repeat, 'attempt': attempt,
                          'processes_before': [list(p) for p in before], 'processes_after': [list(p) for p in after],
                          'record': record})
        print(f'DISCARDED {name} repeat {repeat} attempt {attempt}: processes {before} before, {after} after', flush=True)
        if attempt < max_attempts and wait:
            sleep(wait)
    raise RuntimeError(f'{name} repeat {repeat}: another process on the GPU in all {max_attempts} attempts; '
                       f'see nvidia-smi and rerun when the GPU is free')


# ------------------------------------------------------------------------------------------------ SC15 and gate (viii)
def _read(path):
    return json.loads(Path(path).read_text())


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def read_timing(root, name, repeat, entry, n_images, smoke):
    """The batch file of one accepted repeat (`repeat` an int, or 'e2e'), recomputed from its batches and checked against
    what passes.json and the file itself say."""
    path = Path(root) / 'probe' / 'batches' / entry['file']
    rec = _read(path)
    again = batch_record(rec['pass'], rec['kind'], rec['repeat'], rec['batch_n'], rec['batch_seconds'])
    assert rec['pass'] == name and rec['repeat'] == (None if repeat == 'e2e' else repeat), f'{path}: wrong pass or repeat'
    assert _close(again['ms_per_image'], rec['ms_per_image']) and _close(again['ms_per_image'], entry['ms_per_image']), \
        f'{path}: ms per image of the batches disagrees with the file or with passes.json'
    assert _close(again['total_seconds'], entry['total_seconds']), f'{path}: total seconds disagrees with passes.json'
    assert again['n_images'] == n_images, f'{path}: {again["n_images"]} images, meta says {n_images}'
    assert smoke or n_images == PP.PROBE_N, f'{n_images} images, not {PP.PROBE_N} (only smoke runs may differ)'
    return again['ms_per_image']


def _repeats(root, passes, name, n_images, smoke):
    assert name in passes, f'pass {name!r} was not timed'
    reps = passes[name]['repeats']
    assert sorted(reps) == [str(k) for k in range(PP.PROBE_REPEATS)], f'{name}: need repeats 0..{PP.PROBE_REPEATS - 1}'
    return [read_timing(root, name, k, reps[str(k)], n_images, smoke) for k in range(PP.PROBE_REPEATS)]


def _e2e(root, passes, name, n_images, smoke):
    assert name in passes and passes[name].get('e2e'), f'pass {name!r} has no end-to-end timing'
    return read_timing(root, name, 'e2e', passes[name]['e2e'], n_images, smoke)


def build_cells(root, encoders=None, smoke=None):
    """SC15 cells (index form, `OUT/stage3/cells_cost.json`) from `root/probe/passes.json` and `root/probe/batches/*.json`
    alone: per repeat r, c* = X default pass (r) + networks (r), the teacher = the sum of its three passes (r); ms per image
    is recomputed from every file's per-batch seconds and must agree with passes.json."""
    smoke = bool(PP.SMOKE) if smoke is None else smoke
    doc = _read(Path(root) / 'probe' / 'passes.json')
    passes, meta, log = doc['passes'], doc['meta'], doc.get('repeats_discarded', [])
    n = meta['n_images']
    assert smoke or n == PP.PROBE_N, f'{n} images, not {PP.PROBE_N} (only smoke runs may differ)'
    cells = []
    teacher = [_repeats(root, passes, p, n, smoke) for p in TEACHER_PASSES]
    teacher_ms = [sum(t[r] for t in teacher) for r in range(PP.PROBE_REPEATS)]
    teacher_e2e = _e2e(root, passes, TEACHER_E2E, n, smoke)
    for enc in (ENCODERS if encoders is None else encoders):
        assert enc in ENCODERS, enc
        x_name, h_name = default_pass(enc), heads_pass(enc)
        assert smoke or not passes[h_name].get('standin'), f'{h_name}: timed with stand-in weights, not the seed-0 c*'
        x_ms, h_ms = _repeats(root, passes, x_name, n, smoke), _repeats(root, passes, h_name, n, smoke)
        cstar_ms = [x_ms[r] + h_ms[r] for r in range(PP.PROBE_REPEATS)]
        used = {x_name, h_name, TEACHER_E2E, *TEACHER_PASSES}
        mine = [d for d in log if d['pass'] in used]
        cells.append({'sc': 'SC15', 'encoder': enc, 'cstar_ms': cstar_ms, 'teacher_ms': teacher_ms,
                      'cstar_ms_max': max(cstar_ms), 'teacher_ms_min': min(teacher_ms),
                      'cstar_ms_mean': sum(cstar_ms) / PP.PROBE_REPEATS,
                      'teacher_ms_mean': sum(teacher_ms) / PP.PROBE_REPEATS,
                      'cstar_ms_e2e': _e2e(root, passes, x_name, n, smoke) + sum(h_ms) / PP.PROBE_REPEATS,
                      'teacher_ms_e2e': teacher_e2e, 'repeats_discarded': len(mine), 'repeats_discarded_log': mine,
                      'x_ms': x_ms, 'networks_ms': h_ms, 'n_images': n})
    return cells


def gate_viii(ms):
    """Gate (viii): the Qwen default pass end to end within 20% of the MMArt prep log's 86.3 ms per image."""
    assert isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0, ms
    ref, tol = PP.QWEN_E2E_REF_MS, PP.QWEN_E2E_TOL
    rel = abs(ms - ref) / ref
    return {'pass': bool(rel <= tol), 'qwen_default_e2e_ms': ms, 'reference_ms': ref, 'tolerance': tol,
            'relative_difference': rel, 'low_ms': ref * (1 - tol), 'high_ms': ref * (1 + tol)}


def cost_table(root, cells, smoke=None):
    """Rows for the paper's cost table from the raw timings: per pass ms per image (mean over repeats), images per second,
    peak memory, parameters, end to end; the whole-set time for MMArt's 16,384 test paintings, beside the prep log's
    measured Qwen times (meta.json of its stores)."""
    doc = _read(Path(root) / 'probe' / 'passes.json')
    rows = {}
    for name, p in doc['passes'].items():
        ms = [v['ms_per_image'] for v in p['repeats'].values()]
        row = {k: p[k] for k in ('encoder', 'kind', 'peak_memory_allocated_mib', 'peak_memory_reserved_mib',
                                 'params_loaded', 'params_image_path', 'params_network', 'vision_tower') if k in p}
        if ms:
            row.update(ms_per_image_mean=sum(ms) / len(ms), images_per_second=1000 / (sum(ms) / len(ms)))
        if p.get('e2e'):
            row.update(ms_per_image_e2e=p['e2e']['ms_per_image'], images_per_second_e2e=1000 / p['e2e']['ms_per_image'])
        rows[name] = row
    whole = {}
    for c in cells:
        whole[c['encoder']] = {'cstar_s': c['cstar_ms_mean'] * WHOLE_SET / 1000, 'teacher_s': c['teacher_ms_mean'] * WHOLE_SET / 1000,
                               'cstar_e2e_s': c['cstar_ms_e2e'] * WHOLE_SET / 1000,
                               'teacher_e2e_s': c['teacher_ms_e2e'] * WHOLE_SET / 1000, 'paintings': WHOLE_SET}
    prep = {}
    base = PP.MMART_PRE
    for view, folder in (('qwen_default', 'qwen3vl_emb_2b'), *((v, f'qwen3vl_emb_2b_aspect_prompts/{v}') for v in TEACHER_VIEWS)):
        f = base / folder / 'meta.json'
        if f.is_file():
            m = _read(f)
            prep[view] = {'seconds': m['seconds'], 'paintings': m['n'], 'ms_per_image': 1000 * m['seconds'] / m['n']}
    return {'passes': rows, 'whole_set_16384': whole, 'prep_log_qwen': prep, 'n_images': doc['meta']['n_images']}


# ================================================================================================ GPU plumbing
def now():
    from zoneinfo import ZoneInfo
    return datetime.datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M:%S')


def probe_dir():
    return PP.OUT / 'probe'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    os.replace(tmp, path)


def smoke_guard(n):
    assert n > 0
    assert (n != PP.PROBE_N) == bool(PP.SMOKE), \
        f'{n} keys needs PP_SMOKE=1 with PP_OUTPUTS (and the real probe uses {PP.PROBE_N} keys without PP_SMOKE)'


def probe_keys(n):
    import pp_register  # noqa: F401  (import order of ticket 01: pp_paths, pp_register, then the rest)
    import mt_data
    keys = mt_data.keys('test')[:n]
    assert len(keys) == n and len(set(keys)) == n
    return keys


def keys_sha(keys):
    return hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest()


def environment(keys, n):
    import torch
    import transformers
    return {'n_images': n, 'batch_size': PP.PROBE_BATCH, 'repeats': PP.PROBE_REPEATS, 'warmup_batches': PP.PROBE_WARMUP,
            'keys_sha256': keys_sha(keys), 'first_key': keys[0], 'last_key': keys[-1], 'smoke': bool(PP.SMOKE),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'transformers': transformers.__version__,
            'host': os.uname().nodename, 'started': now()}


def update_passes(keys, n, edit):
    """Read-modify-write OUT/probe/passes.json (the meta must match across jobs: same keys, batch, repeats)."""
    path = probe_dir() / 'passes.json'
    env = environment(keys, n)
    if path.is_file():
        doc = _read(path)
        for k in ('n_images', 'batch_size', 'repeats', 'warmup_batches', 'keys_sha256', 'smoke'):
            assert doc['meta'][k] == env[k], f'passes.json was written with another {k}: {doc["meta"][k]} vs {env[k]}'
    else:
        doc = {'meta': env, 'passes': {}, 'repeats_discarded': []}
    edit(doc)
    write_json(path, doc)


def sync_fn():
    import torch
    return torch.cuda.synchronize


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def image_path_params(enc):
    """(parameters of the image path of one pass, parameters loaded). CLIP and SigLIP: the vision model and its projection
    (the text tower is loaded but not used); DINOv2 and Qwen: the whole model (Qwen's language model is in the pass)."""
    m = enc.m
    loaded = count_params(m)
    if hasattr(m, 'vision_model'):
        proj = getattr(m, 'visual_projection', None)
        return count_params(m.vision_model) + (count_params(proj) if proj is not None else 0), loaded
    return loaded, loaded


def decode_all(paths, workers=8):
    """RGB images at file size, fully decoded, in memory (the MMArt prep's `_open_rgb`)."""
    from extract_features import _open_rgb
    with ThreadPoolExecutor(workers) as pool:
        ims = list(pool.map(_open_rgb, paths))
    return ims


def image_paths(keys):
    import mp_extract
    table = mp_extract.S.read_index().set_index('key')
    paths = table.loc[keys].image_path.tolist()
    assert all(Path(p).is_file() for p in paths), 'missing WikiArt images'
    return paths


def load_qwen_views(enc):
    """Instruction of each Qwen pass, as the extractors set it (`mp_extract.held_identity`, `assert_image_instruction`)."""
    import mp_extract
    ins = {'default': enc.instruction}
    for v in TEACHER_VIEWS:
        ins[v] = mp_extract.held_identity(v)['instruction']
    return ins


def set_instruction(enc, ins):
    import mp_extract
    enc.instruction = ins
    mp_extract.assert_image_instruction(enc, ins)


def file_done(name):
    """All raw files of a pass exist: three repeats (not the teacher's end-to-end pass) and, for default passes and the
    teacher, the end to end."""
    b = probe_dir() / 'batches'
    ok = name == TEACHER_E2E or all((b / f'{name}__r{k}.json').is_file() for k in range(PP.PROBE_REPEATS))
    if name == TEACHER_E2E or name in ENCODERS:
        ok = ok and (b / f'{name}__e2e.json').is_file()
    return ok


def any_file(name):
    b = probe_dir() / 'batches'
    return any(b.glob(f'{name}__*.json'))


def store_discards(doc, discarded):
    """Write each discarded record to OUT/probe/discarded and its summary into passes.json."""
    for d in discarded:
        rec = d.pop('record')
        tag = 'e2e' if d['repeat'] == 'e2e' else f"r{d['repeat']}"
        f = f"{d['pass']}__{tag}__a{d['attempt']}.json"
        write_json(probe_dir() / 'discarded' / f, rec)
        doc['repeats_discarded'].append({**d, 'file': f, 'total_seconds': rec['total_seconds'], 'kind': rec['kind']})


def run_one_pass(name, view, encode, items, repeats, kind='model', e2e=None, poll=poll_gpu, **extra):
    """3 repeats of `encode` over `items` (+ the end to end `e2e() -> record`), polled; returns (entry, discards)."""
    import torch
    own = os.getpid()
    sync, clock = torch.cuda.synchronize, time.perf_counter
    torch.cuda.reset_peak_memory_stats()
    entry = {'encoder': extra.pop('encoder'), 'kind': kind, 'view': view, 'repeats': {}, 'e2e': None, **extra}
    discards = []
    for k in range(repeats):
        rec, d = run_repeat(name, k, lambda: model_repeat(name, encode, items, k, PP.PROBE_BATCH, PP.PROBE_WARMUP, sync,
                                                           clock, kind=kind, view=view), poll, own, wait=30.0)
        discards += d
        write_json(probe_dir() / 'batches' / f'{name}__r{k}.json', rec)
        entry['repeats'][str(k)] = {'file': f'{name}__r{k}.json', 'ms_per_image': rec['ms_per_image'],
                                    'total_seconds': rec['total_seconds'], 'attempt': rec['attempt']}
        print(f'{name} r{k}: {rec["ms_per_image"]:.3f} ms/image (attempt {rec["attempt"]})', flush=True)
    if e2e is not None:
        rec, d = run_repeat(name, 'e2e', e2e, poll, own, wait=30.0)
        discards += d
        write_json(probe_dir() / 'batches' / f'{name}__e2e.json', rec)
        entry['e2e'] = {'file': f'{name}__e2e.json', 'ms_per_image': rec['ms_per_image'],
                        'total_seconds': rec['total_seconds'], 'attempt': rec['attempt']}
        print(f'{name} e2e: {rec["ms_per_image"]:.3f} ms/image', flush=True)
    entry['peak_memory_allocated_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    entry['peak_memory_reserved_mib'] = torch.cuda.max_memory_reserved() / 2 ** 20
    return entry, discards


def e2e_record(name, run, paths, load_images, extra=None):
    """measure() of an end-to-end repeat: batches of `PP.PROBE_BATCH` paths, loaded on the prefetch thread. `extra`: a dict
    or a function returning one, merged into the record (called after the loop)."""
    import torch
    batches = [paths[i:i + PP.PROBE_BATCH] for i in range(0, len(paths), PP.PROBE_BATCH)]

    def measure():
        secs = end_to_end(run, batches, load_images, sync=torch.cuda.synchronize, clock=time.perf_counter)
        more = extra() if callable(extra) else (extra or {})
        return batch_record(name, 'e2e', None, [len(b) for b in batches], secs, batch_size=PP.PROBE_BATCH,
                            warmup_batches=0, **more)
    return measure


def load_images_from_files(batch):
    from extract_features import _open_rgb
    return [_open_rgb(p) for p in batch]


def vision_tower_share(enc, images, own, poll):
    """One extra Qwen default repeat with the vision tower wrapped in synchronised timers: its share of one pass. A separate
    repeat, since the extra synchronisations would perturb the timed ones."""
    import torch
    visual = enc.m.visual
    acc = {'t': 0.0, 'calls': []}
    state = {}

    def pre(_m, _a, _k=None):
        torch.cuda.synchronize()
        state['t0'] = time.perf_counter()

    def post(_m, _a, _o):
        torch.cuda.synchronize()
        acc['t'] += time.perf_counter() - state['t0']

    h1, h2 = visual.register_forward_pre_hook(pre), visual.register_forward_hook(post)

    def encode(batch):
        acc['t'] = 0.0
        enc.encode_images(batch, batch_size=PP.PROBE_BATCH)
        acc['calls'].append(acc['t'])

    def measure():
        acc['calls'].clear()
        rec = model_repeat('qwen_vision', encode, images, 'vision', PP.PROBE_BATCH, PP.PROBE_WARMUP, torch.cuda.synchronize,
                           time.perf_counter, kind='vision')
        rec['visual_batch_seconds'] = acc['calls'][PP.PROBE_WARMUP:]
        assert len(rec['visual_batch_seconds']) == len(rec['batch_n'])
        return rec
    try:
        rec, discards = run_repeat('qwen_vision', 'vision', measure, poll, own, wait=30.0)
    finally:
        h1.remove()
        h2.remove()
    vis = float(sum(rec['visual_batch_seconds']))
    return {'total_seconds': rec['total_seconds'], 'visual_seconds': vis, 'share': vis / rec['total_seconds'],
            'batch_seconds': rec['batch_seconds'], 'visual_batch_seconds': rec['visual_batch_seconds'],
            'params_visual': count_params(visual), 'params_total': count_params(enc.m)}, discards


def cmd_passes(enc_name, n, resume):
    import torch
    import pp_extract
    smoke_guard(n)
    assert torch.cuda.is_available(), 'no GPU'
    names = passes_of(enc_name) + ([TEACHER_E2E] if enc_name == 'qwen' else [])
    todo = []
    for name in names:
        if file_done(name):
            assert resume, f'{name}: raw files exist; use --resume to skip finished passes (nothing is overwritten)'
            continue
        assert not any_file(name), f'{name}: partial raw files in {probe_dir() / "batches"}; inspect, move them away, rerun'
        todo.append(name)
    if not todo:
        print('nothing to do', flush=True)
        return 0
    keys = probe_keys(n)
    paths = image_paths(keys)
    enc, identity, _ = pp_extract.load_encoder(enc_name)
    p_image, p_loaded = image_path_params(enc)
    print(f'{enc_name}: loaded; decoding {n} images', flush=True)
    t0 = time.monotonic()
    images = decode_all(paths)
    mpix = sum(im.size[0] * im.size[1] for im in images) / 1e6
    print(f'decoded {n} images in {time.monotonic() - t0:.0f} s, {mpix:.0f} Mpixel, {3 * mpix / 1000:.1f} GB', flush=True)
    own = os.getpid()
    info = {'encoder': enc_name, 'params_loaded': p_loaded, 'params_image_path': p_image, **identity,
            'decoded_megapixels': mpix}
    ins = load_qwen_views(enc) if enc_name == 'qwen' else None

    def encode(batch):
        enc.encode_images(batch, batch_size=PP.PROBE_BATCH)

    def finish(name, entry, discards, extra=None):
        def edit(doc):
            doc['passes'][name] = {**entry, **(extra or {})}
            store_discards(doc, discards)
        update_passes(keys, n, edit)

    for name in passes_of(enc_name):
        if name not in todo:
            continue
        view = 'default' if name == enc_name else name[len('qwen_'):]
        if enc_name == 'qwen':
            set_instruction(enc, ins[view])
        is_default = name == enc_name
        e2e = e2e_record(name, lambda ims: enc.encode_images(ims, batch_size=PP.PROBE_BATCH), paths,
                         load_images_from_files, {'view': view}) if is_default else None
        entry, discards = run_one_pass(name, view, encode, images, PP.PROBE_REPEATS, e2e=e2e, **info)
        extra = {}
        if enc_name == 'qwen' and is_default:
            vt, d = vision_tower_share(enc, images, own, poll_gpu)
            discards += d
            extra['vision_tower'] = vt
            print(f'qwen vision tower share of one pass: {vt["share"]:.3f}', flush=True)
        finish(name, entry, discards, extra)
    if TEACHER_E2E in todo:
        seconds = {v: [] for v in TEACHER_VIEWS}

        def run_three(ims):
            for v in TEACHER_VIEWS:
                set_instruction(enc, ins[v])
                torch.cuda.synchronize()
                t = time.perf_counter()
                enc.encode_images(ims, batch_size=PP.PROBE_BATCH)
                torch.cuda.synchronize()
                seconds[v].append(time.perf_counter() - t)

        def pass_seconds():
            return {'pass_seconds': {v: list(s) for v, s in seconds.items()}, 'view': 'three instructed passes, one decode'}

        def measure():
            for s in seconds.values():
                s.clear()
            return e2e_record(TEACHER_E2E, run_three, paths, load_images_from_files, pass_seconds)()
        torch.cuda.reset_peak_memory_stats()
        rec, discards = run_repeat(TEACHER_E2E, 'e2e', measure, poll_gpu, own, wait=30.0)
        write_json(probe_dir() / 'batches' / f'{TEACHER_E2E}__e2e.json', rec)
        print(f'teacher e2e: {rec["ms_per_image"]:.3f} ms/image', flush=True)
        entry = {'encoder': 'qwen', 'kind': 'model', 'view': 'three instructed passes, one decode', 'repeats': {},
                 'e2e': {'file': f'{TEACHER_E2E}__e2e.json', 'ms_per_image': rec['ms_per_image'],
                         'total_seconds': rec['total_seconds'], 'attempt': rec['attempt']},
                 'peak_memory_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
        finish(TEACHER_E2E, entry, discards)
    return 0


# ------------------------------------------------------------------------------------------------ the networks
def weights_dir(enc):
    root = Path(PP.CSTAR_ROOT[enc])
    return root / 'weights' / PP.CSTAR_ARM


def load_networks(enc, device, standin):
    """The three seed-0 networks of X's c* (image side, fp32, eval), each a fresh `rb_student.build_model` view as
    `bs_cstar._infer_sides` builds it. Returns (nets, provenance)."""
    import torch
    import rb_rows
    import rb_student
    import pp_stores as S
    width = PP.WIDTH[enc]
    nets, prov = [], {}
    for aspect in PP.ASPECTS:
        _, views = rb_student.build_model(rb_rows.recipe(rb_rows.REFERENCE)['base'], list(PP.ASPECTS), device, width=width)
        view = views[aspect]
        path = weights_dir(enc) / f'{aspect}__s0.pt'
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            side = path.with_suffix('.json')
            if side.is_file():
                assert _read(side)['weights_sha256'] == sha, f'{path}: sidecar sha256 differs'
            view.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            prov[aspect] = {'path': str(path), 'sha256': sha}
        else:
            assert standin, f'{path} does not exist (pass --standin to time random seed-0 stand-ins; not for SC15)'
            torch.manual_seed(0)
            prov[aspect] = {'path': str(path), 'standin': True}
        view.eval()
        nets.append(view['img'])
    return nets, prov


def cmd_heads(encs, n, standin):
    import numpy as np
    import torch
    import pp_stores as S
    smoke_guard(n)
    assert torch.cuda.is_available(), 'no GPU'
    keys = probe_keys(n)
    device = torch.device('cuda')
    own = os.getpid()
    for enc in encs:
        name = heads_pass(enc)
        assert not any_file(name), f'{name}: raw files exist; nothing is overwritten (move them away to rerun)'
        x = np.ascontiguousarray(S.load('mmart', enc, keys), dtype=np.float32)
        assert x.shape == (n, PP.WIDTH[enc]) and np.isfinite(x).all()
        nets, prov = load_networks(enc, device, standin)
        params = [count_params(m) for m in nets]
        if enc == 'qwen':
            assert params == [PP.CSTAR_NET_PARAMS] * 3, params

        def encode(xb):
            with torch.no_grad():
                t = torch.from_numpy(xb).to(device)
                outs = [m(t) for m in nets]
                [o.cpu().numpy() for o in outs]
        entry, discards = run_one_pass(name, 'networks', encode, x, PP.PROBE_REPEATS, kind='heads', encoder=enc,
                                       params_network=params, weights=prov,
                                       standin=any(p.get('standin') for p in prov.values()))

        def edit(doc):
            doc['passes'][name] = entry
            store_discards(doc, discards)
        update_passes(keys, n, edit)
    return 0


# ------------------------------------------------------------------------------------------------ cells
def cmd_cells(encs):
    smoke = bool(PP.SMOKE)
    root = PP.OUT
    cells = build_cells(root, encoders=encs, smoke=smoke)
    doc = _read(root / 'probe' / 'passes.json')
    gate = gate_viii(_e2e(root, doc['passes'], default_pass('qwen'), doc['meta']['n_images'], smoke))
    gate.update(smoke=smoke, n_images=doc['meta']['n_images'])
    write_json(root / 'stage0' / 'gate_viii.json', gate)
    write_json(root / 'stage3' / 'cells_cost.json', cells)
    write_json(root / 'stage3' / 'cost_table.json', cost_table(root, cells, smoke))
    for c in cells:
        verdict = 'holds' if c['cstar_ms_max'] < c['teacher_ms_min'] else 'fails'
        print(f'SC15 {c["encoder"]}: c* max {c["cstar_ms_max"]:.2f} ms, teacher min {c["teacher_ms_min"]:.2f} ms -> {verdict}; '
              f'e2e {c["cstar_ms_e2e"]:.2f} vs {c["teacher_ms_e2e"]:.2f}; discarded {c["repeats_discarded"]}', flush=True)
    print(f'gate (viii): {"pass" if gate["pass"] else "FAIL"}: Qwen default e2e {gate["qwen_default_e2e_ms"]:.2f} ms '
          f'(reference {gate["reference_ms"]} +-{100 * gate["tolerance"]:.0f}%)', flush=True)
    return 0 if gate['pass'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('passes')
    s.add_argument('--enc', choices=ENCODERS, required=True)
    s.add_argument('--keys', type=int, default=PP.PROBE_N)
    s.add_argument('--resume', action='store_true')
    s = sub.add_parser('heads')
    s.add_argument('--enc', choices=ENCODERS, nargs='+', default=list(ENCODERS))
    s.add_argument('--keys', type=int, default=PP.PROBE_N)
    s.add_argument('--standin', action='store_true')
    s = sub.add_parser('cells')
    s.add_argument('--enc', choices=ENCODERS, nargs='+', default=None)
    a = p.parse_args(argv)
    if a.command == 'passes':
        return cmd_passes(a.enc, a.keys, a.resume)
    if a.command == 'heads':
        return cmd_heads(a.enc, a.keys, a.standin)
    assert a.enc is None or PP.SMOKE, '`cells --enc` (a subset) is for smoke runs; SC15 needs all seven encoders'
    return cmd_cells(a.enc)


if __name__ == '__main__':
    sys.exit(main())
