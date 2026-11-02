"""r14 ticket 10: on-node scoring of the knob and photo sweep runs (D15, N7). CPU only, no file read except the validation
labels and names the inherited readers load.

A sweep run keeps no weights and no exports (D15); this module turns the run's validation outputs, held in memory, into
what the paired intervals and the C13 spot check read (index, "Sweep runs"):

  knob (c*'s rows; validation export `dev[aspect]['img']`, float16 [6152, 2048], the trainer's dev order = the sorted
        validation list of r11's reader, checked by key):
     partitions  int32 [3, n] per head: r10's k = 20 k-means, states 0, 1, 2 (`ct_score.output_partitions`, the path of
                 `bs_artclu.validation_ami`), rows in the validation key order stored in the file (`names`)
     AMI         per state (`rb_score.ami`, labels < 0 masked) and their mean, per head
     per-query   AP at R0 (plain cosine, `rt_metrics.retrieval` on `rt_data.embed`'s unit float32 rows), the queries and
                 their rows (indices into the validation list), per head; mAP = their mean (`rt_stage0.validation_seed`)
  photo (r13's rows; validation export = the checkpoint's emotion image side on the 4,251 validation photos, float16):
     partitions  int32 [3, n]: `ac_cluster.partitions` (as `pf_sweep.score`), rows in `ac_data.keys('val')` order
     AMI         `ac_cluster.ami`, mean over the three states
     per-query   AP of r13's sweep read-out (center + k = 10 expansion, `rt_readout` fit on one parity half and applied
                 to the other, cross-fit), the AP of every query kept in its half, NaN for a dropped query; mAP = mean over
                 the two halves of the half mean, which is `rt_readout.crossfit`'s number; the module checks it against
                 `pf_sweep.score` (same AMI, mAP within 1e-12) on every run, then adds r13's gaps (`pf_sweep.gaps`)
Files of a run folder `OUT/sweeps/<kind>/<row>/s<seed>/`: `metrics.json`, `partitions.npz` (`<head>`: int32 [3, n];
`names`: the validation keys), `perquery.npz` (knob: `<head>__ap`, `<head>__rows`; photo: `ap` float64 [n] with NaN for a
dropped query, `keys`), `run.json` (the trainer's record plus attempts and timing), `log.txt` (written by the driver).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import bs_paths as P

for _round in (P.R10_ROUND, P.R11_ROUND, P.R13_ROUND):      # the main checkout's copies, as bs_cstar and bs_photo use
    if str(_round) not in sys.path:
        sys.path.insert(0, str(_round))
import rs_paths  # noqa: E402,F401  (puts r10's and the older rounds' folders on sys.path)
import pf_paths  # noqa: E402,F401
import ct_score  # noqa: E402
import rb_score  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rs_vh  # noqa: E402
import ac_cluster  # noqa: E402
import ac_data  # noqa: E402
import ac_select  # noqa: E402
import pf_sweep  # noqa: E402

ASPECTS = tuple(P.ASPECTS)
TOL_MAP = 1e-12            # the per-query photo mAP against r13's cross-fit number


def _unit32(x16):
    """`rt_data.embed`'s unit float32 rows of a float16 export."""
    x = np.asarray(x16)
    assert x.dtype == np.float16 and x.ndim == 2 and np.isfinite(x).all()
    x = x.astype(np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def validation_names():
    return rt_data.names('val')


def ami_states(x32, y):
    """r10's k = 20 k-means partitions (states 0, 1, 2; `ct_score.output_partitions`, `bs_artclu.validation_ami`'s path)
    of float32 rows `x32` and the AMI of each state against labels `y` (labels < 0 masked). Returns (int partitions
    [3, n], [AMI per state]). Also used by 0(viii) on r10's stored 3090 validation export."""
    parts = np.asarray(ct_score.output_partitions(x32, 'k1')['r20'])
    assert parts.shape == (3, len(y)), parts.shape
    mask = y >= 0
    return parts, [float(rb_score.ami(y[mask], p[mask])) for p in parts]


def score_knob(dev, dev_names, workers=8):
    """Score one knob run from its in-memory validation outputs `dev` {aspect: {'img': f16 [n, 2048], ...}} whose rows
    are `dev_names` (the trainer's dev order). Returns (metrics, partitions, perquery): see the module doc."""
    names, labels = validation_names(), rt_data.labels('val')
    assert [str(n) for n in dev_names] == names, 'dev order is not the sorted validation list'   # joined by key, C6
    metrics, partitions, perquery = {'ami': {}, 'ami_states': {}, 'map_r0': {}, 'n_queries': {}, 'dropped': {}}, {}, {}
    for aspect in ASPECTS:
        x16 = dev[aspect]['img']
        assert x16.shape == (len(names), 2048), (aspect, x16.shape)
        y = labels[aspect]
        parts, states = ami_states(x16.astype(np.float32), y)
        res = rt_metrics.retrieval(_unit32(x16), y, workers=workers)
        partitions[aspect] = parts.astype(np.int32)
        perquery[f'{aspect}__ap'], perquery[f'{aspect}__rows'] = res['ap'], res['rows']
        metrics['ami'][aspect] = float(np.mean(states))
        metrics['ami_states'][aspect] = states
        metrics['map_r0'][aspect] = float(res['ap'].mean())
        metrics['n_queries'][aspect], metrics['dropped'][aspect] = int(len(res['ap'])), int(res['dropped'])
    return metrics, dict(partitions, names=np.asarray(names)), perquery


def score_photo(x_stored, keys=None, workers=8):
    """Score one photo run from its validation export `x_stored` (float16 [4251, 2048], rows in `ac_data.keys('val')`
    order). Returns (metrics, partitions, perquery)."""
    vkeys = list(ac_data.keys('val'))
    assert keys is None or list(keys) == vkeys, 'photo validation keys differ from ac_data.keys(val)'      # C6
    x_stored = np.asarray(x_stored)
    assert x_stored.dtype == np.float16 and x_stored.shape[0] == len(vkeys), (x_stored.dtype, x_stored.shape)
    labels = ac_data.labels('val')
    parts = ac_cluster.partitions(x_stored)
    ami = float(ac_cluster.ami(parts, labels))
    ro = ac_select.readout_by_name(pf_paths.SWEEP_READOUT, x_stored.shape[1])
    x = pf_sweep.unit_rows(x_stored)
    n = len(labels)
    ap = np.full(n, np.nan)
    half_means = []
    for h, (score_idx, fit_idx) in enumerate(zip(rt_readout.halves(n), reversed(rt_readout.halves(n)))):
        params = rt_readout.fit(ro, x[fit_idx])
        applied = rt_readout.apply(params, x[score_idx])
        res = rt_metrics.retrieval(applied, labels[score_idx], qe_k=ro.k, workers=workers)
        ap[score_idx[res['rows']]] = res['ap']
        half_means.append(float(res['ap'].mean()))
    map_ = float(np.mean(half_means))
    ref = pf_sweep.score(x_stored, labels, workers=workers)              # r13's own path: the check of this one
    assert ref['ami'] == ami and abs(ref['map'] - map_) <= TOL_MAP, (ref['ami'], ami, ref['map'], map_)
    metrics = dict(pf_sweep.gaps(ami, map_), half_maps=half_means, n_queries=int(np.isfinite(ap).sum()),
                   readout=pf_paths.SWEEP_READOUT,
                   ami_states=[float(ac_cluster.ami(p[None], labels)) for p in parts])
    return metrics, dict(emotion=parts.astype(np.int32), names=np.asarray(vkeys)), dict(ap=ap, keys=np.asarray(vkeys))


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_run(folder, metrics, partitions, perquery, record):
    """Write a scored run's files (`metrics.json` last: its presence marks the run done). `record`: the run's
    `run.json` content (the driver adds status, attempts, seconds). Refuses to overwrite a finished run."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'metrics.json').exists(), f'{folder}: finished (metrics.json exists); move it aside to rerun'
    np.savez(folder / 'partitions.pending.npz', **partitions)
    (folder / 'partitions.pending.npz').replace(folder / 'partitions.npz')
    np.savez(folder / 'perquery.pending.npz', **perquery)
    (folder / 'perquery.pending.npz').replace(folder / 'perquery.npz')
    (folder / 'run.json').write_text(json.dumps(record, indent=1, allow_nan=False, default=_json_default) + '\n')
    doc = dict(metrics, partitions_sha256=_file_sha(folder / 'partitions.npz'),
               perquery_sha256=_file_sha(folder / 'perquery.npz'), scored_at=time.strftime('%F %T'))
    pending = folder / 'metrics.pending.json'
    pending.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    pending.replace(folder / 'metrics.json')
    return doc


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
