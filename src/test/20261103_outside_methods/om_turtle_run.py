"""r18 ticket 09: the 412 TURTLE fits and the timing smoke (spec §3.3, D9, D21, §3.6).

Fit list (held split of each set, `om_data.plain` / `om_data.cstar` inputs, ticket 02's `om_turtle.fit`):
    plain image: every `om_paths.plain_ks(set)` (8 per set), TURTLE seed = start 0 to 2   -> 2 sets x 4 enc x 24 = 192
    c*'s heads:  every `om_paths.label_ks(set, aspect)` (11 pairs), c* seeds 0 to 4, start 0 -> 4 enc x 55      = 220
Each fit becomes a partition file with a sidecar (`om_data.save_parts`): the fit's loss, entropy, n_clusters, seconds,
device, plus `torch_version`, `turtle` (`om_paths.TURTLE` and the outer steps used), and `input_sha256` (sha256 of the
exact float32 fit input, `om_data.array_sha256`). 16 jobs: set x encoder x arm. A fit whose npz and sidecar exist and
pass `om_data.check_part` (with the same input sha256) is skipped; anything else (an npz alone after a crash, a changed
input) is rewritten with `overwrite=True`.

GPU runs need nvrtc's builtins (`torch.special.entr` compiles a kernel at run time). EVERY GPU command is prefixed:
    LD_LIBRARY_PATH=/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib
Example (one of the 16 jobs):
    LD_LIBRARY_PATH=/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib \\
      /root/miniconda3/envs/CoSiR/bin/python om_turtle_run.py fit --set mmart --enc qwen --arm cstar --device cuda
      (`res-queue gpu` wraps it)

CLI:
    python om_turtle_run.py --list                          the plan with counts (asserts 412), no data read
    python om_turtle_run.py fit --set S --enc E --arm plain|cstar [--device cuda|cpu] [--limit N] [--outer-steps N]
    python om_turtle_run.py timing [--device cuda|cpu]      -> OUT/smoke/turtle_timing.json
    python om_turtle_run.py pack [--pack DIR] [--limit N]   local: export every fit input once to DIR (default OUT/turtle_pack)
    python om_turtle_run.py fit --pack DIR ... [--only-k K --only-start S]   inputs and keys from the pack (node: no stores)
    python om_turtle_run.py check-pulled RESULTS_DIR [--pack DIR]            local: check the 412 pulled fits against the pack
`check-pulled` also requires one non-null host, one non-null torch_version, CUDA devices and full default TURTLE settings.
`--outer-steps` (other than 6000) and `--limit` are for the smoke and need OM_SMOKE=1. `--only-k`/`--only-start` keep
only the fits with that K / start (the full fit is unchanged), so they need no OM_SMOKE.

Pack (DIR): <set>/<enc>/plain.npy, <set>/<enc>/<aspect>__s<seed>.npy (float32, exactly `_input`), <set>/keys.json
(`om_data.keys` of the held split), manifest.json {files: {rel path: {sha256, shape}}, keys: {set: {n, sha256}}}. Node
(DAS6) fits read the pack, not the older rounds' stores: scripts/run_om_turtle_node.sh.
"""
import argparse
import hashlib
import itertools
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as P  # noqa: E402

NVRTC_LIB = '/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib'
GPU_PREFIX = f'LD_LIBRARY_PATH={NVRTC_LIB}'
HEAVY_WATTS = 200.0
TIMING_K = 20
TIMING_SET, TIMING_SPLIT = 'mmart', 'test'
TIMING_ENC = {2048: 'qwen', 1152: 'siglip2', 768: 'clip_l14'}      # one plain space per input width
POWER_EVERY_S = 2.0


def plan(set_=None, enc=None, arm=None):
    """The fits of the plan as dicts {set, enc, arm, aspect, K, start, seed}; asserts 412 for the whole plan."""
    out = []
    for s, e in itertools.product(P.SETS, P.ENCODERS):
        for k in P.plain_ks(s):
            for start in P.TURTLE_PLAIN_STARTS:
                out.append(dict(set=s, enc=e, arm='plain', aspect=None, K=k, start=start, seed=None))
        for a in P.ASPECTS:
            for k in P.label_ks(s, a):
                for seed in P.SEEDS:
                    out.append(dict(set=s, enc=e, arm='cstar', aspect=a, K=k, start=P.TURTLE_CSTAR_START, seed=seed))
    assert len(out) == P.N_TURTLE_FITS, (len(out), P.N_TURTLE_FITS)
    assert len({tuple(sorted(f.items(), key=lambda kv: kv[0])) for f in out}) == len(out)
    return [f for f in out if (set_ in (None, f['set'])) and (enc in (None, f['enc'])) and (arm in (None, f['arm']))]


def job_counts():
    jobs = {}
    for f in plan():
        jobs[(f['set'], f['enc'], f['arm'])] = jobs.get((f['set'], f['enc'], f['arm']), 0) + 1
    assert len(jobs) == 16, len(jobs)
    return jobs


def print_plan():
    jobs = job_counts()
    print(f'{len(plan())} fits in {len(jobs)} jobs (held split; outer steps {P.TURTLE["outer_steps"]})')
    for (s, e, a), n in sorted(jobs.items()):
        print(f'  fit --set {s} --enc {e} --arm {a}: {n} fits')
    print('  plain:', sum(1 for f in plan() if f['arm'] == 'plain'), ' cstar:', sum(1 for f in plan() if f['arm'] == 'cstar'))
    print('GPU runs need the prefix:', GPU_PREFIX)


def _input(f, held):
    import om_data
    if f['arm'] == 'plain':
        return om_data.plain(f['set'], held, f['enc'])
    return om_data.cstar(f['set'], held, f['enc'], f['aspect'], f['seed'])


# ------------------------------------------------------------------------------------------------ pack
def pack_rel(f):
    """Pack path of a fit's input, relative to the pack folder."""
    name = 'plain.npy' if f['arm'] == 'plain' else f'{f["aspect"]}__s{f["seed"]}.npy'
    return f'{f["set"]}/{f["enc"]}/{name}'


def pack_inputs(set_=None, enc=None, arm=None):
    """The distinct inputs of the plan, in plan order: list of (rel path, a fit that uses it)."""
    seen = {}
    for f in plan(set_, enc, arm):
        seen.setdefault(pack_rel(f), f)
    return list(seen.items())


def _pack_manifest(pack):
    m = json.loads((Path(pack) / 'manifest.json').read_text())
    assert set(m) == {'files', 'keys'}, list(m)
    return m


def _pack_keys(pack, set_):
    ks = json.loads((Path(pack) / set_ / 'keys.json').read_text())
    assert isinstance(ks, list) and all(isinstance(k, str) for k in ks), set_
    return ks


def _pack_input(pack, f, man):
    """The fit input from the pack; asserts its sha256 (om_data.array_sha256) against the manifest. Returns (z, sha)."""
    import om_data
    rel = pack_rel(f)
    z = np.load(Path(pack) / rel, allow_pickle=False)
    sha = om_data.array_sha256(z)
    ent = man['files'][rel]
    assert sha == ent['sha256'] and list(z.shape) == ent['shape'] and z.dtype == np.float32, (rel, sha, ent)
    return z, sha


def pack_build(pack=None, limit=None, set_=None, enc=None, arm=None):
    """Local, CPU: write the inputs of the plan to `pack` (default OUT/turtle_pack). A file whose sha256 equals the
    fresh input's is kept. Returns {written, kept}."""
    import om_data
    pack = Path(pack or Path(P.OUT) / 'turtle_pack')
    pack.mkdir(parents=True, exist_ok=True)
    mp = pack / 'manifest.json'
    man = json.loads(mp.read_text()) if mp.exists() else dict(files={}, keys={})
    todo = pack_inputs(set_, enc, arm)
    if limit:
        todo = todo[:limit]

    def save_manifest():
        text = (json.dumps(man, indent=1, sort_keys=True) + '\n').encode()
        tmp = mp.with_name(mp.name + '.tmp')
        tmp.write_bytes(text)
        tmp.replace(mp)

    for s in sorted({f['set'] for _, f in todo}):
        ks = [str(k) for k in om_data.keys(s, om_data.HELD_SPLIT[s])]
        text = json.dumps(ks).encode()
        tmp = pack / s / 'keys.json.tmp'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(text)
        tmp.replace(pack / s / 'keys.json')
        man['keys'][s] = dict(n=len(ks), sha256=hashlib.sha256(text).hexdigest())
    written = kept = 0
    for rel, f in todo:
        z = np.ascontiguousarray(_input(f, om_data.HELD_SPLIT[f['set']]), dtype=np.float32)
        assert z.ndim == 2 and z.shape[0] == man['keys'][f['set']]['n'], (rel, z.shape)
        assert np.isfinite(z).all(), rel
        sha = om_data.array_sha256(z)
        path = pack / rel
        if path.exists() and om_data.array_sha256(np.load(path, allow_pickle=False)) == sha:
            kept += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + '.tmp.npy')
            np.save(tmp, z, allow_pickle=False)
            tmp.replace(path)
            written += 1
        man['files'][rel] = dict(sha256=sha, shape=list(z.shape))
        print(f'pack {rel}: {z.shape} {sha[:12]}', flush=True)
    save_manifest()
    print(f'pack {pack}: {written} written, {kept} kept, {len(man["files"])} files in the manifest', flush=True)
    return dict(written=written, kept=kept)


def check_pulled(results, pack=None):
    """Local, after a pull: the 412 fits under `results`/parts/turtle against the pack manifest (input sha256, n,
    labels in [0, K), keys equal to om_data.keys) via om_data.check_part. Requires one host and torch version,
    CUDA and default 6000-step settings throughout. Prints the counts; returns (ok, bad)."""
    import om_data
    pack = Path(pack or Path(P.OUT) / 'turtle_pack')
    man = _pack_manifest(pack)
    ok, bad = 0, []
    jobs, hosts, torch_v = {}, {}, {}
    for f in plan():
        sha = man['files'].get(pack_rel(f), {}).get('sha256')
        why = 'not in the pack manifest' if sha is None else om_data.check_part(f['set'], f['enc'], f['arm'], f['K'], f['aspect'], f['start'], f['seed'], out=results,
                                 expect=dict(input_sha256=sha))
        jobs.setdefault((f['set'], f['enc'], f['arm']), [0, 0])[0 if why is None else 1] += 1
        if why is None:
            ok += 1
            p = om_data._path(results, 'turtle', f['set'], f['enc'], f['arm'],
                              om_data._name(f['arm'], f['K'], f['aspect'], f['start'], f['seed']))
            side = json.loads(p.with_suffix('.json').read_text())
            hosts[side.get('host')] = hosts.get(side.get('host'), 0) + 1
            torch_v[side.get('torch_version')] = torch_v.get(side.get('torch_version'), 0) + 1
            if side.get('device') != 'cuda':
                bad.append((f, f'device {side.get("device")!r} is not cuda'))
            if side.get('turtle') != dict(P.TURTLE, outer_steps=6000):
                bad.append((f, 'turtle settings differ from the full default settings'))
        else:
            bad.append((f, why))
    if len(hosts) != 1 or None in hosts:
        bad.append((plan()[0], f'fits require one non-null host: {hosts}'))
    if len(torch_v) != 1 or None in torch_v:
        bad.append((plan()[0], f'fits require one non-null torch_version: {torch_v}'))
    for (s, e, a), (g, b) in sorted(jobs.items()):
        print(f'  {s}/{e}/{a}: {g} good, {b} bad')
    for f, why in bad[:10]:
        print('  BAD', pack_rel(f), f['K'], f['start'], why[:160])
    print(f'check-pulled {results}: {ok} of {len(plan())} fits good, {len(bad)} bad; hosts {hosts}; torch {torch_v}')
    if not bad:
        print(f'copy (not run): mkdir -p {P.OUT}/parts && rsync -a {results}/parts/turtle {P.OUT}/parts/')
    return ok, bad


def run_fit(f, z, device, outer_steps, out=None, ks=None, node=False):
    import om_data
    import om_turtle
    res = om_turtle.fit(z, f['K'], f['start'], device, outer_steps=outer_steps)
    extra = dict(torch_version=torch.__version__, turtle=dict(P.TURTLE, outer_steps=int(outer_steps)))
    if node:
        extra.update(host=socket.gethostname(), input_source='pack')
    return om_data.save_parts(res['labels'], method='turtle', set_=f['set'], enc=f['enc'], arm=f['arm'], k=f['K'],
                              aspect=f['aspect'], start=f['start'], seed=f['seed'], input_sha256=om_data.array_sha256(z),
                              loss=res['loss'], entropy=res['entropy'], n_clusters=res['n_clusters'],
                              seconds=res['seconds'], device=str(device), out=out, overwrite=True, extra=extra, ks=ks)


def fit_job(set_, enc, arm, device, outer_steps=P.TURTLE['outer_steps'], limit=None, out=None, pack=None,
            only_k=None, only_start=None):
    """Run (or resume) one job. Returns {done, skipped}. Inputs are loaded once per (c* aspect, seed) or once.
    `pack`: inputs, their sha256 and the keys come from the pack folder (no om_data.plain, cstar or keys)."""
    import om_data
    held = om_data.HELD_SPLIT[set_]
    todo = plan(set_, enc, arm)
    if only_k is not None:
        todo = [f for f in todo if f['K'] == only_k]
    if only_start is not None:
        todo = [f for f in todo if f['start'] == only_start]
    if limit:
        todo = todo[:limit]
    ks = man = None
    if pack:
        man = _pack_manifest(pack)
        ks = _pack_keys(pack, set_)
        assert man['keys'][set_]['n'] == len(ks) == P.N_ROWS[(set_, held)], (set_, len(ks))
    done = skipped = 0
    cache = {}
    for f in todo:
        key = (f['aspect'], f['seed'])
        if key not in cache:
            cache.clear()
            if pack:
                cache[key] = _pack_input(pack, f, man)
            else:
                z0 = _input(f, held)
                cache[key] = (z0, om_data.array_sha256(z0))
        z, want = cache[key]
        why = om_data.check_part(f['set'], f['enc'], f['arm'], f['K'], f['aspect'], f['start'], f['seed'], out=out,
                                 expect=dict(input_sha256=want), ks=ks)
        name = f'{set_}/{enc}/{arm}/{f["aspect"]}/K{f["K"]}/start{f["start"]}/seed{f["seed"]}'
        if why is None:
            skipped += 1
            continue
        p = run_fit(f, z, device, outer_steps, out=out, ks=ks, node=bool(pack))
        done += 1
        side = json.loads(p.with_suffix('.json').read_text())
        print(f'{name}: {side["seconds"]:.1f} s, loss {side["loss"]:.4f}, n_clusters {side["n_clusters"]}'
              + ('' if 'FileNotFoundError' in why else f' (rewritten: {why[:80]})'), flush=True)
    print(f'job {set_}/{enc}/{arm}: {done} fitted, {skipped} skipped (already good), {len(todo)} in the job', flush=True)
    return dict(done=done, skipped=skipped)


# ------------------------------------------------------------------------------------------------ timing
class PowerSampler:
    """nvidia-smi power draw (W) of GPU 0 every `every` seconds in a thread; empty list when nvidia-smi fails."""

    def __init__(self, every=POWER_EVERY_S):
        self.every, self.watts, self._stop = every, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', '0'],
                               capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None

    def _run(self):
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self.watts.append(w)
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def estimate_total(seconds_by_width):
    """Estimated seconds of the 412 fits: each fit costs the timed fit of its input width (K 20; held n is not larger
    than MMArt test's, so this errs high; K changes the cost only slightly)."""
    total = 0.0
    for f in plan():
        total += seconds_by_width[P.WIDTH[f['enc']] if f['arm'] == 'plain' else P.CSTAR_WIDTH]
    return total


def timing(device, outer_steps=P.TURTLE['outer_steps']):
    """One full fit per input width on MMArt test at K 20, timed with the GPU power draw sampled; writes
    OUT/smoke/turtle_timing.json."""
    import om_data
    import om_turtle
    rows, seconds = [], {}
    for width, enc in TIMING_ENC.items():
        z = om_data.plain(TIMING_SET, TIMING_SPLIT, enc)
        assert z.shape[1] == width, (enc, z.shape)
        with PowerSampler() as ps:
            t0 = time.time()
            res = om_turtle.fit(z, TIMING_K, 0, device, outer_steps=outer_steps)
            wall = time.time() - t0
        w = ps.watts
        seconds[width] = res['seconds']
        rows.append(dict(width=width, enc=enc, n=int(z.shape[0]), K=TIMING_K, seconds=res['seconds'], wall_seconds=wall,
                         n_clusters=res['n_clusters'], power_samples=len(w),
                         power_mean_w=float(np.mean(w)) if w else None, power_max_w=float(np.max(w)) if w else None))
        print(rows[-1], flush=True)
    full = outer_steps == P.TURTLE['outer_steps']
    est = estimate_total(seconds)
    means = [r['power_mean_w'] for r in rows if r['power_mean_w'] is not None]
    out = dict(device=str(device), torch_version=torch.__version__, outer_steps=int(outer_steps), full_fit=full, fits=rows,
               n_fits=P.N_TURTLE_FITS, estimated_total_seconds=est, estimated_total_hours=est / 3600,
               power_mean_w=float(np.mean(means)) if means else None,
               power_max_w=max((r['power_max_w'] for r in rows if r['power_max_w'] is not None), default=None),
               heavy_threshold_w=HEAVY_WATTS,
               heavy=bool(means and np.mean(means) > HEAVY_WATTS),
               note='estimate = sum over the 412 fits of the timed fit of that input width, K 20, MMArt test n')
    P.write_json(Path(P.OUT) / 'smoke' / 'turtle_timing.json', out)
    print(f'estimated total for {P.N_TURTLE_FITS} fits: {est / 3600:.2f} h; power mean {out["power_mean_w"]} W, '
          f'max {out["power_max_w"]} W; heavy={out["heavy"]}')
    return out


def _device(name):
    if name == 'cuda':
        assert torch.cuda.is_available(), 'cuda requested but not available'
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--list', action='store_true', help='print the plan with counts and the GPU prefix')
    sub = ap.add_subparsers(dest='cmd')
    f = sub.add_parser('fit')
    f.add_argument('--set', dest='set_', choices=P.SETS, required=True)
    f.add_argument('--enc', choices=P.ENCODERS, required=True)
    f.add_argument('--arm', choices=('plain', 'cstar'), required=True)
    f.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    f.add_argument('--limit', type=int, default=None)
    f.add_argument('--outer-steps', type=int, default=P.TURTLE['outer_steps'])
    f.add_argument('--pack', default=None, help='read inputs and keys from this pack folder (node)')
    f.add_argument('--only-k', type=int, default=None, help='only the fits with this K (full fits)')
    f.add_argument('--only-start', type=int, default=None, help='only the fits with this start (full fits)')
    pk = sub.add_parser('pack')
    pk.add_argument('--pack', default=None, help='pack folder (default OUT/turtle_pack)')
    pk.add_argument('--limit', type=int, default=None, help='smoke: only the first N inputs')
    pk.add_argument('--set', dest='set_', choices=P.SETS, default=None)
    pk.add_argument('--enc', choices=P.ENCODERS, default=None)
    pk.add_argument('--arm', choices=('plain', 'cstar'), default=None)
    cp = sub.add_parser('check-pulled')
    cp.add_argument('results', help='pulled results folder (holds parts/turtle/...)')
    cp.add_argument('--pack', default=None, help='pack folder (default OUT/turtle_pack)')
    t = sub.add_parser('timing')
    t.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    t.add_argument('--outer-steps', type=int, default=P.TURTLE['outer_steps'])
    args = ap.parse_args(argv)
    if args.list:
        print_plan()
        return 0
    if args.cmd is None:
        ap.error('give --list or a command (fit, timing)')
    if getattr(args, 'outer_steps', P.TURTLE['outer_steps']) != P.TURTLE['outer_steps'] or getattr(args, 'limit', None):
        assert P.SMOKE, '--outer-steps and --limit are for the smoke: set OM_SMOKE=1'
    if P.SMOKE:
        assert Path(P.OUT).resolve() != Path(P.DEFAULT_OUT).resolve(), 'smoke needs OM_OUTPUTS in a scratch folder'
    if args.cmd == 'fit':
        fit_job(args.set_, args.enc, args.arm, _device(args.device), args.outer_steps, args.limit, pack=args.pack,
                only_k=args.only_k, only_start=args.only_start)
    elif args.cmd == 'pack':
        assert not args.limit or P.SMOKE, '--limit is for the smoke: set OM_SMOKE=1'
        pack_build(args.pack, args.limit, args.set_, args.enc, args.arm)
    elif args.cmd == 'check-pulled':
        ok, bad = check_pulled(args.results, args.pack)
        return 0 if not bad else 1
    else:
        timing(_device(args.device), args.outer_steps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
