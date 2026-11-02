"""Independent C13 part-b spot derivation from the rule and frozen r10--r13.

Only compare mode reads r14 sheets. No r14 scorer is imported or inspected;
bs_partb, bs_sweepscore, bs_artclu, bs_artret and bs_diag were consulted only
through their module docstrings for file interfaces. All writes stay in SPOT.
Smoke derives have deliberately non-production intervals and are labelled so.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import bs_paths as P

for _folder in (P.R10_ROUND, P.R11_ROUND, P.R12_ROUND, P.R13_ROUND):
    sys.path.insert(0, str(_folder))
import pf_paths  # noqa: E402; installs the frozen transitive paths
import ac_data  # noqa: E402
import ac_paths  # noqa: E402
import ct2_data  # noqa: E402
import ct_score  # noqa: E402
import rb_ami  # noqa: E402
import rs_ablation  # noqa: E402
import rs_graphs  # noqa: E402
import rs_stage3  # noqa: E402
import rs_sweep  # noqa: E402
import rs_vh  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_paths  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

SPOT = P.DEFAULT_OUT / 'c13/partb_spot'
ASPECTS = tuple(P.ASPECTS)
ROWS = (('knob', 'k2=16'), ('knob', 'k11=80'), ('photo', 'lr=0.003'),
        ('ablation', 'content_correction'), ('ablation', 'relational'))
CLUSTER_R = dict(emotion=0.0046, style=0.0082, genre=0.0128)
MAP_R = dict(emotion=0.0014, style=0.0022, genre=0.0038)
TOL = 1e-6
SOURCES = {
    'knob_ami': 'rs_stage3.joint_bootstrap (joint_draws, boot_scores, rb_ami.score); 2000 draws',
    'photo_ami': 'rs_stage3.joint_draws and boot_scores; one-aspect paired seed reduction; 2000 draws',
    'ablation_ami': 'rs_ablation.held_effect on validation and held; 2000 draws',
    'validation_partitions': "ct_score.output_partitions(x, 'k1') on saved float16 exports",
    'held_partitions': 'r10 ablation/cache/held, hash-checked read only',
    'retrieval': 'rt_metrics.retrieval; validation center+k10 via rt_readout.crossfit_all; '
                 'held via stage5_apply fitted on own full validation export',
    'retrieval_interval': 'rt_stats.paired (5 seeds); same I-then-S transcript generalized to 3 seeds for sweeps',
    'photo_map_point': 'r13 cross-fit: metrics.json map, checked against mean of parity-half AP means',
    'M': 'rs_sweep.mean_gain for AMI, identical mean(effect/r) arithmetic for mAP; photo mean of two units',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_array(x):
    x = np.asarray(x)
    h = hashlib.sha256(str((x.dtype.str, x.shape)).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def effect_record(candidate, reference, seeds, r, interval=None, point=None):
    """Row minus reference, averaged over the paired seeds; r is never rounded."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    assert candidate.shape == reference.shape == (len(seeds),)
    assert np.isfinite(candidate).all() and np.isfinite(reference).all() and r > 0
    e = float(candidate.mean() - reference.mean()) if point is None else float(point)
    result = dict(effect=e, effect_r=e / r, r=float(r), arm_mean=float(candidate.mean()),
                  ref_mean=float(reference.mean()),
                  arm_seed={str(s): float(v) for s, v in zip(seeds, candidate)},
                  ref_seed={str(s): float(v) for s, v in zip(seeds, reference)})
    if interval is not None:
        result.update(lo=float(interval[0]), hi=float(interval[1]),
                      lo_r=float(interval[0]) / r, hi_r=float(interval[1]) / r)
    return result


def m_score(effects, ranges):
    assert set(effects) == set(ranges) and effects
    assert all(np.isfinite(v) and v > 0 for v in ranges.values())
    return float(np.mean([effects[a] / ranges[a] for a in effects]))


def paired_n(cand, ref, draws):
    """r11 paired(), changing only the seeded dimension from five to nseed.

    Sweep seed positions are 0..2 in their declared order (actual seeds 1..3
    or 0..2). Five-seed inputs go directly to the frozen function.
    """
    cand, ref = np.asarray(cand), np.asarray(ref)
    assert cand.dtype == ref.dtype == np.float64
    assert cand.shape == ref.shape and cand.ndim == 2 and cand.shape[0] in (3, 5)
    assert np.isfinite(cand).all() and np.isfinite(ref).all() and draws > 0
    ns, n = cand.shape
    assert n > 0
    if ns == 5:
        return rt_stats.paired(cand, ref, draws=draws)
    rng = np.random.default_rng(0)
    cols = np.vstack([cand, ref])
    boot = np.empty(draws, np.float64)
    for start in range(0, draws, rt_stats._CHUNK):
        size = min(rt_stats._CHUNK, draws - start)
        counts = np.empty((size, n), np.float64)
        selected = np.empty((size, ns), np.int64)
        for b in range(size):
            counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            selected[b] = rng.choice(list(range(ns)), size=ns, replace=True)
        means = counts @ cols.T / n
        boot[start:start + size] = (np.take_along_axis(means[:, :ns], selected, axis=1).mean(axis=1)
                                    - np.take_along_axis(means[:, ns:], selected, axis=1).mean(axis=1))
    return dict(d=float(cand.mean() - ref.mean()), interval=np.percentile(boot, [2.5, 97.5]).tolist(), boot=boot)


class Inputs:
    """Content-addressed own caches, bound to arrays, source bytes and versions."""
    def __init__(self):
        self.files, self.arrays = {}, {}
        modules = [m for m in sys.modules.values() if getattr(m, '__file__', None)]
        paths = {Path(m.__file__).resolve() for m in modules}
        self.code = {str(p): sha_file(p) for p in paths if p.suffix == '.py' and
                     (str(p).startswith(str(P.MAIN / 'src/')) and
                      (p.parent != P.ROUND or p.name in ('bs_paths.py', 'bs_c13_partb.py')))}
        self.code[str(Path(__file__).resolve())] = sha_file(__file__)
        self.versions = rb_ami.versions()
        self.code_key = hashlib.sha256(json.dumps([self.code, self.versions], sort_keys=True).encode()).hexdigest()

    def file(self, path):
        path = Path(path)
        key = str(path.resolve())
        if key not in self.files:
            self.files[key] = sha_file(path)
        return self.files[key]

    def json(self, path):
        self.file(path)
        return read_json(path)

    def npz(self, path):
        self.file(path)
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def identity(self, name, value):
        self.arrays[name] = dict(shape=list(np.asarray(value).shape), sha256=sha_array(value))

    def cached(self, operation, arrays, params, compute):
        identity = dict(operation=operation, arrays={k: sha_array(v) for k, v in arrays.items()},
                        params=params, code=self.code_key)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = SPOT / 'cache' / f'{operation}_{key}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                assert json.loads(str(z['_identity'])) == identity, path
                return {k: z[k] for k in z.files if k != '_identity'}
        result = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, _identity=np.asarray(json.dumps(identity, sort_keys=True)), **result)
        return result


def art_labels(inputs, split):
    # Frozen loaders verify split ordering, counts and names digests themselves.
    for p in (rs_vh.ANNOTATIONS, *rs_vh.GENRE_CSVS):
        inputs.file(p)
    if split == 'held':
        inputs.file(rt_paths.CT_OUT / 'held/held_paintings.json')
    else:
        inputs.file(ct2_data.CLIP / 'metadata.json')
        for path in sorted((ct2_data.CLIP / 'shards').glob('*/sample_ids.npy')):
            inputs.file(path)
    y, names = rt_data.labels(split), rt_data.names(split)
    inputs.identity(f'artelingo/{split}/names', np.asarray(names))
    for a in ASPECTS:
        inputs.identity(f'artelingo/{split}/{a}/labels', y[a])
    return names, y


def expected_queries(y):
    keep = np.flatnonzero(y >= 0)
    counts = np.bincount(y[keep])
    return keep[counts[y[keep]] > 1]


def sweep_unit(inputs, kind, row, seed, names, labels):
    root = P.DEFAULT_OUT / 'sweeps' / kind / row / f's{seed}'
    assert not (root / 'failed.json').exists(), root
    run, metrics = inputs.json(root / 'run.json'), inputs.json(root / 'metrics.json')
    assert run['status'] == 'ok' and run['host'] == 'node402' and run['row'] == row and run['seed'] == seed
    assert not run.get('smoke', False)
    parts, perq = inputs.npz(root / 'partitions.npz'), inputs.npz(root / 'perquery.npz')
    for stem in ('partitions', 'perquery'):
        assert metrics[f'{stem}_sha256'] == inputs.file(root / f'{stem}.npz'), root
    assert parts.pop('names').tolist() == names, (root, 'partition order')
    ap, points = {}, {}
    for a, y in labels.items():
        assert parts[a].shape == (3, len(names)) and parts[a].dtype.kind in 'iu'
        ami = rb_ami.score(y, parts[a], np.flatnonzero(y >= 0))
        want = metrics['ami'][a] if kind == 'knob' else metrics['ami']
        assert abs(ami - want) <= 1e-12, (root, a, 'AMI')
        if kind == 'knob':
            assert np.array_equal(perq[a + '__rows'], expected_queries(y)), (root, a, 'queries')
            ap[a] = perq[a + '__ap']
            points[a] = float(ap[a].mean())
            assert abs(points[a] - metrics['map_r0'][a]) <= 1e-12
        else:
            assert perq['keys'].tolist() == names and perq['ap'].shape == (len(names),)
            full = perq['ap']
            mask = np.zeros(len(names), bool)
            for half in rt_readout.halves(len(names)):
                mask[half[expected_queries(y[half])]] = True
            assert np.array_equal(np.isfinite(full), mask), (root, 'cross-fit dropped queries')
            crossfit = float(np.mean([np.nanmean(full[h]) for h in rt_readout.halves(len(names))]))
            assert abs(crossfit - metrics['map']) <= 1e-12, (root, 'cross-fit point')
            points[a], ap[a] = float(metrics['map']), full[mask]
        assert ap[a].dtype == np.float64 and np.isfinite(ap[a]).all()
    return parts, ap, points


def one_aspect_ami(inputs, y, cand, ref, seeds, draws):
    """One-aspect form of the frozen joint bootstrap, sharing I and S."""
    I, selected = rs_stage3.joint_draws(len(y), draws, seeds)
    boot = {}
    for name, side in (('X', cand), ('F', ref)):
        boot[name] = {}
        for s in seeds:
            got = inputs.cached('ami_boot', {'labels': y, 'parts': side[s], 'draws': I}, {},
                                lambda s=s, side=side: {'boot': rs_stage3.boot_scores(y, {'p': side[s]}, I, 8)['p']})
            boot[name][s] = got['boot']
    values = np.asarray([np.mean([boot['X'][int(s)][b] - boot['F'][int(s)][b] for s in selected[b]])
                         for b in range(draws)])
    return np.percentile(values, [2.5, 97.5]).tolist()


def derive_sweep(inputs, kind, row, nclu, nret):
    seeds = (1, 2, 3) if kind == 'knob' else (0, 1, 2)
    if kind == 'knob':
        names, labels = art_labels(inputs, 'val')
        cr, mr = CLUSTER_R, MAP_R
    else:
        inputs.file(ac_paths.ANNOTATIONS)
        names, labels = ac_data.keys('val'), {'emotion': ac_data.labels('val')}
        inputs.identity('affection/val/names', np.asarray(names))
        inputs.identity('affection/val/emotion/labels', labels['emotion'])
        ranges = inputs.json(P.R13_ROUND / 'stage2_records.json')['ranges']
        cr, mr = {'emotion': ranges['clu']['r']}, {'emotion': ranges['map']['r']}
    units = {name: {s: sweep_unit(inputs, kind, name, s, names, labels) for s in seeds} for name in (row, 'ref')}
    part = {a: {name: {s: units[name][s][0][a] for s in seeds} for name in units} for a in labels}
    if kind == 'knob':
        # Content-bound per-run caches make later rows reuse the reference boots.
        caches = {a: {s: BootCache(inputs, labels[a], {n: part[a][n][s] for n in units}, nclu)
                      for s in seeds} for a in labels}
        with redirect_stdout(io.StringIO()):
            joint = rs_stage3.joint_bootstrap(labels, part, row, 'ref', seeds, nclu, workers=8, caches=caches)
        m_ami = joint['M']
    else:
        joint = None
        m_ami = None
    ami, map_ = {}, {}
    for a, y in labels.items():
        means = {n: [rb_ami.score(y, part[a][n][s], np.flatnonzero(y >= 0)) for s in seeds] for n in units}
        ci = ([joint['effects'][a]['lo'], joint['effects'][a]['hi']] if joint else
              one_aspect_ami(inputs, y, part[a][row], part[a]['ref'], seeds, nclu))
        ami[a] = effect_record(means[row], means['ref'], seeds, cr[a], ci)
        ap = {n: np.stack([units[n][s][1][a] for s in seeds]) for n in units}
        paired = paired_n(ap[row], ap['ref'], nret)
        points = {n: [units[n][s][2][a] for s in seeds] for n in units}
        map_[a] = effect_record(points[row], points['ref'], seeds, mr[a], paired['interval'])
    ma = m_score({a: v['effect'] for a, v in ami.items()}, cr)
    mm = m_score({a: v['effect'] for a, v in map_.items()}, mr)
    if kind == 'knob':
        assert abs(ma - rs_sweep.mean_gain({a: v['effect'] for a, v in ami.items()})) < 1e-12
    values = dict(ami=ami, map={'R0' if kind == 'knob' else 'center+k10': map_},
                  M_ami=m_ami if m_ami else {'M': ma}, M_map={'M': mm})
    if kind == 'photo':
        values['M'] = (ma + mm) / 2
    return dict(kind=kind, row=row, reference='ref', seeds=list(seeds), validation=values)


class BootCache:
    """rs_stage3 cache interface, bound to the exact joint I transcript."""
    def __init__(self, inputs, labels, parts, draws):
        self.inputs, self.labels, self.parts, self.draws = inputs, labels, parts, draws

    def path(self, name, tag):
        identity = dict(labels=sha_array(self.labels), parts=sha_array(self.parts[name]),
                        draws=self.draws, tag=tag, code=self.inputs.code_key, stream='joint_draws(0;1,2,3)')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return SPOT / 'cache' / f'joint_boot_{key}.npz'

    def get(self, name, tag):
        p = self.path(name, tag)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                return z['boot']
        return None

    def put(self, name, tag, boot):
        p = self.path(name, tag)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, boot=boot)


def export_path(arm, split, a, s):
    root = P.R10_OUT / 'full'
    removal = arm.removeprefix('k11=40-no_') if '-no_' in arm else None
    graph = rs_graphs.row_names(rs_ablation.GRAPH_VARIANT.get(removal, 'aspect'))[a]
    return (root / f'held/{arm}/{a}__s{s}.npz' if split == 'held' else
            root / f'train/{arm}/R6-5/{a}__{graph}__s{s}.npz')


def ablation_export(inputs, arm, split, a, s, names):
    """r11's _cstar validation/held checks, extended to the removal identity."""
    path = export_path(arm, split, a, s)
    side = inputs.json(path.with_suffix('.json'))
    x = inputs.npz(path)['img']
    assert x.shape == (len(names), 2048) and x.dtype == np.float16
    assert np.isfinite(x).all() and np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max() < 1e-3
    digest = rs_vh.names_sha(names)
    if split == 'held':
        rs_vh.check_sidecar(side, x, a, s, len(names), digest, arm)
        assert side['arrays']['img'] == dict(shape=list(x.shape), dtype='float16', sha256=rs_vh.array_sha(x))
    else:
        cfg = side['config']
        assert cfg['mode'] == 'full' and not cfg['smoke']
        assert cfg['aspect'] == a and cfg['seed'] == s and cfg['rg']['arm'] == arm
        assert path.stem == f'{a}__{cfg["graph_row"]}__s{s}'
        assert cfg['names_sha256']['dev'] == digest and cfg['output_rows'] == len(names)
        assert side['output_sha256'] == inputs.file(path)
    return x


def ablation_parts(inputs, arm, split, a, s, names, raw):
    if split == 'held':
        path = P.R10_OUT / f'ablation/cache/held/{arm}/{a}__s{s}__r20.npz'
        side = inputs.json(path.with_suffix('.json'))
        assert side['partitions_sha256'] == inputs.file(path)
        assert side['identity']['held_sha256'] == rs_vh.names_sha(names)
        assert side['identity']['source_sha256'] == inputs.file(export_path(arm, split, a, s))
        parts = inputs.npz(path)['labels']
    else:
        # r10's dev cache is on P (29232), not the required 6152 val rows.
        parts = inputs.cached('validation_parts', {'export': raw}, {},
                              lambda: {'parts': ct_score.output_partitions(raw, 'k1')['r20']})['parts']
    assert parts.shape == (3, len(names)) and parts.dtype.kind in 'iu'
    return parts


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def ablation_ap(inputs, raw_val, raw, y, split, readout):
    ro = rt_readout.R0 if readout == 'R0' else rt_readout.ReadOut('center', None, 10)

    def compute():
        xval, x = unit(raw_val), unit(raw)
        if split == 'val' and readout == 'center+k10':
            # r11 stage 4: fit each parity half's map on the opposite half;
            # average the two half mAPs. Each call scores one aspect and seed.
            point = rt_readout.crossfit_all(
                {('emotion', None): xval}, {'emotion': y}, [ro], workers=8)[ro]['emotion']
            return dict(point=np.asarray(point))
        # Held read-outs fit on the entire run-specific validation export.
        mapped = x if readout == 'R0' else rt_readout.stage5_apply(ro, xval, x)
        scored = rt_metrics.retrieval(mapped, y, qe_k=ro.k, workers=8)
        assert np.array_equal(scored['rows'], expected_queries(y))
        return dict(ap=scored['ap'], rows=scored['rows'], point=np.asarray(scored['ap'].mean()))

    return inputs.cached('retrieval', {'val': raw_val, 'export': raw, 'labels': y},
                         dict(split=split, readout=readout,
                              validation_center_score='r11_crossfit_all'), compute)


def derive_ablation(inputs, removal, nclu, nret):
    seeds, arm, ref = tuple(range(5)), 'k11=40-no_' + removal, 'k11=40'
    # The old aggregate records supply provenance/identity only, never numbers.
    for split in ('dev', 'held'):
        rec = inputs.json(P.R10_OUT / f'ablation/{split}.json')
        assert rec['final']['name'] == ref and rec['host'] == 'cbd437f8ce92' and not rec['smoke']
        if split == 'held':
            old_held = rec
    result = dict(kind='ablation', row=removal, reference=ref, seeds=list(seeds))
    labels = {sp: art_labels(inputs, sp) for sp in ('val', 'held')}
    # Work per aspect and seed: only the needed exports are held in memory.
    for split in ('val', 'held'):
        names, ys = labels[split]
        ami, maps = {}, {ro: {} for ro in ('R0', 'center+k10')}
        for a in ASPECTS:
            parts = {n: {} for n in (arm, ref)}
            scores = {ro: {n: [] for n in parts} for ro in maps}
            aps = {ro: {n: [] for n in parts} for ro in maps}
            for n in parts:
                for s in seeds:
                    rawval = ablation_export(inputs, n, 'val', a, s, labels['val'][0])
                    raw = rawval if split == 'val' else ablation_export(inputs, n, split, a, s, names)
                    parts[n][s] = {'r20': ablation_parts(inputs, n, split, a, s, names, raw)}
                    for ro in maps:
                        got = ablation_ap(inputs, rawval, raw, ys[a], split, ro)
                        scores[ro][n].append(float(got['point']))
                        if split == 'held':
                            aps[ro][n].append(got['ap'])
            frozen = rs_ablation.held_effect(parts[arm], parts[ref], ys[a], nclu, 8)
            ami[a] = effect_record([frozen['arm_seed_ami'][str(s)] for s in seeds],
                                   [frozen['final_seed_ami'][str(s)] for s in seeds], seeds,
                                   CLUSTER_R[a], [frozen['lo'], frozen['hi']])
            if split == 'held':
                # Ancillary replay column present in the sheet. This reads an
                # older-round record only after independently scoring the data.
                previous = old_held['results'][removal]['aspects'][a]
                keys = ('effect', 'lo', 'hi', 'arm_ami', 'final_ami')
                ami[a]['vs_r10_held_json_max_abs'] = max(abs(frozen[k] - previous[k]) for k in keys)
            for ro in maps:
                ci = None
                if split == 'held':
                    ci = rt_stats.paired(np.stack(aps[ro][arm]), np.stack(aps[ro][ref]), draws=nret)['interval']
                maps[ro][a] = effect_record(scores[ro][arm], scores[ro][ref], seeds, MAP_R[a], ci)
        result['validation' if split == 'val' else 'held'] = dict(
            ami=ami, map=maps, M_ami={'M': rs_sweep.mean_gain({a: ami[a]['effect'] for a in ASPECTS})},
            M_map={ro: {'M': m_score({a: maps[ro][a]['effect'] for a in ASPECTS}, MAP_R)} for ro in maps})
    return result


def derive(rows=None, draws_smoke=None):
    selected = list(ROWS) if rows is None else [r for r in ROWS if r[1] in rows or ':'.join(r) in rows]
    assert selected and (rows is None or all(any(x in (r[1], ':'.join(r)) for r in ROWS) for x in rows)), 'unknown rows'
    assert draws_smoke is None or draws_smoke > 0
    nclu, nret = (2000, 10000) if draws_smoke is None else (draws_smoke, draws_smoke)
    start, inputs = time.monotonic(), Inputs()
    inputs.file(P.MAIN / 'docs/superpowers/specs/2026-11-02-backbones-sweep-design.md')
    result = dict(schema=1, smoke=draws_smoke is not None, draws=dict(ami=nclu, map=nret),
                  selected=[':'.join(r) for r in selected], sources=SOURCES,
                  interfaces_only=[f'{n}.__doc__' for n in
                                   ('bs_partb', 'bs_sweepscore', 'bs_artclu', 'bs_artret', 'bs_diag')],
                  resolutions={
                      'ablation_validation': 'five seeds; cluster validation exports, not r10 dev.json P partitions',
                      'ablation_readout_fit': 'validation center+k10 cross-fit; held fits entire own validation export',
                      'photo_M': 'mean of the one-aspect AMI and mAP effects in r13 ranges',
                      'three_seed_retrieval': 'r11 RNG/draw order with seed count three, no duplicated pseudo-seeds'},
                  rows=[])
    for kind, row in selected:
        stamp = time.monotonic()
        with threadpool_limits(limits=8):
            rec = (derive_ablation(inputs, row, nclu, nret) if kind == 'ablation' else
                   derive_sweep(inputs, kind, row, nclu, nret))
        rec['seconds'] = time.monotonic() - stamp
        result['rows'].append(rec)
        print(f'{kind}:{row}: derived ({rec["seconds"]:.1f}s)', flush=True)
        result.update(inputs_sha256=inputs.files, array_inputs=inputs.arrays, code_sha256=inputs.code,
                      versions=inputs.versions, seconds=time.monotonic() - start)
        write_json(SPOT / 'derived.json', result)
    return result


def flatten_numbers(value, prefix=()):
    """Only numerical row cells; identities and provenance live outside this tree."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten_numbers(v, prefix + (str(k),))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flatten_numbers(v, prefix + (str(i),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def compare_values(derived, sheet, tol=TOL):
    """Strict, bidirectional identity join; duplicates must be rejected by adapters."""
    common = sorted(derived.keys() & sheet.keys())
    items = [dict(identity=list(k), derived=derived[k], sheet=sheet[k],
                  abs_diff=abs(derived[k] - sheet[k])) for k in common]
    mismatches = [x for x in items if not np.isfinite(x['abs_diff']) or x['abs_diff'] > tol]
    return dict(items=len(items), mismatches=mismatches,
                max_abs_diff=max((x['abs_diff'] for x in items), default=0.0),
                missing_derived=[list(k) for k in sorted(sheet.keys() - derived.keys())],
                missing_sheet=[list(k) for k in sorted(derived.keys() - sheet.keys())], details=items)


def compare():
    # The concrete sheet adapter is kept separate from derivation below.
    doc = read_json(SPOT / 'derived.json')
    ours, theirs, sheets = sheet_numbers(doc)
    report = compare_values(ours, theirs)
    report.update(smoke=doc['smoke'], selected=doc['selected'], tolerance=TOL)
    report['rows_missing_derived'] = [':'.join(r) for r in ROWS if not any(k[:2] == r for k in ours)]
    report['rows_missing_sheet'] = [':'.join(r) for r in ROWS if not any(k[:2] == r for k in theirs)]
    report['inputs_sha256'] = {str(SPOT / 'derived.json'): sha_file(SPOT / 'derived.json'),
                              **{str(p): sha_file(p) for p in sheets}}
    report['evidence_only'] = ['per-seed means (arm_seed, ref_seed)',
                               'photo M_ami/M_map components, combined in M']
    report['pass'] = not (report['mismatches'] or report['missing_derived'] or report['missing_sheet'] or doc['smoke']
                          or report['rows_missing_derived'] or report['rows_missing_sheet'])
    write_json(SPOT / 'compare.json', report)
    print(f'items={report["items"]} mismatches={len(report["mismatches"])} max_abs_diff={report["max_abs_diff"]:.12g}'
          f' missing_derived={report["missing_derived"]} missing_sheet={report["missing_sheet"]}'
          f' rows_missing_derived={report["rows_missing_derived"]} rows_missing_sheet={report["rows_missing_sheet"]}'
          + (' SMOKE (not certification)' if doc['smoke'] else ''))
    return 0 if report['pass'] else 1


EFFECT_FIELDS = {'effect': 'd', 'effect_r': 'd_r', 'arm_mean': 'row_mean', 'ref_mean': 'ref_mean',
                 'r': 'r', 'lo': 'lo', 'hi': 'hi', 'lo_r': 'lo_r', 'hi_r': 'hi_r',
                 'vs_r10_held_json_max_abs': 'vs_r10_held_json_max_abs'}


def derived_numbers(doc):
    """Normalize comparable outputs by kind/row/split/aspect/metric/statistic."""
    out = {}
    for rec in doc['rows']:
        kind, row = rec['kind'], rec['row']
        assert (kind, row) in ROWS
        for split in ('validation', 'held'):
            if split not in rec:
                continue
            values = rec[split]
            for metric, aspects in [('ami', values['ami'])] + [
                    ('map_r0' if ro == 'R0' else 'map_center_k10', vs) for ro, vs in values['map'].items()]:
                if kind == 'photo' and metric == 'map_center_k10':
                    metric = 'map'
                for a, effect in aspects.items():
                    for field, stat in EFFECT_FIELDS.items():
                        if field in effect:
                            key = (kind, row, split, a, metric, stat)
                            assert key not in out, f'duplicate derived identity {key}'
                            out[key] = float(effect[field])
            if kind == 'photo':
                out[(kind, row, split, 'all', 'M', 'M')] = values['M']
            else:
                for stat, val in values['M_ami'].items():
                    out[(kind, row, split, 'all', 'M_ami', stat)] = float(val)
                mm = values['M_map']
                if kind == 'knob':
                    mm = {'R0': mm}
                for ro, m in mm.items():
                    for stat, val in m.items():
                        metric = 'M_map_r0' if ro == 'R0' else 'M_map_center_k10'
                        out[(kind, row, split, 'all', metric, stat)] = float(val)
    return out


def adapt_sheet(kind, sheet):
    """Concrete JSON adapter; all numeric row fields are consumed or unmatched.

    Identity fields (seeds, row settings), timing and provenance are metadata.
    Unknown numerical result fields are preserved with an unmapped identity so
    schema growth can never silently pass. Null validation intervals are absent.
    """
    out, consumed = {}, set()

    def put(path, identity, value):
        assert identity not in out, f'duplicate sheet identity {identity}'
        out[identity] = float(value)
        consumed.add(path)

    def effect(node, path, prefix):
        for stat, value in node.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                put(path + (stat,), prefix + (stat,), value)

    def mcell(node, path, prefix):
        if isinstance(node, (int, float)):
            put(path, prefix + ('M',), node)
        else:
            effect(node, path, prefix)

    rows = sheet.get('rows', {})
    if isinstance(rows, list):
        indexed = {}
        for rec in rows:
            name = rec.get('row', rec.get('removal'))
            assert name not in indexed, f'duplicate sheet row {name}'
            indexed[name] = rec
        rows = indexed
    for expected_kind, row in ROWS:
        if kind != expected_kind or row not in rows:
            continue
        rec = rows[row]
        assert rec.get('row', rec.get('removal', row)) == row
        base = (kind, row)
        if kind == 'photo':
            for metric in ('ami', 'map'):
                effect(rec.get(metric, {}), (metric,), base + ('validation', 'emotion', metric))
            if 'M' in rec:
                mcell(rec['M'], ('M',), base + ('validation', 'all', 'M'))
        elif kind == 'ablation':
            for source_split, split in (('val', 'validation'), ('held', 'held')):
                for a, aspects in rec.get(source_split, {}).items():
                    for metric, cell in aspects.items():
                        effect(cell, (source_split, a, metric), base + (split, a, metric))
                for metric, value in rec.get('M_' + source_split, {}).items():
                    mcell(value, ('M_' + source_split, metric), base + (split, 'all', 'M_' + metric))
        else:
            # Knob sheets use aspects -> aspect -> {ami,map_r0}.
            aspects = rec.get('aspects', rec.get('effects', {}))
            if not aspects:
                aspects = {a: rec[a] for a in ASPECTS if a in rec}
            for a, metrics in aspects.items():
                for metric, cell in metrics.items():
                    if isinstance(cell, dict):
                        effect(cell, ('aspects', a, metric), base + ('validation', a, metric))
            for field, metric in (('M', 'M_ami'), ('M_ami', 'M_ami'), ('M_map', 'M_map_r0'), ('M_map_r0', 'M_map_r0')):
                if field in rec:
                    mcell(rec[field], (field,), base + ('validation', 'all', metric))
        # Keep every result number, including an unfamiliar future column.
        skip = {'seeds', 'seed', 'seconds', 'row', 'removal', 'arm', 'config', 'inputs', 'timing'}
        numeric_tree = {k: v for k, v in rec.items() if k not in skip}
        for path, value in flatten_numbers(numeric_tree):
            normalized = path
            if kind == 'knob' and path[0] in ('effects', *ASPECTS):
                normalized = ('aspects',) + (path[1:] if path[0] == 'effects' else path)
            if normalized not in consumed:
                put(path, base + ('unmapped', *path), value)
    return out


def sheet_numbers(doc):
    """The only r14 sheet reader. Missing files become missing-sheet entries."""
    ours, theirs, paths = derived_numbers(doc), {}, []
    for kind in ('knob', 'photo', 'ablation'):
        path = P.DEFAULT_OUT / f'stage4/partb_{kind}.json'
        if not path.exists():
            continue
        paths.append(path)
        sheet = read_json(path)
        assert tuple(sheet['seeds']) == ((1, 2, 3) if kind == 'knob' else (0, 1, 2) if kind == 'photo' else tuple(range(5)))
        theirs.update(adapt_sheet(kind, sheet))
    return ours, theirs, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('derive')
    run.add_argument('--rows', nargs='+', help='row names or kind:row; default the five preselected rows')
    run.add_argument('--draws-smoke', type=int, help='few draws; labels output smoke, compare cannot certify it')
    sub.add_parser('compare')
    args = parser.parse_args(argv)
    if args.mode == 'derive':
        derive(args.rows, args.draws_smoke)
        return 0
    return compare()


if __name__ == '__main__':
    raise SystemExit(main())
