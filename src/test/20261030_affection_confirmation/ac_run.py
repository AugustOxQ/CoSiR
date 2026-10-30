"""r12 stage runner (spec 2026-10-30-affection-confirmation-design.md, 3.3 to 3.7, 6; index "Output files").

  python ac_run.py --stage {0a,0b,1,2,3} [--smoke N] [--draws-clu 2000] [--draws-ret 10000] [--workers 8]

Wiring only: the logic is in ac_data (loaders), ac_cluster (partitions, AMI, bootstrap), ac_select (picks), ac_stats
(ranges, retrieval comparison, SC entries, verdict). Every file goes under ac_paths.OUTPUTS:
  stage0a/gates.json                      {"items": {name: {value, reference, ok}}, "ok"} (+ "smoke", "missing")
  stage0b/ranges.json                     copied to RANGES_FILE (smoke: OUTPUTS/records/)
  stage1/scores.json, stage1/picks.json   picks copied to PICKS_FILE (smoke: OUTPUTS/records/)
  stage2/verdict.json, stage3/verdict.json, stage{2,3}/perquery/*.npz, partitions/*.npz

Stage stop rule (spec 3.5): 0a exits nonzero when an item fails or is missing. Stage 0b needs a passing gates file.
The guard: stages 1 to 3 refuse to start unless the record file (ranges; for 2 and 3 also the picks) exists, equals the
outputs copy byte for byte and equals `git show HEAD:<path>` (smoke: the git part is skipped and logged). Every
stage calls `require_records` before its first held (or, for stage 1, any) data load; the records are committed by
the queue, not by this runner.

0a reading: the CPU re-inference 0a(iv) is read from its JSON (`ac_export.py reinfer` writes it), not run here. The
replays of 0a(iii) are slow (25+ minutes): they run only when every other item is present and ok (otherwise they
are recorded as not run and the stage fails on the other items). In smoke the replays, the FI merge (store.json),
0a(ii)(b) and the label counts (they are full-split counts) are recorded as "skipped in smoke", never as passed.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import ac_paths  # puts r11's round and r10's on sys.path
import ac_cluster
import ac_data
import ac_select
import ac_stats
import rt_data
import rt_metrics
import rt_readout

STAGES = ('0a', '0b', '1', '2', '3')
SKIPPED = 'skipped in smoke'
EXPECTED_LABELS = {
    'val': dict(classes=[798, 54, 622, 1387, 189, 620, 203, 270, 108], ties=907),
    'test': dict(classes=[1651, 81, 1460, 2626, 351, 1185, 448, 501, 198], ties=1917),
}
# spec 3.5 0a(iii), r10 part
R10_EMOTION = dict(A_cand=0.10658729378204021, A_T=0.09207439052277983, d=0.014512903259260385,
                   interval=[0.011288090988306501, 0.017671180133450672])
NOTE_STAGE3 = ('SC5 to SC7 were set after r11\'s descriptive held CRL numbers were seen; the report states this '
               'beside any ArtELingo CRL claim (spec section 6).')
AFF_CLUSTER_ARMS = ac_data.ARMS
ASPECTS = tuple(ac_paths.ASPECTS)
SEEDS = tuple(ac_paths.SEEDS)


# ---- files and paths --------------------------------------------------------------------------------------------------

def outputs():
    return Path(ac_paths.OUTPUTS)


def smoke():
    return int(ac_paths.SMOKE)


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def record_copy(kind):
    """The record file of the committed copy: RANGES_FILE or PICKS_FILE (smoke: the same name under OUTPUTS/records)."""
    f = Path(ac_paths.RANGES_FILE if kind == 'ranges' else ac_paths.PICKS_FILE)
    return outputs() / 'records' / f.name if smoke() else f


def record_output(kind):
    return outputs() / ('stage0b/ranges.json' if kind == 'ranges' else 'stage1/picks.json')


# ---- the guard ----------------------------------------------------------------------------------------------------------

def guard_reason(kind):
    """'' when the record passes, else the reason it does not: exists, equals the outputs copy byte for byte, and (not
    in smoke) equals its content in git HEAD."""
    f, out = record_copy(kind), record_output(kind)
    if not f.is_file():
        return f'{f} does not exist'
    if not out.is_file():
        return f'{out} does not exist'
    if f.read_bytes() != out.read_bytes():
        return f'{f} differs from {out}'
    if smoke():
        print(f'guard: smoke run, git check of {f} skipped', flush=True)
        return ''
    r = subprocess.run(['git', '-C', str(f.parent), 'show', f'HEAD:./{f.name}'], capture_output=True)
    if r.returncode:
        return f'{f} is not in git HEAD: commit it first'
    if r.stdout != f.read_bytes():
        return f'{f} differs from its content in git HEAD: commit it first'
    return ''


def require_records(need_picks):
    """The guard. Stops the stage (SystemExit, nonzero) before any data is loaded when a record fails."""
    for kind in ('ranges', 'picks') if need_picks else ('ranges',):
        why = guard_reason(kind)
        if why:
            raise SystemExit(f'REFUSED: {kind} record: {why}')


# ---- stage 0a -----------------------------------------------------------------------------------------------------------

def _file_item(path, reference, summary=None):
    """An item from a GPU (or CPU) item's JSON: missing -> not ok."""
    path = Path(path)
    if not path.is_file():
        return dict(value='missing', reference=reference, ok=False, missing=True)
    doc = read_json(path)
    value = summary(doc) if summary else doc.get('ok')
    return dict(value=value, reference=reference, ok=bool(doc.get('ok')))


def _cos(doc):
    return dict(min_cosine=doc.get('min_cosine'), max_abs_diff=doc.get('max_abs_diff'), n=doc.get('n'))


def _max_item_value(doc):
    return max(i['value'] for i in doc['items'].values() if isinstance(i['value'], (int, float)))


def _item_oks(doc):
    return {k: bool(v['ok']) for k, v in doc['items'].items()}


def label_items():
    out = {}
    for sp in ('val', 'test'):
        c = ac_data.label_counts(sp)
        want = EXPECTED_LABELS[sp]
        out[f'label_counts_{sp}'] = dict(value=c['classes'], reference=want['classes'], ok=c['classes'] == want['classes'])
        out[f'label_ties_{sp}'] = dict(value=c['ties'], reference=want['ties'], ok=c['ties'] == want['ties'])
    return out


def file_items():
    o = outputs() / 'stage0a'
    s = bool(smoke())
    items = {
        '0a(i)_store': dict(value=SKIPPED, reference=None, ok=None) if s else
        _file_item(o / 'store.json', 'old rows unchanged, FI rows pass the gate', _item_oks),
        '0a(ii)a_extractor_artelingo': _file_item(o / 'extractor_artelingo.json', 'min cosine >= 0.999', _cos),
        '0a(ii)b_extractor_default_affection': dict(value=SKIPPED, reference=None, ok=None) if s else
        _file_item(o / 'extractor_default_affection.json', 'min cosine >= 0.999', _cos),
        '0a(iv)_cstar_reinfer': _file_item(o / 'cstar_reinfer.json', 'max abs diff <= 1e-3', _max_item_value),
        '0a(v)_percept_loader': _file_item(o / 'percept_loader.json', 'seed-0 hash and AMI 0.08742922158584242',
                                           lambda d: dict(seed0_hash=d['seeds']['0']['ok'] if '0' in d['seeds'] else d['seeds'][0]['ok'],
                                                          ami=d['ami']['value'])),
        '0a(vi)_clip_l14_artelingo': _file_item(o / 'clip_l14_artelingo.json', 'min cosine >= 0.999', _cos),
        '0a(vii)_crl': _file_item(o / 'crl.json', 'vector hash and cosine >= 0.999', _item_oks),
    }
    return items


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def replay_r10(workers, draws_clu):
    got = ac_cluster.r10_emotion_replay(workers=workers, draws=draws_clu)
    diff = max(_maxdiff(got[k], R10_EMOTION[k]) for k in R10_EMOTION)
    return dict(value=dict(got, max_abs_diff=diff), reference=R10_EMOTION, ok=bool(diff < 1e-9))


def replay_r11_stage4(workers):
    saved_scores = read_json(ac_paths.R11_OUT / 'stage4/scores.json')
    saved_picks = read_json(ac_paths.R11_OUT / 'stage4/picks.json')
    scores, picks = ac_select.replay_r11_stage4(workers=workers)
    diff, same = 0.0, True
    for arm, table in scores.items():
        saved = saved_scores[arm]
        same &= list(table) == list(saved) and picks[arm] == saved_picks[arm]
        for name, row in saved.items():
            same &= table[name]['kept'] == row['kept']
            diff = max(diff, abs(table[name]['score'] - row['score']),
                       *[abs(table[name]['map'][a] - row['map'][a]) for a in ASPECTS])
    return dict(value=dict(max_abs_diff=diff, picks=picks), reference=saved_picks, ok=bool(same and diff < 1e-9))


def replay_r11_stage5(workers):
    saved = read_json(ac_paths.R11_OUT / 'stage5/verdict.json')
    got = ac_stats.replay_r11_stage5(workers=workers, draws=saved['draws'])
    diff, same = 0.0, list(got['map']) == saved['arms'] and list(got['sc']) == list(saved['sc'])
    for arm, by in saved['map'].items():
        for a, cell in by.items():
            diff = max(diff, abs(got['map'][arm][a]['mean'] - cell['mean']))
            if cell['per_seed'] is not None:
                diff = max(diff, _maxdiff(got['map'][arm][a]['per_seed'], cell['per_seed']))
    for sid, doc in saved['sc'].items():
        same &= got['sc'][sid]['pass'] == doc['pass']
        for g, s in zip(got['sc'][sid]['comparisons'], doc['comparisons']):
            same &= (g['outcome'], g['candidate'], g['comparator'], g['aspect']) == \
                    (s['outcome'], s['candidate'], s['comparator'], s['aspect'])
            diff = max(diff, abs(g['d'] - s['d']), abs(g['r'] - s['r']), _maxdiff(g['interval'], s['interval']))
    return dict(value=dict(max_abs_diff=diff), reference='r11 stage5/verdict.json within 1e-9', ok=bool(same and diff < 1e-9))


def stage0a(args):
    items = {} if smoke() else label_items()
    if smoke():
        for n in ('label_counts_val', 'label_ties_val', 'label_counts_test', 'label_ties_test'):
            items[n] = dict(value=SKIPPED, reference=None, ok=None)
    items.update(file_items())
    ran = {k: v for k, v in items.items() if v['ok'] is not None}
    missing = sorted(k for k, v in items.items() if v.get('missing'))
    if smoke():
        for n in ('0a(iii)_r10_stage4_emotion', '0a(iii)_r11_stage4', '0a(iii)_r11_stage5'):
            items[n] = dict(value=SKIPPED, reference=None, ok=None)
    elif all(v['ok'] for v in ran.values()):
        for n, fn in (('0a(iii)_r10_stage4_emotion', lambda: replay_r10(args.workers, args.draws_clu)),
                      ('0a(iii)_r11_stage4', lambda: replay_r11_stage4(args.workers)),
                      ('0a(iii)_r11_stage5', lambda: replay_r11_stage5(args.workers))):
            t = time.perf_counter()
            items[n] = fn()
            print(f'  {n}: ok={items[n]["ok"]} {time.perf_counter() - t:.0f}s', flush=True)
    else:
        for n in ('0a(iii)_r10_stage4_emotion', '0a(iii)_r11_stage4', '0a(iii)_r11_stage5'):
            items[n] = dict(value='not run: another item is missing or failed', reference=None, ok=False)
    ok = all(v['ok'] for v in items.values() if v['ok'] is not None)
    doc = dict(items=items, ok=bool(ok), smoke=smoke(), missing=missing)
    write_json(outputs() / 'stage0a/gates.json', doc)
    bad = [k for k, v in items.items() if v['ok'] is False]
    if not ok:
        print(f'stage 0a FAILED. missing: {missing}; failed or not run: {bad}', file=sys.stderr, flush=True)
    return doc


# ---- stage 0b -----------------------------------------------------------------------------------------------------------

def stage0b(args):
    g = outputs() / 'stage0a/gates.json'
    if not g.is_file() or read_json(g).get('ok') is not True or read_json(g).get('smoke') != smoke():
        raise SystemExit(f'REFUSED: {g} is missing, failing or from another smoke setting: stage 0a must pass first')
    doc = ac_stats.validation_seed_ranges(workers=args.workers)
    ac_stats.write_ranges(doc, outputs(), copy_to=record_copy('ranges'))
    print('  ranges: ' + ', '.join(f'{k} r={v["r"]:.6g}' for k, v in doc.items()), flush=True)
    return doc


# ---- shared scoring pieces ------------------------------------------------------------------------------------------------

def seeds_of(arm):
    return SEEDS if arm in ac_data.SEEDED else None


def retrieval(get_val, get_eval, readout, labels, seeds, workers):
    """ac_stats.arm_retrieval; a full run must have no query without a relevant item (as r11's score_arm)."""
    res = ac_stats.arm_retrieval(get_val, get_eval, readout, labels, seeds=seeds, workers=workers)
    if not smoke():
        assert res['dropped'] == 0, (readout, 'queries without a relevant item', res['dropped'])
    return res


def save_perquery(stage, dataset, arm, aspect, res, **extra):
    """perquery/<dataset>__<arm>__<aspect>__s<seed>.npz, one file per seed (s- for an unseeded arm): ap, rows."""
    d = outputs() / f'stage{stage}/perquery'
    d.mkdir(parents=True, exist_ok=True)
    ap = res['ap']
    seeded = ap.ndim == 2
    for i, seed in enumerate(SEEDS if seeded else (None,)):
        np.savez(d / f'{dataset}__{arm}__{aspect}__s{"-" if seed is None else seed}.npz',
                 ap=ap[i] if seeded else ap, rows=res['rows'],
                 **{k: (v[i] if seeded else v) for k, v in extra.items()})


def mapped(res):
    return dict(map=res['point'], per_seed=res['per_seed'])


def load_picks():
    return read_json(record_copy('picks'))


def load_ranges():
    return ac_stats.load_ranges(record_copy('ranges'))


def sc_dump(sc_doc):
    return json.loads(json.dumps(sc_doc, allow_nan=False))


# ---- stage 1 ------------------------------------------------------------------------------------------------------------

def stage1(args):
    require_records(need_picks=False)               # the guard comes before any data load
    ranges = load_ranges()
    r_clu, r_ret = ranges['clu']['r'], ranges['map']['r']
    labels = ac_data.labels('val')
    val_keys = ac_data.keys('val')
    aff_tables, aff_picks = {}, {}
    for arm in ac_select.AFFECTION_ARMS:
        t = time.perf_counter()
        xs = {('emotion', s): ac_data.embed(arm, 'val', s) for s in (SEEDS if arm in ac_data.SEEDED else (None,))}
        aff_tables[arm], aff_picks[arm] = ac_select.arm_table(
            xs, {'emotion': labels}, {'emotion': r_ret}, workers=args.workers, keys=val_keys)
        print(f'  stage1 affection {arm}: pick {aff_picks[arm]}, {time.perf_counter() - t:.0f}s', flush=True)
    ami = {w: ac_cluster.ami(ac_cluster.partitions(ac_data.embed(f'teacher_{w}', 'val')), labels)
           for w in ('artwork', 'photo')}
    r_a = ac_stats.r11_r_a()
    art_labels = {a: ac_data.smoke_rows(x) for a, x in rt_data.labels('val').items()}
    clip_xs = {(a, None): ac_data.smoke_rows(rt_data.embed('crl', 'val', a)) for a in ASPECTS}
    qwen_xs = {(a, None): ac_data.smoke_rows(ac_data.artelingo_crl_qwen('val', a)) for a in ASPECTS}
    art_tables, art_picks = {}, {}
    art_tables['crl_clip_painting'], art_picks['crl_clip_painting'] = ac_select.arm_table(
        clip_xs, art_labels, r_a, fixed=ac_select.readout_by_name(ac_select.ARTELINGO_FIXED['crl_clip_painting'], 50),
        workers=args.workers)
    art_tables['crl_qwen_painting'], art_picks['crl_qwen_painting'] = ac_select.arm_table(
        qwen_xs, art_labels, r_a, workers=args.workers)
    picks, scores = ac_select.build_stage1(aff_tables, aff_picks, ami, r_clu, r_ret, art_tables, art_picks, r_a)
    ac_select.write_picks(picks, scores, outputs(), copy_to=record_copy('picks'))
    return picks


# ---- stage 2 ------------------------------------------------------------------------------------------------------------

def _save_parts(name, parts):
    d = outputs() / 'partitions'
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f'{name}.npz', labels=np.asarray(parts, dtype=np.int32))


def aff_partitions(arm, labels, workers):
    """Partitions of an Affection test arm (saved): int32 [5, 3, n] (seeded) or [3, n]; PercepT also its own reading."""
    out = {}
    if arm == 'percept':
        probs = [ac_data.percept_prob('test', s) for s in SEEDS]
        both = [ac_cluster.percept_partitions(p) for p in probs]
        for s, b in zip(SEEDS, both):
            _save_parts(f'affection__test__percept__s{s}', b['r20'])
            _save_parts(f'affection__test__percept_own__s{s}', b['own'])
        out['r20'] = np.stack([b['r20'] for b in both])
        out['own'] = np.stack([b['own'] for b in both])
    elif arm == 'cstar':
        out['r20'] = np.stack([ac_cluster.cached_partitions(ac_data.embed(arm, 'test', s),
                                                            f'affection__test__{arm}__s{s}') for s in SEEDS])
    else:
        out['r20'] = ac_cluster.cached_partitions(ac_data.embed(arm, 'test'), f'affection__test__{arm}__s-')
    return out


def affection_retrieval(arm, ro_name, labels, workers):
    """Picked read-out and R0 for one Affection arm on test: arm_retrieval results (picked, R0) and R0's nDCG mean."""
    seeds = seeds_of(arm)
    get_val = lambda s: ac_data.embed(arm, 'val', s)
    get_eval = lambda s: ac_data.embed(arm, 'test', s)
    picked = retrieval(get_val, get_eval, ro_name, labels, seeds, workers)
    r0 = retrieval(get_val, get_eval, 'none+k0', labels, seeds, workers)
    return picked, r0


def stage2(args):
    require_records(need_picks=True)                # the guard comes before any held load
    ranges, picks = load_ranges(), load_picks()
    aff = picks['affection']
    labels = ac_data.labels('test')
    dist = ac_data.emotion_dist('test')
    parts, rows, ret = {}, {}, {}
    for arm in AFF_CLUSTER_ARMS:
        t = time.perf_counter()
        parts[arm] = aff_partitions(arm, labels, args.workers)
        picked, r0 = affection_retrieval(arm, aff['readout'][arm], labels, args.workers)
        ret[arm] = picked
        save_perquery(2, 'affection', arm, 'emotion', picked)
        save_perquery(2, 'affection', f'{arm}_R0', 'emotion', r0)
        nd = [float(rt_metrics.graded(ac_data.embed(arm, 'test', s), dist, workers=args.workers).mean())
              for s in (SEEDS if arm in ac_data.SEEDED else (None,))]
        rows[arm] = dict(ami=ac_cluster.ami(parts[arm]['r20'], labels), readout=aff['readout'][arm],
                         map_picked=mapped(picked), map_r0=mapped(r0), ndcg_r0=float(np.mean(nd)))
        if arm == 'percept':
            rows[arm]['ami_own'] = ac_cluster.ami(parts[arm]['own'], labels)
        print(f'  stage2 {arm}: {time.perf_counter() - t:.0f}s', flush=True)
    sc = {}
    for sid in ('SC1', 'SC3'):
        comp = ac_stats.comparator_arm(sid, picks)
        res = ac_cluster.paired(parts['cstar']['r20'], parts[comp]['r20'], labels, args.draws_clu, args.workers)
        sc[sid] = ac_stats.cluster_sc(sid, res, picks, ranges)
    for sid in ('SC2', 'SC4'):
        comp = ac_stats.comparator_arm(sid, picks)
        sc[sid] = ac_stats.retrieval_sc(sid, ret['cstar'], ret[comp], picks, ranges, None, args.draws_ret)
    sc = {sid: sc[sid] for sid in ac_stats.STAGE2_IDS}
    doc = ac_stats.stage2_doc(sc_dump(sc), rows)
    doc.update(smoke=smoke(), draws_clu=args.draws_clu, draws_ret=args.draws_ret)
    write_json(outputs() / 'stage2/verdict.json', doc)
    return doc


# ---- stage 3 ------------------------------------------------------------------------------------------------------------

def art_getter(arm, aspect, split):
    if arm == 'cstar':
        return lambda s: ac_data.smoke_rows(rt_data.embed('cstar', split, aspect, s))
    if arm == 'crl_clip_painting':
        return lambda s: ac_data.smoke_rows(rt_data.embed('crl', split, aspect))
    assert arm == 'crl_qwen_painting', arm
    return lambda s: ac_data.smoke_rows(ac_data.artelingo_crl_qwen(split, aspect))


def stage3(args):
    require_records(need_picks=True)                # the guard comes before any held load
    ranges, picks = load_ranges(), load_picks()
    r_a = ac_stats.r11_r_a()
    art, aff = picks['artelingo'], picks['affection']
    held_labels = {a: ac_data.smoke_rows(x) for a, x in rt_data.labels('held').items()}
    res, rows, sc = {}, {'artelingo': {}, 'affection': {}}, {}
    for arm in ('cstar', 'crl_clip_painting', 'crl_qwen_painting'):
        rows['artelingo'][arm] = {}
        for a in ASPECTS:
            t = time.perf_counter()
            r = retrieval(art_getter(arm, a, 'val'), art_getter(arm, a, 'held'), art['readout'][arm],
                          held_labels[a], seeds_of(arm) if arm == 'cstar' else None, args.workers)
            res[('artelingo', arm, a)] = r
            save_perquery(3, 'artelingo', arm, a, r)
            rows['artelingo'][arm][a] = dict(readout=art['readout'][arm], **mapped(r))
            print(f'  stage3 artelingo {arm} {a}: {time.perf_counter() - t:.0f}s', flush=True)
    labels = ac_data.labels('test')
    for arm in ('cstar',) + tuple(f'crl_{v}' for v in ac_paths.CRL_VERSIONS):
        t = time.perf_counter()
        picked, r0 = affection_retrieval(arm, aff['readout'][arm], labels, args.workers)
        res[('affection', arm, 'emotion')] = picked
        save_perquery(3, 'affection', arm, 'emotion', picked)
        rows['affection'][arm] = dict(readout=aff['readout'][arm], map_picked=mapped(picked), map_r0=mapped(r0))
        print(f'  stage3 affection {arm}: {time.perf_counter() - t:.0f}s', flush=True)
    for sid in ac_stats.STAGE3_IDS:
        s = ac_stats.SC[sid]
        comp = ac_stats.comparator_arm(sid, picks)
        sc[sid] = ac_stats.retrieval_sc(sid, res[(s.dataset, 'cstar', s.aspect)], res[(s.dataset, comp, s.aspect)],
                                        picks, ranges, r_a, args.draws_ret)
    doc = ac_stats.stage3_doc(sc_dump(sc), rows)
    doc.update(note=NOTE_STAGE3, smoke=smoke(), draws_ret=args.draws_ret)
    write_json(outputs() / 'stage3/verdict.json', doc)
    return doc


# ---- command line ---------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--smoke', type=int)
    ap.add_argument('--draws-clu', type=int, default=2000)
    ap.add_argument('--draws-ret', type=int, default=10000)
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = ap.parse_args(argv)
    assert ac_paths.SMOKE == (args.smoke or 0), (
        f'--smoke {args.smoke} and AC_SMOKE={ac_paths.SMOKE} must agree (the queue sets both)')
    t = time.perf_counter()
    print(f'stage {args.stage} (smoke {smoke()}) ...', flush=True)
    out = globals()[f'stage{args.stage}'](args)
    print(f'stage {args.stage}: done in {time.perf_counter() - t:.1f}s', flush=True)
    if args.stage == '0a':
        return 0 if out['ok'] else 1
    if args.stage in ('2', '3'):
        print('  ' + ' '.join(f'{k}={v["outcome"]}' for k, v in out['sc'].items())
              + (f' verdict={out["verdict"]} repeats={out["repeats_artelingo_win"]}' if args.stage == '2' else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
