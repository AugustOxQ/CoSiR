"""r26 scoring: PercepT (CLIP L/14 + ModernBERT-GoEmotions, paper settings) on the 12,281 held (test) paintings.

    python po_score.py --b4-check [--record]   scorer check on arm B4's saved held outputs (needs no new store)
    python po_score.py [--seeds ...] [--cuts ...] [--record]   score OUT/held/ for both cuts

Readings, all on r17's scorer (`ns_metrics.percept_partitions`, `ns_metrics.ami`, `ns_data.labels`):
  rclass  k-means (MiniBatchKMeans, n_init 3, batch 4,096, states 0-2) on the L2-normalised probabilities, k = the
          aspect's class count (emotion 9, style 27, genre 10; read from r17's sweep record), fitted per aspect;
          fills the paper's Table 2 row.
  r20     the same with k = 20 for every aspect.
  own     argmax of the probabilities (lowest index on a tie); k-free.
AMI per aspect on the labelled rows (label -1 masked; genre 9,949 labelled). A (cut, reading, aspect) cell is the mean
over the uncollapsed seeds (and the three states). Collapse: `ct_rcol.judge` on the Stage 1 artefacts (k40: one topic
holds over half the fit pairs; strict: also kept count under 10 or over 90); a seed whose held probabilities are
non-finite counts as collapsed too. Collapsed seeds are left out of the mean and counted; their own values are kept.

This module does not import the claim test's training code (r17's checked imports need the main checkout's helpers
first), only `ct_rcol` (numpy) for the collapse judgement.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import po_paths as P

if str(P.NS_ROUND) not in sys.path:
    sys.path.insert(0, str(P.NS_ROUND))
import ns_paths                      # noqa: E402
ns_paths.add_rounds()
import ns_data                       # noqa: E402
import ns_metrics                    # noqa: E402

for _folder in (P.CT_ROUND, P.PC_ROUND):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))
import ct_rcol                       # noqa: E402

RECORDS = P.ROUND / 'records'
CLASS_COUNT_FILE = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json')
EXPECTED_CLASS_COUNT = {'emotion': 9, 'genre': 10, 'style': 27}
READINGS = ('rclass', 'r20', 'own')
B4_HELD = P.MAIN / 'src/test/20261020_claim_test/outputs/stage2/full/held'      # frozen, gitignored (main checkout)
B4_RCOL = P.MAIN / 'src/test/20261020_claim_test/outputs/stage2/full/percept/rcol.json'


def class_counts(path=None):
    """Per-aspect class counts from r17's sweep record; asserted equal to the user's values (9 / 27 / 10)."""
    got = json.loads(Path(path or CLASS_COUNT_FILE).read_text())['class_count']
    assert got == EXPECTED_CLASS_COUNT, f'class counts changed: {got}'
    return {a: int(got[a]) for a in P.ASPECTS}


def sha_bytes(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def masked_ami(y, c):
    keep = y >= 0
    return float(ns_metrics.ami(y[keep], np.asarray(c)[keep]))


def score_probs(prob, labels, counts):
    """{reading: {aspect: [AMI per state]}} for one probability matrix [n, K] in r17's held order."""
    prob = np.asarray(prob, dtype=np.float32)
    out = {'rclass': {}, 'r20': {}, 'own': {}}
    parts20 = ns_metrics.percept_partitions(prob, P.K_SCORE, 'artelingo')
    own = prob.argmax(axis=1)                                   # lowest index on a tie
    for aspect in P.ASPECTS:
        y = labels[aspect]
        parts_c = ns_metrics.percept_partitions(prob, counts[aspect], 'artelingo')
        out['rclass'][aspect] = [masked_ami(y, p) for p in parts_c]
        out['r20'][aspect] = [masked_ami(y, p) for p in parts20]
        out['own'][aspect] = [masked_ami(y, own)]
    return out


def seed_values(values):
    """Mean over states, per reading and aspect (full precision) plus the per-state list."""
    return {r: {a: dict(per_state=v, mean=float(np.mean(v))) for a, v in by.items()} for r, by in values.items()}


def load_labels():
    labels = {a: ns_data.labels('artelingo', 'held', a) for a in P.ASPECTS}
    assert int((labels['genre'] >= 0).sum()) == 9949, 'expected 9,949 labelled genre rows'
    return labels


def aggregate(seed_results):
    """Cell = mean of the seed means over uncollapsed seeds that have values; None when there are none."""
    table = {r: {a: None for a in P.ASPECTS} for r in READINGS}
    used = [s for s, r in seed_results.items() if not r['collapsed'] and r.get('values')]
    for r in READINGS:
        for a in P.ASPECTS:
            if used:
                table[r][a] = float(np.mean([seed_results[s]['values'][r][a]['mean'] for s in used]))
    return table, sorted(used)


# ======================================================================================== B4 scorer check
def b4_check(labels, counts, names):
    """Score arm B4's held outputs (paper K = 40, CLIP B/32 + RoBERTa, P1) with the functions above."""
    held = json.loads((B4_HELD / 'held_paintings.json').read_text())
    assert held['sha256'] == names_sha(held['names']) and len(held['names']) == P.N_TEST
    assert set(held['names']) == set(names)
    lookup = {n: i for i, n in enumerate(held['names'])}
    rows = np.asarray([lookup[n] for n in names], dtype=np.int64)        # join to r17's order by painting name
    rcol = json.loads(B4_RCOL.read_text())['entries']['B4']
    seeds = {}
    for seed in P.SEEDS:
        path = B4_HELD / 'percept' / 'paper__k40' / f'P1__s{seed}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['held_sha256'] == held['sha256'] and meta['config']['arm'] == 'P1' and meta['config']['cut'] == 'k40'
        with np.load(path, allow_pickle=False) as z:
            prob = z['prob']
        assert prob.shape == (P.N_TEST, 40) and prob.dtype == np.float32
        assert meta['array_sha256s']['prob'] == sha_bytes(prob), f'{path}: array hash'
        assert np.isfinite(prob).all()
        prob = prob[rows]
        seeds[seed] = dict(collapsed=seed not in rcol['seeds'], values=seed_values(score_probs(prob, labels, counts)),
                           held_array_sha256=meta['array_sha256s']['prob'])
        print(f'B4 seed {seed}: r20 emotion {seeds[seed]["values"]["r20"]["emotion"]["mean"]:.4f}', flush=True)
    assert rcol['seeds'] == [0, 1, 2, 3, 4], 'B4 R-COL seed list changed'
    table, used = aggregate(seeds)
    result = dict(what='arm B4 (paper K = 40, CLIP B/32 + RoBERTa-GoEmotions, arm P1) held outputs, scored with po_score',
                  reference='docs/reports/auto/percept/2026-12-19_claim_test_stage2.md, B4-r20 = 0.0699 (emotion)',
                  source=str(B4_HELD), eligible_seeds=used, class_count=counts, table=table, seeds=seeds,
                  names_sha256=names_sha(names), reproduces=bool(round(table['r20']['emotion'], 3) == round(P.B4_R20_EMOTION, 3)))
    assert result['reproduces'], f"B4 r20 emotion {table['r20']['emotion']:.6f} != {P.B4_R20_EMOTION}"
    return result


# ============================================================================================== real scoring
def stage1_collapse(root, seed, cut):
    """Collapse judgement for one (cut, seed) from the Stage 1 artefact, with the cut masks checked against the rule."""
    path = Path(root) / 'percept' / 'paper' / 'stage1' / f'{P.VERSION}__s{seed}.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as z:
        norms, counts = z['center_norms'], z['fit_counts']
        kept, topics = z[f'kept_{cut}'], z[f'fit_topics_{cut}']
    expected = ct_rcol.strict_mask(norms) if cut == 'strict' else ct_rcol.k40_mask(counts)
    assert np.array_equal(kept, expected), f'{path}: {cut} mask differs from the rule'
    assert meta['config']['version'] == P.VERSION and meta['config']['seed'] == seed
    return dict(ct_rcol.judge(cut, kept, topics), smoke=meta['config']['smoke'], stop_epoch=meta['stop_epoch'])


def score_cut_seed(root, cut, seed, names, labels, counts):
    judged = stage1_collapse(root, seed, cut)
    res = dict(judged, collapsed_rule=judged['collapsed'], nonfinite=False, dev_auc=None, K=judged['kept_count'],
               held_array_sha256=None, values=None, error=None)
    s2 = Path(root) / 'percept' / 'paper' / 'stage2' / cut / f'{P.ARM}__s{seed}.json'
    held = Path(root) / 'held' / f'{P.ARM}__{cut}__s{seed}.npz'
    if not held.exists():
        res.update(collapsed=True, error='no held output (Stage 2 skipped: fewer than 2 topics)'
                   if judged['kept_count'] < 2 else 'held output missing')
        if judged['kept_count'] >= 2:
            raise FileNotFoundError(held)
        return res
    meta = json.loads(held.with_suffix('.json').read_text())
    assert meta['held_sha256'] == names_sha(names) and meta['rows'] == len(names), f'{held}: held list'
    assert (meta['config']['cut'], meta['config']['seed'], meta['config']['arm']) == (cut, seed, P.ARM)
    assert meta['config']['smoke'] == judged['smoke']
    with np.load(held, allow_pickle=False) as z:
        prob = z['prob']
    assert prob.dtype == np.float32 and prob.shape == (len(names), meta['K']) and meta['K'] == judged['kept_count']
    assert meta['array_sha256s']['prob'] == sha_bytes(prob), f'{held}: array hash'
    res.update(held_array_sha256=meta['array_sha256s']['prob'], dev_auc=json.loads(s2.read_text()).get('dev_auc'))
    if not np.isfinite(prob).all():
        res.update(nonfinite=True, collapsed=True)
        return res
    try:
        res['values'] = seed_values(score_probs(prob, labels, counts))
    except AssertionError as error:
        if not judged['collapsed']:
            raise
        res['error'] = f'scorer refused a collapsed seed: {error}'
    return res


def score_all(root, seeds, cuts, names, labels, counts):
    out = {}
    for cut in cuts:
        seed_results = {}
        for seed in seeds:
            t = time.perf_counter()
            seed_results[seed] = score_cut_seed(root, cut, seed, names, labels, counts)
            r = seed_results[seed]
            e = r['values']['rclass']['emotion']['mean'] if r['values'] else float('nan')
            print(f'score {cut} seed={seed}: collapsed={r["collapsed"]} K={r["K"]} rclass emotion={e:.4f} '
                  f'{time.perf_counter() - t:.1f} s', flush=True)
        table, used = aggregate(seed_results)
        aucs = [seed_results[s]['dev_auc'] for s in used if seed_results[s]['dev_auc'] is not None]
        out[cut] = dict(seeds=seed_results, table=table, seeds_in_mean=used,
                        collapsed_count=sum(r['collapsed'] for r in seed_results.values()),
                        collapsed_seeds=sorted(s for s, r in seed_results.items() if r['collapsed']),
                        kept_counts={s: r['kept_count'] for s, r in seed_results.items()},
                        dev_auc_mean=float(np.mean(aucs)) if aucs else None)
    return out


def markdown(result):
    """po_table.md: class-count cells first (they fill the paper row), then k = 20 and own; three decimals + full."""
    f3 = lambda v: 'n/a' if v is None else f'{v:.3f}'
    ff = lambda v: 'n/a' if v is None else repr(float(v))
    lines = ['# r26 PercepT (CLIP L/14 + ModernBERT-GoEmotions, paper settings): held AMI', '',
             f'Held (test) paintings: {P.N_TEST} (genre labelled: 9,949). Arm {P.ARM}: Stage 2 attention pooling on CLIP L/14 '
             f'patch tokens. Cell = mean over uncollapsed seeds (and 3 k-means states). Smoke: {result["meta"]["smoke"]}.',
             f'Class counts for rclass: {result["meta"]["class_count"]}.', '']
    for cut in result['cuts']:
        c = result['cuts'][cut]
        lines += [f'## Cut {cut}: {len(c["seeds_in_mean"])} of {len(c["seeds"])} seeds in the mean '
                  f'(collapsed: {c["collapsed_count"]}, seeds {c["collapsed_seeds"]})', '',
                  '| reading | emotion | style | genre |', '|---|---:|---:|---:|']
        for r, label in (('rclass', 'rclass (k = class count: 9 / 27 / 10), paper row'), ('r20', 'r20 (k = 20)'),
                         ('own', 'own (argmax)')):
            lines.append(f'| {label} | ' + ' | '.join(f3(c['table'][r][a]) for a in P.ASPECTS) + ' |')
        lines += ['', 'Full precision:', '']
        for r in READINGS:
            lines.append(f'- {r}: ' + ', '.join(f'{a} {ff(c["table"][r][a])}' for a in P.ASPECTS))
        lines += ['', f'Kept topic counts per seed: {c["kept_counts"]}; mean dev AUC (uncollapsed): {ff(c["dev_auc_mean"])}', '',
                  '| seed | collapsed | share (top topic) | K | dev AUC | rclass emo | rclass sty | rclass gen | r20 emo | own emo |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for seed, r in c['seeds'].items():
            v = r['values']
            cells = ['n/a'] * 5 if not v else [f3(v['rclass'][a]['mean']) for a in P.ASPECTS] + \
                [f3(v['r20']['emotion']['mean']), f3(v['own']['emotion']['mean'])]
            lines.append(f'| {seed} | {r["collapsed"]} | {r["share"]:.3f} | {r["K"]} | {f3(r["dev_auc"])} | ' +
                         ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines) + '\n'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b4-check', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int, choices=P.SEEDS, default=list(P.SEEDS))
    ap.add_argument('--cuts', nargs='+', choices=P.CUTS, default=list(P.CUTS))
    ap.add_argument('--record', action='store_true', help='also write the committed copies under records/')
    ap.add_argument('--smoke', action='store_true', help='score smoke outputs; never writes records/')
    a = ap.parse_args(argv)
    if a.smoke and a.record:
        ap.error('--smoke never writes records/')
    start = time.perf_counter()
    counts = class_counts()
    names = ns_data.keys('artelingo', 'held')
    assert len(names) == len(set(names)) == P.N_TEST
    labels = load_labels()
    if a.b4_check:
        result = b4_check(labels, counts, names)
        t = result['table']
        print(json.dumps({r: {k: round(v, 4) for k, v in t[r].items()} for r in READINGS}), flush=True)
        if a.record:
            write_json(RECORDS / 'b4_scorer_check.json', result)
        return 0
    result = dict(meta=dict(smoke=a.smoke, class_count=counts, readings=list(READINGS), names_sha256=names_sha(names),
                            aspects=list(P.ASPECTS), outputs=str(P.OUT), version=P.VERSION, arm=P.ARM,
                            scorer='r17 ns_metrics.percept_partitions / ami, ns_data.labels',
                            r20='k = 20 for all aspects', rclass='k = class count per aspect', own='argmax, lowest index'),
                  cuts=score_all(P.OUT, a.seeds, a.cuts, names, labels, counts))
    write_json(P.OUT / 'score' / 'po_scores.json', result)
    (P.OUT / 'score').mkdir(parents=True, exist_ok=True)
    (P.OUT / 'score' / 'po_table.md').write_text(markdown(result))
    if a.record:
        assert not any(c['seeds'][s]['smoke'] for c in result['cuts'].values() for s in c['seeds']), 'smoke outputs'
        write_json(RECORDS / 'po_scores.json', result)
        (RECORDS / 'po_table.md').write_text(markdown(result))
    print(f'scored in {time.perf_counter() - start:.1f} s -> {P.OUT / "score" / "po_scores.json"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
