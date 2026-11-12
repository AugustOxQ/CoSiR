"""r28 job F, part 1: the settings check on SigLIP 1 (handoff section 4, item 9). Validation only (6,152 paintings).

Rows `k11=10`, `k11=20`, `k11=80`, `k6=0.0005` against c* (`k11=40`, r14's run, not retrained), seed 0.
Exports: `<root>/train/<arm>/R6-5/<aspect>__<graph>__s0.npz|json` (validation export, 6,152 paintings): c* under r14's
root, a knob row under r28's root `/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1`.

Which criterion applies (decision line, 2026-11-12): r10's stage 3 pick criterion cannot be built for these runs, so the
handoff's fallback applies.
  - P (the 29,232 development paintings of `rb_data.p_names()`) contains the 6,152 validation paintings and 23,080
    paintings that the full-scale runs trained on (the full fit set is the 19,889 sample paintings plus P minus the
    validation paintings). No export on P exists for a full-scale SigLIP 1 run, and one would not be held out.
  - r_a (0.0046, 0.0082, 0.0128) are seed ranges of Qwen sample-scale runs at k = 20 on P; no SigLIP 1 value exists.
  - The criterion's draw runs over seeds {1, 2, 3} and P; with the seed set {0} the seed draw is empty, and the
    partitions ("the three states" of k = 20) exist only as the class-count and k = 20 partitions of the validation rows.
Fallback, as the handoff names it: the mean of the six validation cells (AMI at the class counts 9 / 27 / 10 and
plain-cosine mAP, `r28_valpick.py`'s cells, here for seed 0), c* winning ties. A row "wins" when its six-cell mean is
above c*'s by more than 1e-12; otherwise "c* wins or ties".
Context, deciding nothing: c*'s five-seed range of the same six-cell mean (from r14's other seeds, `records/valpick.json`)
and, with `--boot N`, a paired painting bootstrap of the six-cell-mean difference (one `default_rng(0)`, I_b over
the 6,152 rows shared by both arms and the six cells, unlabelled rows dropped, fixed partitions and per-query AP).

Cells are scored as `r28_valpick.py` scores them: stored float16 rows cast to float32 (`ns_data.cstar(unit=False)`), `ns_metrics.partitions`
(MiniBatchKMeans, n_init 3, batch 4,096, states 0 to 2, 8 threads), AMI mean over the 3 states; mAP from unit rows,
`rt_metrics.retrieval(qe_k=0)`, labelled rows. Gate: c*'s seed-0 cells equal `valpick_siglip1.json`'s seed-0 values.

    python r28_settings.py run [--rows ...] [--out DIR] [--records DIR] [--boot N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261103_new_scores'))
sys.path.insert(0, str(HERE))
import r28_valpick as V  # noqa: E402  (cells, class counts, join_check, cell names)

ASPECTS = V.ASPECTS
CELLS = V.CELLS
CLASS_COUNT = V.CLASS_COUNT
REF_ARM = 'k11=40'
ROWS = ('k11=10', 'k11=20', 'k11=80', 'k6=0.0005')
SEED = 0
TIE = 1e-12
TOL = 1e-9
R14_ROOT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1')
R28_ROOT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1')
OUT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/settings')
RECORDS = HERE / 'records'
VALPICK_SIGLIP1 = Path('/data/SSD2/cosir_r28/20261112_siglip_main/valpick/valpick_siglip1.json')


# ------------------------------------------------------------------------------------------------ pure parts
def six_mean(cells):
    """Mean of the six cell values; `cells` {'ami/emotion': v, ...} with exactly V.CELLS."""
    assert tuple(sorted(cells)) == tuple(sorted(CELLS)), sorted(cells)
    v = np.array([cells[c] for c in CELLS], dtype=np.float64)
    assert np.isfinite(v).all()
    return float(v.mean())


def compare(ref_cells, row_cells, tie=TIE):
    """One row against c*: both cell sets, the differences (row minus c*), the six-cell means and the outcome:
    'c* wins or ties' unless the row's mean is above c*'s by more than `tie`."""
    mr, mc = six_mean(ref_cells), six_mean(row_cells)
    d = {c: float(row_cells[c] - ref_cells[c]) for c in CELLS}
    return dict(cstar=dict(ref_cells), row=dict(row_cells), diff=d, cstar_mean=mr, row_mean=mc, mean_diff=mc - mr,
                row_wins=bool(mc - mr > tie))


def outcome(name, cmp):
    return f'{name} wins' if cmp['row_wins'] else 'c* wins or ties'


def overall(results):
    """{row: compare dict} -> dict(outcome, winners): 'c* wins or ties', or '<row> wins' for the row with the largest
    six-cell mean among the winners (ties among rows: the listed order)."""
    names = list(results)
    winners = [n for n in names if results[n]['row_wins']]
    if not winners:
        return dict(outcome='c* wins or ties', winners=[], best=None)
    best = max(winners, key=lambda n: (results[n]['mean_diff'], -names.index(n)))
    return dict(outcome=f'{best} wins', winners=winners, best=best)


def boot_diff(ref, row, n_draws, rng_seed=0):
    """Paired painting bootstrap of (row minus c*) six-cell mean. `ref`, `row`: dict(parts={aspect: int [3, n]},
    ap={aspect: float64 [n], nan where the row is no query}); labels in `ref['labels']`. The same I_b serves both arms
    and the six cells; AMI over the labelled rows of I_b (repeats kept), AP over the queries among them.
    -> dict(mean_diff, lo, hi, n_draws) with numpy.percentile(., [2.5, 97.5])."""
    from sklearn.metrics import adjusted_mutual_info_score as ami
    y = ref['labels']
    n = len(next(iter(y.values())))
    rng = np.random.default_rng(rng_seed)
    out = np.empty(n_draws)
    for b in range(n_draws):
        idx = rng.integers(0, n, size=n)
        tot = {}
        for name, arm in (('ref', ref), ('row', row)):
            vals = []
            for a in ASPECTS:
                keep = idx[y[a][idx] >= 0]
                vals.append(np.mean([ami(y[a][keep], arm['parts'][a][j][keep]) for j in range(len(arm['parts'][a]))]))
            for a in ASPECTS:
                ap = arm['ap'][a][idx]
                vals.append(float(np.nanmean(ap)))
            tot[name] = float(np.mean(vals))
        out[b] = tot['row'] - tot['ref']
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(mean_diff=float(out.mean()), lo=float(lo), hi=float(hi), n_draws=int(n_draws))


# ------------------------------------------------------------------------------------------------ loading and scoring
def _imports():
    import ns_paths
    ns_paths.add_rounds()
    import ns_clu
    import ns_data
    import ns_metrics
    import rs_vh
    import rt_data
    import rt_metrics
    return ns_clu, ns_data, ns_metrics, rs_vh, rt_data, rt_metrics


def export_dir(arm, r14_root=R14_ROOT, r28_root=R28_ROOT):
    """c* from r14's tree, a knob row from r28's: `<root>/train/<arm>/R6-5/`."""
    return Path(r14_root if arm == REF_ARM else r28_root) / f'train/{arm}/R6-5'


def export_path(arm, aspect, seed, r14_root=R14_ROOT, r28_root=R28_ROOT):
    """`<aspect>__<graph row>__s<seed>.npz` found by glob (r10 names a validation export after its graph row: Gres,
    Fres05, R0o for c* and the knob rows, Dview or R0 for the graph removals); exactly one file must match. Returns
    (path, graph row from the file name)."""
    hits = sorted(export_dir(arm, r14_root, r28_root).glob(f'{aspect}__*__s{seed}.npz'))
    assert len(hits) == 1, (arm, aspect, seed, [h.name for h in hits])
    stem = hits[0].stem.split('__')
    assert len(stem) == 3 and stem[0] == aspect and stem[2] == f's{seed}', hits[0]
    return hits[0], stem[1]


def load_val(arm, aspect, seed, names_sha, n, r14_root=R14_ROOT, r28_root=R28_ROOT):
    """float16 validation export [n, 2048] of one run, checked: sidecar names this arm (when it records one), aspect,
    seed, graph, the validation list hash and row count, the output file hash; unit norm."""
    ns_clu, ns_data, ns_metrics, rs_vh, rt_data, rt_metrics = _imports()
    path, graph = export_path(arm, aspect, seed, r14_root, r28_root)
    side = json.loads(path.with_suffix('.json').read_text())
    cfg = side['config']
    assert cfg['aspect'] == aspect and cfg['seed'] == seed and not cfg.get('smoke'), path
    assert cfg['graph_row'] == graph, (path, cfg['graph_row'])          # the run record names the same graph row
    assert graph == rt_data.GRAPHS[aspect], (path, graph)                # c* and the knob rows keep c*'s graph rows
    rg_arm = cfg.get('rg', {}).get('arm')
    assert rg_arm in (None, arm), (path, rg_arm, arm)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img', 'cap'}, (path, z.files)
        x16 = z['img']
    assert x16.dtype == np.float16 and x16.shape[0] == n and x16.shape[1] == 2048, (path, x16.shape, x16.dtype)
    V.join_check(cfg['names_sha256']['dev'], names_sha, len(x16), n, n, str(path))
    assert cfg.get('output_rows', n) == n, path
    if 'output_sha256' in side:
        assert side['output_sha256'] == rs_vh.file_sha(path), f'{path}: output file hash'
    assert np.isfinite(x16.astype(np.float32)).all() and \
        np.abs(np.linalg.norm(x16.astype(np.float32), axis=1) - 1).max() < 1e-3, path
    return x16, path


def score_arm(arm, seed=SEED, keep_objects=False, log=print, r14_root=R14_ROOT, r28_root=R28_ROOT):
    """The six validation cells of one run. -> dict(cells, objects (parts and per-row AP, when asked), path hashes)."""
    ns_clu, ns_data, ns_metrics, rs_vh, rt_data, rt_metrics = _imports()
    names = ns_data.keys('artelingo', 'val')
    assert names == rt_data.names('val') and len(names) == 6152
    names_sha = rs_vh.names_sha(names)
    cells, parts_all, ap_all, labels, files = {}, {}, {}, {}, {}
    for a in ASPECTS:
        t = time.monotonic()
        y = ns_data.labels('artelingo', 'val', a)
        k = CLASS_COUNT[a]
        assert ns_metrics.class_count(y) == k, (a, ns_metrics.class_count(y), k)
        x16, path = load_val(arm, a, seed, names_sha, len(names), r14_root, r28_root)
        files[a] = dict(path=str(path), file_sha256=rs_vh.file_sha(path))
        if arm == REF_ARM and r14_root == R14_ROOT:     # the loader equals r28_valpick's, bit for bit
            assert np.array_equal(x16.astype(np.float32), ns_data.cstar('artelingo', 'val', 'siglip1', a, seed, unit=False))
        xc = x16.astype(np.float32)
        parts = ns_metrics.partitions(xc, k, 'artelingo')
        (row,) = ns_clu.metric_table(f'{arm}', a, parts[None], y, k, metrics=('ami',))
        cells[f'ami/{a}'] = float(row['per_seed'][0])
        xu = ns_data._unit32(x16)
        res = rt_metrics.retrieval(xu, y, qe_k=0, workers=8)
        assert np.isfinite(res['ap']).all()
        cells[f'map/{a}'] = float(res['ap'].mean())
        if keep_objects:
            ap = np.full(len(y), np.nan)
            ap[res['rows']] = res['ap']
            parts_all[a], ap_all[a], labels[a] = parts, ap, y
        log(f'  {arm} {a}: AMI {cells[f"ami/{a}"]:.5f} mAP {cells[f"map/{a}"]:.5f} ({time.monotonic() - t:.1f}s)')
    out = dict(arm=arm, seed=seed, cells=cells, mean=six_mean(cells), files=files)
    if keep_objects:
        out['objects'] = dict(parts=parts_all, ap=ap_all, labels=labels)
    return out


def gate_ref(ref_cells, path=VALPICK_SIGLIP1):
    """c*'s seed-0 cells against r28_valpick's seed-0 values (same code path)."""
    rec = json.loads(Path(path).read_text())
    assert rec['encoder'] == 'siglip1'
    exp = {c: rec['cells'][c]['per_seed'][SEED] for c in CELLS}
    diffs = {c: abs(ref_cells[c] - exp[c]) for c in CELLS}
    return dict(source=str(path), max_abs_diff=max(diffs.values()), diffs=diffs, **{'pass': bool(max(diffs.values()) <= TOL)})


def noise_context(path=VALPICK_SIGLIP1):
    """c*'s six-cell mean per seed (r14's five seeds) and its range: the scale of a seed-to-seed difference."""
    rec = json.loads(Path(path).read_text())
    per_seed = [float(np.mean([rec['cells'][c]['per_seed'][s] for c in CELLS])) for s in V.SEEDS]
    return dict(source=str(path), cstar_six_mean_per_seed=per_seed, range=float(max(per_seed) - min(per_seed)))


def table_md(doc):
    names = list(doc['rows'])
    lines = ['| cell | c* (k11=40) | ' + ' | '.join(f'{n} (diff)' for n in names) + ' |', '|---|---:|' + '---:|' * len(names)]
    for c in list(CELLS) + ['mean of six']:
        ref = doc['cstar_mean'] if c == 'mean of six' else doc['cstar_cells'][c]
        row = []
        for n in names:
            r = doc['rows'][n]
            v, d = (r['row_mean'], r['mean_diff']) if c == 'mean of six' else (r['row'][c], r['diff'][c])
            row.append(f'{v:.4f} ({d:+.4f})')
        lines.append(f'| {c} | {ref:.4f} | ' + ' | '.join(row) + ' |')
    lines.append('| outcome | | ' + ' | '.join(doc['outcomes'][n] for n in names) + ' |')
    return '\n'.join(lines)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def run(rows=ROWS, out=OUT, records=None, boot=0, log=print, r14_root=R14_ROOT, r28_root=R28_ROOT):
    t0 = time.monotonic()
    ref = score_arm(REF_ARM, SEED, keep_objects=bool(boot), log=log, r14_root=r14_root, r28_root=r28_root)
    gate = gate_ref(ref['cells']) if r14_root == R14_ROOT and VALPICK_SIGLIP1.exists() else None
    if gate is not None:
        log(f'gate (c* seed 0 = r28_valpick seed 0): max abs diff {gate["max_abs_diff"]:.2e} pass={gate["pass"]}')
        assert gate['pass'], 'c* cells differ from r28_valpick'
    results, outcomes, files, bootstraps = {}, {}, {}, {}
    for r in rows:
        assert r in ROWS + (REF_ARM,), r      # REF_ARM: the self-check, c* against itself
        sc = score_arm(r, SEED, keep_objects=bool(boot), log=log, r14_root=r14_root, r28_root=r28_root)
        results[r] = compare(ref['cells'], sc['cells'])
        outcomes[r] = outcome(r, results[r])
        files[r] = sc['files']
        if boot:
            o_ref, o_row = ref['objects'], sc['objects']
            bootstraps[r] = boot_diff(o_ref, o_row, boot)
        log(f'{r}: mean diff {results[r]["mean_diff"]:+.5f} -> {outcomes[r]}')
    doc = dict(kind='r28_settings_check', encoder='siglip1', split='artelingo validation', n_val=6152, seed=SEED,
               criterion='fallback: mean of the six validation cells (AMI at 9/27/10, plain-cosine mAP), c* wins ties; '
                         'r10 stage 3 criterion not buildable (P, r_a, seeds 1-3, k=20 partitions)',
               tie_tolerance=TIE, cstar_arm=REF_ARM, cstar_cells=ref['cells'], cstar_mean=ref['mean'],
               cstar_files=ref['files'], rows=results, outcomes=outcomes, row_files=files,
               overall=overall(results) if results else dict(outcome='c* wins or ties', winners=[], best=None),
               gate_cstar_vs_valpick=gate, noise_context=noise_context() if VALPICK_SIGLIP1.exists() else None,
               bootstrap=bootstraps or None, seconds=time.monotonic() - t0)
    write_json(Path(out) / 'settings_check.json', doc)
    if records is not None:
        write_json(Path(records) / 'settings_check.json', doc)
    log(table_md(doc) if results else 'no rows scored')
    log(f'overall: {doc["overall"]["outcome"]}')
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('run',))
    p.add_argument('--rows', nargs='*', default=list(ROWS), choices=ROWS + (REF_ARM,))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--records', type=Path, default=None)
    p.add_argument('--boot', type=int, default=0)
    a = p.parse_args()
    run(tuple(a.rows), a.out, a.records, a.boot)


if __name__ == '__main__':
    main()
