"""Stage ii encoder statistics, correlations and held table (spec §3.6 stage ii, D13; rule §3.2 "Reported", §4.2).

`stats`: for each encoder of `eg_encoders.ENCODERS` whose default store exists, the statistics of its default image
rows for P (the 29,232 paintings, `rb_encoders.full_images`): stored norms before normalization, then on unit rows
rb_sheet's kernels (anisotropy, 20-NN label purity per aspect with chance, cosine silhouette per aspect, hubness);
then Spearman correlations over the encoders scored in `stage_ii/lift.json` between the emotion lift and each of
A_emotion(F), the own arm's best emotion reading and each statistic. Writes `stage_ii/stats.{json,md}`.
`held`: only after `stats.json` exists; per encoder the held frozen AMI, the own arm's best reading, the student and the
lift (student minus own best) at both scales, from 12-24's `reported_stage{3,4}.json` (K2 to K4) and this round's
`part2/reported_stage{3,4}.json` (K5 to K8), and the lift's rank on held against P. Writes `stage_ii/held_table.{json,md}`.
Both decide nothing; the correlations are descriptive only (n is small).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import eg_paths
import eg_encoders

STAGE = 'stage_ii'
THREADS = 8
SMOKE_ROWS = 1000
SCALES = {3: 'sample', 4: 'full'}          # reported_stage{3,4}.json: stage 3 on the sample, stage 4 on the full split
READINGS = ('r20', 'own')
CONSTANT_TOL = 1e-6      # max minus min below this: float noise, no rank correlation
_NOTE = 'descriptive only: n is the number of encoders; no test, no threshold'


# ------------------------------------------------------------------ small helpers
def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=eg_paths.ROUND, capture_output=True, text=True,
                          check=True).stdout.strip()


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(text)
    pending.replace(path)


def write_record(folder, name, record, md):
    """Write `<name>.json` and `<name>.md`; an existing json must hold the same results (code_commit and seconds aside)."""
    record = json.loads(json.dumps(record, allow_nan=False))
    path = Path(folder) / f'{name}.json'
    if path.exists():
        drop = lambda d: json.loads(json.dumps(d, sort_keys=True), object_hook=lambda o: {k: v for k, v in o.items()
                                                                                         if k not in ('code_commit', 'seconds')})
        assert drop(json.loads(path.read_text())) == drop(record), f'{path} exists with other results; move it aside'
        return path
    _atomic(path, json.dumps(record, indent=1) + '\n')
    _atomic(Path(folder) / f'{name}.md', md)
    return path


def spearman(x, y):
    """{rho, n}: Spearman rank correlation (ties averaged); rho None when n < 3 or an input is constant."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all()
    n = int(len(x))
    if n < 3:
        return {'rho': None, 'n': n}
    if np.ptp(x) < CONSTANT_TOL or np.ptp(y) < CONSTANT_TOL:
        return {'rho': None, 'n': n, 'reason': 'constant over the encoders'}
    return {'rho': float(spearmanr(x, y)[0]), 'n': n}


def ranks_desc(values):
    """Rank 1 = largest; ties share the mean rank."""
    from scipy.stats import rankdata
    return (len(values) + 1 - rankdata(values, method='average')).tolist()


# ------------------------------------------------------------------ the default store rows for P
def load_default(store, names, width):
    """(unit float32 [n, width], stored norms float64 [n]) of the store's default image rows for `names`.

    Qwen goes through the CT's loader (as `rb_encoders.full_images`); the others read `artelingo_full/<store>` with the
    same checks as `rb_encoders.read_store`. Raises FileNotFoundError when the store does not exist."""
    import rb_encoders
    import rb_sheet
    import ct2_data
    names = [str(n) for n in names]
    if store == 'qwen3vl_emb_2b':
        x = rb_encoders.full_images(store, names)
        norms = rb_sheet.stored_image_norms([ct2_data.QWEN], names)
    else:
        path = rb_encoders.FULL_ROOT / store
        if not (path / 'index.json').is_file() or not (path / 'img.npy').is_file():
            raise FileNotFoundError(f'{path}: no default store (index.json, img.npy)')
        index = json.loads((path / 'index.json').read_text())
        assert index['backbone'] == store and index['dataset'] == path.parent.name
        assert not index['smoke'] and index['dtype'] == 'float16' and index['dim'] == width, (store, index['dim'])
        stored = [r['painting'] for r in index['images']]
        assert len(stored) == index['n_images'] and len(set(stored)) == len(stored)
        raw = np.load(path / 'img.npy', mmap_mode='r')
        assert raw.shape == (len(stored), width) and raw.dtype == np.float16, (store, raw.shape, raw.dtype)
        lookup = {n: i for i, n in enumerate(stored)}
        assert all(n in lookup for n in names), f'{store}: P paintings missing from the store'
        rows = np.asarray([lookup[n] for n in names], dtype=np.int64)
        v = np.asarray(raw[rows], dtype=np.float32)
        assert np.isfinite(v).all()
        norms = np.linalg.norm(v, axis=1).astype(np.float64)
        assert (norms > 0).all()
        from apc_data import l2
        x = l2(v)
    assert x.dtype == np.float32 and x.shape == (len(names), width) and norms.shape == (len(names),)
    return x, norms


# ------------------------------------------------------------------ the statistics (default view only)
def compute_default(x, norms, labels):
    """rb_sheet's kernels on one set of unit rows `x` [n, w]; `labels` {aspect: int64 [n]} (-1 masked)."""
    import rb_data
    import rb_sheet as sh
    n = x.shape[0]
    rb_data.check_unit(x, n)
    assert tuple(labels) == sh.ASPECTS
    for y in labels.values():
        assert y.shape == (n,) and y.dtype == np.int64
    with threadpool_limits(limits=THREADS):
        nbr = sh.knn(x)
        return {'n': int(n), 'width': int(x.shape[1]), 'norms_as_stored': sh.norm_summary(norms),
                'anisotropy': sh.anisotropy(x),
                'purity_20nn': {a: sh.purity(nbr, labels[a]) for a in sh.ASPECTS},
                'silhouette_cosine': {a: sh.silhouette(x, labels[a]) for a in sh.ASPECTS},
                'hubness_20nn': sh.hubness(nbr, n)}


def flat_statistics(s):
    """The scalar statistics the correlations use, from one encoder's block."""
    out = {'norm_mean': s['norms_as_stored']['mean'], 'mean_pairwise_cosine': s['anisotropy']['mean_pairwise_cosine'],
           'top_pc_variance_share': s['anisotropy']['top_pc_variance_share']}
    for a in eg_paths.ASPECTS:
        out[f'purity_{a}'] = s['purity_20nn'][a]['purity']
        out[f'purity_minus_chance_{a}'] = s['purity_20nn'][a]['purity'] - s['purity_20nn'][a]['chance']
        out[f'silhouette_{a}'] = s['silhouette_cosine'][a]['overall']
    out['hubness_skew'] = s['hubness_20nn']['skew']
    out['hubness_max_in_degree'] = s['hubness_20nn']['max_in_degree']
    out['hubness_share_zero'] = s['hubness_20nn']['share_zero_in_degree']
    return out


# ------------------------------------------------------------------ lift.json
def read_lift(outputs):
    """lift.json (ticket 08) with the checks of every file read here; returns (record, sha256)."""
    path = Path(outputs) / STAGE / 'lift.json'
    assert path.is_file(), f'{path} missing (ticket 08 writes it)'
    lift = json.loads(path.read_text())
    assert lift['p_sha256'] == eg_paths.P_SHA256, 'lift.json: P differs'
    assert lift['n_rows'] == 29232
    assert list(lift['encoders']) == list(eg_encoders.STAGE_II), f"lift.json: encoder order {list(lift['encoders'])}"
    for enc in eg_encoders.ENCODERS:
        b = lift['encoders'][enc.student]
        assert (b['encoder'], b['own'], b['frozen']) == (enc.store, enc.own, enc.frozen), enc.student
        assert b['status'] in ('scored', 'missing')
        if b['status'] == 'scored':
            assert tuple(b['aspects']) == eg_paths.ASPECTS and b['outcome'] in ('L-HOLDS', 'L-QUAL', 'L-FAILS')
        else:
            assert b['outcome'] is None and isinstance(b.get('reason'), str)
    return lift, file_sha256(path)


def lift_outcomes(lift):
    """{store: {lift, a_frozen, own_best_a, outcome}} of the encoders scored in lift.json (emotion)."""
    out = {}
    for enc in eg_encoders.ENCODERS:
        b = lift['encoders'][enc.student]
        if b['status'] != 'scored':
            continue
        e = b['aspects']['emotion']
        own_best = max(e['A']['own_r20'], e['A']['own_own'])
        assert abs((e['A']['student'] - own_best) - e['lift']) < 1e-9, f'{enc.student}: lift != student - own best'
        out[enc.store] = {'student': enc.student, 'lift': e['lift'], 'A_emotion_frozen': e['A']['frozen'],
                          'own_best_emotion': own_best, 'outcome': b['outcome']}
    return out


# ------------------------------------------------------------------ stats
def correlations(per_encoder, outcomes):
    """Spearman of the emotion lift against A_emotion(F), the own arm's best emotion reading and each statistic, over
    the encoders that have both a statistics block and a lift."""
    stores = [e.store for e in eg_encoders.ENCODERS if e.store in per_encoder and e.store in outcomes]
    if not stores:
        return {'stores': [], 'n': 0, 'note': _NOTE, 'rho': {}}
    lift = [outcomes[s]['lift'] for s in stores]
    cols = {'A_emotion_frozen': [outcomes[s]['A_emotion_frozen'] for s in stores],
            'own_best_emotion': [outcomes[s]['own_best_emotion'] for s in stores]}
    flat = {s: flat_statistics(per_encoder[s]) for s in stores}
    for key in flat[stores[0]]:
        cols[key] = [flat[s][key] for s in stores]
    return {'stores': stores, 'n': len(stores), 'note': _NOTE, 'lift': lift,
            'rho': {k: spearman(v, lift) for k, v in cols.items()}}


def fmt(x, d=4):
    return 'n/a' if x is None else f'{x:.{d}f}'


def stats_md(rec):
    lines = ['# Stage ii encoder statistics (default image rows for P)', '',
             f"smoke: {rec['smoke']}. Rows: {rec['n_rows']}. Descriptive only; decides nothing.", '',
             '| encoder | norm mean | mean pair cos | top PC share | purity emo/sty/gen (chance) | silhouette emo/sty/gen '
             '| hub skew | hub max | hub zero |', '|---|---|---|---|---|---|---|---|---|']
    for store, s in rec['encoders'].items():
        if s['status'] != 'ok':
            lines.append(f"| {store} | skipped: {s['reason']} | | | | | | | |")
            continue
        pur = ' / '.join(f"{fmt(s['purity_20nn'][a]['purity'])} ({fmt(s['purity_20nn'][a]['chance'])})" for a in eg_paths.ASPECTS)
        sil = ' / '.join(fmt(s['silhouette_cosine'][a]['overall']) for a in eg_paths.ASPECTS)
        h = s['hubness_20nn']
        lines.append(f"| {store} | {fmt(s['norms_as_stored']['mean'])} | {fmt(s['anisotropy']['mean_pairwise_cosine'])} "
                     f"| {fmt(s['anisotropy']['top_pc_variance_share'])} | {pur} | {sil} | {fmt(h['skew'])} "
                     f"| {h['max_in_degree']} | {fmt(h['share_zero_in_degree'])} |")
    c = rec['correlations']
    lines += ['', f"## Spearman correlation with the emotion lift (n = {c['n']} encoders: {', '.join(c['stores']) or 'none'})", '',
              f"{_NOTE}. With n < 3 no value is given.", '', '| quantity | rho |', '|---|---|']
    for k, v in c['rho'].items():
        cell = 'n/a: constant over the encoders' if 'reason' in v else fmt(v['rho'], 3)
        lines.append(f'| {k} | {cell} |')
    lines += ['', 'Encoders without a lift outcome (not scored in lift.json): '
              + (', '.join(rec['lift_not_scored']) or 'none') + '.', '']
    return '\n'.join(lines)


def run_stats(outputs, labels, names, *, stores=None, smoke=False, loader=load_default):
    """Statistics for each encoder, correlations, and `stage_ii/stats.{json,md}`; returns the record."""
    import rb_data
    outputs = Path(outputs)
    assert rb_data.names_sha256(list(names)) == eg_paths.P_SHA256 and len(names) == 29232
    lift, lift_sha = read_lift(outputs)
    per, skipped = {}, {}
    for enc in eg_encoders.ENCODERS:
        if stores is not None and enc.store not in stores:
            continue
        t0 = time.time()
        try:
            x, norms = loader(enc.store, names, enc.width)
        except FileNotFoundError as e:
            per[enc.store] = {'status': 'skipped', 'reason': str(e)}
            print(f'{enc.store}: skipped ({e})', flush=True)
            continue
        y = labels
        if smoke:
            sel = np.arange(0, len(names), max(1, len(names) // SMOKE_ROWS))[:SMOKE_ROWS]
            x, norms, y = x[sel], norms[sel], {a: v[sel] for a, v in labels.items()}
        block = compute_default(x, norms, y)
        per[enc.store] = {'status': 'ok', **block, 'seconds': round(time.time() - t0, 1)}
        print(f'{enc.store}: n={block["n"]} {per[enc.store]["seconds"]} s', flush=True)
    ok = {s: b for s, b in per.items() if b['status'] == 'ok'}
    outcomes = lift_outcomes(lift)
    rec = {'p_sha256': eg_paths.P_SHA256, 'n_rows': 29232, 'smoke': bool(smoke), 'k': 20, 'lift_sha256': lift_sha,
           'lift_outcomes': outcomes, 'encoders': per, 'correlations': correlations(ok, outcomes),
           'lift_not_scored': [e.store for e in eg_encoders.ENCODERS if e.store not in outcomes],
           'code_commit': code_commit()}
    write_record(outputs / STAGE, 'stats', rec, stats_md(rec))
    return rec


# ------------------------------------------------------------------ held table
def read_reported(path, stage):
    """reported_stage{3,4}.json with the checks; returns {student: block}."""
    d = json.loads(Path(path).read_text())
    assert d['stage'] == stage and d['scale'] == SCALES[stage] and 'blocks' in d, path
    return d['blocks']


def held_row(enc, block, aspect):
    """Held readings of one encoder and aspect from its reported block."""
    assert block['candidate'] == enc.student and block['frozen_arm'] == enc.frozen
    a = block['aspects'][aspect]
    spread = a['point_spread']
    student = spread[enc.student]['r20']['mean']
    c = a['margins']['c_frozen']
    assert abs(c['A_cand'] - student) < 1e-9, f'{enc.student}/{aspect}: c_frozen A_cand != student'
    assert list(c['A_e']) == [f'{enc.frozen}-r20'], c['A_e']
    own = {r: spread[enc.own][r]['mean'] for r in READINGS if r in spread[enc.own]}
    best = max(own, key=own.get)
    return {'frozen': c['A_e'][f'{enc.frozen}-r20'], 'own_best': own[best], 'own_best_reading': best,
            'student': student, 'lift': student - own[best]}


def held_md(rec):
    lines = ['# Stage ii held table', '', f"Descriptive only; decides nothing. {_NOTE}. Lift = student minus own arm's "
             'best reading; P lift from lift.json (emotion).', '']
    for sc, tab in rec['scales'].items():
        lines += [f"## {tab['scale']} scale (stage {sc}; n = {len(tab['rank']['stores'])} encoders with a P lift and a held lift)", '']
        for a in eg_paths.ASPECTS:
            lines += [f'{a}', '', '| encoder | held frozen | own best (reading) | student | lift |'
                      + (' P lift | rank held | rank P |' if a == 'emotion' else ''),
                      '|---|---|---|---|---|' + ('---|---|---|' if a == 'emotion' else '')]
            for store, e in tab['encoders'].items():
                if e['status'] != 'ok':
                    lines.append(f"| {store} | skipped: {e['reason']} |")
                    continue
                r = e[a]
                row = f"| {store} | {fmt(r['frozen'])} | {fmt(r['own_best'])} ({r['own_best_reading']}) | {fmt(r['student'])} | {fmt(r['lift'])} |"
                if a == 'emotion':
                    rk = tab['rank']
                    i = rk['stores'].index(store) if store in rk['stores'] else None
                    row += (f" {fmt(e.get('p_lift'))} | {rk['held_rank'][i]:g} | {rk['p_rank'][i]:g} |" if i is not None
                            else f" {fmt(e.get('p_lift'))} | - | - |")
                lines.append(row)
            lines.append('')
        lines += [f"Rank correlation of the held lift with the P lift (emotion): rho {fmt(tab['rank']['spearman']['rho'], 3)}, "
                  f"n = {tab['rank']['spearman']['n']}.", '']
    return '\n'.join(lines)


def run_held(outputs, *, rb_part2=eg_paths.RB_PART2, part2=None):
    """Held table; refuses unless `stage_ii/stats.json` exists. K2 to K4 from `rb_part2`, K5 to K8 from `part2`."""
    outputs = Path(outputs)
    stats_path = outputs / STAGE / 'stats.json'
    assert stats_path.is_file(), f'{stats_path} missing: the P table comes before the held table (rule §4.2)'
    stats = json.loads(stats_path.read_text())
    assert stats['p_sha256'] == eg_paths.P_SHA256
    lift, lift_sha = read_lift(outputs)
    assert stats['lift_sha256'] == lift_sha, 'stats.json was made from another lift.json; rerun stats'
    outcomes = lift_outcomes(lift)
    part2 = Path(part2 if part2 is not None else outputs / 'part2')
    old = {e.student for e in eg_encoders.ENCODERS if not e.new}
    scales, sources = {}, {}
    for stage, scale in SCALES.items():
        blocks = {}
        for root, who in ((Path(rb_part2), old), (part2, {e.student for e in eg_encoders.ENCODERS if e.new})):
            path = root / f'reported_stage{stage}.json'
            if path.is_file():
                sources[f'{root}/reported_stage{stage}.json'] = file_sha256(path)
                got = read_reported(path, stage)
                blocks.update({k: v for k, v in got.items() if k in who})
        enc_rows = {}
        for enc in eg_encoders.ENCODERS:
            if enc.student not in blocks:
                enc_rows[enc.store] = {'status': 'skipped', 'reason': f'no {enc.student} block in reported_stage{stage}.json'}
                continue
            row = {a: held_row(enc, blocks[enc.student], a) for a in eg_paths.ASPECTS}
            row['status'] = 'ok'
            row['p_lift'] = outcomes[enc.store]['lift'] if enc.store in outcomes else None
            enc_rows[enc.store] = row
        both = [s for s, r in enc_rows.items() if r['status'] == 'ok' and r['p_lift'] is not None]
        p_l = [enc_rows[s]['p_lift'] for s in both]
        h_l = [enc_rows[s]['emotion']['lift'] for s in both]
        scales[str(stage)] = {'scale': scale, 'encoders': enc_rows,
                              'rank': {'stores': both, 'p_lift': p_l, 'held_lift': h_l,
                                       'p_rank': ranks_desc(p_l) if both else [], 'held_rank': ranks_desc(h_l) if both else [],
                                       'spearman': spearman(p_l, h_l), 'note': _NOTE}}
    rec = {'p_sha256': eg_paths.P_SHA256, 'stats_sha256': file_sha256(stats_path), 'lift_sha256': lift_sha,
           'held_sha256': eg_paths.HELD_SHA256, 'sources': sources, 'scales': scales, 'code_commit': code_commit()}
    write_record(outputs / STAGE, 'held_table', rec, held_md(rec))
    return rec


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description='Stage ii encoder statistics, correlations (stats) and held table (held).')
    ap.add_argument('command', choices=('stats', 'held'))
    ap.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    ap.add_argument('--smoke', action='store_true', help='stats: 1,000 rows; outputs under /tmp/claude-0/')
    ap.add_argument('--stores', nargs='*', help='stats: only these stores (default: all seven)')
    ap.add_argument('--rb-part2', type=Path, default=eg_paths.RB_PART2, help='held: 12-24\'s part2 folder (read only)')
    ap.add_argument('--part2', type=Path, help='held: this round\'s part2 folder (default <outputs>/part2)')
    args = ap.parse_args(argv)
    outputs = args.outputs.resolve()
    for frozen in (eg_paths.RB_OUT, eg_paths.CT_OUT):
        assert not outputs.is_relative_to(frozen.resolve()), f'--outputs {outputs} is inside the frozen {frozen}'
    if args.smoke:
        assert eg_paths.smoke_ok(outputs), 'smoke needs --outputs under /tmp/claude-0/'
    if args.stores:
        assert set(args.stores) <= set(eg_encoders.BY_STORE), args.stores
    eg_paths.check_imports()
    eg_encoders.extend_rb_maps()
    if args.command == 'stats':
        import rb_data
        names = rb_data.p_names()
        assert rb_data.names_sha256(names) == eg_paths.P_SHA256, 'P differs from 12-24\'s'
        rec = run_stats(outputs, rb_data.p_labels(), names, stores=args.stores, smoke=args.smoke)
        print(outputs / STAGE / 'stats.json', 'correlations n =', rec['correlations']['n'], flush=True)
    else:
        rec = run_held(outputs, rb_part2=args.rb_part2, part2=args.part2)
        print(outputs / STAGE / 'held_table.json', {k: len(v['rank']['stores']) for k, v in rec['scales'].items()}, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
