"""Stage A and B reported readings (spec §3.6, rule §4.1 and §4.2 "Reported"): decide nothing. `python rg_score.py report`
calls `run_report`; this module is imported lazily there so that editing it does not change the scorer's code hash (the
verdict files carry that hash).

Runs only after `stage_a/verdict.json` (and reads `stage_b/verdict.json` when it exists). Intervals, outcomes, levels
and best epochs come from the verdict files, never recomputed. New readings: Leiden AMI on P and k-means AMI on the old
5,000 per arm and aspect (as 12-25's `eg_score3.run_report`), the loss-scale run, each arm's genre term means, EQ's values.
Writes `<outputs>/reported_ab.{json,md}`.
"""
import json
from pathlib import Path

import numpy as np

import rg_paths
import rg_variants as V
import rg_score as S
import eg_score3 as E
import rb_ami
import rb_data
import rb_pick

ASPECTS = rg_paths.ASPECTS
BASE = V.BASE


def _load(path, smoke, stage):
    path = Path(path)
    assert path.exists(), f'{path} missing'
    x = json.loads(path.read_text())
    assert x['stage'] == stage and x['smoke'] == bool(smoke), f'{path}: stage or smoke flag differs'
    return x


def cached_reading(outputs, arm, aspect, kind, identity, compute):
    """Copy of `eg_score3.cached_reading`, caching under this round's `stage_a/partitions/<arm>/s0/` (the original uses
    12-25's stage_iii folder)."""
    path = S.cache_dir(outputs, arm) / f'{aspect}__{kind}.json'
    identity = dict(identity, kind=kind, variant=arm, aspect=aspect, versions=rb_ami.versions())
    if path.exists():
        old = json.loads(path.read_text())
        assert old['identity'] == identity, f'{path}: cached from other inputs; move it aside'
        return old['value']
    value = json.loads(json.dumps(compute(), allow_nan=False))
    S._json_dump(path, dict(identity=identity, value=value))
    return value


def old5000_export(outputs, arm, aspect, p_sha):
    """Copy of `eg_score3.old5000_export` for this round's arms (the original reads 12-25's stage_iii folders). R6-5: the
    original's frozen read."""
    if arm == BASE:
        return E.old5000_export(outputs, arm, aspect, p_sha)
    with np.load(S.arm_folder(outputs, arm) / f'{aspect}__old5000.npz', allow_pickle=False) as z:
        assert z.files == ['img'], (arm, aspect, z.files)
        x = z['img']
    assert x.shape == (rg_paths.N_OLD, rb_pick.WIDTH) and x.dtype == np.float16, (arm, aspect, x.shape, x.dtype)
    return x


def arm_readings(outputs, arm, imgs, ctx, old_labels, leiden_fn):
    """{leiden, old5000} per aspect (copy of the loop in `eg_score3.report_variant`, without the GA branch)."""
    import rb_part1_report
    p_sha, labels = ctx['p_sha'], ctx['labels']
    out = dict(leiden={}, old5000={})
    for a in ASPECTS:
        ident = dict(p_sha256=p_sha, img_sha256=rb_pick.array_sha256(imgs[a]), label_sha256=rb_pick.array_sha256(labels[a]))
        out['leiden'][a] = cached_reading(outputs, arm, a, 'leiden', ident, lambda: leiden_fn(imgs[a], labels[a]))
        old = old5000_export(outputs, arm, a, p_sha)
        ident = dict(p_sha256=p_sha, img_sha256=rb_pick.array_sha256(old), label_sha256=rb_pick.array_sha256(old_labels[a]))

        def old_ami():
            try:
                return rb_part1_report.kmeans_ami(old, old_labels[a])
            except AssertionError as err:                    # VA3 covers P only; an old-5,000 problem is reported
                return f'not scored: {err}'
        out['old5000'][a] = cached_reading(outputs, arm, a, 'old5000', ident, old_ami)
    return out


def term_means_of(folder, arm, smoke):
    """The genre term means of a run.json (ticket 02's trainer): {term_aspect, term_reduction, epochs: [...]} or None for a
    run that has none (12-25's frozen GR-8 run; a failed run that stopped before an epoch)."""
    path = Path(folder) / 'run.json'
    assert path.exists(), f'{path} missing'
    run = json.loads(path.read_text())
    assert run['variant'] == arm, (arm, run['variant'])
    tm = run.get('term_means')
    if tm is None or tm == []:                 # [] : stopped on a non-finite loss in its first call (no step to average)
        return None
    assert run.get('term_aspect') == 'genre' and isinstance(tm, list), (arm, 'term_means')
    assert all('epoch' in e and 'calls' in e for e in tm), (arm, 'term_means rows')
    return dict(term_aspect='genre', reduction=run.get('term_reduction'), status=run['status'], smoke=bool(run.get('smoke')),
                best_epoch=run.get('best_epoch'), epochs=tm, **({'anchors': run['anchors']} if 'anchors' in run else {}))


def _plain(x):
    """A JSON-safe copy of `x`: a non-finite float becomes its text ('nan', 'inf', '-inf'), shown as such in the report."""
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _loss_scale_run(ls_path, smoke):
    """The loss-scale run's record for the report. The report decides nothing, so a missing file, a failed run or empty
    epoch means is a `problems` line in the report, not an assertion (spec §3.5, §3.6). -> dict(source, problems, ...)."""
    out = dict(source=str(ls_path), problems=[], epoch_means=[])
    if not ls_path.exists():
        out['problems'].append(f'{ls_path} is missing: the loss-scale run left no record (it failed or did not run)')
        return out
    out['sha256'] = S._file_sha256(ls_path)
    try:
        ls = json.loads(ls_path.read_text())
    except ValueError as error:
        out['problems'].append(f'{ls_path} is not readable JSON: {error}')
        return out
    if not isinstance(ls, dict):
        out['problems'].append(f'{ls_path} holds no JSON object')
        return out
    ls = _plain(ls)
    for key in ('status', 'cause', 'variant', 'switches', 'smoke', 'reduction', 'anchors'):
        if key in ls:
            out[key] = ls[key]
    out.update({k: v for k, v in ls.items() if 'finite' in k})       # a recorded count of non-finite terms, whatever its name
    if 'smoke' in ls and ls['smoke'] != bool(smoke):
        out['problems'].append(f"the record is from a {'smoke' if ls['smoke'] else 'real'} run, this report is not")
    if ls.get('status') != 'ok':
        out['problems'].append(f"the run's status is {ls.get('status')!r}; cause: {ls.get('cause')}")
    if 'variant' in ls and (ls['variant'], ls.get('switches')) != (V.LOSS_SCALE, V.LOSS_SCALE_SWITCHES):
        out['problems'].append(f"variant {ls['variant']!r} with switches {ls.get('switches')!r} is not {V.LOSS_SCALE}'s")
    means = ls.get('epoch_means')
    if not isinstance(means, list) or not means:
        out['problems'].append('no epoch means (the run stopped before it finished its first epoch of training calls)')
        means = []
    out['epoch_means'] = [e for e in means if isinstance(e, dict)]
    for e in out['epoch_means']:
        absent = [k for k in ('direct', 'relational') if k not in e]
        if absent:
            out['problems'].append(f"epoch {e.get('epoch')}: no {' or '.join(absent)} term")
    # rg_train writes a non-finite term mean as null (with nonfinite_counts per term); older records may hold 'nan' text
    shown = [f'epoch {e.get("epoch")} {k}' for e in out['epoch_means'] for k, v in e.items() if v is None or isinstance(v, str)]
    if shown:
        out['problems'].append('non-finite term means: ' + '; '.join(shown))
    return out


def loss_scale(outputs, arms, smoke):
    """spec §3.6: the loss-scale run's epoch means, each arm's genre term means, EQ's values."""
    out = Path(outputs)
    ls_path = out / 'loss_scale' / 'loss_scale.json'
    run_record = _loss_scale_run(ls_path, smoke)
    per_arm = {}
    for arm in arms:
        folder = out / 'regression' / 'va1_GR-8_s0' if arm == 'GR-8' else rg_paths.run_dir(out, arm, 0)
        per_arm[arm] = term_means_of(folder, arm, smoke)
    eq_path = out / 'regression' / 'eq.json'
    assert eq_path.exists(), f'{eq_path} missing'
    eq = json.loads(eq_path.read_text())
    assert eq['smoke'] == bool(smoke) and isinstance(eq['pass'], bool) and eq['items'], f'{eq_path}: smoke flag, pass or items'
    return dict(loss_scale=run_record,
                arm_term_means=per_arm,
                eq=dict(source=str(eq_path), sha256=S._file_sha256(eq_path), **{'pass': eq['pass']}, tolerance=eq.get('tolerance'),
                        items=eq['items']))


def stage_b_reading(vb):
    """Stage B's reported part from its verdict file: Ā per arm and aspect with intervals, C levels, and whether the
    confirmed arms sit within one seed range (r_genre) of each other (spread of Ā_genre; decides nothing)."""
    arms = {}
    for arm, r in vb['results'].items():
        if r['status'] != 'ok':
            arms[arm] = dict(status=r['status'], cause=r['cause'])
            continue
        arms[arm] = dict(status='ok', c_outcome=r['c_outcome'], confirmed=r['confirmed'], Abar_genre=r['Abar_genre'],
                         genre_vs_targets=r['genre_vs_targets'],
                         effects={a: dict(Abar=r['effects'][a]['Abar'][arm], Abar_ref=r['effects'][a]['Abar'][BASE],
                                          e=r['effects'][a]['e'], lo=r['effects'][a]['lo'], hi=r['effects'][a]['hi'])
                                  for a in ASPECTS})
    def spread(names):
        v = [arms[n]['Abar_genre'] for n in names]
        return None if len(v) < 2 else dict(arms=list(names), spread=max(v) - min(v), within_one_range=max(v) - min(v) <= V.RANGES['genre'])
    ok = [n for n, x in arms.items() if x['status'] == 'ok']
    return dict(outcome=vb['outcome'], v_held=vb['v_held'], a_outcome=vb['a_outcome'], v_star_a=vb['v_star_a'],
                confirmed=vb['confirmed'], within_one_range_of_v_held=vb['within_one_range'], r_genre=V.RANGES['genre'],
                arms=arms, spread_confirmed=spread(vb['confirmed']), spread_scored=spread(ok), runs=vb['runs'])


def run_report(outputs, smoke=False, leiden_fn=None):
    outputs = Path(outputs)
    va_path = S.stage_dir(outputs) / 'verdict.json'
    va = _load(va_path, smoke, 'a')
    vb_path = S.b_dir(outputs) / 'verdict.json'
    vb = _load(vb_path, smoke, 'b') if vb_path.exists() else None
    if vb is not None:
        assert vb['identity']['stage_a_verdict_sha256'] == S._file_sha256(va_path), 'stage_b/verdict.json is for another stage A verdict'
    ctx = E.load_context()
    p_sha = ctx['p_sha']
    assert va['identity']['p_sha256'] == p_sha, 'stage A\'s verdict is on another P'
    old_labels = rb_data.old5000_labels()
    frozen = json.loads((rg_paths.RB_PART1 / 'reported.json').read_text())
    for a in ASPECTS:
        assert int((old_labels[a] >= 0).sum()) == frozen['label_counts_old5000'][a], f'{a}: old-5,000 labels differ from 12-24\'s'
    if leiden_fn is None:
        import rb_part1_report
        leiden_fn = rb_part1_report.leiden_ami
    # stage A arms: P intervals, outcome, level, best epochs from the verdict; readings new
    entries = {}
    base_img = E.base_images(p_sha)
    entries[BASE] = dict(status='ok', cause=None, best_epoch=E.base_epochs(), level=None,
                         p={a: dict(A=va['base_A'][a], e=0.0, lo=0.0, hi=0.0) for a in ASPECTS},
                         **arm_readings(outputs, BASE, base_img, ctx, old_labels, leiden_fn))
    for arm in va['arms']:
        r = va['results'][arm]
        entry = dict(status=r['status'], cause=r['cause'], best_epoch=r['best_epoch'], level=va['levels'].get(arm),
                     p=r['effects'] if r['status'] == 'ok' else None, A_genre=r['A_genre'], guarded=r['guarded'])
        if r['status'] == 'ok':
            run = S.read_run(outputs, arm, p_sha, smoke=smoke)
            assert run['status'] == 'ok' and run['run_sha256'] == va['identity']['run_sha256'][arm], f'{arm}: run.json changed since the verdict'
            for a in ASPECTS:
                assert rb_pick.array_sha256(run['img'][a]) == va['identity']['img_sha256'][arm][a], f'{arm}/{a}: export changed since the verdict'
            entry.update(arm_readings(outputs, arm, run['img'], ctx, old_labels, leiden_fn))
        else:
            entry.update(leiden=None, old5000=None)
        entries[arm] = entry
        print(f'{arm}: read', flush=True)
    probe = json.loads((rg_paths.RB_PART1 / 'probe.json').read_text())['aspects']['genre']['ami']['P']['ridge']
    assert abs(probe - V.PROBE_GENRE) < 1e-6, probe
    refs = frozen['references_p']
    assert refs['genre_word']['genre'] == V.T_G
    check = dict(leiden_p={a: abs(entries[BASE]['leiden'][a]['mean'] - frozen['p_leiden'][BASE][a]['mean']) for a in ASPECTS},
                 old5000={a: abs(entries[BASE]['old5000'][a] - frozen['old5000'][BASE][a]) for a in ASPECTS})
    record = dict(stage='reported_ab', smoke=bool(smoke), reference=BASE, p_sha256=p_sha, arms=[BASE] + list(va['arms']),
                  a_outcome=va['outcome'], v_star_a=va['winner'], A_genre_winner=va['A_genre_winner'], entries=entries,
                  references=dict(t_g=V.T_G, promising=S.PROMISING, probe_genre_ridge_p=float(probe),
                                  instructed_genre_image_p=dict(refs['genre_word']), default_image_p=dict(refs['default'])),
                  base_vs_12_24=check, stage_b=None if vb is None else stage_b_reading(vb),
                  **loss_scale(outputs, va['arms'], smoke),
                  leiden=dict(graph='union 20-NN cosine', resolution=1.0, seeds=[0, 1, 2]),
                  label_counts_p={a: int((ctx['labels'][a] >= 0).sum()) for a in ASPECTS},
                  label_counts_old5000={a: int((old_labels[a] >= 0).sum()) for a in ASPECTS}, versions=rb_ami.versions(),
                  **S.code_state(), report_sha256=S._file_sha256(Path(__file__)),
                  identity=dict(p_sha256=p_sha, stage_a_verdict_sha256=S._file_sha256(va_path),
                                stage_b_verdict_sha256=S._file_sha256(vb_path) if vb is not None else None,
                                report_sha256=S._file_sha256(Path(__file__))))
    record = json.loads(json.dumps(record, allow_nan=False))
    path = outputs / 'reported_ab.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['identity'] == record['identity'], f'{path} exists with other inputs; move it aside'
        print(f'{path}: exists with the same inputs, kept', flush=True)
        record = old
    else:
        S._json_dump(path, record)
    md = outputs / 'reported_ab.md'
    text = markdown(record) + '\n'
    if md.exists():
        assert md.read_text() == text, f'{md} differs from the record; move it aside'
    else:
        md.write_text(text)
    return record


# ------------------------------------------------------------------ markdown (4 decimals)
def _f(x, d=4):
    return 'n/a' if x is None else x if isinstance(x, str) else f'{x:.{d}f}'


def _term_cells(e):
    """'term mean' for each term of one epoch row of term means; a null (non-finite) mean shows as non-finite, with the
    number of training calls where the term was non-finite (rg_train's nonfinite_counts)."""
    counts = e.get('nonfinite_counts') or {}
    keys = [k for k in e if k not in ('epoch', 'calls', 'pairs', 'nonfinite_counts')]
    return ', '.join(f'{k} ' + ('non-finite' if e[k] is None else _f(e[k]))
                     + (f' ({counts[k]} non-finite calls)' if counts.get(k) else '') for k in sorted(keys))


def _terms_row(name, t):
    if t is None:
        return [f'| {name} | no term means in run.json | | |']
    return [f"| {name} | {e['epoch']} | " + _term_cells(e) + f" | {e['calls']} |" for e in t['epochs']]


def markdown(r):
    asp = list(ASPECTS)
    ent = r['entries']
    arms = r['arms']
    ok = [v for v in arms if ent[v]['status'] == 'ok']
    head = '| arm | ' + ' | '.join(asp) + ' |\n|---|' + '---|' * len(asp)
    lines = ['# Stages A and B: reported readings (decide nothing)', '',
             f"P sha256 {r['p_sha256']}{'; SMOKE RUN' if r['smoke'] else ''}. Seed 0 unless stated; reference {r['reference']}. "
             f"Stage A outcome {r['a_outcome']}; v*_A {r['v_star_a']} (A_genre {_f(r['A_genre_winner'])}). "
             f"Labelled on P: {r['label_counts_p']}; on the old 5,000: {r['label_counts_old5000']}.", '',
             '## Stage A, three aspects on P: A (e against R6-5 [95% interval]); intervals are the verdict file\'s', '', head]
    for v in arms:
        e = ent[v]
        if e['status'] != 'ok':
            lines.append(f"| {v} | failed run: {e['cause']} |  |  |")
        elif v == r['reference']:
            lines.append(f'| {v} | ' + ' | '.join(_f(e['p'][a]['A']) for a in asp) + ' |')
        else:
            lines.append(f'| {v} | ' + ' | '.join(
                f"{_f(e['p'][a]['A'])} ({e['p'][a]['e']:+.4f} [{e['p'][a]['lo']:+.4f}, {e['p'][a]['hi']:+.4f}])" for a in asp) + ' |')
    ref = r['references']
    lines += ['', '## Level against T_g (genre A on P)', '',
              f"T_g {_f(ref['t_g'])}; T_g - r_genre {_f(ref['promising'])}. At target: A_genre >= T_g; promising: T_g - r_genre <= A_genre < T_g.", '',
              '| arm | A_genre | level | guarded |', '|---|---|---|---|']
    lines += [f"| {v} | {_f(ent[v]['A_genre'])} | {ent[v]['level']} | {ent[v]['guarded']} |" for v in arms if v != r['reference'] and ent[v]['status'] == 'ok']
    lines += ['', '## The old 5,000: A per aspect (k-means, as part 1)', '', head]
    lines += [f'| {v} | ' + ' | '.join(_f(ent[v]['old5000'][a]) for a in asp) + ' |' for v in ok]
    lines += ['', '## Leiden AMI on P (union 20-NN cosine graph, resolution 1, mean of seeds 0 to 2)', '', head]
    lines += [f'| {v} | ' + ' | '.join(_f(ent[v]['leiden'][a]['mean']) for a in asp) + ' |' for v in ok]
    lines += ['', 'The old-5,000 and Leiden readings cover the seed-0 arms only. GR-2 and GR-4 have no seed-0 run in this round, so they get '
              'neither (rule §4.1 reports them for stage A; the reported list of rule §4.2 has no such reading for stage B, and the rule governs over spec §3.6).']
    lines += ['', '## Best epochs (seed 0)', '', head]
    lines += [f'| {v} | ' + ' | '.join(str(ent[v]['best_epoch'][a]) if ent[v]['best_epoch'] else '-' for a in asp) + ' |' for v in arms]
    lines += ['', '## References', '',
              f"Ceiling probe (ridge, genre, P) {_f(ref['probe_genre_ridge_p'])}.", '', head,
              '| T_g: instructed genre image | ' + ' | '.join(_f(ref['instructed_genre_image_p'][a]) for a in asp) + ' |',
              '| default image | ' + ' | '.join(_f(ref['default_image_p'][a]) for a in asp) + ' |']
    b = r['stage_b']
    if b is None:
        lines += ['', '## Stage B', '', 'No stage_b/verdict.json yet: not reported.']
    else:
        lines += ['', f"## Stage B: seeds 1 to 3, Abar per arm (e against R6-5 [95% interval]); outcome {b['outcome']}, v_held {b['v_held']}", '',
                  '| arm | ' + ' | '.join(asp) + ' | C level | Abar_genre against T_g |', '|---|' + '---|' * (len(asp) + 2)]
        for n, x in b['arms'].items():
            if x['status'] != 'ok':
                lines.append(f"| {n} | failed run: {x['cause']} |  |  |  |  |")
                continue
            g = x['genre_vs_targets']
            lines.append(f'| {n} | ' + ' | '.join(
                f"{_f(x['effects'][a]['Abar'])} ({x['effects'][a]['e']:+.4f} [{x['effects'][a]['lo']:+.4f}, {x['effects'][a]['hi']:+.4f}])"
                for a in asp) + f" | {x['c_outcome']} | {g['diff_t_g']:+.4f} ({'reaches' if g['reaches_t_g'] else 'below'}) |")
        first = next((y for y in b['arms'].values() if y['status'] == 'ok'), None)
        if first:
            lines += ['', "R6-5's Abar per aspect: " + ', '.join(f"{a} {_f(first['effects'][a]['Abar_ref'])}" for a in asp) + '.']
        for key, label in (('spread_confirmed', 'confirmed arms'), ('spread_scored', 'every scored arm')):
            s = b[key]
            lines.append(f"Seed range: {label} " + ('(fewer than two)' if s is None else
                         f"{s['arms']}: Abar_genre spread {_f(s['spread'])} against r_genre {_f(b['r_genre'])}: "
                         f"{'within one seed range of each other' if s['within_one_range'] else 'further apart than one seed range'}.") + '')
        lines += ['', f"Confirmed arms within one seed range of v_held: {b['within_one_range_of_v_held'] or 'none'}. The pick follows the rule (user, K2b).",
                  '', '### Best epochs on the fresh seeds', '', '| arm | seed | ' + ' | '.join(asp) + ' | rerun of |', '|---|---|' + '---|' * (len(asp) + 1)]
        for n, by in b['runs'].items():
            for s, x in sorted(by.items()):
                cells = [str(x['best_epoch'][a]) for a in asp] if x['status'] == 'ok' else [f"failed: {x['cause']}"] + [''] * (len(asp) - 1)
                lines.append(f'| {n} | {s} | ' + ' | '.join(cells) + f" | {x.get('rerun_of', '-')} |")
    ls = r['loss_scale']
    name = ls.get('variant', V.LOSS_SCALE)
    lines += ['', '## Loss scale (spec §3.6)', '',
              f"Loss-scale run {name} (relational weight 1, seed 0, one epoch, no export on P); status {ls.get('status')}. "
              'Mean over the epoch\'s training calls of each term, before its weight (direct term enters at 0.25, relational term at its lambda).']
    lines += ['', *[f'Loss-scale problem: {p}.' for p in ls['problems']]] if ls['problems'] else []
    lines += [f'Recorded {k}: {v}.' for k, v in sorted(ls.items()) if 'finite' in k]
    if ls['epoch_means']:
        lines += ['', '| run | epoch | term means | calls |', '|---|---|---|---|']
    else:
        lines += ['', 'No epoch means are reported.']
    for e in ls['epoch_means']:
        lines.append(f"| {name} | {e.get('epoch')} | " + _term_cells(e) + f" | {e.get('calls')} |")
    lines += ['', 'Genre term means per arm (GR-8 from regression/va1_GR-8_s0; GQ-8\'s anchors term beside GR-8\'s relational term):', '',
              '| arm | epoch | term means | calls |', '|---|---|---|---|']
    for arm in r['arms']:
        if arm != r['reference']:
            lines += _terms_row(arm, r['arm_term_means'][arm])
    eq = r['eq']
    lines += ['', f"## EQ (anchors loss equals the relational loss when the anchors are the batch): {'pass' if eq['pass'] else 'FAIL'}, tolerance {eq['tolerance']}", '',
              '| seed | pairs | relational | anchors | difference |', '|---|---|---|---|---|']
    lines += [f"| {i['seed']} | {i['pairs']} | {_f(i['relational'], 6)} | {_f(i['anchors'], 6)} | {_f(i['difference'], 8)} |" for i in eq['items']]
    lines += ['', f"Base R6-5 here against 12-24's reported.json (max abs difference): {r['base_vs_12_24']}."]
    return '\n'.join(lines)
