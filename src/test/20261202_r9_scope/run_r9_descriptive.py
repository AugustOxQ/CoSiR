"""Ticket 23: rule §8 after a verdict, never a decision or an overwrite.

    python run_r9_descriptive.py --read H6
    python run_r9_descriptive.py --read H8 --smoke --smoke-dir /path/to/held/smoke [--smoke-tag NAME]

Real mode requires the verdict's phase-2 agreement and its exact pass. Smoke
requires a smoke pass and binds any present verdict to it, uses selection rows and
writes only to this worktree's results/smoke/descriptive. --smoke-tag adds a fresh output suffix, never replacing
an earlier file. Fix paths come from the verdict; --reserve selects
the reserve verdict and writes descriptive_<read>_reserve.json. Missing paired
reads are absent. All input hashes and reproduction checks precede the write.
Exit 4: refused; exit 5: contradictory inputs. Console: file names and hashes only.
"""
import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R
import r9_backbone as BB
import r9_descriptive as D
import r9_episodes as E
import r9_picks as P
import r9_score as S
import r9_stats as ST
import r9_a28_data as AD
import r9_a28_episodes as AE
import run_r9_apply_rule as AR
import run_r9_held as RH
import numpy as np

class Refused(RH.Refused):
    """An owned refusal boundary name, safe for the console."""


require = D.require
# Stage 1's near_duplicates_a28.json: in no pins file; the same value as rederive/rd9_inputs.py (fx-c13) and run_r9_held.py.
NEAR_DUPLICATES_SHA256 = RH.NEAR_DUPLICATES_SHA256  # one pin, in the held runner (rederive keeps its own copy; a test compares)


def refuse_unless(ok, boundary):
    if not ok:
        raise Refused(boundary)


def read_json(path):
    raw = Path(path).read_bytes()
    rec = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    require(isinstance(rec, dict), 'JSON object: ' + str(path))
    return rec


def source(path):
    return {'file': str(path), 'sha256': R.sha256_file(path)}


def bind_read(out, read, smoke, reserve=False):
    """No held arrays loaded until a run verdict and its agreement are established."""
    out = Path(out)
    base = RH.read_names(read, 'reserve' if reserve else 'smoke' if smoke else 'held')
    vpath, verdict, agr = out / base.verdict, None, None
    sources = {}
    if smoke:
        names = base
    else:
        refuse_unless(vpath.is_file(), 'verdict missing: ' + str(vpath))
        verdict = read_json(vpath)
        allowed = ('full claim', 'small claim', 'NO-GO') if read in RH.A28_READS else ('GO', 'NO-GO')
        refuse_unless(verdict.get('status') == 'run' and verdict.get('verdict') in allowed,
                      'read has no run verdict: ' + str(vpath))
        refuse_unless(verdict.get('rule_sha256') == R.RULE_SHA256 and verdict.get('read') == read
                      # The apply step writes the ledger name (run_r9_apply_rule.READS), the pass the code name.
                      and verdict.get('backbone') == RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]
                      and verdict.get('mode') == 'held'
                      and verdict.get('variant') == ('reserve' if reserve else None), 'verdict identity')
        variants = ('reserve',) if reserve else ('held', 'fix1')
        matched = [RH.read_names(read, kind) for kind in variants
                   if RH.read_names(read, kind).pass_ == verdict.get('pass_file')]
        refuse_unless(len(matched) == 1, 'verdict pass filename')
        names = matched[0]
        refuse_unless(verdict.get('fix_pass') is (names.kind == 'fix1'), 'verdict fix flag')
        suffix = '_fix1' if names.kind == 'fix1' else '_reserve' if reserve else ''
        aname = f'rd9_{read}_agreement{suffix}.json'
        refuse_unless(verdict.get('agreement_file') == aname, 'verdict agreement filename')
        # Share the apply step's canonical location; its verdict stores only the basename.
        candidates = [out / aname, AR.AGREEMENT_DIR / aname]
        apath = next((p for p in candidates if p.is_file() and R.sha256_file(p) == verdict.get('agreement_sha256')), None)
        refuse_unless(apath is not None, 'agreement missing or SHA mismatch')
        agr, sha = AR.load_agreement(apath, False, names.pass_, read)
        refuse_unless(sha == verdict['agreement_sha256'], 'agreement SHA binding')
        require(verdict.get('agreement') == {k: agr.get(k) for k in
                ('phase', 'smoke', 'agreement', 'read', 'pass_file', 'held_pass_sha256', 'compared')}, 'embedded agreement')
        sources.update({'verdict': source(vpath), 'agreement': source(apath)})
    ppath = out / names.pass_
    refuse_unless(ppath.is_file(), 'pass missing: ' + str(ppath))
    rec = AR.load_pass(ppath.read_bytes(), ppath.name, read, smoke)
    require(rec.get('read') == read and rec.get('kind') == names.kind, 'pass read/kind')
    backbone = rec.get('backbone')
    require(backbone in BB.BACKBONES and (smoke or backbone == RH.READ_BACKBONES[read]), 'pass backbone')
    if smoke and vpath.is_file():
        verdict = read_json(vpath)
        allowed = ('full claim', 'small claim', 'NO-GO') if read in RH.A28_READS else ('GO', 'NO-GO')
        refuse_unless(verdict.get('status') == 'run' and verdict.get('verdict') in allowed
                      and verdict.get('rule_sha256') == R.RULE_SHA256 and verdict.get('read') == read
                      and verdict.get('backbone') == RH.LEDGER_NAMES[backbone]
                      and verdict.get('mode') == 'smoke' and verdict.get('variant') is None
                      and verdict.get('pass_file') == names.pass_ and verdict.get('fix_pass') is False,
                      'smoke verdict identity')
        refuse_unless(R.sha256_file(ppath) == verdict.get('held_pass_sha256'), 'pass SHA binding')
        require(verdict.get('seeds') == rec['seeds'], 'verdict seeds')
        sources['verdict'] = source(vpath)
    if not smoke:
        refuse_unless(R.sha256_file(ppath) == verdict.get('held_pass_sha256') == agr['held_pass_sha256'], 'pass SHA binding')
        require(verdict.get('seeds') == rec['seeds'], 'verdict seeds')
        require(all(agr['rederived_counts'][c] == rec['checks'][c]['n'] for c in AR.P_CHECKS), 'agreement counts')
    sources['pass'] = source(ppath)
    return SimpleNamespace(out=out, read=read, smoke=smoke, names=names, rec=rec, backbone=backbone,
                           verdict=verdict, vpath=vpath, sources=sources)


def load_bound_inputs(bound):
    r, out, names = bound.rec, bound.out, bound.names
    npp = R.N_SMOKE if bound.smoke else R.N_PER_PAIR
    spath = out / names.started
    attempts = RH.read_attempts(spath)
    attempt = [a for a in attempts if a.get('attempt') == r.get('attempt')]
    require(len(attempt) == 1, 'producing started attempt')
    attempt = attempt[0]
    for k in ('mode', 'read', 'backbone', 'runner_sha256', 'module_sha256', 'input_sha256',
              'pins_sha256', 'coef_sha256', 'episodes_sha256'):
        require(attempt.get(k) == r.get(k), 'started/pass binding: ' + k)
    require(bool(r.get('coef_sha256')), 'coefficient hashes missing')
    apath = out / names.arrays
    require(R.sha256_file(apath) == r.get('outputs', {}).get(apath.name), 'arrays/pass SHA binding')
    arrays, core = D.load_arrays(apath, r['seeds'], npp)
    check = D.check_pass(core, r, bound.read in RH.A28_READS)
    eps = []
    for seed, x in zip(r['seeds'], core):
        path = out / RH.episodes_name(bound.read, seed)
        ep = E.load_episodes(path)
        require(ep.seed == seed and ep.n == len(R.PAIRS) * npp and ep.sha == r['episodes_sha256'][str(seed)]
                and np.array_equal(ep.pair_index, x['pair_index']), 'episodes/pass alignment: ' + str(path))
        bound.sources[path.name] = source(path)
        eps.append(ep)
    sens_path = out / names.sensitivity
    require(R.sha256_file(sens_path) == r['outputs'].get(sens_path.name), 'sensitivity/pass SHA binding')
    sensitivity = read_json(sens_path)
    require(sensitivity.get('seeds') == r['seeds'] and sensitivity.get('episodes_sha256') == r['episodes_sha256'],
            'sensitivity episodes binding')
    if bound.verdict is not None:
        require(bound.verdict.get('sensitivity_file') == sens_path.name
                and bound.verdict.get('sensitivity_sha256') == R.sha256_file(sens_path), 'verdict sensitivity binding')
    bound.sources.update({'arrays': source(apath), 'started': source(spath), 'sensitivity': source(sens_path)})
    return SimpleNamespace(arrays=arrays, core=core, eps=eps, attempt=attempt, npp=npp, checks=check)


def score_inputs(bound, loaded, on_file=None):
    """Re-fit/check the pinned heads, rebuild each seed, re-score with include_pm=True."""
    backbone = bound.backbone
    input_shas = R.assert_inputs(backbone)
    RH.check_pins(backbone)
    require(R.sha256_file(R.pins_path(backbone)) == bound.rec['pins_sha256'], 'backbone pins/pass binding')
    require(all(bound.rec['input_sha256'].get(k) == v for k, v in input_shas.items()), 'input pins/pass binding')
    paths = BB.parts(backbone)
    picks = P.load_picks(paths['picks_seed42'], backbone)
    lambdas = P.frozen_lambdas(backbone)
    cells = S.cells_from_record(P.read_cells(paths['cells_seed42']))
    taus = S.taus_of(backbone)
    env = RH.setup(backbone)
    coef = RH.head_guard(env, RH.refit_reference(backbone))
    require(coef == bound.rec['coef_sha256'] == loaded.attempt['coef_sha256'], 'rebuilt coefficients/pass binding')
    if bound.read in RH.A28_READS:
        require(R.sha256_file(AD.PINS_PATH) == bound.rec['pins_a28_sha256'], 'A28 pins/pass binding')
        files = AD._pin_files(AD.PINS_PATH)
        for key in ('results/value_sets_a28.json', 'results/language_flags_a28.json'):
            AD._pinned(files, key)
        data = RH.setup_a28(env, backbone, bound.smoke)
        require(data.value_record == bound.rec['value_sets'], 'A28 value sets/pass binding')
    else:
        checked = RH.value_sets_guard(env, RH.value_sets_file_guard())
        require(checked == bound.rec['value_sets'], 'ArtELingo value sets/pass binding')
        data = None
    if data is not None:
        require(data.a28.read_set_sha256 == bound.rec['read_set_sha256'], 'A28 row space/pass binding')
    inputs = []
    for seed, ep, core in zip(bound.rec['seeds'], loaded.eps, loaded.core):
        if data is not None:
            rebuilt = AE.build_seed_a28(data.prepared, data.values, seed, loaded.npp)
            part = RH.a28_episode_record(rebuilt)
            require(part == loaded.attempt['a28_episodes'].get(str(seed)), 'A28 rebuilt part records')
            require(rebuilt.sha == ep.sha, 'A28 rebuilt hashes')
            bundle = RH.a28_seed_bundle(env, data, rebuilt, loaded.npp, backbone, coef, bound.smoke)
            groups, emotion = data.a28.painting_index, data.a28.emotion
        else:
            bundle = RH.read_seed_bundle(env, 'selection' if bound.smoke else 'held', seed, loaded.npp, backbone)
            groups, emotion = env.split.groups, env.labels['emotion']
        require(bundle.episodes_sha256 == ep.sha and all(np.array_equal(getattr(bundle.pooled, f), getattr(ep.pooled, f))
                for f in E.FIELDS), 'rebuilt episode fields')
        require(np.array_equal(core['cl'], np.asarray(groups)[ep.pooled.anchor]), 'anchor cluster row join')
        directions, collect = D.direction_rows()
        if bound.smoke and data is not None:
            # These are the held runner's synthetic SELECTION episodes, not a real A28 read.
            # The scorer's admitted selection seed has identical assembly and no verdict gate.
            require(bundle.smoke is True and seed in R.SMOKE_SEEDS, 'synthetic smoke scoring scope')
            bundle = SimpleNamespace(**{**vars(bundle), 'mode': 'selection'})
        sc = S.score_seed(bundle, picks, lambdas, env.readers, include_pm=True, cells=cells, taus=taus,
                          verdict=None if bound.smoke else bound.vpath, on_scores=collect)
        # Reconstruct ALL saved primary/raw arrays, including dtype, shape, gates and reader inputs.
        primary = {k: sc[k] for k in S.PRIMARY_SCORERS}
        primary.update({'gates': {'aff': sc['gates']['aff']}, 'reader': sc['reader'],
                        'cl': sc['cl'], 'pair_index': sc['pair_index']})
        got = RH.read_arrays([primary])
        i = bound.rec['seeds'].index(seed)
        D.check_rebuilt(got, loaded.arrays, slice(i * ep.n, (i + 1) * ep.n), ep.n)
        D.check_directions(sc, directions)
        inputs.append({'seed': seed, 'cl': sc['cl'], 'pair_index': sc['pair_index'],
                       'pa': {s: sc[s] for s in S.ALL_SCORERS}, 'directions': directions,
                       'gates': sc['gates'], 'reader': sc['reader']})
        del bundle, sc
        gc.collect()
        if on_file is not None:
            on_file(bound.sources[RH.episodes_name(bound.read, seed)])
    frozen = {'picks': picks, 'lambdas': lambdas, 'cells': cells, 'taus': taus}
    return inputs, env, data, frozen


# H5's R6 held_arrays.npz layout, read without importing any round-6 module (no r9 module may; ticket fx-desc).
# Copies of r6_score.CORE_SCORERS, r6_descriptive.ARRAY_EXTRA and the two functions below
# (r6_descriptive.load_core, check_pass); the statistics are r9_stats', a copy of r6_stats' (checked equal on H5's files).
H5_CORE = ('cosine', 'rca', 'B', 'B0', 'B1', 'aff_fused', 'aff_cf', 'r1_fused')
H5_EXTRA = ('cl', 'pair_index', 'seed_index')


def h5_load_core(path, seeds, n_per_pair):
    """held_arrays.npz of the R6 read split per seed: {seed: {cl, pair_index, pa {scorer: {metric: float64}}}}."""
    seeds = [int(s) for s in seeds]
    n_seed = len(R.PAIRS) * int(n_per_pair)
    want = tuple(f'{s}__{m}' for s in H5_CORE for m in D.METRICS) + H5_EXTRA
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == set(want), 'H5 arrays keys')
        d = {k: z[k] for k in want}
    n = len(seeds) * n_seed
    require(all(d[k].shape == (n,) for k in want), 'H5 arrays shape')
    require(all(d[k].dtype == np.int64 for k in H5_EXTRA), 'H5 arrays index dtype')
    require(np.array_equal(d['seed_index'], np.repeat(np.arange(len(seeds), dtype=np.int64), n_seed)), 'H5 arrays seed order')
    pi_seed = np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), int(n_per_pair))
    out = {}
    for i, seed in enumerate(seeds):
        sl = slice(i * n_seed, (i + 1) * n_seed)
        require(np.array_equal(d['pair_index'][sl], pi_seed), 'H5 arrays pair order')
        pa = {}
        for s in H5_CORE:
            pa[s] = {}
            for m in D.METRICS:
                x = d[f'{s}__{m}'][sl]
                require(x.dtype == np.float64 and bool(np.isfinite(x).all()), 'H5 arrays float64: ' + f'{s}__{m}')
                pa[s][m] = np.ascontiguousarray(x)
        out[seed] = {'cl': np.ascontiguousarray(d['cl'][sl]), 'pair_index': np.ascontiguousarray(d['pair_index'][sl]),
                     'pa': pa}
    return out


def h5_check_pass(core, rec):
    """The H5 arrays reproduce every P1 to P7, S1 and S2 count, point, interval and quantity of its pass file."""
    seeds = [int(s) for s in rec['seeds']]
    diffs = ST.check_diffs([{'cl': core[s]['cl'], **core[s]['pa']} for s in seeds], True)
    cl = diffs['P1'][1]
    require(rec.get('n_episodes') == len(cl) and rec.get('n_clusters') == len(np.unique(cl)), 'H5 arrays episode counts')
    for c in ST.CHECKS + ST.SECONDARY:
        want = (rec['checks'] if c in ST.CHECKS else rec['secondary'])[c]
        d, cl_c = diffs[c]
        b, int_le0 = ST.bootstrap_draws(d, cl_c)
        pc = ST.point_ci_from_draws(d, b)
        require(int(int_le0.sum()) == want['n'] and pc['point'] == want['point'] and pc['ci95'] == list(want['ci95'])
                and ST.QUANTITIES[c][2] == want['quantity'], 'H5 arrays reproduce pass: ' + c)


def h5_inputs(smoke):
    """H5's read-only R6 layout, its own rule/agreement/pass reproduction."""
    out = R.R6D / 'results'
    if smoke:
        out /= 'smoke'
        ppath = out / 'held_pass.json'
        if not ppath.is_file():
            return None
        rec = read_json(ppath)
        require(rec.get('mode') == 'smoke' and rec.get('rule_sha256') == R.R6_RULE_SHA256
                and rec.get('seeds') == list(R.SMOKE_SEEDS), 'H5 smoke pass identity')
        suffix = ''
        sources = {}
    else:
        vpath = out / 'held_verdict.json'
        if not vpath.is_file():
            return None
        verdict = read_json(vpath)
        require(verdict.get('mode') == 'held' and verdict.get('rule_sha256') == R.R6_RULE_SHA256
                and verdict.get('verdict') in ('GO', 'NO-GO'), 'H5 verdict identity')
        allowed = {'held_pass.json': '', 'held_pass_fix1.json': '_fix1'}
        require(verdict.get('pass_file') in allowed, 'H5 pass filename')
        suffix = allowed[verdict['pass_file']]
        ppath = out / verdict['pass_file']
        require(verdict.get('agreement_file') == f'rederive_agreement{suffix}.json', 'H5 agreement filename')
        apath = out / verdict['agreement_file']
        agreement = read_json(apath)
        require(R.sha256_file(apath) == verdict.get('agreement_sha256') and agreement.get('phase') == 2
                and agreement.get('all_agree') is True and agreement.get('disagreements') == []
                and agreement.get('smoke') is False and agreement.get('pass_file') == ppath.name
                and R.sha256_file(ppath) == verdict.get('held_pass_sha256') == agreement.get('held_pass_sha256'),
                'H5 phase-2 pass binding')
        rec = read_json(ppath)
        require(rec.get('mode') == 'held' and rec.get('rule_sha256') == R.R6_RULE_SHA256
                and rec.get('seeds') == list(R.HELD_SEEDS), 'H5 pass identity')
        sources = {'verdict': source(vpath), 'agreement': source(apath)}
    apath = out / f'held_arrays{suffix}.npz'
    npp = R.N_SMOKE if smoke else R.N_PER_PAIR
    core = h5_load_core(apath, rec['seeds'], npp)
    h5_check_pass(core, rec)
    eps, inputs = [], []
    for seed in rec['seeds']:
        path = out / f'held_episodes_seed{seed}.npz'
        ep = E.load_episodes(path)
        require(ep.seed == seed and ep.n == len(R.PAIRS) * npp and ep.sha == rec['episodes_sha256'][str(seed)], 'H5 episodes/pass')
        require(np.array_equal(core[seed]['pair_index'], ep.pair_index), 'H5 pair alignment')
        inputs.append({'seed': seed, **core[seed]})
        eps.append(ep)
        sources[path.name] = source(path)
    sources.update({'pass': source(ppath), 'arrays': source(apath)})
    return SimpleNamespace(rec=rec, inputs=inputs, eps=eps, sources=sources)


def paired_reads(bound, inputs):
    reads = ('H5', 'H6', 'H7') if bound.read not in RH.A28_READS else ('H8', 'H9', 'H10')
    out = {}
    for read in reads:
        if read == bound.read:
            continue
        if read == 'H5':
            other = h5_inputs(bound.smoke)
            if other is None:
                out[read] = D.absent('paired read is missing')
                continue
        else:
            names = RH.read_names(read, 'smoke' if bound.smoke else 'held')
            gate = bound.out / (names.pass_ if bound.smoke else names.verdict)
            if not gate.is_file():
                out[read] = D.absent('paired read is missing')
                continue
            if not bound.smoke and read_json(gate).get('status') != 'run':
                out[read] = D.absent('paired read was not run')
                continue
            other_bound = bind_read(bound.out, read, bound.smoke)
            loaded = load_bound_inputs(other_bound)
            other = SimpleNamespace(rec=other_bound.rec, sources=other_bound.sources,
                                    inputs=[{'seed': s, 'pa': {k: c[k] for k in S.PRIMARY_SCORERS},
                                             **{k: c[k] for k in ('cl', 'pair_index')}}
                                            for s, c in zip(other_bound.rec['seeds'], loaded.core)])
        # Never pair merely by array position: the entire episodes (all members) must be identical.
        require(other.rec['seeds'] == bound.rec['seeds'] and
                other.rec['episodes_sha256'] == bound.rec['episodes_sha256'], 'cross-backbone episode hashes')
        out[read] = {'episode_hashes_equal': True, 'aff_paired': D.compare_backbones(inputs, other.inputs),
                     'sources': other.sources}
    return out


def near_counts(rec):
    """-> (path, counts) of stage 1's near-duplicates file, bound to the pass and to its pin, counts checked.

    The file is in no pins file (ticket 21c LOW-5): the pass's source SHA and this pin are its only binding."""
    path = Path(rec['near_duplicates_source']['file'])
    sha = R.sha256_file(path)
    require(sha == rec['near_duplicates_source']['sha256'] == NEAR_DUPLICATES_SHA256, 'near-duplicates source binding')
    near = read_json(path)
    require(all(type(near[k]) is int for k in ('cos_099', 'cos_098', 'cos_092'))
            and 0 <= near['cos_099'] <= near['cos_098'] <= near['cos_092'] <= 746
            and {k: near[k] for k in ('cos_099', 'cos_098')} == rec['near_duplicates'], 'near-duplicate counts')
    return path, near


def a28_report(bound, inputs, env, data, episodes):
    if data is None:
        return {'status': 'not applicable', 'reason': 'ArtELingo read'}
    files = AD._pin_files(AD.PINS_PATH)
    tab = AD.load_read_set()
    fpath = AD._pinned(files, 'results/language_flags_a28.json')
    flags = read_json(fpath)
    cpath = AD._pinned(files, f'translations/{AD.MERGED_NAME}/coverage.json')
    coverage = read_json(cpath)
    require(set(flags['languages']) == set(coverage) == set(AE.LANGUAGES), 'translation language alignment')
    require(flags['flagged'] == bound.rec['flagged_languages'], 'flags/pass binding')
    near_path, near = near_counts(bound.rec)
    prepared = AE.prepare(tab)
    language = [np.asarray(data.a28.language)[ep.pooled.anchor] for ep in episodes]
    require(all(x.shape == d['cl'].shape for x, d in zip(language, inputs)), 'language/episode alignment')
    set_aside = [{'language': lang, 'pair': pair, 'seed': int(seed)}
                 for seed, rec in loaded_parts(bound).items() for pair, langs in rec['set_aside'].items() for lang in langs]
    require(set_aside == bound.rec['set_aside'], 'set-aside/pass binding')
    # H9/H10 use H8's episode disclosures; compare when that reference exists.
    h8_path = bound.out / RH.read_names('H8').started
    if bound.read != 'H8' and not bound.smoke:
        h8 = bind_read(bound.out, 'H8', False)
        require(h8.rec['set_aside'] == set_aside and RH.a28_attempt_records(RH.read_attempts(h8_path)) == loaded_parts(bound),
                'H8 disclosure alignment')
    languages = {}
    for i, lang in enumerate(AE.LANGUAGES):
        r = prepared.R[i]
        kept = prepared.kept[i]
        labels = prepared.labels[i]
        pool = kept[np.all(np.stack([labels[a][kept] >= 0 for a in R.ASPECTS]), axis=0)]
        thin = {}
        for a in R.ASPECTS:
            thin[a] = []
            for value in data.value_record[a]:
                count = len(np.unique(prepared.groups[i][pool[labels[a][pool] == value['code']]]))
                if count < 30:
                    thin[a].append({**value, 'n_paintings': int(count)})
        _, counts = np.unique(prepared.groups[i][kept], return_counts=True)
        require(len(counts) > 0 and coverage[lang]['rows'] == flags['languages'][lang]['rows'], 'language coverage/rows')
        languages[lang] = {'median_rows_per_painting': float(np.median(counts)), 'values_on_fewer_than_30_paintings': thin,
                           'note': 'thin languages; eligibility of 30 paintings applies to the pooled pool only (rule §5.1)',
                           'episodes': D.scope(inputs, [x == i + 1 for x in language]),
                           'translation_coverage': coverage[lang], 'language_flags': flags['languages'][lang],
                           'read_rows_before_drops': int(len(r)), 'read_rows_kept': int(len(kept))}
    exclude = [AE.LANGUAGES.index(lang) + 1 for lang in flags['flagged']]
    unflagged = D.scope(inputs, [~np.isin(x, exclude) for x in language])
    # The smoke's synthetic fixture has held=False throughout, as in run_r9_held.setup_a28.
    held = np.zeros(data.a28.n, dtype=bool) if bound.smoke else np.asarray(data.a28.held)
    hmask = [held[ep.pooled.anchor] for ep in episodes]
    require(len(tab.painting.unique()) == 746 and tab[tab.held].painting.nunique() == 248
            and tab[~tab.held].painting.nunique() == 498, 'A28 painting partition')
    h5 = h5_inputs(bound.smoke)
    h5_dist = D.absent('H5 read is missing') if h5 is None else D.emotion_distribution(h5.eps, env.labels['emotion'])
    if h5 is not None:
        groups = np.asarray(env.split.groups)
        require(all(np.array_equal(x['cl'], groups[ep.pooled.anchor]) for x, ep in zip(h5.inputs, h5.eps)), 'H5 anchor row join')
    return {'per_language': languages,
            'per_language_note': ('Smoke episodes use the synthetic language column (ArtELingo selection row ID mod 27); '
                                  'median rows, thin values, flags and translation coverage use real stage-1 data.'
                                  if bound.smoke else 'Episodes use each language\'s real read rows; metadata come from stage-1 files.'),
            'without_flagged_languages': {'excluded': flags['flagged'], **unflagged},
            'new_against_held': {'painting_universe': {'new': 498, 'held': 248},
                                'note': 'synthetic selection fixture: held=False' if bound.smoke else 'episodes by anchor painting',
                                'new': D.scope(inputs, [~x for x in hmask]), 'held': D.scope(inputs, hmask)},
            'anchor_emotion_distribution': {'emotion_labels': list(R.value_names(env.data)['emotion']),
                                           'this_read': D.emotion_distribution(episodes, data.a28.emotion), 'H5': h5_dist,
                                           'H5_sources': None if h5 is None else h5.sources},
            'set_aside': set_aside, 'set_aside_reasons': {s: x['set_aside_reasons'] for s, x in loaded_parts(bound).items()},
            'flagged_languages': flags['flagged'],
            'near_duplicates': {k: near[k] for k in ('cos_099', 'cos_098', 'cos_092')},
            'near_duplicates_note': 'CLIP image version gap: the zip contains re-encoded 600 px copies; same-painting '
                                    'cosines span 0.92–0.99 (rule §3.3). cos_092 is disclosed beside cos_099/cos_098; none is excluded.',
            'stage1_sources': {'read_set': source(AD._pinned(files, 'results/read_set_a28.parquet')),
                               'flags': source(fpath), 'coverage': source(cpath), 'near_duplicates': source(near_path)}}


def loaded_parts(bound):
    attempts = RH.read_attempts(bound.out / bound.names.started)
    mine = [a for a in attempts if a['attempt'] == bound.rec['attempt']]
    require(len(mine) == 1 and set(mine[0]['a28_episodes']) == set(map(str, bound.rec['seeds'])), 'A28 started seed records')
    return mine[0]['a28_episodes']


def stage0b(bound):
    paths = [bound.out / 'stage0b.json'] if bound.smoke else [R.RESULTS / 'stage0b.json',
             R.MAIN / HERE.relative_to(R.CHECKOUT) / 'results/stage0b.json']
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return D.absent('stage0b.json is missing')
    rec = read_json(path)
    require(rec.get('rule_sha256') == R.RULE_SHA256 and rec.get('scored') is True
            and isinstance(rec.get('item3'), dict) and bool(rec['item3']), 'stage0b record')
    counts = {}
    for chain, values in rec['item3'].items():
        counts[chain] = {k: values[k] for k in ('changed_episodes', 'cpu_changed_episodes')}
        count = counts[chain]['changed_episodes']
        require(type(count['n_episodes']) is int and count['n_episodes'] > 0 and type(count['any_a3_fed']) is int
                and isinstance(count['per_scorer'], dict)
                and all(type(n) is int and 0 <= n <= count['n_episodes'] for n in
                        [count['any_a3_fed'], *count['per_scorer'].values()]), 'stage0b count shape')
        # The CPU field is a SUM of per-scorer counts, not another union/count record.
        cpu = counts[chain]['cpu_changed_episodes']
        require(type(cpu) is int and 0 <= cpu <= sum(count['per_scorer'].values()), 'stage0b CPU count shape')
    return {'source': source(path), 'case': rec['case'], 'passed': rec['passed'], 'chains': counts}


def run(read, smoke=False, smoke_dir=None, reserve=False, on_file=None, smoke_tag=None):
    refuse_unless(smoke == (smoke_dir is not None), '--smoke requires --smoke-dir; held mode does not take it')
    refuse_unless(not (smoke and reserve), 'reserve is a held mode')
    refuse_unless(smoke_tag is None or (smoke and RH.SUBDIR_RE.fullmatch(smoke_tag)),
                  'smoke tag requires smoke mode and a plain name')
    out = Path(smoke_dir).resolve() if smoke else R.RESULTS
    target_dir = R.RESULTS / 'smoke/descriptive' if smoke else R.RESULTS
    suffix = f'_{smoke_tag}' if smoke_tag is not None else '_reserve' if reserve else ''
    target = target_dir / f'descriptive_{read}{suffix}.json'
    refuse_unless(not target.exists(), 'output exists: ' + str(target))
    bound = bind_read(out, read, smoke, reserve)
    loaded = load_bound_inputs(bound)
    if on_file is not None:
        on_file(bound.sources['pass'])
        on_file(bound.sources['arrays'])
    inputs, env, data, frozen = score_inputs(bound, loaded, on_file)
    report = {'what': 'rule §8 items 1–6: after the verdict; decides nothing', 'read': read,
              'mode': 'smoke' if smoke else 'held', 'backbone': bound.backbone, 'variant': bound.names.kind,
              'seeds': bound.rec['seeds'], 'units': 'R@1 points; gate/pick shares in percent; emotion shares in fractions',
              **D.describe(inputs, frozen['cells']), 'aff_across_backbones': paired_reads(bound, inputs),
              'artelingo28': a28_report(bound, inputs, env, data, loaded.eps), 'stage0b_changed_episodes': stage0b(bound),
              'frozen': frozen, 'coef_sha256': bound.rec['coef_sha256'],
              'consistency': {'arrays_reproduce_pass': loaded.checks, 'rebuilt_primary_and_raw_arrays_equal': True,
                              'episode_fields_equal': True, 'direction_averages_equal': True},
              'sources': bound.sources, 'rule_sha256': R.RULE_SHA256,
              'runner_sha256': R.sha256_file(Path(__file__).resolve()), 'module_sha256': R.r9_module_shas(),
              'time': R.amsterdam_now()}
    payload = json.dumps(RH._plain(report), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    target_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation; serialize and validate every boundary before touching the output.
    with target.open('x', encoding='utf-8') as f:
        f.write(payload)
    return target, bound.sources


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--read', choices=tuple(RH.READ_BACKBONES), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-dir', type=Path)
    p.add_argument('--reserve', action='store_true')
    p.add_argument('--smoke-tag', help='fresh output suffix under results/smoke/descriptive; smoke only')
    args = p.parse_args(argv)
    console = sys.stdout

    def on_file(entry):
        print(f'{entry["file"]} {entry["sha256"]}', file=console, flush=True)

    # Upstream setup/bundle functions log durations and diagnostics. Keep those off the console.
    try:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            path, sources = run(args.read, args.smoke, args.smoke_dir, args.reserve, on_file, smoke_tag=args.smoke_tag)
    except AR.Stop as stop:
        print(f'{"refused" if stop.code == 4 else "contradictory inputs"} descriptive_{args.read}.json', flush=True)
        return stop.code
    except Refused as refusal:
        print(f'refused descriptive_{args.read}.json: {refusal}', flush=True)
        return 4
    except (RH.Refused, FileExistsError):
        print(f'refused descriptive_{args.read}.json', flush=True)
        return 4
    except D.BoundaryError as boundary:
        print(f'contradictory inputs descriptive_{args.read}.json: {boundary}', flush=True)
        return 5
    except (Exception, SystemExit):
        print(f'contradictory inputs descriptive_{args.read}.json', flush=True)
        return 5
    for entry in sources.values():
        print(f'{entry["file"]} {entry["sha256"]}', flush=True)
    print(f'{path} {R.sha256_file(path)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
