"""Read-only audit of the main checkout's §8 artifact bindings; JSON on stdout.

No scoring, refitting, descriptive run or artifact write is performed here.
Run with the ticket's Python environment and redirect stdout in this worktree.
"""
import json
from functools import lru_cache
from pathlib import Path

import run_r9_descriptive as RD

R, RH, AD, AR = RD.R, RD.RH, RD.AD, RD.AR
MAIN = Path('/project/CoSiR/src/test/20261202_r9_scope')
OUT = MAIN / 'results'
KEYS = ('mode', 'read', 'backbone', 'runner_sha256', 'module_sha256',
        'input_sha256', 'pins_sha256', 'coef_sha256', 'episodes_sha256')


@lru_cache(None)
def sha(path):
    return R.sha256_file(path)


def equal(left, right):
    rec = {'equal': left == right}
    if not rec['equal']:
        rec.update(left=str(left)[:120], right=str(right)[:120])
    return rec


def probe(fn):
    try:
        fn()
    except (Exception, SystemExit) as exc:
        return {'equal': False, 'error': str(exc)[:120], 'type': type(exc).__name__}
    return {'equal': True}


def main():
    reads = {}
    # All nine raw started/pass fields are checked first, without normalization.
    for read in ('H8', 'H9', 'H10', 'H6', 'H7'):
        names = RH.read_names(read)
        ppath, spath = OUT / names.pass_, OUT / names.started
        rec = RD.read_json(ppath)
        attempts = RH.read_attempts(spath)
        mine = [a for a in attempts if a.get('attempt') == rec.get('attempt')]
        RD.require(len(mine) == 1, 'producing started attempt')
        reads[read] = {'binding': {k: equal(mine[0].get(k), rec.get(k)) for k in KEYS},
                       'sources': {p.name: sha(p) for p in (ppath, spath)}}

    # Use the main agreement directory, as the controller's real run will.
    AR.AGREEMENT_DIR = MAIN / 'rederive/rd9_out'
    for read, report in reads.items():
        names = RH.read_names(read)
        rec = RD.read_json(OUT / names.pass_)
        attempts = RH.read_attempts(OUT / names.started)
        att = next(a for a in attempts if a['attempt'] == rec['attempt'])
        verdict = RD.read_json(OUT / names.verdict)
        checks = report['other_comparisons'] = {}
        # Covers identity, pass/agreement filenames, fix/reserve flags, agreement
        # schema, embedded agreement, SHA bindings, seeds, counts and backbone.
        checks['bind_read (all verdict/pass/agreement comparisons)'] = probe(
            lambda: RD.bind_read(OUT, read, False))
        for filename in (names.verdict, names.sensitivity):
            report['sources'][filename] = sha(OUT / filename)
        apath = AR.AGREEMENT_DIR / verdict['agreement_file']
        report['sources'][apath.name] = sha(apath)
        agreement = RD.read_json(apath)
        allowed = ('full claim', 'small claim', 'NO-GO') if read in RH.A28_READS else ('GO', 'NO-GO')
        checks['verdict status/allowed outcome'] = equal(
            (verdict.get('status'), verdict.get('verdict') in allowed), ('run', True))
        checks['verdict rule/read/ledger backbone/mode/variant'] = equal(
            tuple(verdict.get(k) for k in ('rule_sha256', 'read', 'backbone', 'mode', 'variant')),
            (R.RULE_SHA256, read, RH.LEDGER_NAMES[RH.READ_BACKBONES[read]], 'held', None))
        checks['verdict pass filename/fix flag'] = equal(
            (verdict.get('pass_file'), verdict.get('fix_pass')), (names.pass_, False))
        checks['verdict agreement filename/SHA'] = equal(
            (verdict.get('agreement_file'), verdict.get('agreement_sha256')),
            (f'rd9_{read}_agreement.json', sha(apath)))
        checks['agreement schema/phase/smoke/read/pass'] = probe(
            lambda: AR.load_agreement(apath, False, names.pass_, read))
        checks['embedded agreement'] = equal(verdict.get('agreement'), {k: agreement.get(k) for k in
            ('phase', 'smoke', 'agreement', 'read', 'pass_file', 'held_pass_sha256', 'compared')})
        checks['pass schema/rule/mode/seeds/design'] = probe(
            lambda: AR.load_pass((OUT / names.pass_).read_bytes(), names.pass_, read, False))
        checks['pass read/kind/code backbone'] = equal(
            tuple(rec.get(k) for k in ('read', 'kind', 'backbone')), (read, names.kind, RH.READ_BACKBONES[read]))
        checks['verdict/agreement/pass SHA'] = equal(
            (verdict.get('held_pass_sha256'), agreement['held_pass_sha256']),
            (sha(OUT / names.pass_), sha(OUT / names.pass_)))
        checks['verdict/pass seeds'] = equal(verdict.get('seeds'), rec['seeds'])
        checks['agreement/pass counts'] = equal(agreement['rederived_counts'],
            {c: rec['checks'][c]['n'] for c in AR.P_CHECKS})
        sens_path = OUT / names.sensitivity
        sensitivity = RD.read_json(sens_path)
        checks['coefficient hashes present'] = equal(bool(rec.get('coef_sha256')), True)
        for filename in (names.arrays, names.sensitivity):
            checks[f'outputs SHA: {filename}'] = equal(sha(OUT / filename), rec['outputs'][filename])
        for key in ('seeds', 'episodes_sha256'):
            checks[f'sensitivity/pass: {key}'] = equal(sensitivity.get(key), rec[key])
        checks['verdict sensitivity filename'] = equal(verdict.get('sensitivity_file'), names.sensitivity)
        checks['verdict sensitivity SHA'] = equal(verdict.get('sensitivity_sha256'), sha(sens_path))
        pins_path = MAIN / R.pins_path(rec['backbone']).name
        checks['backbone pins/pass SHA'] = equal(sha(pins_path), rec['pins_sha256'])
        checks['worktree/main backbone pins SHA'] = equal(sha(R.pins_path(rec['backbone'])), sha(pins_path))
        report['sources'][pins_path.name] = sha(pins_path)
        checks['input pins/pass hashes'] = equal(
            {k: rec['input_sha256'].get(k) for k in R.input_pins(rec['backbone'])},
            {k: v for k, (_, v) in R.input_pins(rec['backbone']).items()})
        # Arrays and episodes need no score rebuilding for these joins.
        arrays, core = RD.D.load_arrays(OUT / names.arrays, rec['seeds'], R.N_PER_PAIR)
        for seed, row in zip(rec['seeds'], core):
            path = OUT / RH.episodes_name(read, seed)
            ep = RD.E.load_episodes(path)
            checks[f'episodes/pass seed {seed}'] = equal(
                (ep.seed, ep.n, ep.sha),
                (seed, len(R.PAIRS) * R.N_PER_PAIR, rec['episodes_sha256'][str(seed)]))
            checks[f'episode pair alignment seed {seed}'] = equal(
                bool(RD.np.array_equal(ep.pair_index, row['pair_index'])), True)
        del arrays, core
        if read not in RH.A28_READS:
            continue
        a28_pins_path = MAIN / 'pins_a28.json'
        checks['A28 pins/pass SHA'] = equal(sha(a28_pins_path), rec['pins_a28_sha256'])
        checks['worktree/main A28 pins SHA'] = equal(sha(AD.PINS_PATH), sha(a28_pins_path))
        report['sources'][a28_pins_path.name] = sha(a28_pins_path)
        files = AD._pin_files(a28_pins_path)
        for key in ('results/value_sets_a28.json', 'results/language_flags_a28.json',
                    'results/read_set_a28.parquet', f'translations/{AD.MERGED_NAME}/coverage.json'):
            checks[f'A28 pinned file: {key}'] = probe(lambda key=key: AD._pinned(files, key))
        values = RD.read_json(Path(files['results/value_sets_a28.json']['path']))
        flags = RD.read_json(Path(files['results/language_flags_a28.json']['path']))
        coverage = RD.read_json(Path(files[f'translations/{AD.MERGED_NAME}/coverage.json']['path']))
        checks['A28 value sets/pass'] = equal(values, rec['value_sets'])
        checks['A28 read row-space/pass SHA'] = equal(
            sha(Path(files['results/read_set_a28.parquet']['path'])), rec['read_set_sha256'])
        checks['translation language alignment'] = equal(
            (set(flags['languages']), set(coverage)), (set(RD.AE.LANGUAGES), set(RD.AE.LANGUAGES)))
        checks['flags/pass'] = equal(flags['flagged'], rec['flagged_languages'])
        checks['language coverage/rows'] = equal(
            {k: coverage[k]['rows'] for k in RD.AE.LANGUAGES},
            {k: flags['languages'][k]['rows'] for k in RD.AE.LANGUAGES})
        checks['near-duplicate source SHA and counts'] = probe(lambda: RD.near_counts(rec))
        parts = att['a28_episodes']
        checks['A28 started seed records'] = equal(set(parts), set(map(str, rec['seeds'])))
        set_aside = [{'language': lang, 'pair': pair, 'seed': int(seed)}
                     for seed, part in parts.items() for pair, langs in part['set_aside'].items() for lang in langs]
        checks['set-aside/pass'] = equal(set_aside, rec['set_aside'])
        for other in RH.A28_READS:
            if other == read:
                continue
            other_rec = RD.read_json(OUT / RH.read_names(other).pass_)
            checks[f'cross-backbone seeds/episode hashes: {other}'] = equal(
                (other_rec['seeds'], other_rec['episodes_sha256']), (rec['seeds'], rec['episodes_sha256']))
        if read != 'H8':
            h8 = RD.bind_read(OUT, 'H8', False)
            reference = RH.a28_attempt_records(RH.read_attempts(OUT / RH.read_names('H8').started))
            checks['H8 set-aside disclosure'] = equal(h8.rec['set_aside'], set_aside)
            # Exact pre-fix expression: helper uses int keys, JSON uses str keys.
            checks['H8 part disclosures (raw int vs JSON str seed keys)'] = equal(reference, parts)
            checks['H8 part disclosures (canonical int seed keys)'] = equal(
                reference, {int(s): part for s, part in parts.items()})
        tab = AD.load_read_set()
        checks['A28 painting partition'] = equal(
            (tab.painting.nunique(), tab[tab.held].painting.nunique(), tab[~tab.held].painting.nunique()),
            (746, 248, 498))

    print(json.dumps({'main_results': str(OUT), 'binding_keys': KEYS, 'reads': reads,
                      'not_checked': [
                          'Current full recipe input/backbone file hashes (input pin declarations checked).',
                          'Refitted coefficient hashes; rebuilt A28 episode parts/hashes/fields and anchor joins.',
                          'Numerical pass reproduction, rebuilt score arrays and direction averages.',
                          'Paired anchor joins; language mask shapes and nonempty kept-row counts.',
                          'Optional H5 reproduction/anchor join and stage0b payload (no A28 mode comparison).',
                          'Full real-mode descriptive run and output write; controller runs after merge.'
                      ]}, indent=2))


if __name__ == '__main__':
    main()
