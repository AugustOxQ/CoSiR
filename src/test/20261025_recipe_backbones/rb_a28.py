"""Stage 5: ArtELingo-28 labels, the reading (A-HOLDS / A-QUAL / A-FAILS), reported margins and the culture split.

Rule §3.3, §4.3 (spec §3.4 "Stage 5"). Labels come from the public CSV's strings, joined to the read-set rows by
`row_id` (the CSV's data-row position), never from the read set's codes. The statistic is the part 2 scorer
(`rb_score`) at n = 746; the verdict file is written before any reported file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

import rb_paths
import ct_score
import rb_encoders
import rb_score

CSV = Path('/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv')
CSV_ROWS = 173745
# Our codes: names sorted (`encode_labels` over the full annotations); genre as `wikiart_genre.GENRE_NAMES`.
EMOTION_NAMES = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness',
                 'something else']
STYLE_NAMES = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque',
               'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism',
               'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism',
               'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
               'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']
GENRE_NAMES = ['abstract_painting', 'cityscape', 'genre_painting', 'illustration', 'landscape', 'nude_painting',
               'portrait', 'religious_painting', 'sketch_and_study', 'still_life']


# --- labels (rule §3.3) -------------------------------------------------------------------------------------------

def join_rows(read_set, csv):
    """The read-set rows with the public CSV's strings, joined by `row_id` = the CSV's data-row position."""
    assert read_set.row_id.is_unique and read_set.row_id.between(0, len(csv) - 1).all(), 'row_id outside the CSV'
    rows = csv.iloc[read_set.row_id.to_numpy()].reset_index(drop=True)
    assert (rows.painting.to_numpy() == read_set.painting.to_numpy()).all(), 'painting differs from the CSV row'
    assert (rows.language.to_numpy() == read_set.language.to_numpy()).all(), 'language differs from the CSV row'
    out = pd.DataFrame(dict(row_id=read_set.row_id.to_numpy(), painting=rows.painting, language=rows.language,
                            emotion=rows.emotion.replace({'other': 'something else'}), art_style=rows.art_style,
                            genre=rows.genre, held=read_set.held.to_numpy()))
    assert not out.isna().any().any()
    return out


def plurality(codes, groups, n_groups, n_classes):
    """(label, tied) per group: the most frequent code, ties to the smallest code; every group needs a member."""
    counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(counts, (groups, codes), 1)
    top = counts.max(axis=1)
    assert (top > 0).all(), 'a group without members'
    return counts.argmax(axis=1).astype(np.int64), (counts == top[:, None]).sum(axis=1) > 1, counts.sum(axis=1)


def _codes(values, names, what):
    lookup = {name: code for code, name in enumerate(names)}
    unknown = sorted(set(values) - set(lookup))
    assert not unknown, f'{what}: names outside our codes: {unknown[:3]}'
    return np.asarray([lookup[v] for v in values], dtype=np.int64)


def _painting_index(rows, names):
    names = list(names)
    assert names == sorted(set(names)) and set(names) == set(rows.painting), 'painting list is not the rows\' paintings'
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup[p] for p in rows.painting], dtype=np.int64)


def build_labels(rows, names, emotion_names=EMOTION_NAMES, style_names=STYLE_NAMES, genre_names=GENRE_NAMES):
    """Per painting (in `names` order): emotion plurality (ties to the smallest code), style and genre by name
    (constant per painting, asserted), held flag (constant). No label is masked."""
    pid = _painting_index(rows, names)
    n = len(names)
    emotion, tied, n_captions = plurality(_codes(rows.emotion, emotion_names, 'emotion'), pid, n, len(emotion_names))
    out = dict(emotion=emotion, ties=tied, n_captions=n_captions, n_ties=int(tied.sum()), n_unique=int((~tied).sum()))
    for key, column, aspect_names in (('style', 'art_style', style_names), ('genre', 'genre', genre_names)):
        codes = _codes(rows[column], aspect_names, key)
        first = np.zeros(n, dtype=np.int64)
        first[pid[::-1]] = codes[::-1]
        assert (codes == first[pid]).all(), f'{key} must be constant per painting'
        out[key] = first
    held = np.zeros(n, dtype=bool)
    held[pid[::-1]] = rows.held.to_numpy()[::-1]
    assert (rows.held.to_numpy() == held[pid]).all(), 'held must be constant per painting'
    out['held'] = held
    return out


def culture_labels(rows, names, emotion_names=EMOTION_NAMES):
    """Per language: the plurality emotion of that language's captions on the paintings that have one (same rule).

    {language: dict(idx (indices into `names`), emotion, ties, n_captions, agree)}; `agree` = share of those paintings
    whose language label equals the pooled label."""
    pid = _painting_index(rows, names)
    codes = _codes(rows.emotion, emotion_names, 'emotion')
    pooled, _, _ = plurality(codes, pid, len(names), len(emotion_names))
    out = {}
    for language in sorted(set(rows.language)):
        sel = (rows.language == language).to_numpy()
        idx, local = np.unique(pid[sel], return_inverse=True)
        emotion, tied, n_captions = plurality(codes[sel], local, len(idx), len(emotion_names))
        out[language] = dict(idx=idx, emotion=emotion, ties=tied, n_captions=n_captions,
                             agree=float((emotion == pooled[idx]).mean()))
    return out


# --- the reading (rule §4.3) --------------------------------------------------------------------------------------

def outcome(d, lower):
    """A-HOLDS: d > 0 and lower bound > 0; A-QUAL: d > 0 and lower bound <= 0; A-FAILS: d <= 0 (full precision)."""
    assert np.isfinite(d) and np.isfinite(lower)
    return ('A-HOLDS' if lower > 0 else 'A-QUAL') if d > 0 else 'A-FAILS'


SEEDS = rb_score.SEEDS
A28_ROWS = rb_encoders.A28_COUNT
PERCEPT_FILE = {'B1': 'P3', 'B6': 'P2', 'M2': 'P3', 'M3': 'P3', 'M5': 'P3'}   # B1 = core P3, B6 = core P2 (index)
MATCHED = {'K1': 'B1', 'K2': 'M2', 'K3': 'M3', 'K4': 'B1', 'K5': 'M5'}          # K1's and K4's matched arm is B1
ENCODER = {'K1': 'qwen3vl_emb_2b', 'K2': 'siglip2_so400m', 'K3': 'clip_b32', 'K4': 'qwen3vl_emb_2b',
           'K5': 'dinov2_large'}


def arms_for(model):
    """Rule order, stage 5 entries: the model, B1, B6, then its own matched arm (K1 and K4 have none beyond B1)."""
    assert model in ENCODER
    return (model, 'B1', 'B6') + ((MATCHED[model],) if MATCHED[model] != 'B1' else ())


def output_path(outputs, scale, arm, seed, aspect='emotion'):
    """`part2/<scale>/a28/<arm>/<aspect>__s{s}.npz` for K1..K5, `.../P{2,3}__s{s}.npz` for B1, B6, M2, M3, M5."""
    assert scale in ('sample', 'full') and seed in SEEDS
    stem = f'{aspect}__s{seed}' if arm[0] == 'K' else f'{PERCEPT_FILE[arm]}__s{seed}'
    assert arm[0] != 'K' or arm in ENCODER
    return Path(outputs) / f'part2/{scale}/a28/{arm}/{stem}.npz'


def read_output(path, arm, scale, seed, aspect, list_sha256, n_rows=A28_ROWS):
    """One ArtELingo-28 output as float32, with the index's sidecar checked here for every arm: `rb_score.load_output`
    treats K1, B1 and B6 as the CT's and would not check (or would misread) their sidecars. Students: unit norm,
    2048 wide float16; PercepT arms: sigmoid probabilities in [0, 1], 40 wide float32. No non-finite row is tolerated."""
    path = Path(path)
    is_k = arm[0] == 'K'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == (['img'] if is_k else ['prob']), f'{path}: array keys {z.files}'
        x = z['img' if is_k else 'prob']
    assert x.ndim == 2 and x.shape[0] == n_rows, f'{path}: shape {x.shape}'
    rb_score.check_sidecar(path, x, arm, n_rows, list_sha256, scale, seed, aspect)
    assert (x.dtype, x.shape[1]) == ((np.float16, 2048) if is_k else (np.float32, 40)), f'{path}: {x.dtype} {x.shape}'
    x = x.astype(np.float32)
    assert np.isfinite(x).all(), f'{path}: non-finite rows'
    if is_k:
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), f'{path}: not unit norm'
    else:
        assert ((x >= 0) & (x <= 1)).all(), f'{path}: outside [0, 1]'
    return x


# --- setup: the 746 list, labels, culture labels, images -----------------------------------------------------------

def make_setup(rows, names, sha256, images, instructed, emotion_names=EMOTION_NAMES, style_names=STYLE_NAMES,
               genre_names=GENRE_NAMES):
    """Everything the reading needs, by painting name order `names`. `images(encoder)` and `instructed(key)` give
    float32 unit rows in that order (the frozen encoder image Fk and the instructed Qwen image T)."""
    labels = build_labels(rows, names, emotion_names, style_names, genre_names)
    return dict(names=list(names), sha256=sha256, labels=labels, images=images, instructed=instructed,
                culture=culture_labels(rows, names, emotion_names))


def real_setup(read_set=rb_encoders.READ_SET, csv=CSV):
    """The 746 read-set paintings (hash pinned in `rb_encoders`), labels from the 74,569 rows and the public CSV."""
    names, sha256 = rb_encoders.a28_names(), rb_encoders.a28_sha256()
    frame, strings = pd.read_parquet(read_set), pd.read_csv(csv)
    assert len(frame) == 74569 and len(strings) == CSV_ROWS and sorted(frame.painting.unique()) == names
    setup = make_setup(join_rows(frame, strings), names, sha256, rb_encoders.a28_images, rb_encoders.a28_instructed)
    labels = setup['labels']
    assert (labels['n_unique'], labels['n_ties']) == (729, 17), (labels['n_unique'], labels['n_ties'])
    assert (int(labels['held'].sum()), int((~labels['held']).sum())) == (248, 498)
    return setup


def default_models(outputs):
    """K1, K2, K3, and K4 / K5 when `part1/pick.json` says they exist (its `triggers`, rule §4.1)."""
    path = Path(outputs) / 'part1/pick.json'
    assert path.exists(), f'{path}: the default model list comes from the triggers of part 1'
    triggers = json.loads(path.read_text())['triggers']
    assert set(triggers) == {'K4', 'K5'}, triggers
    return ['K1', 'K2', 'K3'] + [m for m in ('K4', 'K5') if triggers[m]]


def model_scale(model, outputs):
    """(Rule §2, stage 5) Full scale when the model's stage 4 ran (a block with an outcome in `verdict_stage4.json`),
    else the sample scale. K1's stage 4 is the CT's stage 2: always full (its verdict file must exist); missing
    outputs then fail V3, they never change the scale."""
    return scale_and_reason(model, outputs)[0]


def scale_and_reason(model, outputs):
    if model == 'K1':
        path = rb_score.CT_VERDICTS['full']
        assert path.exists(), f'{path}: the CT full verdict (K1 runs at full scale)'
        return 'full', None
    path = Path(outputs) / 'part2/verdict_stage4.json'
    if path.exists() and json.loads(path.read_text())['blocks'].get(model, {}).get('outcome_id'):
        return 'full', None
    return 'sample', 'no stage 4 outcome'


VERDICT, REPORTED = 'part2/verdict_stage5.json', 'part2/reported_stage5.json'
ASPECTS = ('emotion', 'style', 'genre')


class OutputFault(AssertionError):
    """A fault of an output file (missing, or failing its checks): V3 for the model, nothing else is."""


def partitions_of(arm, aspect, scale, outputs, setup, seeds):
    """(fixed partitions per seed, sources): read, check and cluster one arm's outputs once (cached on disk)."""
    n, sha = len(setup['names']), setup['sha256']
    part_aspect = aspect if arm[0] == 'K' else 'percept'
    folder = Path(outputs) / f'part2/partitions/a28_{scale}/{arm}'
    fixed, sources = {}, {}
    for seed in seeds:
        path = output_path(outputs, scale, arm, seed, aspect)
        try:
            assert path.exists(), f'missing output: {path}'
            x = read_output(path, arm, scale, seed, part_aspect, sha, n)
        except (AssertionError, FileNotFoundError) as err:
            raise OutputFault(f'{type(err).__name__}: {err}') from err
        digest = rb_score.sha_file(path)
        fixed[seed] = rb_score.fixed_partitions(x, arm, folder, part_aspect, seed, sha, digest)
        sources[str(seed)] = dict(path=str(path), output_sha256=digest)
    return fixed, sources


def verdict_block(model, scale, outputs, setup, seeds, n_resamples, workers):
    arms = arms_for(model)
    for aspect in ASPECTS:                                   # V3 on all three aspects; style and genre only reported
        partitions_of(model, aspect, scale, outputs, setup, seeds)
    fixed, sources = {}, {}
    for arm in arms:
        fixed[arm], sources[arm] = partitions_of(arm, 'emotion', scale, outputs, setup, seeds)
    eligible = {arm: list(seeds) for arm in arms}
    labels = setup['labels']['emotion']
    result = rb_score.paired_bootstrap(fixed, labels, eligible, arms, n_resamples, workers)
    values, counts = np.unique(labels, return_counts=True)
    return dict(candidate=model, arms=list(arms), scale=scale, **result, eligibility=dict(eligible_seeds=eligible),
                sources=sources, label_counts=dict(classes={str(v): int(c) for v, c in zip(values, counts)}, missing=0,
                                                   ties=setup['labels']['n_ties']),
                outcome_id=outcome(result['d'], result['interval'][0]),
                bootstrap=dict(resamples=n_resamples, rng='numpy.random.default_rng(0) per model',
                               arm_order=list(arms), n_rows=len(labels), percentiles=[2.5, 97.5], method='linear'))


def header_of(setup):
    return dict(stage=5, list_sha256=setup['sha256'], n_rows=len(setup['names']))


def run_verdict(models, outputs, setup=None, workers=8, n_resamples=2000, seeds=SEEDS, replace=None):
    """d_A28 and the reading of each model, into `part2/verdict_stage5.json` (one block per model)."""
    outputs = Path(outputs)
    setup = setup or real_setup()
    path = outputs / VERDICT
    record = None
    for model in models:
        scale, reason = scale_and_reason(model, outputs)
        try:
            block = verdict_block(model, scale, outputs, setup, list(seeds), n_resamples, workers)
            if reason:
                block['scale_reason'] = reason
            line = f'{block["outcome_id"]} d={block["d"]:.4f} interval={block["interval"]} ({scale})'
        except OutputFault as err:                           # V3: only this model's outcome is removed
            block = dict(candidate=model, arms=list(arms_for(model)), scale=scale, outcome_id=None, v_failure=str(err))
            if reason:
                block['scale_reason'] = reason
            line = f'NO OUTCOME ({block["v_failure"]})'
        record = rb_score.write_verdict(path, header_of(setup), {model: block}, replace)
        print(f'{model}: {line}', flush=True)
    return record


# --- reported, deciding nothing --------------------------------------------------------------------------------------

def frozen_fixed(x):
    """One partition set (r20, three states) of a frozen image: the seedless arm 'F'."""
    return {0: {'r20': np.asarray(ct_score.output_partitions(x, 'k1')['r20'])}}


def margin(fixed, labels, arms, seeds, n_resamples, workers, seedless=()):
    """The candidate (arms[0]) minus the maximum over the other arms' readings, with its own bootstrap."""
    eligible = {arm: list(seeds) for arm in arms if arm not in seedless}
    r = rb_score.paired_bootstrap({a: fixed[a] for a in arms}, labels, eligible, arms, n_resamples, workers, seedless)
    return dict(A=r['A_cand'], d=r['d'], interval=r['interval'], A_e=r['A_e'], arms=list(arms))


def subset(fixed, idx):
    return {arm: {seed: {r: p[:, idx] for r, p in readings.items()} for seed, readings in by_seed.items()}
            for arm, by_seed in fixed.items()}


INSTRUCTED = dict(emotion='emotion_focus', style='style_word', genre='genre_word')


def aspect_report(model, aspect, scale, outputs, setup, fixed_other, seeds, n_resamples, workers):
    """Model's `aspect` output against (a) B1, B6, (b) its matched arm, (c) Fk, max of (a) and (b); emotion also F1 and T.
    K1 and K4 have B1 as matched arm, so (max of a and b) is (a)."""
    labels = setup['labels'][aspect]
    matched = MATCHED[model]
    fixed = dict(fixed_other)
    fixed[model], _ = partitions_of(model, aspect, scale, outputs, setup, seeds)
    out = dict(A=None)
    plan = {'vs_a': (model, 'B1', 'B6'), 'vs_b': (model, matched)}
    if matched != 'B1':
        plan['vs_max_ab'] = arms_for(model)
    for name, arms in plan.items():
        out[name] = margin(fixed, labels, arms, seeds, n_resamples, workers)
    fixed['F'] = frozen_fixed(setup['images'](ENCODER[model]))
    out['vs_Fk'] = margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
    if aspect == 'emotion':
        if ENCODER[model] != 'qwen3vl_emb_2b':
            fixed['F'] = frozen_fixed(setup['images']('qwen3vl_emb_2b'))
            out['vs_F1'] = margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
        fixed['F'] = frozen_fixed(setup['instructed'](INSTRUCTED[aspect]))
        out['vs_T'] = margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
    out['A'] = out['vs_a']['A']
    return out


def culture_report(setup, parts):
    """Descriptive: AMI of the model's emotion partition (r20, fit on all 746; mean over its 3 states, then seeds)
    against each language's plurality label on the paintings with a caption in it, and that label's agreement with
    the pooled label."""
    pooled = setup['labels']['emotion']
    languages = {}
    for language, c in setup['culture'].items():
        idx = c['idx']
        score = lambda y: [float(np.mean([ami(y, p[idx]) for p in parts[seed]])) for seed in sorted(parts)]
        by_seed = score(c['emotion'])
        languages[language] = dict(
            n_paintings=int(len(idx)), n_captions=int(c['n_captions'].sum()), n_ties=int(c['ties'].sum()),
            ami=float(np.mean(by_seed)), ami_by_seed=by_seed,
            ami_vs_pooled_label_same_paintings=float(np.mean(score(pooled[idx]))), agree_pooled=c['agree'])
    return dict(languages=languages, mean_ami=float(np.mean([v['ami'] for v in languages.values()])))


def reported_block(model, verdict, outputs, setup, seeds, n_resamples, workers):
    scale = verdict['scale']
    others = {}
    for arm in arms_for(model)[1:]:
        others[arm], _ = partitions_of(arm, 'emotion', scale, outputs, setup, seeds)
    block = dict(model=model, scale=scale, verdict_outcome=verdict['outcome_id'], verdict_d=verdict['d'],
                 verdict_interval=verdict['interval'], arms=list(arms_for(model)))
    with threadpool_limits(limits=8):
        block['aspects'] = {aspect: aspect_report(model, aspect, scale, outputs, setup, others, seeds, n_resamples,
                                                  workers) for aspect in ASPECTS}
        fixed = dict(others)
        fixed[model], _ = partitions_of(model, 'emotion', scale, outputs, setup, seeds)
        held = setup['labels']['held']
        block['subsets'] = {}
        for name, idx in (('held_248', np.flatnonzero(held)), ('outside_498', np.flatnonzero(~held))):
            block['subsets'][name] = dict(n=int(len(idx)), **margin(
                subset(fixed, idx), setup['labels']['emotion'][idx], arms_for(model), seeds, n_resamples, workers))
        block['culture'] = culture_report(setup, {s: fixed[model][s]['r20'] for s in seeds})
    return block


def run_reported(models, outputs, setup=None, workers=8, n_resamples=2000, seeds=SEEDS, replace=None):
    """The reported items of each model with a stage 5 outcome, into `part2/reported_stage5.json` (after the verdict)."""
    outputs = Path(outputs)
    setup = setup or real_setup()
    vpath = outputs / VERDICT
    assert vpath.exists(), f'{vpath}: the verdict file is written before any reported file'
    verdicts = json.loads(vpath.read_text())['blocks']
    record = None
    for model in models:
        assert verdicts.get(model, {}).get('outcome_id'), f'{model}: no outcome in the verdict file'
        block = reported_block(model, verdicts[model], outputs, setup, list(seeds), n_resamples, workers)
        record = rb_score.write_verdict(outputs / REPORTED, header_of(setup), {model: block}, replace)
        print(f'{model}: reported', flush=True)
    return record


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='seed 0 only, 20 draws; outputs under /tmp/claude-0/')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--models', nargs='+', choices=sorted(ENCODER), help='default: K1, K2, K3 and the triggered K4, K5')
    p.add_argument('--part', choices=('verdict', 'reported', 'both'), default='both')
    p.add_argument('--replace', metavar='REASON', help='V-check reason for replacing a changed block')
    args = p.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    models = args.models or default_models(args.outputs)
    assert models, f'no ArtELingo-28 outputs under {args.outputs}/part2'
    kw = dict(workers=args.workers, n_resamples=20 if args.smoke else 2000, seeds=[0] if args.smoke else SEEDS,
              replace=args.replace)
    setup = real_setup()
    failed = []
    if args.part in ('verdict', 'both'):
        record = run_verdict(models, args.outputs, setup, **kw)
        failed = [m for m in models if record['blocks'][m].get('outcome_id') is None]
    if args.part in ('reported', 'both'):
        run_reported([m for m in models if m not in failed], args.outputs, setup, **kw)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
