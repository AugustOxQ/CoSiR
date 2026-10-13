"""Recipe III (ticket 09; rule §3.2 item 1, code map §1.11 and §1.12): cross-fitted heads, practice banks, half-readers
and tau, re-fitted from one backbone's features. The reader chain of `20261117_reader_fix_csd/` (rb_build.py stages,
run_rc.py), configuration A0 only.

    python r9_recipe_reader.py --backbone B [--out DIR] [--upstream DIR] --stage {heads,bank,train,tau,all}
                               [--grouping G]

Stages, and the original code each calls (round 1's functions are imported from the main checkout, not copied; the
bodies of the stages, which sit inside `main`-level stage functions of rb_build.py that load CLIP and write CLIP's
folder, are re-run here in the same order with the same arguments):
1. halves: CLIP's `rb_halves.npz`, used as it is (frozen; SHA-256 pinned here, equal to `pins_clip.json`). It is checked
   against the split (scorer-train rows, painting of each row, the partition into two halves of 18,259 paintings).
2. `heads` -> `rb_heads_<G>.npz` for G in affect, image, caption, csd, rand (all five roles exist in r9_backbone; the
   reader needs the first three): `rb_build.fit_head_pair` per half, 60,000-row draw from the half, posteriors on the
   other half's rows (cross-fit). The check of `stage_heads` that the recipe copy equals
   `run_told_oracle.fit_one_head` (3,000-row draw) is kept.
3. `bank` -> `rb_bank_A0_half0/1.npz`: `build_episode_bank` on each half's rows, 16,384 episodes per block, seeds
   11,700 and 11,800, validated as in rb_build (block layout on every episode, `validate_aspect_episodes` on 1,000).
4. `train` -> `rb_reader_A0.pkl` and `.json`: features from the cross-fitted heads (`rb_features.both_conditions`),
   `rb_build.fit_half_reader` per half (C by 5-fold CV over episodes). The pickle and the JSON are the formats
   `r9_bundle.load_readers` reads.
5. `tau` -> `rc_tau.json`: the A0 features of the seed-42 episodes (`rb_eval.seed42_features`) from this backbone's
   refit heads, the two half-readers' averaged probabilities, the top-two margins (24,576 of them), and
   `rc_core.thresholds` (numpy.percentile, linear, 0, 25, 50, 75). The bundle's other terms (A3, PM) do not enter
   tau, so they are not built: the features come from the same `RowContext` and posteriors that
   `r9_bundle.build_bundle_r6` calls `seed42_features` on (its step 8), without the A3 and PM terms.

Sources of the group labels (`source_pins`): `partition_L`, `style_csd`, `style_rand` stay CLIP's (frozen roles, round
1's SHA-256); `image` and `caption` are the backbone's `partitions.npz` (ticket 07's output; a pin is that file's
SHA-256 at the time the stage starts, kept in the record; the train stage compares the heads' and the banks' label
hashes, the tau stage compares the label hashes and the partitions SHA-256 of the groupings its refit heads are fit on
with the `rb_heads_<G>` records of the readers' heads).
`--upstream DIR` names the folder that holds `partitions.npz`, `n6_posteriors.npz` and `step1_heads_style.npz`
(default: the backbone's parts; for the CLIP replay: `results/replay/clip_b32`). Every source is checked against its
pin before it is read (`# guard:source_pin`). Any `partitions.npz` that is not CLIP's pinned original must have a
record whose `backbone` field is `--backbone` (ticket 07 writes it).

rb_build's size constants are PARAMS, asserted equal to rb_build's, rb_features' and run_rc's at import and at the
start of a run (`check_params`); the functions take them as arguments.

Outputs go to --out (default `results/recipe/<B>/`; the CLIP replay of stage 0b: `results/replay/clip_b32/`). No file
is overwritten: all names are checked before any work. Every file gets `<file>.record.json` (SHA-256 of the feature
store files, the sources, the halves, this script and the modules it imports; the keys made and copied; the file's
own SHA-256). There is no resume: a crashed run leaves files that block the rerun, and the refusal names them.

    cd /project/CoSiR-wt/r9-t09 && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_recipe_reader.py --backbone clip_b32 \
        --out src/test/20261202_r9_scope/results/replay/clip_b32_reader --stage all

Guards carry a `# guard:<name>` marker; test_r9_recipe_reader.py deletes each on a copy and shows that its scenario
then goes through.
"""
import argparse
import functools
import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
import r9_context as X  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_recipe_groups as G  # noqa: E402
from r9_backbone import BACKBONES, parts  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

C, rb, rbe, rf, RC, rto = R.C, R.rb, R.rbe, R.rf, R.RC, R.rto
sha256_file = R.sha256_file

CONFIG = "A0"
A0 = tuple(rb.CONFIGS[CONFIG])
GROUPINGS = tuple(rb.GROUPINGS)
STAGES = ("heads", "bank", "train", "tau")
CONDITIONS = ("a", "b")
HALVES_SHA256 = "ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da"     # pins_clip.json, rb_halves
HALF_KEYS = ("scorer_train", "painting_of_local_row", "g_st", "half_of_local_row", "local_rows_half0",
             "local_rows_half1", "global_rows_half0", "global_rows_half1")
HEADS_KEYS = ("img", "txt", "filled_by_half", "classes_half0", "classes_half1")
T0 = time.monotonic()

# rb_build's size constants, written out (rule §4.2 items 1 to 3, 6; run_rc: tau); asserted against the originals
PARAMS = {
    "n_st": 183_694, "n_paintings": 36_518, "half_paintings": 18_259,         # item 1
    "head_rows": 60_000, "check_rows": 10_000, "copy_check_rows": 3_000,      # item 2 (copy_check: rb SMOKE_HEAD_ROWS)
    "bank_n": 16_384, "bank_size": 49_152, "bank_seeds": (11_700, 11_800),    # item 3 (bank_size: A0, per half)
    "min_paintings": 30, "validate_n": 1_000,
    "reader_max_iter": 2000, "c_grid": (0.01, 0.1, 1.0, 10.0, 100.0), "n_folds": 5,   # item 6
    "n_per_pair_seed42": 4096, "n_tau_values": 24_576, "tau_percentiles": (0, 25, 50, 75),   # rc_core / run_rc
}


def log(*a):
    print(f"[{time.monotonic() - T0:7.1f}s]", *a, flush=True)


def rule_values() -> dict:
    """PARAMS' keys as the originals define them."""
    return {"n_st": rb.N_ST, "n_paintings": rb.N_PAINTINGS, "half_paintings": rb.HALF_PAINTINGS,
            "head_rows": rb.HEAD_ROWS, "check_rows": rb.CHECK_ROWS, "copy_check_rows": rb.SMOKE_HEAD_ROWS,
            "bank_n": rb.BANK_N, "bank_size": rb.BANK_SIZE[CONFIG], "bank_seeds": tuple(rb.BANK_SEEDS),
            "min_paintings": rb.MIN_PAINTINGS, "validate_n": rb.VALIDATE_N, "reader_max_iter": rb.READER_MAX_ITER,
            "c_grid": tuple(rf.C_GRID), "n_folds": rf.N_FOLDS, "n_per_pair_seed42": R.N_PER_PAIR,
            "n_tau_values": 2 * len(R.PAIRS) * R.N_PER_PAIR, "tau_percentiles": tuple(RC.PCTS)}


def check_params(params):
    """Every size parameter equals the original's (and no parameter is missing or extra)."""
    want = rule_values()
    assert set(params) == set(want), f"parameters differ: missing {sorted(set(want) - set(params))}, " \
                                     f"extra {sorted(set(params) - set(want))}"
    for key, value in want.items():
        got = tuple(params[key]) if isinstance(value, tuple) else params[key]
        assert got == value, f"{key}: {params[key]!r} is not the rule's {value!r}"


check_params(PARAMS)
if not (A0 == ("affect", "image", "caption") and GROUPINGS == ("affect", "image", "caption", "csd", "rand")
        and rb.BLOCKS[CONFIG] == [("affect", "caption"), ("affect", "image"), ("caption", "image")]):
    raise ImportError("rb_build's A0 configuration, groupings or blocks differ from the ones this module was written for")


# ---------------------------------------------------------------- pins and sources

def check_pin(path, sha, what):
    """Stops the run when ``path``'s SHA-256 is not the pin."""
    got = sha256_file(path)
    if got != sha:  # guard:source_pin
        raise SystemExit(f"{what}: {path} has SHA-256 {got}, the pin is {sha}")
    return got


def clip_pinned_sha(role) -> str:
    """The SHA-256 of one of CLIP's parts in pins_clip.json (the replay's reference)."""
    return json.loads((HERE / "pins_clip.json").read_text())["parts"][role]["sha256"]


def abs_path(p) -> Path:
    """``p`` as an absolute path (against the working folder; symbolic links kept): record keys never depend on it."""
    return Path(os.path.abspath(p))


def upstream_part(backbone, upstream, role) -> Path:
    """``role``'s file: the backbone's part, or the same file name in the ``upstream`` folder (absolute)."""
    path = parts(backbone)[role]
    return Path(path) if upstream is None else abs_path(upstream) / path.name


def check_part_backbone(path, backbone):
    """A partitions file that is not CLIP's pinned original has a record (ticket 07's) that names this backbone."""
    rec_path = Path(str(path) + ".record.json")
    assert rec_path.is_file(), f"{path}: no record {rec_path.name} next to it, cannot tell which backbone it was made for"
    made_for = json.loads(rec_path.read_text()).get("backbone")
    assert made_for == backbone, f"{path} was made for {made_for}, not {backbone}"


def source_pins(backbone, upstream=None) -> dict:
    """{grouping: (path, SHA-256, key)} (rb_build.SOURCES). partition_L, style_csd and style_rand are CLIP's frozen
    files at round 1's SHA-256; image and caption are the backbone's partitions.npz, pinned to its SHA-256 now (round
    1's, when the file is CLIP's own)."""
    src = {name: tuple(entry) for name, entry in rb.SOURCES.items()}
    if backbone != "clip_b32" or upstream is not None:
        path = upstream_part(backbone, upstream, "partitions")
        check_part_backbone(path, backbone)
        for name in ("image", "caption"):
            clip_path, clip_sha, _ = src[name]
            src[name] = (path, clip_sha if Path(path) == Path(clip_path) else sha256_file(path), name)
    return src


def load_groupings(names, st, g_st, sources, n_groups, *, e2):
    """{name: local label array (n_st,)}: rb_build.load_groupings with the pins of ``sources`` ({name: (path, sha,
    key)}) and the group counts ``n_groups``. ``e2`` = (path, SHA-256) of CLIP's partitions.npz, the frozen affect-km
    labels that D1's affect partition is cross-checked with. Every file is checked against its pin first."""
    st, g_st = np.asarray(st), np.asarray(g_st)
    assert (np.diff(st) > 0).all(), "scorer_train must be strictly ascending"
    out, cache = {}, {}
    e2_path, e2_sha = e2
    for name in names:
        path, sha, key = sources[name]
        if path not in cache:
            check_pin(path, sha, f"grouping source {name}")
            cache[path] = {k: np.asarray(v) for k, v in np.load(path).items()}
        z = cache[path]
        lab = np.asarray(z[key], dtype=np.int64)
        if name in ("image", "caption"):
            ok = np.array_equal(z["local_groups"], g_st)
            if "affect" in z and ok:
                check_pin(e2_path, e2_sha, "E2 partitions")
                with np.load(e2_path) as e:
                    ok = np.array_equal(z["affect"], e["affect"])
        elif name in ("csd", "rand"):
            ok = np.array_equal(z["scorer_train"], st) and np.array_equal(z["g_st"], g_st)
        else:   # affect: the told-oracle file's R0 partition is E2's affect partition, row for row
            check_pin(e2_path, e2_sha, "E2 partitions")
            with np.load(e2_path) as e:
                ok = (np.array_equal(e["local_groups"], g_st)
                      and np.array_equal(np.asarray(z["partition_R0"], dtype=np.int64), e["affect"]))
        assert ok, f"{name}: label array is not aligned with the ascending scorer-train rows"
        counts = np.bincount(lab) if lab.min() >= 0 else None
        assert lab.shape == (len(st),) and counts is not None and len(counts) == n_groups[name] \
            and not (counts == 0).any(), \
            f"{name}: expected {n_groups[name]} non-empty groups 0..{n_groups[name] - 1} on {len(st)} local rows"
        if name in ("csd", "rand"):
            first = np.unique(g_st, return_index=True)[1]
            assert np.array_equal(lab[first][g_st], lab), f"{name}: labels are not constant within a painting"
        out[name] = lab
    return out


def rto_labels(local, scorer_train, n_rows_all):
    """run_told_oracle.global_labels: -1 outside the scorer-train rows."""
    return rto.global_labels(np.asarray(local), np.asarray(scorer_train), n_rows_all)


# ---------------------------------------------------------------- halves

def check_halves(h, split, params):
    """The halves file against the split and the rule's sizes (item 1)."""
    st = np.asarray(split.scorer_train)
    assert len(st) == params["n_st"] and (np.diff(st) > 0).all(), "scorer_train: size or order"
    assert set(HALF_KEYS) <= set(h), f"halves lack {sorted(set(HALF_KEYS) - set(h))}"
    assert np.array_equal(h["scorer_train"], st), "halves: scorer_train differs from the split's"
    paint = np.asarray(split.groups)[st]
    assert np.array_equal(h["painting_of_local_row"], paint), "halves: painting of each row differs from the split's"
    paintings, g_st = np.unique(paint, return_inverse=True)
    assert len(paintings) == params["n_paintings"] and 2 * params["half_paintings"] == params["n_paintings"], \
        f"halves: {len(paintings)} scorer-train paintings"
    assert np.array_equal(h["g_st"], g_st), "halves: local painting codes differ from the split's"
    half = np.asarray(h["half_of_local_row"])
    seen = []
    for k in (0, 1):
        local = np.asarray(h[f"local_rows_half{k}"])
        assert np.array_equal(local, np.flatnonzero(half == k)), f"halves: rows of half {k} do not follow the half index"
        assert np.array_equal(h[f"global_rows_half{k}"], st[local]), f"halves: global rows of half {k}"
        codes = np.unique(g_st[local])
        assert len(codes) == params["half_paintings"], f"halves: half {k} has {len(codes)} paintings"
        seen.append(codes)
    assert not np.intersect1d(*seen).size, "halves: a painting lies in both halves"
    assert len(h["local_rows_half0"]) + len(h["local_rows_half1"]) == params["n_st"], "halves must partition the rows"


def load_halves(backbone="clip_b32") -> dict:
    """CLIP's rb_halves.npz (the frozen role of every backbone), SHA-256 pinned."""
    path = parts(backbone)["rb_halves"]
    assert path == parts("clip_b32")["rb_halves"], "rb_halves is a frozen part: CLIP's file"
    check_pin(path, HALVES_SHA256, "rb_halves")
    with np.load(path) as z:
        h = {k: np.asarray(z[k]) for k in HALF_KEYS}
    h["sha256"] = HALVES_SHA256
    h["path"] = str(path)
    return h


# ---------------------------------------------------------------- heads (item 2)

def fit_crossfit_heads(F, lab, halves, grouping, n_groups, *, n_rows, check_rows, backbone):
    """rb_build.stage_heads' loop: per half k, `fit_head_pair` on a ``n_rows`` draw of half k's global rows, posteriors
    of the other half's rows. -> ({img, txt: (n_st, G) float32, filled_by_half, classes_half0, classes_half1}, {"0":
    provenance, "1": ...}). ``backbone`` decides whether CLIP's reference accuracies are written."""
    n_st = len(halves["half_of_local_row"])
    cf = {m: np.full((n_st, n_groups), np.nan, dtype=np.float32) for m in ("img", "txt")}
    filled_by = np.full(n_st, -1, dtype=np.int8)
    heads, classes_of = {}, {}
    for k in (0, 1):
        own_g, other_g = halves[f"global_rows_half{k}"], halves[f"global_rows_half{1 - k}"]
        other_l = halves[f"local_rows_half{1 - k}"]
        post, prov, classes = rb.fit_head_pair(F, lab, own_g, n_rows, other_g, other_rows=other_g, check_rows=check_rows)
        missing = sorted(set(range(n_groups)) - set(classes.tolist()))
        prov.update(n_groups=n_groups, all_groups_in_draw=not missing, missing_groups=missing)
        if backbone == "clip_b32":      # rb_build.STANDARD_ACC: the accuracies of CLIP's standard heads (a reference)
            prov["standard_heldout_accuracy"] = {"img": rb.STANDARD_ACC[grouping][0],
                                                 "txt": rb.STANDARD_ACC[grouping][1]}
        else:                           # the code does not compute the backbone's own standard heads
            prov["standard_heldout_accuracy"] = None
            prov["standard_heldout_accuracy_unavailable"] = (
                "rb_build.STANDARD_ACC holds CLIP's standard heads' accuracies; they do not describe this backbone, "
                "and no standard head is fitted for it")
        prov["posteriors_on"] = f"all rows of half {1 - k}"
        if missing:
            raise SystemExit(f"{grouping}, half {k}: groups {missing} are missing from the {n_rows}-row draw; stopping "
                             "(DECISION_RULE.md §4.2 item 2: the step goes to the user)")
        for m in ("img", "txt"):
            full = np.zeros((len(other_l), n_groups), dtype=np.float32)
            full[:, classes] = post[m]
            cf[m][other_l] = full
        filled_by[other_l] = k
        heads[str(k)], classes_of[k] = prov, classes
    assert np.array_equal(filled_by, 1 - halves["half_of_local_row"]) and all(np.isfinite(cf[m]).all() for m in cf), \
        "every local row must carry finite posteriors from the other half's heads"
    arrays = {"img": cf["img"], "txt": cf["txt"], "filled_by_half": filled_by,
              "classes_half0": classes_of[0], "classes_half1": classes_of[1]}
    return arrays, heads


def check_head_copy(data, split, lab, *, n_rows) -> bool:
    """rb_build.stage_heads' check: `fit_head_pair` equals `run_told_oracle.fit_one_head` on scorer-train, an
    ``n_rows`` draw (bit for bit, same draw, same held-out accuracy). Stops the run otherwise."""
    st, sel, groups = np.asarray(split.scorer_train), np.asarray(split.selection), np.asarray(split.groups)
    in_sel = np.zeros(len(groups), dtype=bool)
    in_sel[sel] = True
    ctx_like = SimpleNamespace(data=data, selection=sel, groups=groups, in_sel=in_sel)
    ref_post, ref_prov = rto.fit_one_head(ctx_like, lab, st, n_rows)
    F = {"img": data.img_features, "txt": data.txt_features}
    mine, mprov, _ = rb.fit_head_pair(F, lab, st, n_rows, sel)
    ok = bool(all(np.array_equal(mine[m], ref_post[m][sel]) for m in ("img", "txt"))
              and mprov["draw_rows_sha256"] == ref_prov["draw_rows_sha256"]
              and mprov["heldout_accuracy_own_half"] == ref_prov["heldout_accuracy"])
    if not ok:
        raise SystemExit("the head recipe copy differs from run_told_oracle.fit_one_head: stopping")
    return True


# ---------------------------------------------------------------- banks (item 3)

def fit_bank_half(labels, halves, j, *, params, config=CONFIG):
    """rb_build.stage_bank for one half: `build_episode_bank` on half j's rows with the half's seed, validated as
    there. ``labels`` = {grouping: local labels} of the configuration. -> SimpleNamespace(arrays, prov)."""
    from src.eval.aspect_episodes import PaintingValueIndex, eligible_values, validate_aspect_episodes
    from src.train.pseudo_partitions import build_episode_bank
    parts_ = rb.CONFIGS[config]
    partitions = {k: np.asarray(labels[k], dtype=np.int64) for k in parts_}
    assert set(labels) >= set(parts_), f"labels lack {sorted(set(parts_) - set(labels))}"
    paint = np.asarray(halves["painting_of_local_row"])
    blocks = rf.bank_blocks(partitions)
    assert blocks == rb.BLOCKS[config], f"{config}: builder blocks {blocks} differ from the rule's {rb.BLOCKS[config]}"
    n, seed = params["bank_n"], params["bank_seeds"][j]
    rows = np.asarray(halves[f"local_rows_half{j}"])
    assert (np.diff(rows) > 0).all(), "half rows must be ascending"
    index = PaintingValueIndex(partitions, paint)
    t0 = time.time()
    try:
        bank = build_episode_bank(partitions, paint, rows, n_per_pair=n, seed=seed,
                                  min_paintings=params["min_paintings"])
    except RuntimeError as e:
        raise SystemExit(f"{config} half {j}: a block cannot be filled ({e}); stopping (DECISION_RULE.md §4.2 "
                         "item 3 and §8: the step goes to the user)")
    build_s = time.time() - t0
    total = len(blocks) * n
    assert len(bank.anchor) == total == params["bank_size"], \
        f"{config} half {j}: {len(bank.anchor)} episodes, bank_size {params['bank_size']}, blocks x bank_n {total}"
    in_half = np.zeros(len(paint), dtype=bool)
    in_half[rows] = True
    assert in_half[bank.rows()].all(), f"{config} half {j}: a bank row lies outside half {j}"
    layout, checked = [], {}
    sample = np.sort(np.random.default_rng(42).choice(total, size=min(params["validate_n"], total), replace=False))
    for i, (a, b) in enumerate(blocks):
        sl = slice(i * n, (i + 1) * n)
        la, lb = partitions[a], partitions[b]
        ok = bool((la[bank.pairs_a_img[sl]] == la[bank.pairs_a_txt[sl]]).all()
                  and (lb[bank.pairs_b_img[sl]] == lb[bank.pairs_b_txt[sl]]).all()
                  and (la[bank.pairs_a_img[sl]] != la[bank.anchor[sl]][:, None]).all()
                  and (la[bank.candidates[sl, 0]] == la[bank.anchor[sl]]).all()
                  and (lb[bank.candidates[sl, 1]] == lb[bank.anchor[sl]]).all())
        assert ok, f"{config} half {j} block {i} ({a}, {b}): episodes do not follow the block layout"
        idx = sample[(sample >= sl.start) & (sample < sl.stop)]
        validate_aspect_episodes(rb.subset_episodes(bank, a, b, idx), partitions, paint, index,
                                 third=rb.THIRD[config].get((a, b)))
        checked[f"{a}__{b}"] = int(len(idx))
        layout.append({"block": i, "first": a, "second": b, "third_controlled": rb.THIRD[config].get((a, b)),
                       "seed": seed + i, "episodes": [i * n, (i + 1) * n - 1]})
    arrays = {f: getattr(bank, f) for f in rb.EPISODE_FIELDS}
    arrays.update(block_pairs=np.array([f"{a}__{b}" for a, b in blocks]), block_size=np.array(n),
                  seed=np.array(seed), half=np.array(j))
    prov = {"config": config, "half": j, "groupings": list(parts_), "n_per_pair": n, "seed": seed,
            "min_paintings": params["min_paintings"], "n_episodes": int(total), "blocks": layout,
            "validated_per_block": checked,
            "eligible_values": {k: len(eligible_values(partitions[k], paint, rows, params["min_paintings"]))
                                for k in parts_},
            "rows_used": int(len(np.unique(bank.rows()))), "build_s": round(build_s, 1),
            "label_sha256": {k: rb.sha_array(partitions[k]) for k in parts_}}
    return SimpleNamespace(arrays=arrays, prov=prov)


def bank_episodes(arrays):
    """The AspectEpisodes of a bank file's arrays (rb_build.load_bank)."""
    from src.eval.aspect_episodes import AspectEpisodes
    return AspectEpisodes("mixed", "mixed", *(np.asarray(arrays[f]).astype(np.int64) for f in rb.EPISODE_FIELDS))


def bank_blocks_of(arrays):
    blocks = [tuple(s.split("__")) for s in np.asarray(arrays["block_pairs"]).tolist()]
    assert blocks == rb.BLOCKS[CONFIG], f"bank block order {blocks} differs from the rule's"
    return blocks


def stack_conditions_for(F, arrays):
    """(X, y, episode) of a bank: rb_build.bank_features' stacking of both conditions."""
    ya, yb = rf.bank_labels(bank_blocks_of(arrays), int(arrays["block_size"]), A0)
    return rf.stack_conditions(F["a"], F["b"], ya, yb)


# ---------------------------------------------------------------- readers (items 4 to 6)

def fit_readers(post, banks, halves, *, params):
    """rb_build.stage_train for A0: features of half j's bank from the cross-fitted posteriors, `fit_half_reader` per
    half. ``post`` = {grouping: {img, txt, filled_by_half}}, ``banks`` = {j: bank arrays}. -> (pickle dict, record)."""
    other_half = 1 - np.asarray(halves["half_of_local_row"])
    for h in A0:
        assert np.array_equal(post[h]["filled_by_half"], other_half), \
            f"rb_heads_{h}: rows are not scored by the other half's heads"
    pp = {h: {"img": post[h]["img"], "txt": post[h]["txt"]} for h in A0}
    half_models, rec_h = [], {}
    for j in (0, 1):
        ep = bank_episodes(banks[j])
        assert (np.asarray(halves["half_of_local_row"])[ep.rows()] == j).all(), \
            f"bank half {j}: a row outside half {j}, its posteriors would not be cross-fitted"
        F = rf.both_conditions(pp, A0, ep)
        Xj, y, epi = stack_conditions_for(F, banks[j])
        assert np.isfinite(Xj).all(), f"bank half {j}: non-finite features"
        counts = np.bincount(y, minlength=len(A0))
        assert not (counts != counts[0]).any(), f"half {j}: classes are not balanced ({counts})"
        scaler, model, rec, _ = rb.fit_half_reader(Xj, y, epi, len(ep.anchor), len(A0))
        half_models.append({"scaler": scaler, "model": model, "C": rec["chosen_C"]})
        rec_h[str(j)] = rec
        log(f"half {j}: chosen C {rec['chosen_C']}")
    pkl = {"config": CONFIG, "groupings": list(A0), "feature_names": rf.feature_names(A0), "halves": half_models,
           "sklearn_version": sklearn.__version__, "rule_sha256": C.RULE_SHA, "smoke": False}
    return pkl, {"halves": rec_h}


# ---------------------------------------------------------------- tau (run_rc)

def reader_margins(pk, F):
    """({c: top-two margin of the averaged half-reader probabilities}, {c: those probabilities}) for the features
    ``F`` = {a, b} (rb_eval.stage_config's steps)."""
    probs, margins = {}, {}
    for c in CONDITIONS:
        per_half = rbe.half_reader_probs(pk, F[c], len(A0))
        P = rf.average_probs(per_half)
        assert np.allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12), "averaged probabilities do not sum to 1"
        margins[c] = rf.picks_and_margins(P)[1]
        assert np.array_equal(margins[c], C.top_two_margin(P)), "top-two margin differs from common.top_two_margin"
        probs[c] = P
    return margins, probs


def thresholds(margins, *, n_expected):
    """(tau_0..tau_3, number of margins): rc_core.thresholds, with the count asserted (run_rc: 24,576)."""
    for c in CONDITIONS:
        assert np.isfinite(np.asarray(margins[c])).all(), f"margins of condition {c} are not finite"
    taus, n_vals = RC.thresholds(margins)
    assert n_vals == n_expected, f"expected {n_expected:,} (episode, condition) margins, got {n_vals:,}"
    assert len(taus) == 4 and taus == sorted(taus), f"taus {taus}"
    return taus, n_vals


def tau_record(backbone, taus, n_values, *, reader_sha256):
    return {"backbone": backbone, "config": CONFIG, "taus": [float(t) for t in taus],
            "percentiles": list(RC.PCTS), "n_values": int(n_values),
            "method": "numpy.percentile, linear, of the readers' top-two margins on the seed-42 episodes (both "
                      "conditions)",
            "reader_pkl_sha256": reader_sha256}


# ---------------------------------------------------------------- files, records, refusal

def output_names(stage, groupings=None) -> list:
    assert stage in STAGES + ("all",), f"--stage must be one of {STAGES + ('all',)}"
    names = {"heads": [f"rb_heads_{g}.npz" for g in (groupings or GROUPINGS)],
             "bank": [f"rb_bank_{CONFIG}_half{j}.npz" for j in (0, 1)],
             "train": [f"rb_reader_{CONFIG}.pkl", f"rb_reader_{CONFIG}.json"],
             "tau": ["rc_tau.json"]}
    return [n for s in (STAGES if stage == "all" else (stage,)) for n in names[s]]


def _refuse(out, names):
    present = [str(Path(out) / n) for name in names for n in (name, name + ".record.json") if (Path(out) / n).exists()]
    if present:
        raise FileExistsError(
            f"refusing to overwrite. There is no resume: a crashed run leaves files that block the rerun. To rerun "
            f"in {out}, remove these files first: {' '.join(present)}")


def preflight(out, backbone, stage, groupings=None):
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    _refuse(out, output_names(stage, groupings))


@functools.cache
def _store_shas(backbone) -> tuple:
    return tuple(sorted(G.read_shas(backbone).items()))


def code_record() -> dict:
    modules = {}
    for name, mod in (("r9_recipe_reader", sys.modules[__name__]), ("r9_recipe_groups", G), ("r9_heads", H),
                      ("r9_common", R), ("r9_context", X), ("rb_build", rb), ("rb_features", rf), ("rb_eval", rbe),
                      ("rc_core", RC), ("common", C), ("run_told_oracle", rto)):
        modules[name] = {"file": str(Path(mod.__file__).resolve()), "sha256": sha256_file(mod.__file__)}
    return {"modules": modules, "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                                             "sklearn": sklearn.__version__}}


def write_file(env, out, name, kind, payload, *, inputs, made, copied=(), extra=None):
    """One output (``kind`` npz, json or pkl) and its record. Exclusive creation: an existing file is never replaced."""
    out = Path(out)
    path, rec_path = out / name, out / (name + ".record.json")
    _refuse(out, [name])
    out.mkdir(parents=True, exist_ok=True)
    if kind == "npz":
        assert set(payload) == set(made) | set(copied), f"{name}: keys made+copied differ from keys written"
        with path.open("xb") as f:
            np.savez(f, **payload)
    elif kind == "json":
        with path.open("x") as f:
            json.dump(payload, f, indent=1, allow_nan=False)
            f.write("\n")
    else:
        assert kind == "pkl"
        with path.open("xb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    sha = sha256_file(path)
    script = Path(__file__)
    rec = {"backbone": env.backbone, "file_sha256": sha, "input_sha256": {str(abs_path(k)): v for k, v in {**env.input_sha256, **dict(inputs)}.items()},
           "halves_sha256": env.halves_sha256, "params": C.jsonable(env.params), "script": str(script),
           "script_sha256": sha256_file(script), "code": code_record(), "keys_made": list(made),
           "keys_copied": list(copied), **(extra or {})}
    with rec_path.open("x") as f:
        json.dump(C.jsonable(rec), f, indent=2, allow_nan=False)
        f.write("\n")
    return rec


def load_record(env, out, name):
    """The record of an earlier stage's file: both exist, the file's SHA-256 is its record's, the record names this
    backbone and these halves."""
    out = Path(out)
    path, rec_path = out / name, out / (name + ".record.json")
    assert path.is_file() and rec_path.is_file(), f"missing {name} (or its record) in {out}: run the earlier stage"
    rec = json.loads(rec_path.read_text())
    assert sha256_file(path) == rec["file_sha256"], f"{name}: SHA-256 differs from its record"
    assert rec["backbone"] == env.backbone, f"{name}: made for {rec['backbone']}, not {env.backbone}"
    assert rec["halves_sha256"] == env.halves_sha256, f"{name}: built on other halves"
    return rec


def load_part(env, out, name, *, kind="npz"):
    """(arrays or object, record) of an earlier stage's file (checked as `load_record` does)."""
    out = Path(out)
    path = out / name
    rec = load_record(env, out, name)
    if kind == "npz":
        with np.load(path) as z:
            return {k: np.asarray(z[k]) for k in z.files}, rec
    if kind == "json":
        return json.loads(path.read_text()), rec
    with path.open("rb") as f:
        return pickle.load(f), rec


def _need(out, names):
    gone = [n for n in names if not ((Path(out) / n).is_file() and (Path(out) / (n + ".record.json")).is_file())]
    assert not gone, f"missing in {out}: {', '.join(gone)} (run the earlier stage)"


def _source_inputs(env, names):
    """{path: SHA-256} of the files the groupings ``names`` were read from, and the frozen E2 file."""
    files = {str(env.sources[n][0]): env.sources[n][1] for n in names}
    files[str(env.e2[0])] = env.e2[1]
    return files


# ---------------------------------------------------------------- stages

def stage_heads(env, out, groupings):
    """Items 2: one `rb_heads_<G>.npz` per grouping."""
    _refuse(out, output_names("heads", groupings))
    st, g_st = env.split.scorer_train, env.halves["g_st"]
    F = {"img": env.data.img_features, "txt": env.data.txt_features}
    n_all = len(env.split.groups)
    for g in groupings:
        t0 = time.time()
        local = load_groupings([g], st, g_st, env.sources, env.n_groups, e2=env.e2)[g]
        lab = rto_labels(local, st, n_all)
        check_head_copy(env.data, env.split, lab, n_rows=env.params["copy_check_rows"])
        log(f"{g}: recipe copy equals fit_one_head")
        arrays, prov = fit_crossfit_heads(F, lab, env.halves, g, env.n_groups[g], n_rows=env.params["head_rows"],
                                          check_rows=env.params["check_rows"], backbone=env.backbone)
        for k in ("0", "1"):
            log(f"{g} half {k}: img own/other {prov[k]['heldout_accuracy_own_half']['img']:.2f}/"
                f"{prov[k]['accuracy_other_half']['img']:.2f}, txt {prov[k]['heldout_accuracy_own_half']['txt']:.2f}/"
                f"{prov[k]['accuracy_other_half']['txt']:.2f}")
        # filled_by_half (1 - the frozen half index) and classes_half0/1 (0..G-1, as all groups are in the draw) are
        # CLIP's values for every backbone: copied; the posteriors are made
        write_file(env, out, f"rb_heads_{g}.npz", "npz", arrays, inputs=_source_inputs(env, [g]),
                   made=["img", "txt"], copied=["filled_by_half", "classes_half0", "classes_half1"],
                   extra={"grouping": g, "n_groups": env.n_groups[g], "head_rows": env.params["head_rows"],
                          "label_sha256": rb.sha_array(local), "source": [str(abs_path(env.sources[g][0])), env.sources[g][2]],
                          "heads": prov, "recipe_copy_equals_fit_one_head": True,
                          "runtime_s": round(time.time() - t0, 1)})
        log(f"{g} heads done [{time.time() - t0:.0f}s]")


def stage_bank(env, out):
    """Item 3: `rb_bank_A0_half0/1.npz`."""
    _refuse(out, output_names("bank"))
    labels = load_groupings(list(A0), env.split.scorer_train, env.halves["g_st"], env.sources, env.n_groups, e2=env.e2)
    for j in (0, 1):
        t0 = time.time()
        bank = fit_bank_half(labels, env.halves, j, params=env.params)
        # block layout, block size, seed and half are the rule's constants (CLIP's values for every backbone): copied;
        # the episodes are made
        write_file(env, out, f"rb_bank_{CONFIG}_half{j}.npz", "npz", bank.arrays,
                   inputs=_source_inputs(env, A0), made=list(rb.EPISODE_FIELDS),
                   copied=["block_pairs", "block_size", "seed", "half"], extra=bank.prov)
        log(f"bank half {j} done ({bank.prov['n_episodes']:,} episodes) [{time.time() - t0:.0f}s]")


def stage_train(env, out):
    """Items 4 to 6: `rb_reader_A0.pkl` and `.json`."""
    _refuse(out, output_names("train"))
    heads_names = [f"rb_heads_{g}.npz" for g in A0]
    bank_names = output_names("bank")
    _need(out, heads_names + bank_names)
    post, head_recs, banks, bank_recs = {}, {}, {}, {}
    for g, name in zip(A0, heads_names):
        post[g], head_recs[g] = load_part(env, out, name)
    for j, name in enumerate(bank_names):
        banks[j], bank_recs[j] = load_part(env, out, name)
        assert bank_recs[j]["half"] == j and bank_recs[j]["config"] == CONFIG, f"{name}: another half or configuration"
    for g in A0:                         # the heads and the banks were made from the same label arrays
        for j in (0, 1):
            assert head_recs[g]["label_sha256"] == bank_recs[j]["label_sha256"][g], \
                f"rb_heads_{g} and bank half {j} were made from different {g} labels"
    t0 = time.time()
    pkl, rec = fit_readers(post, banks, env.halves, params=env.params)
    inputs = {str(Path(out) / n): head_recs[g]["file_sha256"] for g, n in zip(A0, heads_names)}
    inputs.update({str(Path(out) / n): bank_recs[j]["file_sha256"] for j, n in enumerate(bank_names)})
    write_file(env, out, f"rb_reader_{CONFIG}.pkl", "pkl", pkl, inputs=inputs, made=["config", "groupings",
               "feature_names", "halves", "sklearn_version", "rule_sha256", "smoke"])
    pkl_sha = sha256_file(Path(out) / f"rb_reader_{CONFIG}.pkl")
    body = {"stage": "train", "config": CONFIG, "groupings": list(A0), "smoke": False, "rule_sha256": C.RULE_SHA,
            "feature_names": rf.feature_names(A0), "classes": {str(i): h for i, h in enumerate(A0)},
            "halves": rec["halves"], "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "heads_files_sha256": {g: head_recs[g]["file_sha256"] for g in A0},
            "banks_files_sha256": {str(j): bank_recs[j]["file_sha256"] for j in (0, 1)},
            "model_spec": {"scaler": "StandardScaler fitted once per half on all of that half's bank features",
                           "reader": f"LogisticRegression(solver='lbfgs', max_iter={env.params['reader_max_iter']}, C "
                                     "from grid, multinomial)", "C_grid": list(env.params["c_grid"]),
                           "cv": "KFold(5, shuffle=True, random_state=0) over episode indices; both conditions of an "
                                 "episode share a fold; criterion mean held-out log loss; ties to the smaller C"},
            "provenance": {"pkl_sha256": pkl_sha, "script_sha256": sha256_file(__file__),
                           "finished_amsterdam": R.amsterdam_now()}}
    write_file(env, out, f"rb_reader_{CONFIG}.json", "json", C.jsonable(body), inputs=inputs,
               made=list(body), extra={"runtime_s": round(time.time() - t0, 1)})
    log(f"readers done [{time.time() - t0:.0f}s]")


def load_reader(env, out):
    """The pickled A0 half-readers with the checks of r9_bundle.load_readers (JSON, SHA-256, rule, scikit-learn)."""
    out = Path(out)
    _need(out, output_names("train"))
    rec = json.loads((out / f"rb_reader_{CONFIG}.json").read_text())
    assert rec["rule_sha256"] == C.RULE_SHA and rec["smoke"] is False, "reader JSON: another rule or smoke flag"
    pkl_path = out / f"rb_reader_{CONFIG}.pkl"
    assert sha256_file(pkl_path) == rec["provenance"]["pkl_sha256"], "reader pickle: SHA-256 differs from its JSON"
    pk, prec = load_part(env, out, pkl_path.name, kind="pkl")
    assert pk["config"] == CONFIG and tuple(pk["groupings"]) == A0 and pk["rule_sha256"] == C.RULE_SHA, \
        "reader pickle: other configuration or rule"
    assert pk["sklearn_version"] == sklearn.__version__, \
        f"reader pickle was written with scikit-learn {pk['sklearn_version']}, running {sklearn.__version__}"
    return pk, prec["file_sha256"]


def check_tau_groupings(env, out, head_groupings):
    """The groupings the tau stage's refit (n6) heads are fit on are the ones the readers' heads were trained on: for
    each grouping of A0, the label hash equals the `rb_heads_<G>` record's, and for image and caption the SHA-256 of the
    partitions file equals the one the record lists for its source. -> {heads file: SHA-256} (inputs of the tau
    record)."""
    out = Path(out)
    inputs = {}
    for g in A0:
        name = f"rb_heads_{g}.npz"
        rec = load_record(env, out, name)
        assert rec["label_sha256"] == head_groupings["label_sha256"][g], \
            f"tau stage: the {g} labels of the refit heads are not the ones {name} was trained on"
        if g in ("image", "caption"):
            trained_on = rec["input_sha256"][rec["source"][0]]
            assert trained_on == head_groupings["partitions_sha256"], \
                f"tau stage: partitions {head_groupings['partitions_path']} (SHA-256 {head_groupings['partitions_sha256']}) " \
                f"are not the file {name} was trained on ({rec['source'][0]}, SHA-256 {trained_on})"
        inputs[str(out / name)] = rec["file_sha256"]
    return inputs


def stage_tau(env, out, F42, checks):
    """run_rc's thresholds from the seed-42 features ``F42`` = {a, b} (n episodes x 18): `rc_tau.json`. ``checks`` =
    `seed42_features`' second result (the groupings and the input hashes of the refit heads)."""
    _refuse(out, output_names("tau"))
    pk, pkl_sha = load_reader(env, out)
    assert pk["feature_names"] == rf.feature_names(A0), "the readers were trained on another feature layout"
    heads_inputs = check_tau_groupings(env, out, checks["head_groupings"])
    margins, _ = reader_margins(pk, F42)
    taus, n_vals = thresholds(margins, n_expected=env.params["n_tau_values"])
    rec = tau_record(env.backbone, taus, n_vals, reader_sha256=pkl_sha)
    inputs = {str(Path(out) / f"rb_reader_{CONFIG}.pkl"): pkl_sha, **heads_inputs, **checks["head_input_sha256"],
              **checks["upstream_posterior_sha256"]}
    write_file(env, out, "rc_tau.json", "json", rec, inputs=inputs, made=["taus"],
               extra={"seed42_feature_checks": checks})
    reread = json.loads((Path(out) / "rc_tau.json").read_text())["taus"]
    assert reread == taus, "tau did not round-trip at full precision"
    log(f"tau_0..3 written: {taus}")


# ---------------------------------------------------------------- the seed-42 features (the real run)

def check_stored_posteriors(heads, data, selection, backbone, upstream):
    """The refit heads' selection-row posteriors equal the upstream n6_posteriors.npz and step1_heads_style.npz bit for
    bit (rule §6.2's eight arrays); a new backbone's also the coefficient hashes of the n6 record."""
    files = {H.N6_REL: upstream_part(backbone, upstream, "n6_posteriors"),
             H.CSD_HEADS_REL: upstream_part(backbone, upstream, "step1_heads_style")}
    if backbone == "clip_b32" and upstream is None:
        for rel in files:
            R.assert_input(rel)
    selection = np.asarray(selection)
    post = H.predict(heads, data, selection)
    items = {}
    for rel, keys in H.TARGETS.items():
        with np.load(files[rel], allow_pickle=False) as z:
            assert np.array_equal(z["selection"], selection), f"{files[rel]} was computed on another selection"
            for key, (h, m) in keys.items():
                items[f"{Path(rel).stem}/{key}"] = H.compare_bits(post[h][m][selection], z[key])
    assert tuple(items) == H.ITEM_KEYS, "the checked items differ from rule §6.2's eight arrays"
    bad = [k for k, v in items.items() if not v["equal"]]
    assert not bad, f"refit posteriors differ from {', '.join(map(str, files.values()))}: {bad}"
    if backbone != "clip_b32":
        stored = json.loads(Path(str(files[H.N6_REL]) + ".record.json").read_text())
        assert stored["coef_sha256"] == H.coef_sha256(heads), "refit head coefficients differ from the n6 record's"
    return {"items": items, "files": {k: str(v) for k, v in files.items()}}


def rows_from(data, split) -> SimpleNamespace:
    """r9_common.load_rows' result (but the value index, which the tau stage does not use) from the ``data`` and
    ``split`` that are already loaded: the stage loads the features once."""
    from src.data.artelingo_splits import artelingo_aspect_labels
    labels = artelingo_aspect_labels(data)
    assert set(labels) == set(R.ASPECTS) and all(np.asarray(v).shape == (R.N_ROWS,) for v in labels.values()), \
        "aspect label columns or rows differ from ArtELingo"
    values = R.development_value_sets(labels, split.groups, split.selection)
    return SimpleNamespace(data=data, split=split, labels=labels, value_sets=values)


def seed42_features(backbone, upstream, data, split):
    """(F = {a, b}: (12288, 18), checks): the refit heads of this backbone (checked against the upstream posterior
    files), the seed-42 selection-row context, and `rb_eval.seed42_features` on its A0 posteriors, as
    r9_bundle.build_bundle_r6 calls it (without the A3 and PM terms). ``data`` and ``split`` are the loaded features
    and split of the run. ``checks`` also holds what the tau record needs: `head_groupings` (label hash of each A0
    grouping, partitions file and SHA-256: compared with the readers' heads), `head_input_sha256` (every input of the
    head fit) and `upstream_posterior_sha256`."""
    rows = rows_from(data, split)
    pins = H.recipe_head_pins(backbone, None if upstream is None else abs_path(upstream))
    heads = H.fit_heads_r9(rows.data, rows.split.groups, rows.split.scorer_train, backbone=backbone, input_pins=pins)
    log("heads refit")
    st = np.asarray(rows.split.scorer_train)
    head_labels = H.head_labels(rows.split.groups, st, backbone, input_pins=pins)      # the labels the heads were fit on
    part_path, part_sha = pins[H.E2_PARTITIONS_REL]
    head_groupings = {"partitions_path": str(part_path), "partitions_sha256": part_sha,
                      "label_sha256": {g: rb.sha_array(head_labels[g][st]) for g in A0}}
    stored = check_stored_posteriors(heads, rows.data, rows.split.selection, backbone, upstream)
    ctx = X.RowContext("selection", R.DEV_SEED, rows.data, rows.split, rows.labels, heads, rows.value_sets,
                       R.N_PER_PAIR)
    post = {h: ctx.post[h] for h in A0}
    F, checks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0)
    assert all(F[c].shape == (ctx.n, 6 * len(A0)) for c in CONDITIONS)
    log(f"seed-42 features ({ctx.n} episodes)")
    return F, {**checks, "stored_posteriors_equal": True, "episodes_sha256": dict(ctx.episode_sha),
               "stored_posterior_files": stored["files"], "head_groupings": head_groupings,
               "head_input_sha256": dict(heads["affect"]["fit_inputs"]["input_sha256"]),
               "upstream_posterior_sha256": {v: sha256_file(v) for v in stored["files"].values()}}


# ---------------------------------------------------------------- main

def split_indices():
    """groups and scorer_train from prepare.npz (SHA-256 pinned): the stages without features need nothing else."""
    R.assert_input(R.PREPARE_REL)
    with np.load(R.INPUT_PATHS[R.PREPARE_REL], allow_pickle=False) as z:
        return SimpleNamespace(groups=np.asarray(z["groups"]), scorer_train=np.asarray(z["scorer_train"]))


def build_env(backbone, upstream, stage):
    params = dict(PARAMS)
    check_params(params)
    halves = load_halves(backbone)
    needs_features = stage in ("heads", "tau", "all")
    if needs_features:
        data = R.load_data(backbone)
        split = R.load_split(data)
        log(f"{backbone}: data and split loaded")
    else:
        data, split = None, split_indices()
    check_halves(halves, split, params)
    sources = source_pins(backbone, upstream)
    clip_e2 = rb.SOURCES["image"]
    e2 = (clip_e2[0], clip_e2[1])
    reads = dict(_store_shas(backbone)) if needs_features else {}
    reads[halves["path"]] = HALVES_SHA256
    return SimpleNamespace(backbone=backbone, data=data, split=split, halves=halves, sources=sources, params=params,
                           n_groups=dict(rb.N_GROUPS), e2=e2, input_sha256=reads, halves_sha256=HALVES_SHA256)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default results/recipe/<backbone>/")
    ap.add_argument("--stage", choices=STAGES + ("all",), required=True)
    ap.add_argument("--grouping", choices=GROUPINGS, default=None,
                    help="heads only: this grouping alone (default: all five)")
    ap.add_argument("--upstream", type=Path, default=None,
                    help="folder with partitions.npz, n6_posteriors.npz, step1_heads_style.npz (default: the "
                         "backbone's parts)")
    args = ap.parse_args(argv)
    assert args.grouping is None or args.stage == "heads", "--grouping goes with --stage heads"
    out = args.out or R.RESULTS / "recipe" / args.backbone
    H.require_threads()
    groupings = [args.grouping] if args.grouping else list(GROUPINGS)
    preflight(out, args.backbone, args.stage, groupings if args.stage == "heads" else None)
    env = build_env(args.backbone, args.upstream, args.stage)
    if args.stage in ("heads", "all"):
        stage_heads(env, out, groupings)
    if args.stage in ("bank", "all"):
        stage_bank(env, out)
    if args.stage in ("train", "all"):
        stage_train(env, out)
    if args.stage in ("tau", "all"):
        F42, checks = seed42_features(args.backbone, args.upstream, env.data, env.split)
        stage_tau(env, out, F42, checks)
    log(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
