"""Round 6 episodes (DECISION_RULE.md of this folder: section 5 items 2 and 3, section 6 item 4's last item, section 10
item 1): the held-capable episode builder with the development value restriction, the per-seed builder, the episode
hash guards and the episode file.

The builder. build_aspect_episodes_r6 is src/eval/aspect_episodes.build_aspect_episodes with one keyword,
`values=(V_a, V_b)`. After ok_a and ok_b are formed it asserts V_a <= ok_a and V_b <= ok_b and replaces them with
sorted(set(ok_a) & set(V_a)) and the same for b. Nothing else changes (a test compares the two functions statement by
statement). The restriction reaches only the anchors and the example pairs' shared values: a value outside V (on held
rows, New_Realism) keeps its rows in the pool, so they can still be negatives, p_a or p_b, example items whose shared
value is on the other aspect, and anchors of a pair where that aspect is the third one. On selection rows the
development sets equal ok_a and ok_b, so the lists, the random stream and the episodes are those of the original; the
identity check (identity_check) shows it on the stored hashes of seeds 42, 9001, 9002 and 9003.

The guards. build_seed validates every pair (validate_aspect_episodes) and asserts that every member row (anchor, the
4 + 4 example pairs in both modalities, the 13 candidates) lies in `rows`. assert_distinct is rule section 5 item 3's
hash guard: the new per-pair hashes differ from each other and from every recorded non-smoke hash. In selection mode
seed 42 reproduces its recorded hashes by design, so the guard belongs to the held path (and the smoke seeds, whose
records are smoke records, pass it).

The episode file (save_episodes / load_episodes) is CPU-internal: it is never given to a GPU job.

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (before anything that imports src: it puts MAIN's src first and checks it)

import numpy as np  # noqa: E402

from src.data.sampling import draw_distinct  # noqa: E402
from src.eval.aspect_episodes import (NUM_NEGATIVES, NUM_PAIRS, AspectEpisodes, PaintingValueIndex,  # noqa: E402
                                      _pairs, concat_episodes, eligible_values, episodes_sha256,
                                      validate_aspect_episodes)

FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")   # run_baselines'
N_CANDIDATES = 2 + NUM_NEGATIVES                                                                # p_a, p_b, negatives
FILE_KEYS = ("seed", "pair_index", *FIELDS, "pair_names", *(f"sha__{p}" for p in R.PAIR_NAMES))
# rule section 5 item 3: the non-smoke records the new hashes must differ from
AB_RESULTS = R.AB / "results"
R3_RESULTS = R.R3D / "results"
RECORDED_AB_SEEDS = (42, 43, 45, 47, 48, 49, 50, 51)      # AB/results/baselines_seed*.json present on 2026-10-09
R3_BUILD_SEEDS = (49, 50, 51)                              # R3D/results/build_seed{s}.json


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _codes(values) -> set:
    """A value set as Python ints (label codes); anything but an integer is refused."""
    out = set()
    for v in values:
        _require(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)),
                 f"value {v!r} is not an integer label code")
        out.add(int(v))
    return out


# ---------------------------------------------------------------- the builder (rule section 5 item 3)

def build_aspect_episodes_r6(labels: dict, groups: np.ndarray, rows: np.ndarray, aspect_a: str, aspect_b: str,
                             n_episodes: int, seed: int, third: str | None = None, min_paintings: int = 30,
                             index: PaintingValueIndex | None = None, values=None) -> AspectEpisodes:
    """build_aspect_episodes plus rule section 5 item 3's one keyword, values=(V_a, V_b)."""
    groups = np.asarray(groups)
    index = index or PaintingValueIndex(labels, groups)
    la, lb = index.labels[aspect_a], index.labels[aspect_b]
    lt = index.labels[third] if third else None
    rows = np.asarray(rows, dtype=np.int64)
    known = (la[rows] >= 0) & (lb[rows] >= 0)
    if third:
        known &= lt[rows] >= 0
    pool = rows[known]
    ok_a = eligible_values(la, groups, pool, min_paintings)
    ok_b = eligible_values(lb, groups, pool, min_paintings)
    if values is not None:
        v_a, v_b = (_codes(v) for v in values)
        _require(v_a <= set(ok_a), f"{aspect_a} x {aspect_b}: values {sorted(v_a - set(ok_a))} of {aspect_a} "
                                   f"are not eligible on the pool")  # guard:values_subset
        _require(v_b <= set(ok_b), f"{aspect_a} x {aspect_b}: values {sorted(v_b - set(ok_b))} of {aspect_b} "
                                   f"are not eligible on the pool")  # guard:values_subset
        ok_a = sorted(set(ok_a) & v_a)
        ok_b = sorted(set(ok_b) & v_b)
    if len(ok_a) <= NUM_PAIRS or len(ok_b) <= NUM_PAIRS:
        raise RuntimeError(f"{aspect_a} x {aspect_b}: need more than {NUM_PAIRS} eligible values per aspect, "
                           f"got {len(ok_a)} and {len(ok_b)}")
    anchors = pool[np.isin(la[pool], ok_a) & np.isin(lb[pool], ok_b)]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")}
    failures = 0
    while len(out["anchor"]) < n_episodes:
        anchor = int(anchors[rng.integers(len(anchors))])
        a, b = int(la[anchor]), int(lb[anchor])
        lack_a, lack_b = index.lacks(aspect_a, a)[pool], index.lacks(aspect_b, b)[pool]
        lack_t = index.lacks(third, int(lt[anchor]))[pool] if third else np.ones(len(pool), dtype=bool)
        used = {groups[anchor]}
        try:
            p_a = draw_distinct(rng, pool[(la[pool] == a) & lack_b & lack_t], groups, used, 1)
            p_b = draw_distinct(rng, pool[(lb[pool] == b) & lack_a & lack_t], groups, used, 1)
            negatives = draw_distinct(rng, pool[lack_a & lack_b & lack_t], groups, used, NUM_NEGATIVES)
            base = pool[lack_a & lack_b]
            va = rng.choice([v for v in ok_a if v != a], NUM_PAIRS, replace=False)
            vb = rng.choice([v for v in ok_b if v != b], NUM_PAIRS, replace=False)
            pa_img, pa_txt = _pairs(rng, base, la, lb, va, groups, used)
            pb_img, pb_txt = _pairs(rng, base, lb, la, vb, groups, used)
        except ValueError:
            failures += 1
            if failures > 10 * n_episodes:
                raise RuntimeError(f"{aspect_a} x {aspect_b}: could not fill {n_episodes} episodes "
                                   f"({failures} failed draws); the pools are too small for these constraints")
            continue
        for key, value in zip(out, (anchor, p_a + p_b + negatives, pa_img, pa_txt, pb_img, pb_txt)):
            out[key].append(value)
    return AspectEpisodes(aspect_a, aspect_b, *(np.asarray(out[k], dtype=np.int64) for k in out))


# ---------------------------------------------------------------- one seed (rule section 5 item 3)

def members(ep: AspectEpisodes) -> np.ndarray:
    """Every member row of every episode, shapes asserted: anchor, the 13 candidates, the 4 + 4 example pairs in
    both modalities."""
    n = len(ep.anchor)
    want = {"anchor": (n,), "candidates": (n, N_CANDIDATES), "pairs_a_img": (n, NUM_PAIRS),
            "pairs_a_txt": (n, NUM_PAIRS), "pairs_b_img": (n, NUM_PAIRS), "pairs_b_txt": (n, NUM_PAIRS)}
    for f in FIELDS:
        x = getattr(ep, f)
        _require(x.shape == want[f] and x.dtype == np.int64, f"{f}: shape {x.shape} {x.dtype}, not {want[f]} int64")
    return np.concatenate([getattr(ep, f).ravel() for f in FIELDS])


def build_seed(labels, groups, rows, index, value_sets, seed, n_per_pair) -> SimpleNamespace:
    """The three pairs of run_baselines.PAIRS for one seed, each built by build_aspect_episodes_r6 on ``rows`` with
    the development value sets, validated, and every member row asserted to lie in ``rows``.

    -> SimpleNamespace(seed, per_pair (3 AspectEpisodes, PAIRS order), pooled (concat_episodes), sha {pair: sha},
    n, pair_index (n,) int64, parity (n,) int64 = arange(n) % 2)."""
    groups = np.asarray(groups)
    rows = np.asarray(rows, dtype=np.int64)
    _require(len(groups) == R.N_ROWS, f"groups has {len(groups)} rows, not ArtELingo's {R.N_ROWS}")
    _require(np.array_equal(index.groups, groups)
             and all(np.array_equal(index.labels[x], np.asarray(labels[x], dtype=np.int64)) for x in R.ASPECTS),
             "index is not PaintingValueIndex(labels, groups)")
    _require(rows.ndim == 1 and rows.size > 0 and (np.diff(rows) > 0).all(),
             "rows must be sorted, unique row positions")
    _require(set(value_sets) >= set(R.ASPECTS), f"value sets lack an aspect: {sorted(value_sets)}")
    n_per_pair = int(n_per_pair)
    _require(n_per_pair > 0, "n_per_pair must be positive")
    in_rows = np.zeros(len(groups), dtype=bool)
    in_rows[rows] = True
    per_pair, sha = [], {}
    for (a, b, third), name in zip(R.PAIRS, R.PAIR_NAMES):
        ep = build_aspect_episodes_r6(labels, groups, rows, a, b, n_per_pair, int(seed), third=third, index=index,
                                      values=(value_sets[a], value_sets[b]))
        _require((ep.aspect_a, ep.aspect_b) == (a, b) and len(ep.anchor) == n_per_pair,
                 f"{name}: {len(ep.anchor)} episodes of {ep.aspect_a} x {ep.aspect_b}")
        validate_aspect_episodes(ep, labels, groups, index, third=third)  # guard:validate
        every = members(ep)
        _require(in_rows[every].all(), f"{name}: {int((~in_rows[every]).sum())} member rows lie outside "
                                       f"the given rows")  # guard:members_in_rows
        sha[name] = episodes_sha256(ep)
        per_pair.append(ep)
    pooled = concat_episodes(per_pair)
    n = len(pooled.anchor)
    return SimpleNamespace(seed=int(seed), per_pair=per_pair, pooled=pooled, sha=sha, n=n,
                           pair_index=np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), n_per_pair),
                           parity=np.arange(n, dtype=np.int64) % 2)


# ---------------------------------------------------------------- the identity check (rule section 6 item 4)

def _baselines_record(path) -> dict:
    rec = json.loads(Path(path).read_text())
    _require(tuple(rec["pair_order"]) == R.PAIR_NAMES and tuple(rec["episodes_sha256"]) == R.PAIR_NAMES,
             f"{path}: pair order differs from {R.PAIR_NAMES}")
    return rec


def identity_targets() -> dict:
    """{seed: {"n_per_pair", "episodes_sha256": {pair: sha}}} for seed 42 (AB/results/baselines_seed42.json, 4,096
    per pair) and the smoke seeds 9001 to 9003 (AB/results/smoke/, 64 per pair); each file's SHA-256 asserted. AB's
    smoke seed-42 file (64 per pair) is not a target."""
    files = {R.DEV_SEED: (f"20261030_aspect_baselines/results/baselines_seed{R.DEV_SEED}.json", R.N_PER_PAIR)}
    files.update({s: (f"20261030_aspect_baselines/results/smoke/baselines_seed{s}.json", R.N_SMOKE)
                  for s in R.SMOKE_SEEDS})
    out = {}
    for seed, (rel, n) in files.items():
        R.assert_input(rel)
        rec = _baselines_record(R.INPUT_PATHS[rel])
        _require(int(rec["episodes_seed"]) == seed and int(rec["n_per_pair"]) == n,
                 f"{rel}: seed {rec['episodes_seed']}, {rec['n_per_pair']} per pair; expected {seed}, {n}")
        out[seed] = {"n_per_pair": n, "episodes_sha256": {p: str(rec["episodes_sha256"][p]) for p in R.PAIR_NAMES}}
    return out


def identity_check(labels, groups, selection, index, value_sets) -> dict:
    """Rule section 6 item 4, last item: build_seed on selection rows with the development value sets reproduces the
    stored per-pair hashes of seeds 42, 9001, 9002, 9003. -> {"passed", "items": {"episodes_seed<s>__<pair>":
    {"equal", "got", "want"}}} (the regression's item format)."""
    items = {}
    for seed, t in identity_targets().items():
        eps = build_seed(labels, groups, selection, index, value_sets, seed, t["n_per_pair"])
        for p in R.PAIR_NAMES:
            got, want = eps.sha[p], t["episodes_sha256"][p]
            items[f"episodes_seed{seed}__{p}"] = {"equal": got == want, "got": got, "want": want}
    return {"passed": all(v["equal"] for v in items.values()), "items": items}


# ---------------------------------------------------------------- hash distinctness (rule section 5 item 3)

def recorded_hashes(ab_results=None, r3_results=None) -> dict:
    """{seed: {pair: sha}} of every non-smoke AB/results/baselines_seed*.json and R3D/results/build_seed{49,50,51}.json.

    Each AB file's SHA-256 is asserted where one is recorded (R3 rule D15 for 42 to 48, round 3's build records for
    49 to 51); a seed found in both sources must carry the same hashes. The directories are parameters for the tests
    only."""
    ab = Path(ab_results or AB_RESULTS)
    r3 = Path(r3_results or R3_RESULTS)
    builds = {}
    for s in R3_BUILD_SEEDS:
        path = r3 / f"build_seed{s}.json"
        _require(path.is_file(), f"{path} is missing")
        rec = json.loads(path.read_text())
        _require(int(rec["seed"]) == s and rec["smoke"] is False and rec["passed"] is True,
                 f"{path}: not the passed non-smoke build of seed {s}")
        builds[s] = rec
    out, ab_seeds = {}, set()
    for path in sorted(ab.glob("baselines_seed*.json")):
        rec = _baselines_record(path)
        seed = int(rec["episodes_seed"])
        _require(path.name == f"baselines_seed{seed}.json" and seed not in out, f"{path}: seed {seed}")
        rel = f"20261030_aspect_baselines/results/{path.name}"
        want = (R.INPUT_SHA256.get(rel) if seed not in builds else builds[seed]["sha256"]["baselines"])
        got = R.sha256_file(path)
        _require(want is None or got == want, f"{path}: SHA-256 {got} differs from the recorded {want}",
                 SystemExit)  # guard:recorded_file_sha
        out[seed] = {p: str(rec["episodes_sha256"][p]) for p in R.PAIR_NAMES}
        ab_seeds.add(seed)
    _require(set(RECORDED_AB_SEEDS) - set(R3_BUILD_SEEDS) <= ab_seeds,
             f"AB records missing: {sorted(set(RECORDED_AB_SEEDS) - set(R3_BUILD_SEEDS) - ab_seeds)} in {ab}"
             )  # guard:recorded_complete
    for s, rec in builds.items():
        h = {p: str(rec["episode_pair_sha256"][p]) for p in R.PAIR_NAMES}
        _require(out.setdefault(s, h) == h, f"seed {s}: the build record and the baselines record "
                                            f"disagree on the episode hashes")  # guard:recorded_agree
    return out


def assert_distinct(new: dict, recorded=None) -> dict:
    """Rule section 5 item 3's guard. ``new`` = {seed: {pair: sha}} (the three pairs per seed): every new hash
    differs from every other new hash and from every recorded non-smoke hash (default recorded_hashes()).
    -> {"passed": True, "new_seeds", "recorded_seeds"}."""
    recorded = recorded_hashes() if recorded is None else recorded
    seen = {}
    for s, hs in new.items():
        _require(tuple(hs) == R.PAIR_NAMES, f"seed {s}: pairs {tuple(hs)}, not {R.PAIR_NAMES}")
        for p, h in hs.items():
            _require(isinstance(h, str) and len(h) == 64, f"seed {s} {p}: {h!r} is not a SHA-256")
            _require(h not in seen, f"seed {s} {p} repeats the episode hash of seed "
                                    f"{seen.get(h, ('?', '?'))[0]} {seen.get(h, ('?', '?'))[1]}")  # guard:distinct_new
            seen[h] = (s, p)
    clash = [f"new seed {seen[h][0]} {seen[h][1]} equals recorded seed {s} {p}"
             for s, hs in recorded.items() for p, h in hs.items() if h in seen]
    _require(not clash, "episode hashes equal recorded ones: " + "; ".join(clash))  # guard:distinct_recorded
    return {"passed": True, "new_seeds": sorted(int(s) for s in new), "recorded_seeds": sorted(int(s) for s in recorded)}


# ---------------------------------------------------------------- the episode file (contracts section 2)

def save_episodes(path, eps) -> Path:
    """npz: seed, pair_index, anchor (n,), candidates (n,13), pairs_{a,b}_{img,txt} (n,4), all int64 global row ids,
    plus pair_names and sha__<pair>. ``eps`` is build_seed's namespace; its hashes are recomputed first."""
    path = Path(path)
    _require(path.suffix == ".npz", f"{path}: the episode file must end in .npz")
    _require(len(eps.per_pair) == len(R.PAIRS) and tuple(eps.sha) == R.PAIR_NAMES, "eps does not hold the three pairs")
    for ep, p in zip(eps.per_pair, R.PAIR_NAMES):
        _require(f"{ep.aspect_a}__{ep.aspect_b}" == p and episodes_sha256(ep) == eps.sha[p],
                 f"{p}: the per-pair episodes do not match their hash")
    pooled = concat_episodes(eps.per_pair)
    n_per = [len(ep.anchor) for ep in eps.per_pair]
    pair_index = np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), n_per)
    _require(np.array_equal(eps.pair_index, pair_index) and eps.n == len(pair_index), "pair_index is not the pairs'")
    for f in FIELDS:
        _require(np.array_equal(getattr(eps.pooled, f), getattr(pooled, f)), f"pooled {f} is not the pairs' concat")
    arrays = {"seed": np.int64(eps.seed), "pair_index": pair_index,
              **{f: np.ascontiguousarray(getattr(pooled, f), dtype=np.int64) for f in FIELDS},
              "pair_names": np.array(R.PAIR_NAMES), **{f"sha__{p}": np.array(eps.sha[p]) for p in R.PAIR_NAMES}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_episodes(path) -> SimpleNamespace:
    """The namespace build_seed returns, read from save_episodes' file; each pair's hash recomputed and asserted."""
    with np.load(path, allow_pickle=False) as z:
        _require(set(z.files) == set(FILE_KEYS), f"{path}: keys {sorted(z.files)} differ from {sorted(FILE_KEYS)}")
        d = {k: z[k] for k in z.files}
    _require(tuple(str(p) for p in d["pair_names"]) == R.PAIR_NAMES, f"{path}: pair names differ")
    n = len(d["anchor"])
    for k in ("seed", "pair_index", *FIELDS):
        _require(d[k].dtype == np.int64, f"{path}: {k} is {d[k].dtype}, not int64")
    pair_index = d["pair_index"]
    counts = np.bincount(pair_index, minlength=len(R.PAIRS))
    _require(pair_index.shape == (n,) and len(counts) == len(R.PAIRS)
             and np.array_equal(pair_index, np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), counts)),
             f"{path}: pair_index is not three consecutive blocks")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    per_pair, sha = [], {}
    for i, p in enumerate(R.PAIR_NAMES):
        a, b = p.split("__")
        ep = AspectEpisodes(a, b, *(np.ascontiguousarray(d[f][bounds[i]:bounds[i + 1]]) for f in FIELDS))
        members(ep)
        sha[p] = str(d[f"sha__{p}"])
        _require(episodes_sha256(ep) == sha[p], f"{path}: {p} episodes do not match their stored hash")  # guard:load_sha
        per_pair.append(ep)
    return SimpleNamespace(seed=int(d["seed"]), per_pair=per_pair, pooled=concat_episodes(per_pair), sha=sha, n=n,
                           pair_index=pair_index, parity=np.arange(n, dtype=np.int64) % 2)
