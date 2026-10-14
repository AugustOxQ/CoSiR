"""ArtELingo-28 per-language episode builder (ticket 17; DECISION_RULE.md section 5.1 "language-homogeneous (option B)",
section 5 item 3, section 12 row C4; R6 rule section 5 item 3's hash guard).

Labels only: no feature file is opened and nothing is scored. One seed's three pairs are built by section 5.1:
  1. languages in Python sort order, l = 1 to L (27 in the data);
  2. shares n_l by largest remainders of the pool sizes (ties to the lower l), summing to n_per_pair (4,096);
  3. per language, pair and seed, r9_episodes.build_aspect_episodes_r6 (unchanged) on the language's sub-arrays:
     labels[a][R_l], groups[R_l] (global painting index), PaintingValueIndex over the sub-arrays, rows = the kept
     positions in R_l, n_episodes = n_l, values = (V_a, V_b), min_paintings = 1, seed 1000 * s + l;
     validate_aspect_episodes on every part; member rows in R_l; global row positions = R_l[.];
  4. the set-aside passes (section 5.1 item 6): only the values-subset AssertionError and the failed-draw RuntimeError
     set a language aside, the shares are re-split over the rest and a new pass starts; more than two set aside for
     one pair and seed raises SetAsideStop (the read stops before any score);
  5. assembly by concat_episodes in l order; per-pair SHA-256 (episodes_sha256 of the concatenation) and the part
     hashes (episodes_sha256 of each language's part, in global row positions);
  6. the hash guard (hash_guard).

Row space and formats (index "Shared interfaces"; ticket 25's C13 re-derivation compares against these):
  * the table is results/read_set_a28.parquet: one row per read row in `row_id` order. A row's global position k is
    its position in that table (0 to 74,568), NOT its CSV `row_id`. Every episode array holds positions k.
  * the bootstrap cluster code is cl = np.unique(painting names, return_inverse=True)[1] over all table rows (the
    inverse index of the sorted painting names); it is also the `groups` the builder sees.
  * part hash: r9_episodes' episodes_sha256 of the part, whose arrays are int64 global positions in the builder's
    field order, with the pair's aspect names; the per-pair hash is the same function on the l-ordered concatenation.
  * the empty-output drops: pool sizes and `rows` count kept rows only (dropped False); the label index counts all of
    R_l, so a dropped row's labels still count for `lacks`. The feasibility run passes dropped = all False.

Guards carry a `# guard:<name>` marker; the tests replace each on a copy and show the scenario then fails.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (before anything that imports src: it puts MAIN's src first and checks it)
import r9_episodes as E  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_episodes import (AspectEpisodes, PaintingValueIndex, concat_episodes,  # noqa: E402
                                      episodes_sha256, validate_aspect_episodes)

MAX_SET_ASIDE = 2                                    # rule section 5.1 item 6 and section 10
TABLE_COLUMNS = ("row_id", "painting", "language", "emotion", "style", "genre", "held", "dropped")
LEDGER_DIR = R.MAIN / "docs/superpowers"
SEED_LEDGER = LEDGER_DIR / "episode_seed_ledger.md"
HELD_LEDGER = LEDGER_DIR / "held_ledger.md"
H5_SEEDS = (52, 53, 54)
LANGUAGES = ("Burmese", "Darija", "Emakhuwa", "Hausa", "Hindi", "Igbo", "Indonesian", "IsiNdebele", "IsiXhosa",
             "IsiZulu", "Kazakh", "Kinyarwanda", "Korean", "Kyrgyz", "Malay", "Setswana", "Swahili", "Tagalog",
             "Tamil", "Thai", "Turkish", "Urdu", "Uzbek", "Vietnamese", "Yoruba", "arabic", "chinese")  # rule 5.1 item 1
N_LANGUAGES = len(LANGUAGES)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


class SetAsideStop(RuntimeError):
    """More than two languages set aside for one pair and seed: the read stops before any score (rule section 10)."""


# ---------------------------------------------------------------- languages, shares, the set-aside classifier

def languages_of(language) -> list:
    """The language names in Python sort order (index l = position + 1)."""
    return sorted(set(str(x) for x in language))


def shares(sizes, total) -> list:
    """n_l = floor(total * size_l / sum(sizes)); the rest, one each, to the largest remainders, ties to the lower l.
    Integer arithmetic only. Sum = total (asserted by the caller and here)."""
    sizes = [int(x) for x in sizes]
    _require(sizes and min(sizes) >= 0 and sum(sizes) > 0, f"pool sizes {sizes}")
    S = sum(sizes)
    base = [total * x // S for x in sizes]
    rem = [total * x % S for x in sizes]
    left = total - sum(base)
    order = sorted(range(len(sizes)), key=lambda i: (-rem[i], i))
    for i in order[:left]:
        base[i] += 1
    _require(sum(base) == total, f"shares {base} do not sum to {total}")
    return base


def set_aside_reason(exc):
    """'values_subset' for the builder's values-subset AssertionError, 'failed_draws' for its failed-draw
    RuntimeError, None for everything else (those propagate). Told apart by the builder's own messages."""
    if isinstance(exc, AssertionError) and "are not eligible on the pool" in str(exc):
        return "values_subset"
    if isinstance(exc, RuntimeError) and "could not fill" in str(exc):
        return "failed_draws"
    return None  # mut:no_other


# ---------------------------------------------------------------- the row space

def prepare(table, dropped=None, expected_languages=LANGUAGES) -> SimpleNamespace:
    """Per-language sub-arrays of the ArtELingo-28 table (read_set_a28.parquet or a frame with its columns).

    ``dropped`` overrides the table's column (the feasibility run passes all False). ``expected_languages``: the
    language names the table must hold, exactly (the rule's 27; the unit tests pass their synthetic names).
    -> SimpleNamespace(languages,
    R (global positions per language), labels, groups, index, kept (local positions of kept rows), pool_size,
    cl, n_rows)."""
    _require(set(TABLE_COLUMNS) <= set(table.columns), f"table columns {list(table.columns)}")
    n = len(table)
    row_id = table["row_id"].to_numpy()
    _require((np.diff(row_id) > 0).all(), "row_id is not strictly increasing")
    drop = table["dropped"].to_numpy(dtype=bool) if dropped is None else np.asarray(dropped, dtype=bool)
    _require(drop.shape == (n,), f"dropped has shape {drop.shape}, not ({n},)")
    lang = table["language"].to_numpy().astype(str)
    languages = languages_of(lang)
    _require(languages == sorted(expected_languages),
             f"language names {languages} are not the expected {sorted(expected_languages)}")
    lab = {a: table[a].to_numpy(dtype=np.int64) for a in R.ASPECTS}
    cl = np.unique(table["painting"].to_numpy().astype(str), return_inverse=True)[1].astype(np.int64)
    out = SimpleNamespace(languages=languages, R=[], labels=[], groups=[], index=[], kept=[], pool_size=[], cl=cl,
                          n_rows=n)
    for name in languages:
        r = np.flatnonzero(lang == name).astype(np.int64)
        sub = {a: lab[a][r] for a in R.ASPECTS}
        kept = np.flatnonzero(~drop[r]).astype(np.int64)
        known = (sub["emotion"][kept] >= 0) & (sub["style"][kept] >= 0) & (sub["genre"][kept] >= 0)
        out.R.append(r)
        out.labels.append(sub)
        out.groups.append(cl[r])
        out.index.append(PaintingValueIndex(sub, cl[r]))
        out.kept.append(kept)
        out.pool_size.append(int(known.sum()))
    return out


# ---------------------------------------------------------------- one pair, one seed

def _build_pair(ctx, aspects, name, value_sets, s, n_per_pair) -> SimpleNamespace:
    a, b, third = aspects
    L = len(ctx.languages)
    active = list(range(1, L + 1))                  # l, 1-based, in sort order; set-aside languages leave it
    set_aside = []                                  # (name, reason) in the order set aside
    while True:
        sizes = [ctx.pool_size[l - 1] for l in active]
        share = shares(sizes, n_per_pair)
        _require(sum(share) == n_per_pair, f"{name}: shares sum to {sum(share)}, not {n_per_pair}")
        built, failed = {}, {}
        for l, n in zip(active, share):
            _require(n > 0, f"{name}: {ctx.languages[l - 1]} gets no episodes (share {n}, pool {ctx.pool_size[l - 1]})")
            seed = 1000 * int(s) + l
            try:
                built[l] = E.build_aspect_episodes_r6(
                    labels=ctx.labels[l - 1], groups=ctx.groups[l - 1], rows=ctx.kept[l - 1], aspect_a=a, aspect_b=b,
                    n_episodes=n, seed=seed, third=third, min_paintings=1, index=ctx.index[l - 1],
                    values=(value_sets[a], value_sets[b]))
            except Exception as exc:
                reason = set_aside_reason(exc)
                if reason is None:
                    raise
                failed[l] = reason
                continue
            # every part of every pass is validated, not only the last pass's (rule 5.1 item 6: a validation failure
            # stops the step)
            validate_aspect_episodes(built[l], ctx.labels[l - 1], ctx.groups[l - 1], ctx.index[l - 1], third=third)
        if not failed:  # mut:last_pass
            break
        set_aside += [(ctx.languages[l - 1], r) for l, r in failed.items()]
        if len(set_aside) > MAX_SET_ASIDE:  # guard:stop
            raise SetAsideStop(f"{name}, seed {s}: {len(set_aside)} languages set aside (more than "
                               f"{MAX_SET_ASIDE}): {set_aside}; the read stops before any score")
        active = [l for l in active if l not in failed]
    parts, part_sha, lang_of = {}, {}, []
    for l, n in zip(active, share):
        ep = built[l]
        r = ctx.R[l - 1]
        every = E.members(ep)
        in_kept = np.zeros(len(r), dtype=bool)
        in_kept[ctx.kept[l - 1]] = True
        _require(in_kept[every].all(), f"{name}: {ctx.languages[l - 1]}: member rows outside the language's "
                                       f"kept rows")  # guard:members_in_language
        glob = AspectEpisodes(a, b, *(r[getattr(ep, f)] for f in E.FIELDS))
        parts[ctx.languages[l - 1]] = glob
        part_sha[ctx.languages[l - 1]] = episodes_sha256(glob)
        lang_of.append(np.full(n, l, dtype=np.int64))
    ordered = [parts[ctx.languages[l - 1]] for l in active]
    pooled = concat_episodes(ordered)
    _require(len(pooled.anchor) == n_per_pair, f"{name}: {len(pooled.anchor)} episodes, not {n_per_pair}")
    return SimpleNamespace(ep=pooled, sha=episodes_sha256(pooled), parts=parts, part_sha=part_sha,
                           shares={ctx.languages[l - 1]: n for l, n in zip(active, share)},
                           set_aside=[nm for nm, _ in set_aside], set_aside_reasons=dict(set_aside),
                           language=np.concatenate(lang_of))


def build_seed_a28(ctx, value_sets, s, n_per_pair=R.N_PER_PAIR) -> SimpleNamespace:
    """The three pairs of run_baselines.PAIRS for read seed ``s`` (builder seed 1000 * s + l).

    -> SimpleNamespace(seed, per_pair (3 AspectEpisodes, PAIRS order, global positions), pooled (concat_episodes),
    sha {pair: sha}, n, pair_index, parity (as r9_episodes.build_seed), shares {pair: {language: n}}, set_aside
    {pair: [language]}, part_sha {pair: {language: sha}}, parts {pair: {language: AspectEpisodes}},
    language {pair: (n_per_pair,) int64 of l per episode})."""
    _require(set(value_sets) >= set(R.ASPECTS), f"value sets lack an aspect: {sorted(value_sets)}")
    n_per_pair = int(n_per_pair)
    _require(n_per_pair > 0, "n_per_pair must be positive")
    res = {name: _build_pair(ctx, aspects, name, value_sets, s, n_per_pair)
           for aspects, name in zip(R.PAIRS, R.PAIR_NAMES)}
    per_pair = [res[p].ep for p in R.PAIR_NAMES]
    pooled = concat_episodes(per_pair)
    n = len(pooled.anchor)
    return SimpleNamespace(
        seed=int(s), per_pair=per_pair, pooled=pooled, sha={p: res[p].sha for p in R.PAIR_NAMES}, n=n,
        pair_index=np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), n_per_pair),
        parity=np.arange(n, dtype=np.int64) % 2, shares={p: res[p].shares for p in R.PAIR_NAMES},
        set_aside={p: res[p].set_aside for p in R.PAIR_NAMES},
        set_aside_reasons={p: res[p].set_aside_reasons for p in R.PAIR_NAMES},
        part_sha={p: res[p].part_sha for p in R.PAIR_NAMES}, parts={p: res[p].parts for p in R.PAIR_NAMES},
        language={p: res[p].language for p in R.PAIR_NAMES})


# ---------------------------------------------------------------- the hash guard (rule section 5.1 item 5)

def builder_seeds(read_seeds, n_languages) -> list:
    """The builder seeds 1000 * s + l, l = 1 to n_languages (81 for seeds 55 to 57 and 27 languages)."""
    return [1000 * int(s) + l for s in read_seeds for l in range(1, int(n_languages) + 1)]


def h5_hashes(path=None) -> dict:
    """{seed: {pair: sha}} of the nine per-pair hashes in held_ledger.md's row H5 (seeds 52 to 54)."""
    text = Path(path or HELD_LEDGER).read_text()
    rows = [ln for ln in text.splitlines() if ln.startswith("| H5 |")]
    _require(len(rows) == 1, f"{path or HELD_LEDGER}: {len(rows)} rows for H5, not one")
    out = {}
    for s, *hs in re.findall(r"seed (\d+): emotion × style ([0-9a-f]{64}), emotion × genre ([0-9a-f]{64}), "
                             r"style × genre ([0-9a-f]{64})", rows[0]):
        out[int(s)] = dict(zip(R.PAIR_NAMES, hs))
    flat = [h for hs in out.values() for h in hs.values()]
    _require(len(flat) == 9 and tuple(sorted(out)) == H5_SEEDS,
             f"H5 row holds {len(flat)} per-pair hashes for seeds {sorted(out)}, not the nine of seeds 52 to 54")
    return out


def ledger_seeds(path=None) -> dict:
    """{seed: seed cell} of the integers in the Seed column of every row of episode_seed_ledger.md except round 9's
    rows and the rows whose status is 'free'."""
    out = {}
    for line in Path(path or SEED_LEDGER).read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Seed" or set(cells[0]) <= set("-: "):
            continue
        status = cells[1]
        if "round 9" in line.lower():  # guard:round9
            continue
        if status == "free":  # guard:free
            continue
        for m in re.findall(r"\d+", cells[0]):
            out[int(m)] = cells[0]
    return out


def assert_seeds_free(seeds, path=None) -> dict:
    """The builder seeds are distinct and absent from every ledger row except round 9's."""
    seeds = [int(x) for x in seeds]
    _require(len(set(seeds)) == len(seeds), f"builder seeds repeat: {sorted(x for x in set(seeds) if seeds.count(x) > 1)}")
    used = ledger_seeds(path)
    clash = sorted(x for x in seeds if x in used)
    _require(not clash, "builder seeds in the ledger's rows of other rounds: "
                        + "; ".join(f"{x} (row '{used[x]}')" for x in clash))  # guard:seeds_free
    return {"passed": True, "n_seeds": len(seeds)}


def hash_guard(new: dict, builder_seed_list, held_ledger=None, seed_ledger=None, recorded=None,
               n_languages=N_LANGUAGES) -> dict:
    """Rule section 5.1 item 5 (R6 rule section 5 item 3): the per-pair hashes ``new`` ({seed: {pair: sha}}, nine for
    the read) differ from each other, from H5's nine and from every recorded non-smoke hash; the builder seeds are
    absent from every ledger row except round 9's. ``builder_seed_list`` must be exactly the read's own
    builder_seeds(sorted(new), n_languages) (an empty or partial list would check nothing). ``recorded`` defaults to
    r9_episodes.recorded_hashes()."""
    expected_seeds = builder_seeds(sorted(new), n_languages)
    _require(list(builder_seed_list) == expected_seeds,
             f"the builder seed list ({len(list(builder_seed_list))} seeds) is not the read's 1000 s + l for read "
             f"seeds {sorted(new)} and {n_languages} languages ({len(expected_seeds)} seeds)")  # guard:seed_list
    h5 = h5_hashes(held_ledger)
    recorded = E.recorded_hashes() if recorded is None else recorded
    seeds = assert_seeds_free(builder_seed_list, seed_ledger)  # guard:seeds
    new_set = {h: (s, p) for s, hs in new.items() for p, h in hs.items()}
    h5_clash = [f"new seed {new_set[h][0]} {new_set[h][1]} equals H5 seed {s} {p}"
                for s, hs in h5.items() for p, h in hs.items() if h in new_set]
    _require(not h5_clash, "episode hashes equal H5's: " + "; ".join(h5_clash))  # guard:h5
    distinct = E.assert_distinct(new, recorded)  # guard:recorded
    return {"passed": True, "h5_hashes": 9, "builder_seeds": seeds["n_seeds"], **distinct}
