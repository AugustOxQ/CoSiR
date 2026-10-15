"""Final review A2, item 2: rule section 5.1 re-derived from its text, compared with r9_a28_episodes.

Row space: the real read set's row_id, painting and dropped columns (no label or language column is read).
Language: synthetic, LANGUAGES[row_id % 27]. Labels: synthetic (style, genre per painting; emotion per row; a few
unknown). Scenarios: base (no set-aside), set-aside (values-subset and few paintings), stop (three set aside).
Writes fr_a2_s51.json beside this file.
"""
import dataclasses
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(F))
import r9_common as R  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_a28_episodes as AE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from src.eval.aspect_episodes import PaintingValueIndex, concat_episodes, episodes_sha256, validate_aspect_episodes  # noqa: E402

READ_SET = Path("/project/CoSiR/src/test/20261202_r9_scope/results/read_set_a28.parquet")
# rule 5.1 item 1, copied from the rule's text
RULE_LANGS = ["Burmese", "Darija", "Emakhuwa", "Hausa", "Hindi", "Igbo", "Indonesian", "IsiNdebele", "IsiXhosa",
              "IsiZulu", "Kazakh", "Kinyarwanda", "Korean", "Kyrgyz", "Malay", "Setswana", "Swahili", "Tagalog",
              "Tamil", "Thai", "Turkish", "Urdu", "Uzbek", "Vietnamese", "Yoruba", "arabic", "chinese"]
# rule 5 item 2
V = {"emotion": list(range(8)), "style": [9, 12, 17, 20, 21, 23, 24], "genre": [0, 1, 2, 4, 6, 7, 8, 9]}
PAIRS = [("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion")]
NAMES = ["emotion__style", "emotion__genre", "style__genre"]


class MyStop(Exception):
    pass


def my_shares(sizes, total=4096):
    tot = sum(sizes)
    exact = [Fraction(total * x, tot) for x in sizes]
    base = [int(e) for e in exact]                       # floor (non-negative)
    rem = [e - b for e, b in zip(exact, base)]
    left = total - sum(base)
    order = sorted(range(len(sizes)), key=lambda i: (-rem[i], i))   # largest remainder; ties to the lower l
    for i in order[:left]:
        base[i] += 1
    assert sum(base) == total
    return base


def my_pair(T, a, b, third, s, n=4096):
    lang = T["l"]
    active = list(range(1, 28))
    gone = []
    while True:
        pools = []
        for l in active:
            m = (lang == l) & ~T["dropped"] & (T["emotion"] >= 0) & (T["style"] >= 0) & (T["genre"] >= 0)
            pools.append(int(m.sum()))
        sh = my_shares(pools, n)
        built, failed = {}, []
        for l, nl in zip(active, sh):
            Rl = np.flatnonzero(lang == l)
            lab = {k: T[k][Rl] for k in ("emotion", "style", "genre")}
            grp = T["cl"][Rl]
            idx = PaintingValueIndex(lab, grp)
            rows = np.flatnonzero(~T["dropped"][Rl])
            try:
                ep = E.build_aspect_episodes_r6(lab, grp, rows, a, b, nl, 1000 * s + l, third=third, min_paintings=1,
                                                index=idx, values=(V[a], V[b]))
            except AssertionError as e:
                if "not eligible" not in str(e):
                    raise
                failed.append((l, "values_subset"))
                continue
            except RuntimeError as e:
                if "could not fill" not in str(e):
                    raise
                failed.append((l, "failed_draws"))
                continue
            validate_aspect_episodes(ep, lab, grp, idx, third=third)
            assert np.isin(E.members(ep), rows).all()
            built[l] = dataclasses.replace(ep, **{f: Rl[getattr(ep, f)] for f in E.FIELDS})
        if not failed:
            break
        gone += failed
        if len(gone) > 2:
            raise MyStop([RULE_LANGS[l - 1] for l, _ in gone])
        active = [l for l in active if l not in dict(failed)]
    pooled = concat_episodes([built[l] for l in active])
    return {"sha": episodes_sha256(pooled), "shares": {RULE_LANGS[l - 1]: x for l, x in zip(active, sh)},
            "part_sha": {RULE_LANGS[l - 1]: episodes_sha256(built[l]) for l in active},
            "set_aside": [RULE_LANGS[l - 1] for l, _ in gone], "reasons": {RULE_LANGS[l - 1]: r for l, r in gone},
            "language": np.concatenate([np.full(x, l) for l, x in zip(active, sh)])}


def table(scenario):
    t = pq.read_table(READ_SET, columns=["row_id", "painting", "dropped"]).to_pandas()
    rng = np.random.default_rng(20261014)
    rid = t["row_id"].to_numpy()
    lang = (rid % 27).astype(np.int64) + 1
    paint = t["painting"].to_numpy().astype(str)
    if scenario == "draws":                                              # Kyrgyz rows on two paintings only
        u0 = np.unique(paint)
        k = np.flatnonzero(lang == 14)
        paint = paint.copy()
        paint[k] = u0[k % 2]
    up, cl = np.unique(paint, return_inverse=True)
    st_p = rng.choice(list(range(27)), size=len(up))                   # style per painting (V and non-V codes)
    st_p[: 200] = rng.choice(V["style"], size=200)
    gn_p = rng.choice(list(range(11)), size=len(up))
    emo = rng.integers(0, 8, size=len(rid))
    emo[rng.random(len(rid)) < .02] = -1
    style, genre = st_p[cl].astype(np.int64), gn_p[cl].astype(np.int64)
    dropped = t["dropped"].to_numpy(dtype=bool).copy()
    dropped[rng.random(len(rid)) < .01] = True                          # a few empty-output drops (synthetic)
    if scenario in ("set_aside", "stop"):
        m = lang == 3                                                    # Emakhuwa: style 9 absent (values subset)
        style[m & (style == 9)] = 0
        m = lang == 14                                                   # Kyrgyz: genre 0 kept on one painting only
        p0 = np.flatnonzero(m & (genre == 0))
        keep = cl[p0[0]] if len(p0) else -1
        genre[m & (genre == 0) & (cl != keep)] = 3
    if scenario == "stop":
        for l in (5, 9):                                                 # Hindi, IsiXhosa: emotion 7 absent
            emo[(lang == l) & (emo == 7)] = 6
    names = np.asarray(RULE_LANGS)[lang - 1]
    tab = pd.DataFrame({"row_id": rid, "painting": paint, "language": names, "emotion": emo, "style": style,
                        "genre": genre, "held": False, "dropped": dropped})
    T = {"l": lang, "cl": cl.astype(np.int64), "dropped": dropped, "emotion": emo, "style": style, "genre": genre}
    return tab, T


out = {"rule_languages_in_python_sort_order": RULE_LANGS == sorted(RULE_LANGS),
       "code_languages_equal_rule": list(AE.LANGUAGES) == RULE_LANGS,
       "builder_seeds_55_57": AE.builder_seeds((55, 56, 57), 27) == [1000 * s + l for s in (55, 56, 57)
                                                                       for l in range(1, 28)],
       "default_rng_tuple_note": bool(np.random.default_rng([55, 0]).integers(0, 2**31) ==
                                      np.random.default_rng(55).integers(0, 2**31))}
# shares on adversarial sizes (ties, zeros excluded, large)
rng = np.random.default_rng(1)
mism = 0
for _ in range(2000):
    k = int(rng.integers(1, 28))
    sizes = [int(x) for x in rng.integers(1, 5000, size=k)]
    if rng.random() < .3:
        sizes = [int(rng.integers(1, 4)) * 64 for _ in range(k)]       # many exact ties
    mism += my_shares(sizes) != AE.shares(sizes, 4096)
out["shares_random_2000_mismatches"] = mism

SCEN = sys.argv[1:] or ["base", "set_aside", "stop"]
for scenario in SCEN:
    tab, T = table(scenario)
    t0 = time.time()
    ctx = AE.prepare(tab)
    res = {}
    for s in ((55, 56, 57) if scenario == "base" else (55,)):
        try:
            code = AE.build_seed_a28(ctx, V, s)
            code_err = None
        except AE.SetAsideStop as e:
            code, code_err = None, f"SetAsideStop: {str(e)[:160]}"
        mine, my_err = {}, None
        try:
            for (a, b, th), nm in zip(PAIRS, NAMES):
                mine[nm] = my_pair(T, a, b, th, s)
        except MyStop as e:
            my_err = f"stop: {e.args[0]}"
        r = {"code_stop": code_err, "my_stop": my_err}
        if code is not None and my_err is None:
            for nm in NAMES:
                m = mine[nm]
                r[nm] = {"sha_equal": m["sha"] == code.sha[nm], "shares_equal": m["shares"] == code.shares[nm],
                         "part_sha_equal": m["part_sha"] == code.part_sha[nm],
                         "set_aside_equal": m["set_aside"] == code.set_aside[nm],
                         "reasons_equal": m["reasons"] == code.set_aside_reasons[nm],
                         "language_equal": bool(np.array_equal(m["language"], code.language[nm])),
                         "set_aside": m["set_aside"], "reasons": m["reasons"],
                         "share_range": [min(m["shares"].values()), max(m["shares"].values())],
                         "n_languages": len(m["shares"])}
            r["pooled_n"] = int(code.n)
            r["parity_is_index_mod2"] = bool(np.array_equal(code.parity, np.arange(code.n) % 2))
        res[str(s)] = r
    res["seconds"] = round(time.time() - t0)
    out[scenario] = res
    print(scenario, json.dumps(res, default=str)[:600], flush=True)

(HERE / ("fr_a2_s51.json" if not sys.argv[1:] else f"fr_a2_s51_{'_'.join(SCEN)}.json")).write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
