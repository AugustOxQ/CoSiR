"""fr_c: re-derive rule section 7 items 2 to 6 (parsing, CRL, fusion with the failure fallback, DTS-CF, DTS-N, the
tuning, the chosen run, the stop count) with my own code from the rule's text, on a crafted smoke-shaped world
(seed 9001: 192 episodes, tuning subset 16 per pair), and compare with run_r6_dts.py's stage outputs, run through its
main() (load_context replaced by the crafted world). Then the held DTS path (frozen setting and lambdas) on the same
crafted world. Reviewer C; writes only under the scratch dir given as argv[1]."""
import json
import re
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402  (path setup only)
import run_r6_dts as RD  # noqa: E402  (the code under review, run through main())
import r6_dts as D  # noqa: E402  (only to give the code under review its fake encoder)
import r6_episodes as E  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores, fused_scores, LAMBDA_GRID, EDGE_EXTENSION  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SEED, NPER, NC, NP = 9001, 64, 13, 4
N = 3 * NPER
PAIRS = [("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion")]
assert [tuple(p) for p in R.PAIRS] == PAIRS
BLOCK = {"emotion": 0, "style": 64, "genre": 128}
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
rng = np.random.default_rng(777)

# ------------------------------------------------------------------ the crafted world
pair_index = np.repeat(np.arange(3), NPER)
n_rows = 14 * N + 8 * N + 50
lab = {a: rng.integers(0, 16, size=n_rows) for a in BLOCK}
VAL = {a: rng.random((16, 16)) + 0.2 for a in BLOCK}


def vec(a, v):
    x = np.zeros(512)
    x[BLOCK[a] + 16 * (v % 4): BLOCK[a] + 16 * (v % 4) + 16] = VAL[a][v]
    return x


rows = rng.permutation(14 * N).reshape(N, 14)
for e in range(N):
    a, b, t = PAIRS[pair_index[e]]
    va, vb = rng.integers(0, 16, size=2)
    lab[a][rows[e, 0]], lab[b][rows[e, 0]] = va, vb
    lab[a][rows[e, 1]], lab[b][rows[e, 1]] = va, (vb + 1 + rng.integers(0, 15)) % 16
    lab[a][rows[e, 2]], lab[b][rows[e, 2]] = (va + 1 + rng.integers(0, 15)) % 16, vb
    for j in range(3, 14):
        lab[a][rows[e, j]] = (va + 1 + rng.integers(0, 15)) % 16
        lab[b][rows[e, j]] = (vb + 1 + rng.integers(0, 15)) % 16
base = np.stack([sum(vec(a, lab[a][r]) * (1 if len(sys.argv) < 4 else float(sys.argv[3])) for a in BLOCK) for r in range(n_rows)])
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
noise = NOISE * rng.standard_normal((n_rows, 512)) * (np.arange(512) >= 192)
img = (base + noise + 0.05 * rng.standard_normal((n_rows, 512))).astype(np.float32)
txt = (base + noise + 0.05 * rng.standard_normal((n_rows, 512))).astype(np.float32)
ex = lambda: rng.integers(14 * N, n_rows, size=(N, NP)).astype(np.int64)  # noqa: E731
pooled = AspectEpisodes("mixed", "mixed", rows[:, 0].astype(np.int64), rows[:, 1:].astype(np.int64), ex(), ex(), ex(),
                        ex())
parity = np.arange(N) % 2
cos = cosine_scores(EvalInputs(img, txt), pooled)
eps = SimpleNamespace(seed=SEED, n=N, pooled=pooled, pair_index=pair_index, parity=parity,
                      sha={p: "0" * 64 for p in R.PAIR_NAMES})
ctx = SimpleNamespace(mode="selection", seed=SEED, n=N, n_total=N, eps=eps, pooled=pooled, parity=parity,
                      pair_index=pair_index, img=img, txt=txt, cos=cos, anchor_group=np.arange(N),
                      episodes_file_sha256="e" * 64)

# verbaliser answers: per wording a quality; some answers name the target aspect (with formatting noise), some junk,
# some empty after normalisation; listings with markers, duplicates, blanks, short lists.
NAMES_FMT = ["{n}", "  The {N}!\nsecond line", "\n\n'{n}'.", "{N}   of  painting", "“{n}”"]
JUNK = ["Colour use", "brush-work?", "!!!", "", "  \n ", "light", "One", "two"]


def answer(i, c, w):
    a, b, _ = PAIRS[pair_index[i]]
    tgt = a if c == "a" else b
    q = {"W1": 0.2, "W2": 0.7, "W3": 0.45, "W4": 0.6}[w]
    r = np.random.default_rng([i, ord(c), int(w[1])]).random(3)
    if r[0] < q:
        return NAMES_FMT[int(r[1] * len(NAMES_FMT))].format(n=tgt, N=tgt.title())
    return JUNK[int(r[2] * len(JUNK))]


# ------------------------------------------------------------------ my parsing, from the rule's text and the settings
def my_norm(s):
    s = re.sub(r"\s+", " ", s.lower())
    strip = lambda ch: ch.isspace() or unicodedata.category(ch)[0] == "P"  # noqa: E731
    while s and strip(s[0]):
        s = s[1:]
    while s and strip(s[-1]):
        s = s[:-1]
    return s


def my_phrase(ans):
    s = ans.strip()
    return my_norm(s.splitlines()[0]) if s else ""


MARK = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")


def my_values(ans, K):
    out = []
    for line in ans.splitlines():
        v = my_norm(MARK.sub("", line, count=1))
        if v and v not in out:
            out.append(v)
    return out[:K]


def listing_text(p, K):
    """A crafted listing for phrase p: names an aspect -> its block's words with mixed markers; 'one'/'two' -> short
    lists; else hashed words."""
    asp = [a for a in BLOCK if a in p]
    if p in ("one", "two"):
        return "1. word 300\n\n- word 300\n" if p == "one" else "1) word 301\n* word 302"
    if asp:
        a = asp[0]
        marks = ["1. ", "2) ", "- ", "* ", "• ", "  7.", ""]
        lines = [f"{marks[j % len(marks)]}Word {BLOCK[a] + 16 * (j % 4) + (j // 4)}" for j in range(K + 3)]
        lines.insert(2, lines[1])                       # a repeated line
        lines.insert(4, "   ")                          # a blank line
        return "\n".join(lines)
    k = sum(map(ord, p))
    return "\n".join(f"- word {200 + (k + 3 * j) % 300}" for j in range(K + 1))


class Enc:
    meta = {"class": "fr_c_onehot", "batch_size": 1}

    def encode_texts(self, texts, batch_size=1):
        out = []
        for t in texts:
            m = re.fullmatch(r"word (\d+)", t)
            if m:
                v = np.zeros(512)
                v[int(m.group(1))] = 1.0
            else:
                v = np.random.default_rng(abs(hash(t)) % (2 ** 32)).standard_normal(512)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


def my_emb(s):
    return Enc().encode_texts([s])[0].astype(np.float64)


# ------------------------------------------------------------------ my CRL term, fusion, cross-fit
def my_T(phr_by_cond, K, positions):
    T, failed, fails = {}, {}, {}
    an, cand = pooled.anchor[positions], pooled.candidates[positions]
    for c in CONDS:
        T[c] = {d: np.zeros((len(positions), NC), np.float32) for d in DIRS}
        failed[c] = np.zeros(len(positions), bool)
        fails[c] = {"empty_phrase": 0, "short_listing": 0}
        for k, p in enumerate(phr_by_cond[c]):
            if p == "":
                failed[c][k] = True
                fails[c]["empty_phrase"] += 1
                continue
            vals = my_values(listing_text(p, K), K)
            if len(vals) < 2:
                failed[c][k] = True
                fails[c]["short_listing"] += 1
                continue
            V = np.stack([my_emb(v) for v in vals])
            for d in DIRS:
                Q, C = (img, txt) if d == "i2t" else (txt, img)
                q = Q[an[k]].astype(np.float64)
                q /= np.linalg.norm(q)
                rq = V @ q
                for j in range(NC):
                    x = C[cand[k, j]].astype(np.float64)
                    x /= np.linalg.norm(x)
                    rc = V @ x
                    den = np.linalg.norm(rq) * np.linalg.norm(rc)
                    T[c][d][k, j] = np.float32(rq @ rc / den) if den > 0 else 0.0
    return T, failed, fails


def sub(x, pos):
    return {c: {d: np.asarray(x[c][d])[pos] for d in DIRS} for c in CONDS}


def my_fused(cs, T, lam, failed):
    f = fused_scores(cs, T, lam)
    z = fused_scores(cs, T, 0.0)
    for c in CONDS:
        for d in DIRS:
            f[c][d] = np.where(failed[c][:, None], z[c][d], f[c][d])
    return f


def crit(s, rows):
    m = per_anchor(sub(s, rows))
    return 0.5 * (m["r1"].mean() + m["gain"].mean())


def my_crossfit(cs, T, par, failed):
    picks, out = {}, {c: {d: np.zeros_like(cs[c][d]) for d in DIRS} for c in CONDS}
    for h in (0, 1):
        tune = par == h
        grid = list(LAMBDA_GRID)
        vals = [crit(my_fused(cs, T, lam, failed), tune) for lam in grid]
        best = grid[int(np.argmax(vals))]
        if best == 16.0:
            grid = grid + list(EDGE_EXTENSION)
            vals = [crit(my_fused(cs, T, lam, failed), tune) for lam in grid]
            best = grid[int(np.argmax(vals))]
        picks[h] = best
        f = my_fused(cs, T, best, failed)
        for c in CONDS:
            for d in DIRS:
                out[c][d][par != h] = f[c][d][par != h]
    return out, picks


def my_frozen(cs, T, picks, par, failed):
    out = {c: {d: np.zeros_like(cs[c][d]) for d in DIRS} for c in CONDS}
    for h in (0, 1):
        f = my_fused(cs, T, picks[h], failed)
        for c in CONDS:
            for d in DIRS:
                out[c][d][par == 1 - h] = f[c][d][par == 1 - h]
    return out


def my_cf(T):
    z = {c: {d: zscore_rows(torch.as_tensor(T[c][d], dtype=torch.float32)).numpy() for d in DIRS} for c in CONDS}
    m = {d: ((z["a"][d].astype(np.float64) + z["b"][d].astype(np.float64)) / 2).astype(np.float32) for d in DIRS}
    return {c: {d: m[d].copy() for d in DIRS} for c in CONDS}


def int4(x):
    y = np.asarray(x) * 4
    assert np.array_equal(y, np.round(y))
    return int(np.round(y).astype(np.int64).sum())


def jl(lam):
    return "inf" if lam == float("inf") else float(lam)


# ------------------------------------------------------------------ my computation
res = {}
tpos = np.concatenate([np.flatnonzero(pair_index == k)[:16] for k in range(3)])
mine = {"settings": {}}
for w in ("W1", "W2", "W3", "W4"):
    phr = {c: [my_phrase(answer(int(i), c, w)) for i in tpos] for c in CONDS}
    for K in (8, 16):
        T, failed, fails = my_T(phr, K, tpos)
        s, pk = my_crossfit(sub(cos, tpos), T, parity[tpos], failed)
        pa = per_anchor(s)
        mine["settings"][f"{w} K{K}"] = {"score_int": int4(pa["r1"]) + int4(pa["gain"]), "picks": pk, "fails": fails}
order = [f"W{w} K{k}" for w in range(1, 5) for k in (8, 16)]
best = order[0]
for nm in order[1:]:
    if mine["settings"][nm]["score_int"] > mine["settings"][best]["score_int"]:
        best = nm
mine["chosen"] = best
w, K = best.split()[0], int(best.split()[1][1:])
allpos = np.arange(N)
phr = {c: [my_phrase(answer(i, c, w)) for i in range(N)] for c in CONDS}
T, failed, fails = my_T(phr, K, allpos)
s_dts, pk_dts = my_crossfit(cos, T, parity, failed)
tcf = my_cf(T)
both = {c: failed["a"] & failed["b"] for c in CONDS}
s_cf, pk_cf = my_crossfit(cos, tcf, parity, both)
nphr = {c: [PAIRS[pair_index[i]][0 if c == "a" else 1] for i in range(N)] for c in CONDS}
TN, failedN, failsN = my_T(nphr, K, allpos)
s_n, pk_n = my_crossfit(cos, TN, parity, failedN)
san = {}
for KK in (8, 16):
    TT, ff, _ = my_T(nphr, KK, allpos)
    ss, pp = my_crossfit(cos, TT, parity, ff)
    san[KK] = {"gain_int4": int4(per_anchor(ss)["gain"]), "picks": pp}
pa_dts, pa_cf, pa_n = per_anchor(s_dts), per_anchor(s_cf), per_anchor(s_n)
mine.update(picks={"dts": pk_dts, "dts_cf": pk_cf, "dts_n": pk_n}, fails=fails, failsN=failsN,
            hits=int4(pa_dts["r1"]), sanity=san)

# ------------------------------------------------------------------ the code under review, through main()
work = OUT / "dts_world"
jobs = work / "jobs"
settings, ssha = D.load_settings()


def write_job(d, positions):
    d.mkdir(parents=True)
    arr = {"seed": np.int64(SEED), "episode_index": np.asarray(positions, np.int64),
           **{f: np.ascontiguousarray(getattr(pooled, f)[positions], np.int64)
              for f in ("pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")}}
    np.savez(d / "verbalise_input.npz", **arr)
    np.savez(d / "rows_manifest.npz", rows=np.unique(np.concatenate([arr[f].ravel() for f in arr if "pairs" in f])))
    import hashlib
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("rows_manifest.npz", "verbalise_input.npz")}


def write_ver(d, positions, wordings, ins):
    d.mkdir(parents=True)
    for wd in wordings:
        with open(d / f"phrases_{wd}.jsonl", "w", encoding="utf-8") as f:
            for i in positions:
                for c in CONDS:
                    f.write(json.dumps({"seed": SEED, "episode_index": int(i), "condition": c, "wording": wd,
                                        "answer": answer(int(i), c, wd)}) + "\n")
    (d / "provenance.json").write_text(json.dumps({"fingerprint": {"job": "r6_gpu_verbalise", "settings_sha256": ssha,
                                                                   "inputs_sha256": ins}}))


def write_lst(d, items):
    d.mkdir(parents=True)
    with open(d / "listings.jsonl", "w", encoding="utf-8") as f:
        for p, KK in items:
            f.write(json.dumps({"phrase": p, "K": KK, "answer": listing_text(p, KK)}) + "\n")
    (d / "provenance.json").write_text(json.dumps({"fingerprint": {"job": "r6_gpu_listing", "settings_sha256": ssha}}))


def items_of(jobdir):
    return [(r["phrase"], r["K"]) for r in map(json.loads, (jobdir / "listing_input.jsonl").read_text().splitlines())]


RD.load_context = lambda seed, path: ctx                    # the crafted world in place of the real data
RD.load_episodes_checked = lambda seed, path: eps
D.clip_cpu = lambda: Enc()
res_dir = work / "res"
common = ["--seed", str(SEED), "--out", str(res_dir), "--episodes", str(work / "eps.npz")]
codes = {}
codes["li_sanity"] = RD.main(["--stage", "list-input", "--for", "sanity", "--job-out", str(jobs / "l_san"), *common])
write_lst(work / "o/l_san", items_of(jobs / "l_san"))
codes["sanity"] = RD.main(["--stage", "sanity", "--listing-out", str(work / "o/l_san"), *common])
ins_t = write_job(jobs / "v_tune", tpos)
write_ver(work / "o/v_tune", tpos, ("W1", "W2", "W3", "W4"), ins_t)
vt = ["--verbalise-out", str(work / "o/v_tune"), "--verbalise-job", str(jobs / "v_tune")]
(work / "eps.npz").write_bytes(b"x")
codes["li_tune"] = RD.main(["--stage", "list-input", "--for", "tune", "--job-out", str(jobs / "l_tune"), *vt, *common])
write_lst(work / "o/l_tune", items_of(jobs / "l_tune"))
codes["tune"] = RD.main(["--stage", "tune", *vt, "--listing-out", str(work / "o/l_san"), "--listing-out",
                         str(work / "o/l_tune"), *common])
tune = json.loads((res_dir / "dts_tune.json").read_text())
ins_f = write_job(jobs / "v_full", allpos)
cw = tune["chosen"]["wording_id"]
write_ver(work / "o/v_full", allpos, (cw,), ins_f)
vf = ["--verbalise-out", str(work / "o/v_full"), "--verbalise-job", str(jobs / "v_full")]
codes["li_chosen"] = RD.main(["--stage", "list-input", "--for", "chosen", "--job-out", str(jobs / "l_ch"), *vf,
                              *common])
write_lst(work / "o/l_ch", items_of(jobs / "l_ch"))
codes["chosen"] = RD.main(["--stage", "chosen", *vf, "--listing-out", str(work / "o/l_ch"), *common])
codes["stop"] = RD.main(["--stage", "stop", "--seed", str(SEED), "--out", str(res_dir), "--clock-start",
                         "2026-10-09 12:29"])
ch = json.loads((res_dir / "dts_seed9001.json").read_text())
stop = json.loads((res_dir / "dts_stop.json").read_text())
san_rec = json.loads((res_dir / "dts_sanity.json").read_text())

# ------------------------------------------------------------------ compare
cmp = {"codes": codes}
cmp["tune_scores_equal"] = all(tune["settings"][k]["score_int"] == v["score_int"] for k, v in mine["settings"].items())
cmp["tune_picks_equal"] = all(tune["settings"][k]["picks"] == {str(h): jl(p) for h, p in v["picks"].items()}
                              for k, v in mine["settings"].items())
cmp["tune_fails_equal"] = all(tune["settings"][k]["failures"] == v["fails"] for k, v in mine["settings"].items())
cmp["chosen"] = [tune["chosen"]["setting"], mine["chosen"]]
cmp["picks_equal"] = {s: ch["picks"][s] == {str(h): jl(p) for h, p in mine["picks"][s].items()} for s in mine["picks"]}
cmp["picks"] = ch["picks"]
cmp["fails_equal"] = [ch["failures"]["dts"] == mine["fails"], ch["failures"]["dts_n"] == mine["failsN"]]
cmp["sanity_equal"] = all(san_rec["per_K"][str(k)]["gain_int4_sum"] == v["gain_int4"]
                          and san_rec["per_K"][str(k)]["picks"] == {str(h): jl(p) for h, p in v["picks"].items()}
                          for k, v in mine["sanity"].items())
with np.load(res_dir / "dts_seed9001_per_anchor.npz") as z:
    cmp["per_anchor_exact"] = {nm: all(np.array_equal(z[f"{nm}__{m}"], pa[m]) for m in ("r1", "gain", "other", "swap",
                                                                                         "strict"))
                               for nm, pa in (("dts", pa_dts), ("dts_cf", pa_cf), ("dts_n", pa_n))}
cmp["hits"] = [stop["hits"], mine["hits"]]
cmp["stop"] = {k: stop[k] for k in ("built", "stop", "dts_above_aff", "reason")}
cmp["n_failed_rows"] = {c: int(failed[c].sum()) for c in CONDS}
cmp["any_inf_pick"] = any(p == float("inf") for s in mine["picks"].values() for p in s.values()) or any(
    p == float("inf") for v in mine["settings"].values() for p in v["picks"].values())
cmp["inf_with_failures"] = [k for k, v in mine["settings"].items()
                            if any(p == float("inf") for p in v["picks"].values())
                            and sum(sum(f.values()) for f in v["fails"].values()) > 0]

# ------------------------------------------------------------------ held path: frozen setting and picks applied as is
# A 'held' world: the same crafted rows, other episodes (a permutation of the same episodes; held seed key 52).
perm = np.random.default_rng(5).permutation(N)
hpooled = AspectEpisodes("mixed", "mixed", *(np.asarray(getattr(pooled, f))[perm] for f in
                                             ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img",
                                              "pairs_b_txt")))
hpi = pair_index[perm]
hcos = cosine_scores(EvalInputs(img, txt), hpooled)
hctx = SimpleNamespace(seed=52, n=N, pooled=hpooled, img=img, txt=txt, cos=hcos, parity=np.arange(N) % 2,
                       pair_index=hpi)
answers = {(52, i, c, cw): answer(int(perm[i]), c, cw) for i in range(N) for c in CONDS}
lst_all = {}
for p in {my_phrase(a) for a in answers.values()} | set(BLOCK):
    for KK in (8, 16):
        if p:
            lst_all[(p, KK)] = listing_text(p, KK)
emb = D.ValueEmbedder(None, encoder_factory=lambda: Enc())
held = D.held_dts_scores(hctx, answers, lst_all, ch, emb)
pk = {s: {int(h): (float("inf") if v == "inf" else float(v)) for h, v in ch["picks"][s].items()} for s in ch["picks"]}
Kc = int(ch["K"])
pooled_saved, pooled = pooled, hpooled
hphr = {c: [my_phrase(answers[(52, i, c, cw)]) for i in range(N)] for c in CONDS}
HT, hf, _ = my_T(hphr, Kc, np.arange(N))
hn = {c: [PAIRS[hpi[i]][0 if c == "a" else 1] for i in range(N)] for c in CONDS}
HTN, hfn, _ = my_T(hn, Kc, np.arange(N))
pooled = pooled_saved
hpar = np.arange(N) % 2
m_dts = per_anchor(my_frozen(hcos, HT, pk["dts"], hpar, hf))
m_cf = per_anchor(my_frozen(hcos, my_cf(HT), pk["dts_cf"], hpar, {c: hf["a"] & hf["b"] for c in CONDS}))
m_n = per_anchor(my_frozen(hcos, HTN, pk["dts_n"], hpar, hfn))
cmp["held_exact"] = {nm: all(np.array_equal(held[nm][m], mm[m]) for m in ("r1", "gain", "other", "swap", "strict"))
                     for nm, mm in (("dts", m_dts), ("dts_cf", m_cf), ("dts_n", m_n))}
cmp["held_cf_gain_zero"] = bool((m_cf["gain"] == 0).all())
print(json.dumps(cmp, default=str))
(F / "final_review" / ("fr_c_dts" + ("" if len(sys.argv) < 3 else "_" + sys.argv[2]) + ".json")).write_text(json.dumps(cmp, indent=1, default=str))
