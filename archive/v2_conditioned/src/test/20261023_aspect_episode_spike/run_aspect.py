"""Throwaway spike: aspect episodes. The condition names an aspect (emotion or art style) through example pairs whose
values differ from the anchor's. Can factor models (and simple baselines) pick the right aspect?

Run from the repo root:  python src/test/20261023_aspect_episode_spike/run_aspect.py
Only selection rows are read (all other feature/code rows NaN). Nothing here is part of the pipeline.
"""
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
_p = ROOT / "src/test/20261018_affect_factor_learning/run_affect.py"
_s = importlib.util.spec_from_file_location("run_affect", _p)
ra = importlib.util.module_from_spec(_s)
sys.modules["run_affect"] = ra
_s.loader.exec_module(ra)
grid, sel = ra.grid, ra.sel

from src.data.artelingo import ANNOTATIONS_PATH, join_captions  # noqa: E402
from src.data.sampling import draw_distinct  # noqa: E402
from src.eval.condition_eval import _target_first, paired_bootstrap  # noqa: E402
from src.eval.label_episodes import EMOTION_CATCH_ALL, STANDARD_MIN_PAINTINGS_PER_LABEL, tie_aware_rank  # noqa: E402
from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes  # noqa: E402

CONDS, DIRS = ("emotion", "style"), ("i2t", "t2i")
N_EP, SEED, BETA = 4096, 42, 0.3
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
OUT = HERE / "results"
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


def tt(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ------------------------------------------------------------------ setup
data = ra.load_artelingo()
cache, prep, meta, _ = grid.load_grid()
prep_record = json.loads((ra.CACHE / "affect_prepare.json").read_text())
sl = np.asarray(cache["selection"], dtype=np.int64)
groups = np.asarray(cache["groups"])
in_sel = grid._row_mask(len(groups), sl)
img_raw, txt_raw = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
assert np.isnan(img_raw[~in_sel]).all() and np.isnan(txt_raw[~in_sel]).all()
emotions, styles = np.asarray(data.emotions), np.asarray(data.art_styles)
checks = {}

SE_ic, SE_tc, _ = ra.model_codes("SE", data, cache, prep_record)
C0_ic, C0_tc, _ = ra.model_codes("C0", data, cache, prep_record)
R3_ic, R3_tc, _ = ra.model_codes("R3", data, cache, prep_record)
codes = {"SE": (SE_ic, SE_tc), "C0": (C0_ic, C0_tc), "R3": (R3_ic, R3_tc)}
log("loaded features and codes")


def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


img_n, txt_n = nrm(img_raw), nrm(txt_raw)

# ------------------------------------------------------------------ CHECK 2: SE naive b0.3 reproduces stored ranks
episodes_old, _, _ = ra.selection_episodes(data, cache)
refs = dict(np.load(ra.RESULTS / "selection_ranks.npz"))
for lab in ("emotion", "art_style"):
    e = episodes_old[lab]
    assert in_sel[np.column_stack([e.anchor, e.positive, e.supports, e.contrasts, e.distractors])].all()
    w = ra.label_episode_weights(SE_ic, SE_tc, e)
    c = np.concatenate([e.positive[:, None], e.distractors], axis=1)
    for d in DIRS:
        qf, cf, qc, cc = (img_raw, txt_raw, SE_ic, SE_tc) if d == "i2t" else (txt_raw, img_raw, SE_tc, SE_ic)
        r = tie_aware_rank(conditional_score(tt(qf[e.anchor]), tt(cf[c]), tt(qc[e.anchor]), tt(cc[c]), w, BETA)).numpy()
        assert np.array_equal(r, refs[f"naive__SE__0.3__{lab}__{d}"]), f"SE naive mismatch {lab} {d}"
log("CHECK 2 passed: SE naive beta 0.3 reproduces stored selection ranks exactly (both labels, both directions)")
checks["check2_se_naive_reproduces_stored"] = "exact"

# ------------------------------------------------------------------ aspect episodes
rows = sl
row_em, row_st = emotions[rows], styles[rows]


def n_paintings(values, v):
    return len(np.unique(groups[rows[values == v]]))


ok_em = sorted(e for e in np.unique(row_em) if e != EMOTION_CATCH_ALL and n_paintings(row_em, e) >= STANDARD_MIN_PAINTINGS_PER_LABEL)
ok_st = sorted(s for s in np.unique(row_st) if n_paintings(row_st, s) >= STANDARD_MIN_PAINTINGS_PER_LABEL)
log(f"eligible emotions {len(ok_em)}: {ok_em}")
log(f"eligible styles {len(ok_st)}: {ok_st}")
has_emotion = {e: np.isin(groups, np.unique(groups[emotions == e])) for e in np.unique(emotions)}
anchors = rows[np.isin(row_em, ok_em) & np.isin(row_st, ok_st)]
rng = np.random.default_rng(SEED)
F_ = {k: [] for k in ("anchor", "cands", "pe_x", "pe_y", "ps_x", "ps_y")}
failures = 0
while len(F_["anchor"]) < N_EP:
    a = int(anchors[rng.integers(len(anchors))])
    e, s = emotions[a], styles[a]
    clean_e = ~has_emotion[e][rows]
    used = {groups[a]}
    try:
        p_emo = draw_distinct(rng, rows[(row_em == e) & (row_st != s)], groups, used, 1)
        p_sty = draw_distinct(rng, rows[(row_st == s) & clean_e], groups, used, 1)
        negs = draw_distinct(rng, rows[(row_st != s) & clean_e], groups, used, 11)
        pe_x, pe_y, ps_x, ps_y = [], [], [], []
        vs = rng.choice([v for v in ok_em if v != e], 4, replace=False)
        for v in vs:
            pool = rows[(row_em == v) & (row_st != s)]
            x = draw_distinct(rng, pool, groups, used, 1)[0]
            y = draw_distinct(rng, pool[styles[pool] != styles[x]], groups, used, 1)[0]
            pe_x.append(x), pe_y.append(y)
        ts = rng.choice([t for t in ok_st if t != s], 4, replace=False)
        for t in ts:
            pool = rows[(row_st == t) & clean_e & (row_em != e)]
            x = draw_distinct(rng, pool, groups, used, 1)[0]
            y = draw_distinct(rng, pool[emotions[pool] != emotions[x]], groups, used, 1)[0]
            ps_x.append(x), ps_y.append(y)
    except ValueError:
        failures += 1
        if failures > 10 * N_EP:
            raise RuntimeError("could not build enough aspect episodes")
        continue
    for k, v in zip(F_, [a, p_emo + p_sty + negs, pe_x, pe_y, ps_x, ps_y]):
        F_[k].append(v)
EP = {k: np.asarray(v, dtype=np.int64) for k, v in F_.items()}
n = N_EP
log(f"built {n} aspect episodes, {failures} draw failures")

# constraint assertions on the final arrays
anc, cand = EP["anchor"], EP["cands"]
allrows = np.concatenate([anc[:, None], cand, EP["pe_x"], EP["pe_y"], EP["ps_x"], EP["ps_y"]], axis=1)
assert allrows.shape == (n, 30) and in_sel[allrows].all(), "episode row outside selection"
g = groups[allrows]
assert all(len(set(r)) == 30 for r in g.tolist()), "paintings not all distinct (incl. anchor)"
okE, okS = set(ok_em), set(ok_st)
for i in range(n):
    a = anc[i]
    e, s = emotions[a], styles[a]
    assert e in okE and s in okS
    clean = ~has_emotion[e]
    c = cand[i]
    assert emotions[c[0]] == e and styles[c[0]] != s
    assert styles[c[1]] == s and clean[c[1]]
    assert (styles[c[2:]] != s).all() and clean[c[2:]].all()
    px, py, qx, qy = EP["pe_x"][i], EP["pe_y"][i], EP["ps_x"][i], EP["ps_y"][i]
    v = emotions[px]
    assert (emotions[py] == v).all() and len(set(v)) == 4 and all(x in okE and x != e for x in v)
    assert (styles[px] != styles[py]).all() and (styles[px] != s).all() and (styles[py] != s).all()
    t = styles[qx]
    assert (styles[qy] == t).all() and len(set(t)) == 4 and all(x in okS and x != s for x in t)
    assert (emotions[qx] != emotions[qy]).all() and (emotions[qx] != e).all() and (emotions[qy] != e).all()
    assert clean[qx].all() and clean[qy].all()
log("CHECK 1 passed: every episode row is a selection row; all construction constraints hold on the final arrays")
checks["check1_constraints"] = "all asserted"

# condition tensors: (cands target-first, supports (x,y), contrasts (x,y))
COND = {
    "emotion": dict(cands=_target_first(cand, 0), sup=(EP["pe_x"], EP["pe_y"]), con=(EP["ps_x"], EP["ps_y"])),
    "style": dict(cands=_target_first(cand, 1), sup=(EP["ps_x"], EP["ps_y"]), con=(EP["pe_x"], EP["pe_y"])),
}
assert (COND["emotion"]["cands"][:, 0] == cand[:, 0]).all() and (COND["style"]["cands"][:, 0] == cand[:, 1]).all()
assert (COND["emotion"]["cands"][:, 1] == cand[:, 1]).all() and (COND["style"]["cands"][:, 1] == cand[:, 0]).all()


def zs(t):
    t = t.double()
    mu, sd = t.mean(1, keepdim=True), t.std(1, unbiased=False, keepdim=True)
    return torch.where(sd > 0, (t - mu) / sd.clamp_min(1e-300), torch.zeros_like(t))


def cos_q(q, c):
    return F.cosine_similarity(q[:, None, :], c, dim=-1)


def gather(cond, d):
    """Feature views for (cond, direction): query feat, cand feats, plus the example rows."""
    C = COND[cond]
    a = anc
    if d == "i2t":
        q, c = img_n[a], txt_n[C["cands"]]
    else:
        q, c = txt_n[a], img_n[C["cands"]]
    return tt(q), tt(c)


def agree_weights(ic, tc, cond, relu=True, l1=True):
    C = COND[cond]
    sx, sy = C["sup"]
    kx, ky = C["con"]
    w = (ic[sx] * tc[sy]).mean(1) - (ic[kx] * tc[ky]).mean(1)
    w = tt(w)
    if relu:
        w = F.relu(w)
    if l1:
        s = w.abs().sum(1, keepdim=True)
        w = torch.where(s > 0, w / s.clamp_min(1e-30), torch.zeros_like(w))
    return w


# ------------------------------------------------------------------ CHECK 4 + names embeddings
from transformers import AutoModel, AutoProcessor  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
clip = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)


@torch.no_grad()
def encode_txt(texts):
    out = []
    for i in range(0, len(texts), 64):
        t = proc(text=list(texts[i:i + 64]), return_tensors="pt", padding="max_length", truncation=True).to(dev)
        o = clip.text_model(**t)
        out.append(clip.text_projection(o.pooler_output).float().cpu())
    return torch.cat(out).numpy()


import json as _json  # noqa: E402
ann = _json.load(open(ANNOTATIONS_PATH))
r20 = np.random.default_rng(0).choice(sl, 20, replace=False)
caps = join_captions(np.asarray(data.sample_ids)[r20], ann)
enc = nrm(encode_txt(list(caps)))
cs = (enc * txt_n[r20]).sum(1)
log(f"CHECK 4: cosine of re-encoded captions vs cached txt_features: min {cs.min():.5f} mean {cs.mean():.5f}")
assert cs.min() > 0.99, "prompt-encoding check failed"
checks["check4_caption_reencode_cos_min"] = float(cs.min())
checks["check4_caption_reencode_cos_mean"] = float(cs.mean())

N_NAMES = {"emotion": nrm(encode_txt([f"a painting that evokes {e}" for e in ok_em])),
           "style": nrm(encode_txt([f"a painting in the style of {s.replace('_', ' ')}" for s in ok_st]))}
del clip
zero_shot = {}
m_em = np.isin(row_em, ok_em)
pe = (txt_n[rows[m_em]] @ N_NAMES["emotion"].T).argmax(1)
zero_shot["emotion_caption_acc"] = float((np.asarray(ok_em)[pe] == row_em[m_em]).mean())
zero_shot["emotion_chance_uniform"] = 1 / len(ok_em)
zero_shot["emotion_majority_share"] = float(max((row_em[m_em] == e).mean() for e in ok_em))
m_st = np.isin(row_st, ok_st)
ps_ = (img_n[rows[m_st]] @ N_NAMES["style"].T).argmax(1)
zero_shot["style_image_acc"] = float((np.asarray(ok_st)[ps_] == row_st[m_st]).mean())
zero_shot["style_chance_uniform"] = 1 / len(ok_st)
zero_shot["style_majority_share"] = float(max((row_st[m_st] == s).mean() for s in ok_st))
log("zero-shot", zero_shot)


def softmax_p(f, N):
    return torch.softmax(100 * tt(f) @ tt(N).T, dim=-1)


# ------------------------------------------------------------------ terms: T (n,13) per (cond, direction)
def term(name, cond, d):
    C = COND[cond]
    qf, cf = gather(cond, d)
    sx, sy = C["sup"]
    kx, ky = C["con"]
    if name in ("SE", "C0", "R3"):
        ic, tc = codes[name]
        w = agree_weights(ic, tc, cond)
        qc, cc = (ic[anc], tc[C["cands"]]) if d == "i2t" else (tc[anc], ic[C["cands"]])
        return (w[:, None, :] * tt(qc)[:, None, :] * tt(cc)).sum(-1)
    if name in ("raw_agree", "raw_agree_relu"):
        w = tt((img_n[sx] * txt_n[sy]).mean(1) - (img_n[kx] * txt_n[ky]).mean(1))
        if name == "raw_agree_relu":
            w = F.relu(w)
        return (w[:, None, :] * qf[:, None, :] * cf).sum(-1)
    if name == "proto":
        # candidate modality: images are x, captions are y
        fs, fk = (sx, kx) if d == "t2i" else (sy, ky)
        feats = img_n if d == "t2i" else txt_n
        ms, mk = tt(feats[fs]).mean(1), tt(feats[fk]).mean(1)
        return cos_q(ms, cf) - cos_q(mk, cf)
    if name == "names":
        N = N_NAMES[cond]
        return (softmax_p(qf, N)[:, None, :] * softmax_p(cf.reshape(-1, cf.shape[-1]), N).reshape(n, 13, -1)).sum(-1)
    raise KeyError(name)


SCORES = {}   # scorer -> setting -> {(cond,d): (n,13) tensor, target in column 0, other-aspect candidate in column 1}
SETTINGS = {}


def put(scorer, setting, cond, d, s):
    SCORES.setdefault(scorer, {}).setdefault(setting, {})[(cond, d)] = s.double()


COS = {(c, d): cos_q(*gather(c, d)) for c in CONDS for d in DIRS}
for c in CONDS:
    for d in DIRS:
        put("clip", "cos", c, d, COS[(c, d)])
SETTINGS["clip"] = ["cos"]

for name in ("SE", "C0", "R3", "raw_agree", "raw_agree_relu", "proto", "names"):
    sc = {"SE": "SE_agree", "C0": "C0_agree", "R3": "R3_agree"}.get(name, name)
    t1 = perf_counter()
    SETTINGS[sc + "_cv"] = [f"lam{l:g}" for l in LAMS]
    for c in CONDS:
        for d in DIRS:
            T = term(name, c, d)
            base = zs(COS[(c, d)])
            for l in LAMS:
                put(sc + "_cv", f"lam{l:g}", c, d, zs(T) if l == float("inf") else base + l * zs(T))
            if name in ("SE", "C0", "R3"):
                put(sc + "_b0.3", "b0.3", c, d, BETA * COS[(c, d)].double() + T.double())
    if name in ("SE", "C0", "R3"):
        SETTINGS[sc + "_b0.3"] = ["b0.3"]
    log(f"term {name} done ({perf_counter() - t1:.1f}s)")

# SE_value: naive value rule on pair codes (mean support minus mean contrast), beta 0.3
ic, tc = codes["SE"]
for c in CONDS:
    C = COND[c]
    pcs = tt(pair_codes(tt(ic[C["sup"][0]]), tt(tc[C["sup"][1]])))
    pck = tt(pair_codes(tt(ic[C["con"][0]]), tt(tc[C["con"][1]])))
    w = naive_condition_weights(pcs, pck)
    for d in DIRS:
        qf, cf = gather(c, d)
        qc, cc = (ic[anc], tc[C["cands"]]) if d == "i2t" else (tc[anc], ic[C["cands"]])
        put("SE_value", "b0.3", c, d, conditional_score(qf, cf, tt(qc), tt(cc), w, BETA))
SETTINGS["SE_value"] = ["b0.3"]

# ------------------------------------------------------------------ metrics
def hit_arrays(S):
    """per-episode quantities from a scores dict {(cond,d): (n,13)}."""
    out = {}
    for (c, d), s in S.items():
        out[("hit", c, d)] = (tie_aware_rank(s).numpy() <= 1).astype(np.float64)
        other = torch.cat([s[:, 1:2], s[:, :1], s[:, 2:]], 1)         # other-aspect candidate as the positive
        out[("other", c, d)] = (tie_aware_rank(other).numpy() <= 1).astype(np.float64)
    for d in DIRS:
        out[("swap", d)] = ((S[("emotion", d)][:, 0] > S[("emotion", d)][:, 1]) &
                            (S[("style", d)][:, 0] > S[("style", d)][:, 1])).numpy().astype(np.float64)
        out[("strict", d)] = out[("hit", "emotion", d)] * out[("hit", "style", d)]
    return out


def per_anchor(h):
    """per-anchor, mean-of-directions vectors."""
    m = lambda f: 0.5 * (f("i2t") + f("t2i"))
    return {
        "r1_emotion": m(lambda d: h[("hit", "emotion", d)]),
        "r1_style": m(lambda d: h[("hit", "style", d)]),
        "r1_pooled": m(lambda d: 0.5 * (h[("hit", "emotion", d)] + h[("hit", "style", d)])),
        "other_pooled": m(lambda d: 0.5 * (h[("other", "emotion", d)] + h[("other", "style", d)])),
        "swap": m(lambda d: h[("swap", d)]),
        "strict": m(lambda d: h[("strict", d)]),
    }


def summary(h):
    pa = per_anchor(h)
    out = {k: float(v.mean() * 100) for k, v in pa.items()}
    for d in DIRS:
        out[f"r1_emotion_{d}"] = float(h[("hit", "emotion", d)].mean() * 100)
        out[f"r1_style_{d}"] = float(h[("hit", "style", d)].mean() * 100)
        out[f"swap_{d}"] = float(h[("swap", d)].mean() * 100)
        out[f"strict_{d}"] = float(h[("strict", d)].mean() * 100)
        for c in CONDS:
            out[f"other_{c}_{d}"] = float(h[("other", c, d)].mean() * 100)
    return out


results = {"checks": checks, "zero_shot": zero_shot, "eligible_emotions": ok_em, "eligible_styles": ok_st,
           "draw_failures": failures, "settings": {}, "cv_picks": {}, "main": {}, "diffs": {}}
H = {}          # final scorer -> hit arrays
FINAL = {}      # final scorer -> {(cond,d): (n,13) scores actually used}
par = np.arange(n) % 2
for scorer, sets in SETTINGS.items():
    per_set = {s: hit_arrays(SCORES[scorer][s]) for s in sets}
    if len(sets) > 1:
        results["settings"][scorer] = {s: summary(per_set[s]) for s in sets}
    if scorer.endswith("_cv"):
        comp = {k: torch.zeros(n, 13, dtype=torch.float64) for k in SCORES[scorer][sets[0]]}
        folds = {}
        for tune in (0, 1):
            m = par == tune
            tune_r1 = [float(np.mean([per_set[s][("hit", c, d)][m].mean() for c in CONDS for d in DIRS])) * 100 for s in sets]
            pick = sets[int(np.argmax(tune_r1))]
            folds[f"tuned_on_parity{tune}"] = {"picked": pick, "tune_pooled_r1": max(tune_r1), "all": dict(zip(sets, tune_r1))}
            ap = torch.as_tensor(~m)
            for k in comp:
                comp[k][ap] = SCORES[scorer][pick][k][ap]
        results["cv_picks"][scorer] = folds
        H[scorer] = hit_arrays(comp)
        FINAL[scorer] = comp
    else:
        H[scorer] = per_set[sets[0]]
        FINAL[scorer] = SCORES[scorer][sets[0]]
    results["main"][scorer] = summary(H[scorer])

# ties / CLIP checks
cl = SCORES["clip"]["cos"]
tie_eps = {f"{c}_{d}": int((cl[(c, d)][:, 0] == cl[(c, d)][:, 1]).sum()) for c in CONDS for d in DIRS}
n_tie_any = sum(tie_eps.values())
clip_swap = {d: int(H["clip"][("swap", d)].sum()) for d in DIRS}
assert all(v == 0 for v in clip_swap.values()), f"CLIP swap success is nonzero: {clip_swap}"
checks["check3_clip_swap_successes"] = clip_swap
checks["check3_clip_exact_ties_p_emo_vs_p_style"] = tie_eps
log(f"CHECK 3 passed: CLIP swap successes {clip_swap}; exact ties between p_emo and p_style {tie_eps}")

# paired bootstrap
CVS = [s for s in H if s.endswith("_cv")]
metrics = ["r1_pooled", "r1_emotion", "r1_style", "swap", "strict"]


def diff(a, b):
    pa, pb = per_anchor(H[a]), per_anchor(H[b])
    res = {}
    for m in metrics:
        bt = paired_bootstrap(pa[m] - pb[m], 5000, SEED)
        res[m] = {"point_pp": bt["point"] * 100, "ci95_pp": [bt["ci95"][0] * 100, bt["ci95"][1] * 100]}
    return res


for s in H:
    if s != "clip":
        results["diffs"][f"{s}_minus_clip"] = diff(s, "clip")
    if s != "SE_agree_cv":
        results["diffs"][f"{s}_minus_SE_agree_cv"] = diff(s, "SE_agree_cv")

np.savez_compressed(OUT / "aspect_ranks.npz", **{f"{s}__{c}__{d}": tie_aware_rank(FINAL[s][(c, d)]).numpy()
                                                     for s in FINAL for c in CONDS for d in DIRS})
np.savez_compressed(OUT / "aspect_episodes.npz", **EP)
results["runtime_seconds"] = perf_counter() - T0
(OUT / "aspect_results.json").write_text(json.dumps(results, indent=1))

# ------------------------------------------------------------------ tables
log("MAIN TABLE (mean of directions, %): scorer | R@1 emo / style / pooled | other-aspect pooled | swap / strict")
for s, v in results["main"].items():
    log(f"  {s:18s} {v['r1_emotion']:6.2f} {v['r1_style']:6.2f} {v['r1_pooled']:6.2f} | {v['other_pooled']:6.2f} | {v['swap']:6.2f} {v['strict']:6.2f}"
        f" | swap i2t/t2i {v['swap_i2t']:.2f}/{v['swap_t2i']:.2f}")
for base in ("clip", "SE_agree_cv"):
    log(f"DIFFS vs {base} (pp, 95% CI): pooled R@1 | swap")
    for k, v in results["diffs"].items():
        if k.endswith(f"_minus_{base}") and k.split("_minus_")[0].endswith("_cv"):
            log(f"  {k.split('_minus_')[0]:16s} " + " | ".join(
                f"{v[m]['point_pp']:+.2f} [{v[m]['ci95_pp'][0]:+.2f},{v[m]['ci95_pp'][1]:+.2f}]" for m in ("r1_pooled", "swap")))
log("PICKS:")
for s, f in results["cv_picks"].items():
    log(f"  {s:18s} tuned p0 -> {f['tuned_on_parity0']['picked']}, tuned p1 -> {f['tuned_on_parity1']['picked']}")
log(f"runtime {perf_counter() - T0:.1f}s")
