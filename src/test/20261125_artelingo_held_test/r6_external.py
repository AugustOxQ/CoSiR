"""Round 6 descriptive pass, external rows (DECISION_RULE.md of this folder: section 10 item 5, section 8 item 3,
section 7 items 2 to 4, section 2; spec D10, D15, D22; contracts section 9; ticket 14). Called only through
run_r6_descriptive.py's two hooks (external_seed_rows per seed, external_rows at the end), i.e. only after
held_verdict.json records the phase-2 agreement; decides nothing.

Rows (each gets r6_descriptive.describe's rows: R@1, gain and swap success with 95% intervals, and AFF minus it; pooled,
per seed, per pair):
  DTS, DTS-CF, DTS-N  r6_dts.held_dts_scores on each seed's RowContext with the chosen setting and the frozen picks of
                      the seed-42 record (results/dts_seed42.json; smoke: the smoke seed's record), whose stop record
                      (dts_stop.json beside it) passed. DTS-N is the true-name ceiling (privileged). Info: parsing
                      failures per seed and condition (and totals), the phrases' most frequent wordings (per seed and
                      pooled, overall and per pair and condition), the setting, wording, K and picks.
  FT-LP               the LP lr 3e-4 epoch 9 maps (best_params.pt of the clipft run, SHA-256 pinned) applied on the
                      CPU to the seed's cached CLIP features (the RowContext's masked img and txt) as nn.Linear without bias,
                      x W^T, computed in float64 and rounded to float32 (batch-independent; within float32 rounding of
                      the run's own GPU features, test_r6_external.py).
  FT-LB, FT-LoRA      features_<variant>.npz of the FT feature job (r6_gpu_ft_features.py), joined by row id.
                      FT rows are scored as ft_eval.score_features: per_anchor(cosine_scores(EvalInputs(img, txt), ep))
                      with the features placed by row id in arrays of all rows (NaN elsewhere, as ft_eval
                      .place_features): the plain cosine of L2-normalised features.
  MLLM                the reranker's scores_shown (r6_gpu_rerank.py) on the first seed of the mode only (held: 52),
                      un-permuted with the job's CPU-side permutations (<job>.perms.npz; letter j of a prompt showed
                      candidate column perm[j], mllm_reranker.unpermute), per_anchor on the 13 candidate scores.

Sources: <out>/external_sources.json, written by the run chat once the GPU outputs are pulled (its SHA-256 and each
entry are recorded with every row):
  {"dts":   {"record": path, "verbalise_job": [paths], "verbalise_out": [paths], "listing_out": [paths],
             "embeddings": path (optional; default <out>/dts_value_embeddings_descriptive.npz, a value-embedding
             cache of r6_dts.ValueEmbedder)},
   "ft_lp": {} (the pinned checkpoint; or {"checkpoint": path} of a byte copy),
   "ft":    {"job": path, "out": [paths]},
   "mllm":  {"job": path, "out": [paths]}}
  Relative paths resolve against the main checkout (r6_common.MAIN). Any entry may instead be {"missing": "<reason>"}.
  An absent entry (or an absent file) marks its rows "missing" with the reason; so do an FT variant whose features file
  no listed folder holds, and a seed with no listed verbaliser job or output. A missing row has no number: its pa is
  dropped, and so is that of a scorer that was not computed on every seed it is reported on (MLLM: the first seed;
  the others: every seed). A listed path that does not exist, an unknown key, or any failed join stops the pass
  (AssertionError, exit 5): descriptive.json is written once, so nothing is guessed.

Joins (C6; rule section 8 item 3: the GPU outputs are joined to the episodes only here), each asserted:
  - DTS: answers by (seed, episode_index, condition, wording), listings by (phrase, K), through r6_dts (a key missing,
    twice in one file, or twice across folders with different answers, a key of another seed or beyond the seed's
    episodes, a record outside its job, a job whose example pairs are not the episodes': each raises); every listed
    verbaliser job is of a seed of this mode, and every listed output folder is the output of a listed job.
  - FT: the job's ft_rows equal the union of the member rows (anchor, 13 candidates, 4 + 4 example pairs) of the
    mode's episode files; each features file's rows are sorted, unique and equal to the job's (a row missing, twice or
    extra raises); every member of the seed's episodes is present; its provenance names the job's inputs, the pinned
    checkpoint and these file bytes.
  - MLLM: the job holds every episode of the seed (episode_index 0 .. n - 1) with the episode file's query row, pairs
    and candidates in the shown order of its permutations; the permutations are the probe's formula and the job
    record's SHA-256; the merged scores hold each episode index exactly once (twice in one file, or twice across
    folders with different scores, raises; an index the job does not hold raises; one missing raises).

Guards carry a `# guard:<name>` marker; test_r6_external.py deletes each on a copy and shows that its scenario then
goes through.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_descriptive as D  # noqa: E402
import r6_dts as DT  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_ft_features as FF  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import r6_gpu_t12 as T  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402

SOURCES_NAME = "external_sources.json"
SOURCE_KEYS = ("dts", "ft_lp", "ft", "mllm")
NAMES = ("DTS", "DTS-CF", "DTS-N", "FT-LP", "FT-LB", "FT-LoRA", "MLLM")
FAMILY = {"DTS": "dts", "DTS-CF": "dts", "DTS-N": "dts", "FT-LP": "ft_lp", "FT-LB": "ft", "FT-LoRA": "ft",
          "MLLM": "mllm"}
DTS_ROWS = {"dts": "DTS", "dts_cf": "DTS-CF", "dts_n": "DTS-N"}
LABELS = {"DTS": "DTS (describe-then-score, fused with cosine at the frozen seed-42 picks)",
          "DTS-CF": "DTS-CF (DTS's matched counterpart, condition-free)",
          "DTS-N": "DTS-N (DTS with the target aspect's true name: the ceiling, privileged)",
          "FT-LP": "FT-LP (fine-tuned CLIP, linear probe, lr 3e-4 epoch 9)",
          "FT-LB": "FT-LB (fine-tuned CLIP, last block, lr 3e-5 epoch 8)",
          "FT-LoRA": "FT-LoRA (fine-tuned CLIP, LoRA rank 16, lr 1e-4 epoch 10)",
          "MLLM": "MLLM (in-context reranker, Qwen3-VL-8B-Instruct, first seed only)"}
FT_CKPTS = {   # rule section 11: the fine-tune checkpoints are frozen (gpu_path section 4)
    "LP": {"path": "res/cluster_jobs/20261007-070950-65bb2f4/code/outputs/clipft/LP_lr3e-4/best_params.pt",
           "sha256": "c39e3af6af4fa3561b4d415089cec3a0fd50ed3a352e41c1a031132285950fc8", "lr": 3e-4, "epoch": 9},
    "LB": {"path": "res/cluster_jobs/20261007-071151-65bb2f4/code/outputs/clipft/LB_lr3e-5/best_params.pt",
           "sha256": "0d94f4c0613bb84d214d9df69a705a1bb897ca24961eb810923a3a62413d8b98", "lr": 3e-5, "epoch": 8},
    "LoRA": {"path": "res/cluster_jobs/20261007-071251-65bb2f4/code/outputs/clipft/LoRA_lr1e-4/best_params.pt",
             "sha256": "3250274b5da85c6b89ad7fc929da51fc3cf1f143dc523a192cd9088c2c5f9517", "lr": 1e-4, "epoch": 10}}
FT_VARIANTS = ("LB", "LoRA")
LP_PARAMS = ("img_map.weight", "logit_scale", "txt_map.weight")
DTS_ENTRY = ("record", "verbalise_job", "verbalise_out", "listing_out")
EMB_DEFAULT = "dts_value_embeddings_descriptive.npz"
STOP_NAME = "dts_stop.json"            # run_r6_dts.STOP
N_CAND = 13
DIM = 512
TOP_N, TOP_N_CELL = 10, 5
STATE = "_r6_external"                 # the run's state, kept on env between the two hooks
RERANK_CONDITIONS, RERANK_DIRECTIONS = G.CONDITIONS, I.DIRECTIONS     # the reranker's (cond, dir) axis order
SCORING = "plain cosine of L2-normalised features placed by row id (ft_eval.score_features), per_anchor"
encoder_factory = None                 # the value-string encoder of DTS (None: r6_dts.clip_cpu); tests set a fake

if not (all(FF.SELECTED[v] == {"lr": FT_CKPTS[v]["lr"], "epoch": FT_CKPTS[v]["epoch"]} for v in FT_VARIANTS)
        and set(RERANK_CONDITIONS) == set(CONDITIONS) and set(RERANK_DIRECTIONS) == set(DIRECTIONS)
        and set(DTS_ROWS) == set(DT.SCORERS)):
    raise ImportError("r6_external: the selected fine-tunes, the reranker's axes or DTS's scorers differ")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def mode_seeds(smoke: bool) -> list:
    return list(R.SMOKE_SEEDS if smoke else R.HELD_SEEDS)


def expected_seeds(name, seeds) -> list:
    """The seeds a row is reported on: MLLM the first seed of the mode (held 52), every other row every seed."""
    return [seeds[0]] if name == "MLLM" else list(seeds)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else R.MAIN / p


def listed(p, what) -> Path:
    """A path of external_sources.json, resolved; it must exist (a typo stops the pass, nothing is guessed)."""
    path = resolve(p)
    _require(path.exists(), f"{what} {p} is listed in {SOURCES_NAME} but does not exist")  # guard:listed_exists
    return path


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- sources

def load_sources(out) -> SimpleNamespace:
    """<out>/external_sources.json -> SimpleNamespace(path, present, sha256, entries)."""
    path = Path(out) / SOURCES_NAME
    if not path.is_file():
        return SimpleNamespace(path=path, present=False, sha256=None, entries={})
    raw = path.read_bytes()
    rec = json.loads(raw.decode("utf-8"))
    _require(isinstance(rec, dict) and set(rec) <= set(SOURCE_KEYS) and all(isinstance(v, dict) for v in rec.values()),
             f"{SOURCES_NAME}: entries {sorted(rec) if isinstance(rec, dict) else rec!r} are not objects keyed by "
             f"{SOURCE_KEYS}")  # guard:source_keys
    return SimpleNamespace(path=path, present=True, sha256=R.sha256_bytes(raw), entries=rec)


def missing_reason(src, key):
    """The reason the family ``key`` has no rows (None: its entry lists outputs to join)."""
    if not src.present:
        return f"no {SOURCES_NAME} in the results folder: no output of the {key} job was given (job not run)"
    e = src.entries.get(key)
    if e is None:
        return f"{SOURCES_NAME} lists no {key} entry (job not run)"
    if "missing" in e:
        _require(set(e) == {"missing"} and isinstance(e["missing"], str) and e["missing"],
                 f"{SOURCES_NAME}: a missing {key} entry holds only a non-empty reason")
        return e["missing"]
    return None


def _paths(entry, key, what) -> list:
    v = entry.get(key)
    _require(isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x for x in v),
             f"{SOURCES_NAME}: {what} must be a non-empty list of paths")
    return [listed(x, what) for x in v]


def _keys(entry, need, allowed, what):
    _require(set(need) <= set(entry) <= set(allowed),
             f"{SOURCES_NAME}: the {what} entry has keys {sorted(entry)}, not {sorted(need)} "
             f"(+ {sorted(set(allowed) - set(need))})")


# ---------------------------------------------------------------- DTS, DTS-CF, DTS-N

def load_dts(entry, out, smoke) -> SimpleNamespace:
    """The frozen record (and its passed stop record), the listings, the value embedder, and the verbaliser jobs and
    output folders, each output joined to its listed job by its fingerprint's input SHA-256s."""
    _keys(entry, DTS_ENTRY, DTS_ENTRY + ("embeddings",), "dts")
    settings, settings_sha = DT.load_settings()
    rpath = listed(entry["record"], "the DTS record")
    raw = rpath.read_bytes()
    rec = json.loads(raw.decode("utf-8"))
    seed_ok = rec.get("seed") in R.SMOKE_SEEDS if smoke else rec.get("seed") == R.DEV_SEED
    n_want = len(R.PAIRS) * (R.N_SMOKE if smoke else R.N_PER_PAIR)
    _require(rec.get("stage") == "chosen" and rec.get("rule_sha256") == R.RULE_SHA256 and seed_ok
             and rec.get("n_episodes") == n_want and rec.get("settings_sha256") == settings_sha
             and rec.get("wording_id") in DT.WORDINGS and rec.get("K") in DT.KS
             and rec.get("setting") == DT.setting_name(rec.get("wording_id"), rec.get("K") or 0),
             f"{rpath.name}: not the chosen stage's record of this rule on "
             f"{'a smoke seed' if smoke else 'seed 42'} ({n_want} episodes) with this "
             f"dts_settings.json")  # guard:dts_record
    spath = rpath.parent / STOP_NAME
    _require(spath.is_file(), f"{STOP_NAME} is missing beside {rpath.name}: DTS's stop was not evaluated on it")
    stop = _json(spath)
    _require(stop.get("stage") == "stop" and stop.get("stop") is False and stop.get("built") is True
             and stop.get("setting") == rec["setting"]
             and (stop.get("input_sha256") or {}).get(rpath.name) == R.sha256_bytes(raw),
             f"{STOP_NAME}: no passed stop (built, no stop) on these bytes of {rpath.name} (rule section 7 items 5 and "
             f"6)")  # guard:dts_stop
    seeds = mode_seeds(smoke)
    jobs = {}
    for j in _paths(entry, "verbalise_job", "a verbaliser job"):
        inp = G.load_verbalise_input(j / "verbalise_input.npz")
        _require(inp["seed"] in seeds, f"{j}: a verbaliser job of seed {inp['seed']}, not of this mode's seeds "
                                       f"{seeds}")  # guard:dts_job_seed
        shas = {f: G.sha256_file(j / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}
        jobs[(shas["rows_manifest.npz"], shas["verbalise_input.npz"])] = SimpleNamespace(path=j, seed=inp["seed"],
                                                                                         shas=shas)
    outs = {}
    for d in _paths(entry, "verbalise_out", "a verbaliser output"):
        fp = DT._fingerprint(d, "r6_gpu_verbalise", settings_sha)
        ins = fp.get("inputs_sha256") or {}
        key = (ins.get("rows_manifest.npz"), ins.get("verbalise_input.npz"))
        _require(key in jobs, f"{d}: the output of a verbaliser job that {SOURCES_NAME} does not "
                              f"list")  # guard:dts_out_job
        outs[d] = jobs[key]
    listings = DT.merge_listings(_paths(entry, "listing_out", "a listing output"), settings_sha)
    emb_path = resolve(entry["embeddings"]) if "embeddings" in entry else Path(out) / EMB_DEFAULT
    embedder = DT.ValueEmbedder(emb_path, encoder_factory=encoder_factory)
    return SimpleNamespace(record=rec, record_path=rpath, record_sha256=R.sha256_bytes(raw), stop_path=spath,
                           stop_sha256=G.sha256_file(spath), settings=settings, settings_sha=settings_sha, jobs=jobs,
                           outs=outs, listings=listings, embedder=embedder, emb_path=emb_path, counts=Counter(),
                           failures={}, below_K={})


def _top(counter, n) -> list:
    return [[p, int(k)] for p, k in sorted(counter.items(), key=lambda t: (-t[1], t[0]))[:n]]


def phrase_table(cells) -> dict:
    """The phrases' most frequent wordings from a Counter over (pair name, condition, normalised phrase); the empty
    phrase (a parsing failure) is counted as ""."""
    total = Counter()
    for (_, _, p), k in cells.items():
        total[p] += k
    by = {pair: {c: _top(Counter({p: k for (q, d, p), k in cells.items() if q == pair and d == c}), TOP_N_CELL)
                 for c in CONDITIONS} for pair in R.PAIR_NAMES}
    return {"top": _top(total, TOP_N), "by_pair_condition": by, "n_distinct": int(len(total)),
            "n_phrases": int(sum(total.values()))}


def dts_seed(dts, seed, ctx, episodes) -> dict:
    """{"DTS", "DTS-CF", "DTS-N": row} of one seed (row = {"pa", "info"}; pa None with info["missing"])."""
    w = dts.record["wording_id"]
    jobs = [j for j in dts.jobs.values() if j.seed == seed]
    outs = [d for d, j in dts.outs.items() if j.seed == seed]
    reason = (f"no verbaliser job of seed {seed} is listed in {SOURCES_NAME}" if not jobs else
              f"no output of seed {seed}'s verbaliser job is listed in {SOURCES_NAME} (job not run)" if not outs
              else None)
    if reason is not None:
        return {name: {"pa": None, "info": {"missing": reason}} for name in DTS_ROWS.values()}
    merged = DT.merge_verbaliser(outs, [w], dts.settings_sha, seeds={seed})
    job_shas = DT.check_verbaliser_jobs(merged, [j.path for j in jobs], episodes)
    res = DT.held_dts_scores(ctx, merged.answers, dts.listings.answers, dts.record, dts.embedder)
    phrases = DT.phrases_for(ctx, merged.answers, w)
    pi = np.asarray(episodes.pair_index, dtype=np.int64)
    cells = Counter((R.PAIR_NAMES[int(k)], c, p) for c in CONDITIONS for k, p in zip(pi.tolist(), phrases[c]))
    dts.counts.update(cells)
    dts.embedder.save()
    fail = {k: {c: {kind: int(v) for kind, v in res["failures"][k][c].items()} for c in CONDITIONS}
            for k in ("dts", "dts_n")}
    below = {k: {c: int(res["below_K"][k][c]) for c in CONDITIONS} for k in ("dts", "dts_n")}
    dts.failures[seed], dts.below_K[seed] = fail, below
    info_dts = {"parsing_failures": fail["dts"], "below_K": below["dts"], "phrases": phrase_table(cells),
                "verbaliser_files_sha256": dict(merged.files), "verbaliser_jobs_sha256": job_shas}
    return {"DTS": {"pa": res["dts"], "info": info_dts},
            "DTS-CF": {"pa": res["dts_cf"], "info": {}},
            "DTS-N": {"pa": res["dts_n"], "info": {"parsing_failures": fail["dts_n"], "below_K": below["dts_n"]}}}


def _sum_failures(per_seed) -> dict:
    out = {c: dict.fromkeys(DT.FAIL_KINDS, 0) for c in CONDITIONS}
    for f in per_seed.values():
        for c in CONDITIONS:
            for kind in DT.FAIL_KINDS:
                out[c][kind] += int(f[c][kind])
    return out


def dts_info(dts, name) -> dict:
    rec = dts.record
    out = {"setting": rec["setting"], "wording_id": rec["wording_id"],
           "wording": dts.settings["verbaliser"]["wordings"][rec["wording_id"]], "K": int(rec["K"]),
           "picks": rec["picks"][{v: k for k, v in DTS_ROWS.items()}[name]],
           "pick_convention": "key h = tuned on seed-42 parity half h, scores parity 1 - h",
           "record": {"file": str(dts.record_path), "sha256": dts.record_sha256},
           "stop_record": {"file": str(dts.stop_path), "sha256": dts.stop_sha256},
           "settings_sha256": dts.settings_sha}
    if name == "DTS":
        out.update(parsing_failures={str(s): f["dts"] for s, f in dts.failures.items()},
                   parsing_failures_total=_sum_failures({s: f["dts"] for s, f in dts.failures.items()}),
                   phrases=phrase_table(dts.counts), listings_sha256=dict(dts.listings.files),
                   embeddings={"file": str(dts.emb_path),
                               "sha256": G.sha256_file(dts.emb_path) if dts.emb_path.is_file() else None,
                               "n_strings": len(dts.embedder.index), "encoder": dts.embedder.meta})
    elif name == "DTS-N":
        out.update(role="the ceiling: the phrase replaced by the target aspect's name (privileged)",
                   parsing_failures={str(s): f["dts_n"] for s, f in dts.failures.items()},
                   parsing_failures_total=_sum_failures({s: f["dts_n"] for s, f in dts.failures.items()}))
    else:
        out.update(role="DTS's matched counterpart: (z(T^a) + z(T^b)) / 2, condition-free (gain 0 asserted); rows "
                        "where both conditions failed take cosine's")
    return out


# ---------------------------------------------------------------- the fine-tuned CLIP rows

def score_placed(rows, img, txt, pooled, n_rows=R.N_ROWS) -> dict:
    """ft_eval.score_features on features placed by row id (NaN elsewhere): per_anchor of the plain cosine of the
    L2-normalised features (cosine_scores normalises inside EvalInputs)."""
    rows = np.asarray(rows, dtype=np.int64)
    full = []
    for x in (img, txt):
        a = np.full((int(n_rows), DIM), np.nan, dtype=np.float32)
        a[rows] = np.asarray(x, dtype=np.float32)
        full.append(a)
    return per_anchor(cosine_scores(EvalInputs(full[0], full[1]), pooled))


def lp_maps(path) -> SimpleNamespace:
    """The LP checkpoint's two 512 x 512 maps (float64), its identity and SHA-256 asserted (rule section 11)."""
    import torch
    path = Path(path)
    sha = G.sha256_file(path)
    want = FT_CKPTS["LP"]
    _require(sha == want["sha256"], f"{path}: SHA-256 {sha[:12]} is not the selected LP checkpoint's "
                                    f"{want['sha256'][:12]}")  # guard:lp_ckpt
    ck = torch.load(path, map_location="cpu", weights_only=True)
    _require(ck.get("variant") == "LP" and float(ck.get("lr")) == want["lr"] and int(ck.get("epoch")) == want["epoch"]
             and tuple(sorted(ck["params"])) == LP_PARAMS, f"{path}: not LP lr {want['lr']} epoch {want['epoch']}")
    W = {m: ck["params"][f"{m}_map.weight"].detach().numpy().astype(np.float64) for m in ("img", "txt")}
    _require(all(w.shape == (DIM, DIM) and bool(np.isfinite(w).all()) for w in W.values()), f"{path}: map shapes")
    return SimpleNamespace(path=path, sha256=sha, W=W)


def apply_map(x, W) -> np.ndarray:
    """nn.Linear(512, 512, bias=False) on rows x: x W^T, in float64, rounded to float32."""
    return (np.asarray(x, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T).astype(np.float32)


def load_lp(entry, out, smoke) -> SimpleNamespace:
    _keys(entry, (), ("checkpoint",), "ft_lp")
    return lp_maps(listed(entry["checkpoint"], "the LP checkpoint") if "checkpoint" in entry
                   else resolve(FT_CKPTS["LP"]["path"]))


def lp_seed(lp, ctx, episodes) -> dict:
    p = episodes.pooled
    rows = np.unique(np.concatenate([np.asarray(p.anchor), np.asarray(p.candidates).ravel()])).astype(np.int64)
    x_img, x_txt = np.asarray(ctx.img)[rows], np.asarray(ctx.txt)[rows]
    _require(np.asarray(ctx.img).shape == (R.N_ROWS, DIM) and bool(np.isfinite(x_img).all())
             and bool(np.isfinite(x_txt).all()),
             "FT-LP: an anchor or candidate has no cached feature in the context (outside its rows)")  # guard:lp_rows
    pa = score_placed(rows, apply_map(x_img, lp.W["img"]), apply_map(x_txt, lp.W["txt"]), p)
    return {"pa": pa, "info": {"n_rows_mapped": int(len(rows))}}


def member_union(out, seeds) -> np.ndarray:
    """Sorted unique member rows (D.member_rows: anchor, candidates, example pairs) of the mode's episode files."""
    return np.unique(np.concatenate([D.member_rows(E.load_episodes(Path(out) / f"held_episodes_seed{s}.npz").pooled)
                                     .ravel() for s in seeds])).astype(np.int64)


def _diff(a, b) -> str:
    return f"{len(np.setdiff1d(b, a))} missing, {len(np.setdiff1d(a, b))} extra"


def load_ft_features(folder, variant, job, job_rows, job_shas) -> SimpleNamespace:
    """features_<variant>.npz of an FT output folder, joined to its job by row id (module docstring)."""
    folder = Path(folder)
    prov = _json(folder / "provenance.json")
    fp = prov.get("fingerprint") or {}
    _require(fp.get("job") == "r6_gpu_ft_features" and fp.get("inputs_sha256") == job_shas,
             f"{folder}: not an output of the FT job {Path(job).name}")  # guard:ft_out_job
    want = FT_CKPTS[variant]
    _require((fp.get("ckpt_sha256") or {}).get(variant) == want["sha256"]
             and (fp.get("selected") or {}).get(variant) == {"lr": want["lr"], "epoch": want["epoch"]},
             f"{folder}: features_{variant}.npz was not made from the selected {variant} checkpoint")  # guard:ft_ckpt
    path = folder / f"features_{variant}.npz"
    sha = G.sha256_file(path)
    _require(any(((r or {}).get("features_sha256") or {}).get(variant) == sha for r in prov.get("runs") or []),
             f"{path}: no run of its provenance wrote these bytes")  # guard:ft_run
    with np.load(path, allow_pickle=False) as z:
        _require(sorted(z.files) == ["img", "rows", "txt"], f"{path}: keys {sorted(z.files)}")
        rows, img, txt = z["rows"], z["img"], z["txt"]
    _require(rows.dtype == np.int64 and rows.ndim == 1 and bool((np.diff(rows) > 0).all()),
             f"{path}: rows are not sorted and unique (a row twice)")  # guard:ft_unique
    _require(np.array_equal(rows, job_rows),
             f"{path}: its rows are not the job's ({_diff(rows, job_rows)})")  # guard:ft_rows
    _require(img.dtype == txt.dtype == np.float32 and img.shape == txt.shape == (len(rows), DIM)
             and bool(np.isfinite(img).all() and np.isfinite(txt).all()), f"{path}: img, txt not finite float32 "
                                                                         f"({len(rows)}, {DIM})")
    return SimpleNamespace(folder=folder, path=path, sha256=sha, rows=rows, img=img, txt=txt)


def load_ft(entry, out, smoke) -> SimpleNamespace:
    """The FT job (its rows = the member rows of the mode's episodes) and, per variant, its one features file (a
    str: the reason it is missing)."""
    _keys(entry, ("job", "out"), ("job", "out"), "ft")
    job = listed(entry["job"], "the FT job")
    rows = T.load_ft_rows(job / "ft_rows.npz", G.Manifest(job / "rows_manifest.npz"))
    shas = {f: G.sha256_file(job / f) for f in ("rows_manifest.npz", "ft_rows.npz")}
    members = member_union(out, mode_seeds(smoke))
    _require(np.array_equal(rows, members), f"{job}: its rows are not the member rows of the episodes of seeds "
                                            f"{mode_seeds(smoke)} ({_diff(rows, members)})")  # guard:ft_job_rows
    outs = _paths(entry, "out", "an FT output")
    variants = {}
    for v in FT_VARIANTS:
        found = [d for d in outs if (d / f"features_{v}.npz").is_file()]
        _require(len(found) <= 1, f"features_{v}.npz is in {len(found)} listed FT output folders; list "
                                  f"one")  # guard:ft_one_file
        variants[v] = (load_ft_features(found[0], v, job, rows, shas) if found else
                       f"no features_{v}.npz in the FT output folders listed in {SOURCES_NAME} (job not run)")
    return SimpleNamespace(job=job, rows=rows, shas=shas, variants=variants)


def ft_seed(f, episodes, seed) -> dict:
    """One variant's row of one seed: every member of the seed's episodes present, scored as ft_eval."""
    members = np.unique(D.member_rows(episodes.pooled))
    pos = np.minimum(np.searchsorted(f.rows, members), len(f.rows) - 1)
    absent = int((f.rows[pos] != members).sum())
    _require(absent == 0, f"{f.path.name}: {absent} member rows of seed {seed}'s episodes are "
                          f"missing")  # guard:ft_members
    return {"pa": score_placed(f.rows, f.img, f.txt, episodes.pooled), "info": {}}


def ft_info(lp_or_ft, name) -> dict:
    if name == "FT-LP":
        lp = lp_or_ft
        return {"checkpoint": {"file": str(lp.path), "sha256": lp.sha256, "lr": FT_CKPTS["LP"]["lr"],
                               "epoch": FT_CKPTS["LP"]["epoch"]},
                "features": "the seed's cached CLIP features x W^T (float64, rounded to float32), on the CPU",
                "scoring": SCORING}
    v = name.split("-", 1)[1]
    f = lp_or_ft.variants[v]
    return {"checkpoint": {"sha256": FT_CKPTS[v]["sha256"], "lr": FT_CKPTS[v]["lr"], "epoch": FT_CKPTS[v]["epoch"]},
            "features": {"file": str(f.path), "sha256": f.sha256, "n_rows": int(len(f.rows))},
            "job": {"folder": str(lp_or_ft.job), "files_sha256": dict(lp_or_ft.shas)}, "scoring": SCORING}


# ---------------------------------------------------------------- the in-context reranker

def unpermute_scores(shown, perms) -> np.ndarray:
    """(n, 2, 2, 13): letter j of (episode, condition, direction) showed candidate column perms[..., j], so column
    perms[..., j] receives shown[..., j] (mllm_reranker.unpermute, vectorised)."""
    shown, perms = np.asarray(shown), np.asarray(perms)
    _require(shown.shape == perms.shape and shown.shape[1:] == (2, 2, N_CAND), f"shown {shown.shape}, perms "
                                                                                f"{perms.shape}")
    _require(np.array_equal(np.sort(perms, axis=-1), np.broadcast_to(np.arange(N_CAND), perms.shape)),
             "perms are not permutations of the 13 columns")
    out = np.empty_like(shown)
    np.put_along_axis(out, perms, shown, axis=-1)
    return out


def load_rerank_scores(path, allowed) -> tuple:
    """(episode_index (m,), scores_shown (m, 2, 2, 13) float32) of a scores.npz, each index once and one the job
    holds."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        _require(sorted(z.files) == ["episode_index", "scores_shown"], f"{path}: keys {sorted(z.files)}")
        idx, sc = z["episode_index"], z["scores_shown"]
    _require(idx.dtype == np.int64 and idx.ndim == 1 and sc.dtype == np.float32
             and sc.shape == (len(idx), 2, 2, N_CAND) and bool(np.isfinite(sc).all()),
             f"{path}: episode_index (m,) int64 and finite scores_shown (m, 2, 2, 13) float32")
    _require(len(np.unique(idx)) == len(idx), f"{path}: an episode index is held twice")  # guard:mllm_unique
    extra = np.setdiff1d(idx, allowed)
    _require(len(extra) == 0, f"{path}: scores of {len(extra)} episodes the job does not hold, e.g. "
                              f"{extra[:3].tolist()}")  # guard:mllm_extra
    return idx, sc


def load_mllm(entry, out, smoke) -> SimpleNamespace:
    """The reranker job of the mode's first seed (its CPU-side permutations checked) and its merged scores."""
    _keys(entry, ("job", "out"), ("job", "out"), "mllm")
    seed = mode_seeds(smoke)[0]
    job = listed(entry["job"], "the reranker job")
    inp = T.load_rerank_input(job / "rerank_input.npz", G.Manifest(job / "rows_manifest.npz"))
    shas = {f: G.sha256_file(job / f) for f in ("rows_manifest.npz", "rerank_input.npz")}
    rec = _json(job / "job_record.json")
    pp = I.perms_path(job)
    _require(pp.is_file(), f"{pp}: the job's CPU-side permutations are missing")
    with np.load(pp, allow_pickle=False) as z:
        _require(sorted(z.files) == ["episode_index", "perms", "seed"], f"{pp}: keys {sorted(z.files)}")
        p = {k: z[k] for k in z.files}
    _require(inp["seed"] == int(p["seed"]) == rec.get("seed") == seed,
             f"{job}: a reranker job of seed {inp['seed']}, not of {seed}")  # guard:mllm_seed
    perms = p["perms"]
    _require(np.array_equal(p["episode_index"], inp["episode_index"]) and perms.dtype == np.int64
             and np.array_equal(perms, I.rerank_permutations(seed, inp["episode_index"]))
             and rec.get("perms_sha256") == G.sha256_bytes(np.ascontiguousarray(perms).tobytes())
             and all((rec.get("files_sha256") or {}).get(f) == s for f, s in shas.items()),
             f"{pp.name}: not the job's recorded permutations (the probe's formula, job_record.json's "
             f"SHA-256s)")  # guard:mllm_perms
    merged, files = {}, {}
    for d in _paths(entry, "out", "a reranker output"):
        fp = _json(d / "provenance.json").get("fingerprint") or {}
        _require(fp.get("job") == "r6_gpu_rerank" and fp.get("inputs_sha256") == shas,
                 f"{d}: not an output of the reranker job {job.name}")  # guard:mllm_out_job
        idx, sc = load_rerank_scores(d / "scores.npz", inp["episode_index"])
        files[str(d)] = G.sha256_file(d / "scores.npz")
        for i, s in zip(idx.tolist(), sc):
            _require(i not in merged or np.array_equal(merged[i], s),
                     f"{d}: episode {i} is held twice with different scores")  # guard:mllm_conflict
            merged[i] = s
    return SimpleNamespace(seed=seed, job=job, inp=inp, shas=shas, perms=perms, merged=merged, files=files,
                           perms_sha256=rec["perms_sha256"])


def mllm_seed(m, episodes, seed) -> dict:
    """The reranker's row of its seed: the job is the seed's episodes in full, every episode scored once."""
    p, n = episodes.pooled, len(episodes.pooled.anchor)
    inp = m.inp
    _require(int(seed) == m.seed and np.array_equal(inp["episode_index"], np.arange(n)),
             f"{m.job}: holds {len(inp['episode_index'])} episodes, not the {n} of seed {seed} in "
             f"order")  # guard:mllm_cover_job
    shown_rows = np.take_along_axis(np.broadcast_to(np.asarray(p.candidates)[:, None, None, :], m.perms.shape),
                                    m.perms, axis=-1)
    _require(np.array_equal(inp["query_row"], p.anchor) and np.array_equal(inp["cand_shown"], shown_rows)
             and all(np.array_equal(inp[f], getattr(p, f)) for f in G.PAIR_FIELDS),
             f"{m.job}: its query rows, example pairs or shown candidates are not seed {seed}'s episodes under its "
             f"permutations")  # guard:mllm_episodes
    absent = [i for i in range(n) if i not in m.merged]
    _require(not absent, f"{len(absent)} episodes of seed {seed} have no reranker scores, e.g. "
                         f"{absent[:3]}")  # guard:mllm_missing
    scores = unpermute_scores(np.stack([m.merged[i] for i in range(n)]), m.perms)
    by = {c: {d: scores[:, RERANK_CONDITIONS.index(c), RERANK_DIRECTIONS.index(d)] for d in DIRECTIONS}
          for c in CONDITIONS}
    return {"pa": per_anchor(by), "info": {"n_episodes": int(n)}}


def mllm_info(m) -> dict:
    return {"seed": m.seed, "job": {"folder": str(m.job), "files_sha256": dict(m.shas)},
            "outputs_sha256": dict(m.files), "permutations_sha256": m.perms_sha256,
            "permutation": "default_rng([seed, i]).permuted(tile(arange(13), (2, 2, 1)), axis=-1), i the episode "
                           "index; letter j showed candidate column perm[j]"}


# ---------------------------------------------------------------- the two hooks

LOADERS = {"dts": load_dts, "ft_lp": load_lp, "ft": load_ft, "mllm": load_mllm}


def new_state(out, smoke) -> SimpleNamespace:
    """Sources read and every listed family loaded and joined once (a str: the reason the family is missing)."""
    src = load_sources(out)
    st = SimpleNamespace(out=Path(out), smoke=bool(smoke), seeds=mode_seeds(smoke), src=src, fam={}, seen=[])
    for key in SOURCE_KEYS:
        reason = missing_reason(src, key)
        st.fam[key] = reason if reason is not None else LOADERS[key](src.entries[key], st.out, st.smoke)
    return st


def run_state(env, seed, out, smoke) -> SimpleNamespace:
    """The run's state on env: a new one at the mode's first seed (each run of the seed loop starts there)."""
    st = getattr(env, STATE, None)
    if int(seed) == mode_seeds(smoke)[0] or st is None or st.out != Path(out) or st.smoke is not bool(smoke):
        st = new_state(out, smoke)
        setattr(env, STATE, st)
    return st


def _missing(reason) -> dict:
    return {"pa": None, "info": {"missing": reason}}


def seed_rows(env, seed, ctx, episodes, out, smoke) -> dict:
    """run_r6_descriptive.external_seed_rows: {name: {"pa", "label", "info"}} of one seed (MLLM on the first seed
    only). A listed input that cannot be read (a missing file in a listed folder, a malformed JSON or npz) stops the
    pass like a failed join (AssertionError: exit 5), never a crash."""
    try:
        return _seed_rows(env, int(seed), ctx, episodes, out, smoke)
    except (OSError, KeyError, ValueError, TypeError) as e:
        raise AssertionError(f"external rows of seed {seed}: a listed input could not be read: "
                             f"{type(e).__name__}: {e}") from e


def _seed_rows(env, seed, ctx, episodes, out, smoke) -> dict:
    st = run_state(env, seed, out, smoke)
    st.seen.append(seed)
    p = episodes.pooled
    _require(int(ctx.seed) == seed == int(episodes.seed) and int(ctx.n) == len(p.anchor)
             and all(np.array_equal(getattr(ctx.pooled, f), getattr(p, f)) for f in E.FIELDS),
             f"seed {seed}: the context's episodes are not the episode file's")  # guard:ctx_episodes
    fam, rows = st.fam, {}
    if isinstance(fam["dts"], str):
        rows.update({name: _missing(fam["dts"]) for name in DTS_ROWS.values()})
    else:
        rows.update(dts_seed(fam["dts"], seed, ctx, episodes))
    rows["FT-LP"] = _missing(fam["ft_lp"]) if isinstance(fam["ft_lp"], str) else lp_seed(fam["ft_lp"], ctx, episodes)
    for v in FT_VARIANTS:
        f = fam["ft"] if isinstance(fam["ft"], str) else fam["ft"].variants[v]
        rows[f"FT-{v}"] = _missing(f) if isinstance(f, str) else ft_seed(f, episodes, seed)
    if seed == st.seeds[0]:
        rows["MLLM"] = _missing(fam["mllm"]) if isinstance(fam["mllm"], str) else mllm_seed(fam["mllm"], episodes,
                                                                                            seed)
    return {name: {"pa": x["pa"], "label": LABELS[name], "info": x["info"]} for name, x in rows.items()}


def family_info(st, name) -> dict:
    """The row's sources (file SHA-256 and its entry) and, when computed, its family's inputs and settings."""
    key = FAMILY[name]
    out = {"sources": {"file": SOURCES_NAME, "sha256": st.src.sha256, "entry": st.src.entries.get(key)}}
    fam = st.fam[key]
    if isinstance(fam, str):
        return out
    if key == "dts":
        out.update(dts_info(fam, name))
    elif key == "ft_lp":
        out.update(ft_info(fam, name))
    elif key == "ft":
        if not isinstance(fam.variants[name.split("-", 1)[1]], str):
            out.update(ft_info(fam, name))
    else:
        out.update(mllm_info(fam))
    return out


def family_reason(st, name):
    """The reason the row's family (or FT variant) is missing, None when it was loaded."""
    fam = st.fam[FAMILY[name]]
    if FAMILY[name] == "ft" and not isinstance(fam, str):
        fam = fam.variants[name.split("-", 1)[1]]
    return fam if isinstance(fam, str) else None


def finish(env, out, smoke, collected) -> dict:
    """run_r6_descriptive.external_rows: describe's external argument in NAMES order (then any other name given). A
    row not computed on every seed it is reported on (expected_seeds) keeps no number: its pa is dropped and its
    info says why ("missing"); every row gets its family's info when the state on env is this run's (made for this
    results folder and mode, and the per-seed hook saw every seed in order); any state is dropped."""
    seeds = mode_seeds(smoke)
    st = getattr(env, STATE, None)
    if st is not None:
        delattr(env, STATE)
        if not (st.out == Path(out) and st.smoke is bool(smoke) and st.seen == seeds):
            st = None                               # left by another run (another folder, or a run that stopped)
    result = {}
    for name in list(NAMES) + [n for n in collected if n not in NAMES]:
        x = collected.get(name) or {"pa": {}, "label": LABELS.get(name, name), "info": {"per_seed": {}}}
        pa, info = dict(x.get("pa") or {}), dict(x.get("info") or {})
        if sorted(pa) != sorted(expected_seeds(name, seeds)):
            per = info.get("per_seed") or {}
            reasons = list(dict.fromkeys(str(v["missing"]) for v in per.values()
                                         if isinstance(v, dict) and "missing" in v))
            fallback = family_reason(st, name) if st is not None and name in FAMILY else None
            info["missing"] = "; ".join(reasons) or fallback or "not computed on every seed it is reported on"
            if pa:
                info["seeds_computed_but_dropped"] = sorted(int(s) for s in pa)
            pa = {}
        if st is not None and name in FAMILY:
            info.update(family_info(st, name))
        result[name] = {"pa": pa, "label": x.get("label", LABELS.get(name, name)), "info": info}
    return result
