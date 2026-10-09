"""Round 6 describe-then-score (DTS), CPU side (DECISION_RULE.md of this folder: section 7 items 2 to 7, section 10
item 5; contracts section 9; spec D10, D11, D12, D18; ticket 11): parsing of the GPU jobs' raw answers, the CRL
projection, DTS and its two controls, the seed-42 tuning and stop arithmetic, and the held scoring with frozen picks.

Parsing (dts_settings.json, asserted equal to the rule). The phrase of a verbaliser answer is its first line
(answer.strip(), then str.splitlines()[0]) normalised: lower-cased, every run of Unicode whitespace one space, and
whitespace or Unicode punctuation (category P*) removed at both ends until none is left. A listing answer is split by
str.splitlines(); one leading list marker (^\\s*(\\d+[.)]|[-*•])\\s*) is removed from each line, which is then
normalised as a phrase; empty and repeated lines are dropped (the first occurrence kept) and the first K kept. A phrase
that is empty, or whose listing gives fewer than 2 values, is a parsing failure: T_DTS^c = 0 on that episode and
condition, counted per setting and condition. With a finite lambda the fused score of such a row is z(cos) exactly;
with lambda = inf (never picked so far) it would be the all-zero row, a miss.

GPU outputs, joined by key (C6). Verbaliser answers are keyed (seed, episode_index, condition, wording), listings
(phrase, K). Outputs of several folders (DAS6 tags, shards) are merged: a key seen twice must carry the identical raw
answer (greedy), else the merge refuses; a key the computation needs and no folder holds refuses; a record of another
seed, or of an episode index beyond the seed's episodes, refuses. Each output folder's provenance fingerprint must name
the settings file the CPU parses with, and each verbaliser folder must come from a job folder whose example pairs equal
the episodes' at its episode indices.

Value embeddings. Each value string (normalised, no template) goes alone (batch of one, so no padding enters) through
src.data.feature_extract.ClipB32("cpu"), float32, unit norm as returned. A cache file holds every string embedded so
far with the encoder's identity (model commit, library versions); a cache made by another encoder is refused.

CRL projection (Liu et al., NeurIPS 2025). For condition c of an episode the K value embeddings of its phrase form a
basis V (k x 512, k <= K). An item maps to R = V x (x its unit CLIP feature: image items through the image features,
caption items through the text features); T_DTS^c(query, candidate) = cos(R_query, R_candidate), 0 if either vector is
zero (a failed parse has an empty basis, so its row is 0). Directions as cosine_scores: i2t queries the anchor's image
against the candidates' captions, t2i the anchor's caption against their images. Computed in float64, stored float32.

Scores. DTS: fused_scores(cos, T, lambda) with crossfit_lambda (LAMBDA_GRID, edge extension). DTS-CF: the mean of
z(T^a) and z(T^b) per ranking row (zscore_rows in float32, averaged in float64, cast to float32), equal under both
conditions (asserted, and its gain asserted 0 on every episode), fused by crossfit_lambda. DTS-N: the phrase replaced
by the target aspect's name (condition a: the pair's first aspect), listed and scored as DTS. Frozen picks (held): the
pick of tune half h scores the episodes of parity 1 - h, exactly what crossfit_lambda does on seed 42.

Integers. The tuning score of a setting is (sum of 4 R@1 + sum of 4 gain) as int64 (r2_fusion.as_int4 per episode),
which orders the settings exactly as (R@1 + gain) / 2; ties go to the first setting in W1 K8, W1 K16, W2 K8, ...,
W4 K16. The stop's hit count is the int64 sum of as_int4(R@1) of DTS's cross-fitted fused scores; the build stops iff
it exceeds AFF's 9,406. The budget is 24 hours of elapsed time from the first DTS commit (an argument, Amsterdam time).

Guards carry a `# guard:<name>` marker; test_r6_dts.py deletes each on a copy and shows that its scenario then goes
through.
"""
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_gpu_common as G  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free  # noqa: E402
from src.eval.aspect_scorers import EDGE_EXTENSION, LAMBDA_GRID, crossfit_lambda, fused_scores  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

RULE_MARKER = r"^\s*(\d+[.)]|[-*•])\s*"          # rule section 7 item 2, verbatim
WORDINGS = ("W1", "W2", "W3", "W4")
KS = (8, 16)
SETTINGS_ORDER = tuple((w, k) for w in WORDINGS for k in KS)   # W1 K8, W1 K16, W2 K8, ..., W4 K16 (the tie order)
MIN_VALUES = 2
TUNE_PER_PAIR = 1024
TARGET_NAME = {"emotion": "emotion", "style": "style", "genre": "genre"}
N_CANDIDATES = 13
DIM = 512
VERBALISE_FIELDS = ("seed", "episode_index", "condition", "wording", "answer")
LISTING_FIELDS = ("phrase", "K", "answer")
FAIL_KINDS = ("empty_phrase", "short_listing")
LAMBDAS = tuple(LAMBDA_GRID) + tuple(EDGE_EXTENSION)
SCORERS = ("dts", "dts_cf", "dts_n")
METRICS = ("r1", "gain", "other", "swap", "strict")
CHUNK = 512
BUDGET_HOURS = 24
AMSTERDAM = "Europe/Amsterdam"
EMB_KEYS = ("strings", "emb", "meta")
as_int4 = R.F2.as_int4


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def setting_name(wording, K) -> str:
    return f"{wording} K{int(K)}"


# ---------------------------------------------------------------- settings (rule section 7)

def load_settings(path=None):
    """(settings, SHA-256 of the file): r6_gpu_common's checks plus the parts the CPU side relies on, each equal to
    the rule."""
    s, sha = G.load_settings(path)
    lst, order = s["listing"], s["seed42_order"]
    _require(lst["marker_regex"] == RULE_MARKER, f"settings: marker regex {lst['marker_regex']!r} is not the rule's")
    _require(lst["K"] == list(KS) and lst["min_values"] == MIN_VALUES, "settings: K or min_values differ from the rule")
    _require(order["tuning_settings"] == [setting_name(w, k) for w, k in SETTINGS_ORDER],
             "settings: the tuning order differs from W1 K8, W1 K16, ..., W4 K16")
    _require(order["tuning_subset_first_per_pair"] == TUNE_PER_PAIR, "settings: the tuning subset is not 1,024")
    _require(s["controls"]["DTS-N"]["phrase_of_target_aspect"] == TARGET_NAME, "settings: DTS-N's names differ")
    _require(s["stop"]["aff_hits_seed42"] == R.AFF_HITS_SEED42
             and s["stop"]["n_rankings_seed42"] == R.N_RANKINGS_SEED42, "settings: the stop's numbers differ")
    _require(s["budget"]["hours"] == BUDGET_HOURS, "settings: the budget is not 24 hours")
    return s, sha


# ---------------------------------------------------------------- parsing (rule section 7 items 1 and 2)

def first_line(answer) -> str:
    s = answer.strip()
    return s.splitlines()[0] if s else ""


def _edge(c) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


def normalise(s) -> str:
    """Lower-case, Unicode whitespace runs to one space, whitespace and punctuation removed at both ends."""
    s = re.sub(r"\s+", " ", s.lower())
    i, j = 0, len(s)
    while i < j and _edge(s[i]):
        i += 1
    while j > i and _edge(s[j - 1]):
        j -= 1
    return s[i:j]


def phrase_of(answer) -> str:
    """The verbaliser's phrase: the first line of the raw answer, normalised ('' when nothing is left)."""
    return normalise(first_line(answer))


def parse_listing(answer, K, marker=RULE_MARKER) -> list:
    """The values of a listing answer: per line one leading list marker removed, normalised; empty and repeated
    values dropped (first occurrence kept); the first K kept."""
    out, seen = [], set()
    for line in answer.splitlines():
        v = normalise(re.sub(marker, "", line, count=1))
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out[:int(K)]


def name_phrases(pair_index) -> dict:
    """DTS-N's phrases: condition a's target is the pair's first aspect, b's its second."""
    pi = np.asarray(pair_index, dtype=np.int64)
    return {c: [TARGET_NAME[R.PAIRS[int(k)][j]] for k in pi] for j, c in enumerate(CONDITIONS)}


# ---------------------------------------------------------------- GPU outputs, merged by key (C6)

def _fingerprint(d, job, settings_sha) -> dict:
    d = Path(d)
    path = d / "provenance.json"
    _require(path.is_file(), f"{d}: no provenance.json; not a GPU output folder")
    fp = json.loads(path.read_text(encoding="utf-8")).get("fingerprint", {})
    _require(fp.get("job") == job, f"{d}: an output of {fp.get('job')!r}, not of {job}")
    _require(fp.get("settings_sha256") == settings_sha,
             f"{d}: made with settings {str(fp.get('settings_sha256'))[:12]}, not this dts_settings.json "
             f"{settings_sha[:12]}")  # guard:settings_sha
    return fp


def _put(merged, key, answer, where):
    if key in merged:
        _require(merged[key] == answer, f"{where}: key {key} seen twice with different answers")  # guard:same_answer
    merged[key] = answer


def merge_verbaliser(dirs, wordings, settings_sha, seeds=None) -> SimpleNamespace:
    """Verbaliser outputs of several folders -> SimpleNamespace(answers {(seed, episode_index, condition, wording):
    raw answer}, by_dir {folder: set of keys}, fingerprints {folder: fingerprint}, files {name: sha}). ``seeds``
    (optional) refuses records of any other seed."""
    _require(len(dirs) > 0, "no verbaliser output folder given")
    answers, by_dir, fps, files = {}, {}, {}, {}
    for d in dirs:
        d = Path(d)
        fps[str(d)] = _fingerprint(d, "r6_gpu_verbalise", settings_sha)
        files[f"{d.name}/provenance.json"] = G.sha256_file(d / "provenance.json")
        keys = by_dir.setdefault(str(d), set())
        for w in wordings:
            _require(w in WORDINGS, f"wording {w!r} is not one of {WORDINGS}")
            path = d / f"phrases_{w}.jsonl"
            if not path.is_file():
                continue
            files[f"{d.name}/{path.name}"] = G.sha256_file(path)
            st = G.KeyedJsonl(path, VERBALISE_FIELDS, VERBALISE_FIELDS[:4])
            st.load()                                                   # unique within a file (its own guard)
            for rec in st.records():
                key = st.key(rec)
                _require(rec["wording"] == w and rec["condition"] in CONDITIONS and isinstance(rec["answer"], str)
                         and type(rec["seed"]) is int and type(rec["episode_index"]) is int,
                         f"{path}: malformed record {key}")
                _require(seeds is None or rec["seed"] in seeds,
                         f"{path}: a record of seed {rec['seed']}, not of {sorted(seeds or [])}")  # guard:foreign_seed
                _put(answers, key, rec["answer"], path)
                keys.add(key)
    return SimpleNamespace(answers=answers, by_dir=by_dir, fingerprints=fps, files=files)


def select_answers(answers, seed, wording, episode_index, n) -> dict:
    """{condition: [raw answer per episode_index]}: every (seed, i, c, wording) present; no key of this seed and
    wording beyond the seed's n episodes."""
    beyond = [k for k in answers if k[0] == seed and k[3] == wording and not 0 <= k[1] < n]
    _require(not beyond, f"{len(beyond)} answers of seed {seed}, {wording} lie beyond its {n} episodes, e.g. "
                         f"{beyond[:2]}")  # guard:key_range
    out = {c: [] for c in CONDITIONS}
    for i in np.asarray(episode_index, dtype=np.int64).tolist():
        for c in CONDITIONS:
            key = (int(seed), i, c, wording)
            _require(key in answers, f"no verbaliser answer for {key}")  # guard:missing_key
            out[c].append(answers[key])
    return out


def check_verbaliser_jobs(merged, job_dirs, eps) -> dict:
    """Each output folder of ``merged`` was made from one of ``job_dirs`` (fingerprint input SHA-256s), whose
    verbalise_input.npz holds this seed and, at each of its episode indices, the episodes' own example pairs; every
    record of the folder is a key of that job. -> {job folder name: {file: sha}}."""
    _require(len(job_dirs) > 0, "no verbaliser job folder given")
    jobs, out = {}, {}
    for j in job_dirs:
        j = Path(j)
        shas = {f: G.sha256_file(j / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}
        inp = G.load_verbalise_input(j / "verbalise_input.npz")
        pos = inp["episode_index"]
        _require(inp["seed"] == int(eps.seed) and int(pos[-1]) < len(eps.pooled.anchor),
                 f"{j}: a job of seed {inp['seed']} with {len(pos)} episodes, not of seed {eps.seed}")
        same = all(np.array_equal(inp[f], getattr(eps.pooled, f)[pos]) for f in G.PAIR_FIELDS)
        _require(same, f"{j}: its example pairs are not the episodes' at its episode indices")  # guard:job_pairs
        jobs[(shas["rows_manifest.npz"], shas["verbalise_input.npz"])] = (j, set(pos.tolist()))
        out[j.name] = shas
    for d, fp in merged.fingerprints.items():
        ins = fp.get("inputs_sha256", {})
        key = (ins.get("rows_manifest.npz"), ins.get("verbalise_input.npz"))
        _require(key in jobs, f"{d}: made from a job folder that is not given (input SHA-256s "
                              f"{str(key[1])[:12]})")  # guard:job_of_output
        j, pos = jobs[key]
        stray = [k for k in merged.by_dir[d] if k[0] != int(eps.seed) or k[1] not in pos]
        _require(not stray, f"{d}: records outside its job {j.name}, e.g. {stray[:2]}")
    return out


def merge_listings(dirs, settings_sha) -> SimpleNamespace:
    """Listing outputs of several folders -> SimpleNamespace(answers {(phrase, K): raw answer}, fingerprints,
    files)."""
    _require(len(dirs) > 0, "no listing output folder given")
    answers, fps, files = {}, {}, {}
    for d in dirs:
        d = Path(d)
        fps[str(d)] = _fingerprint(d, "r6_gpu_listing", settings_sha)
        path = d / "listings.jsonl"
        _require(path.is_file(), f"{d}: no listings.jsonl")
        files[f"{d.name}/provenance.json"] = G.sha256_file(d / "provenance.json")
        files[f"{d.name}/listings.jsonl"] = G.sha256_file(path)
        st = G.KeyedJsonl(path, LISTING_FIELDS, LISTING_FIELDS[:2])
        st.load()
        for rec in st.records():
            _require(isinstance(rec["phrase"], str) and type(rec["K"]) is int and rec["K"] in KS
                     and isinstance(rec["answer"], str), f"{path}: malformed record {st.key(rec)}")
            _put(answers, st.key(rec), rec["answer"], path)
    return SimpleNamespace(answers=answers, fingerprints=fps, files=files)


def listing_items(phrases, Ks) -> list:
    """The distinct (phrase, K) pairs a listing job must answer, sorted; empty phrases need no listing."""
    out = {(p, int(K)) for p in phrases if p for K in Ks}
    for p, _ in out:
        _require(p == normalise(p) and len(p.splitlines()) == 1, f"phrase {p!r} is not normalised")
    return sorted(out, key=lambda x: (x[1], x[0]))


# ---------------------------------------------------------------- value embeddings

def encoder_meta(enc) -> dict:
    """The identity of a text encoder: what a cache file must share with the encoder that extends it."""
    if hasattr(enc, "meta"):
        return dict(enc.meta)
    import transformers
    from src.data.feature_extract import CLIP_REPO
    return {"class": f"{type(enc).__module__}.{type(enc).__name__}", "repo": CLIP_REPO,
            "commit": getattr(enc.m.config, "_commit_hash", None), "device": enc.device, "dtype": str(enc.dtype),
            "batch_size": 1, "template": None, "torch": torch.__version__, "transformers": transformers.__version__}


def clip_cpu():
    """src.data.feature_extract.ClipB32("cpu"), loaded without progress bars (logs must hold no decimal)."""
    import transformers
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    from src.data.feature_extract import ClipB32
    return ClipB32("cpu")


class ValueEmbedder:
    """Unit float32 embeddings of value strings, one string per encoder call, cached by string (in memory and, with
    ``cache_path``, in an npz: strings, emb, meta)."""

    def __init__(self, cache_path=None, encoder_factory=None):
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.encoder_factory, self._enc = encoder_factory, None
        self.meta, self.index, self.rows, self.n_new = None, {}, [], 0
        if self.cache_path is not None and self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as z:
                _require(sorted(z.files) == sorted(EMB_KEYS), f"{self.cache_path}: keys {sorted(z.files)}")
                strings, emb, meta = z["strings"].tolist(), z["emb"], json.loads(str(z["meta"]))
            _require(emb.dtype == np.float32 and emb.shape == (len(strings), DIM) and len(set(strings)) == len(strings),
                     f"{self.cache_path}: not {len(strings)} distinct strings with float32 ({DIM},) rows")
            self._check_rows(emb, self.cache_path)
            self.meta = meta
            self.index = {s: i for i, s in enumerate(strings)}
            self.rows = list(emb)
        self.n_loaded = len(self.rows)

    @staticmethod
    def _check_rows(emb, what):
        norms = np.linalg.norm(emb.astype(np.float64), axis=1)
        _require(bool(np.isfinite(emb).all()) and bool((np.abs(norms - 1) < 1e-5).all()),
                 f"{what}: embeddings must be finite and of unit norm")

    def encoder(self):
        if self._enc is None:
            enc = (self.encoder_factory or clip_cpu)()
            meta = encoder_meta(enc)
            _require(self.meta is None or self.meta == meta,
                     f"{self.cache_path}: made by another encoder {self.meta}, not {meta}; use a new cache "
                     f"file")  # guard:embed_meta
            self._enc, self.meta = enc, meta
        return self._enc

    def embed(self, strings) -> np.ndarray:
        """(len(strings), 512) float32, rows in the given order."""
        strings = list(strings)
        for s in dict.fromkeys(strings):
            if s in self.index:
                continue
            _require(isinstance(s, str) and s != "", f"value {s!r} cannot be embedded")
            e = np.asarray(self.encoder().encode_texts([s], batch_size=1))
            _require(e.shape == (1, DIM) and e.dtype == np.float32, f"encoder returned {e.shape} {e.dtype}")
            self._check_rows(e, f"embedding of {s!r}")
            self.index[s] = len(self.rows)
            self.rows.append(e[0])
            self.n_new += 1
        if not strings:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([self.rows[self.index[s]] for s in strings]).astype(np.float32, copy=False)

    def save(self):
        """Write the cache (atomically) when strings were added; -> its SHA-256 (None without a cache file)."""
        if self.cache_path is None:
            return None
        if self.n_new or not self.cache_path.exists():
            _require(self.meta is not None or not self.rows, "cache rows without an encoder identity")
            strings = sorted(self.index, key=self.index.get)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_name(self.cache_path.stem + ".partial.npz")
            np.savez(tmp, strings=np.array(strings, dtype=str),
                     emb=np.stack(self.rows).astype(np.float32) if self.rows else np.zeros((0, DIM), np.float32),
                     meta=np.array(json.dumps(self.meta, sort_keys=True)))
            os.replace(tmp, self.cache_path)
            self.n_new = 0
        return G.sha256_file(self.cache_path)


# ---------------------------------------------------------------- bases and the CRL projection

def bases(phrases, K, listings) -> SimpleNamespace:
    """Per item the value list of its phrase at K, or None on a parsing failure. -> SimpleNamespace(values,
    fail {"empty_phrase", "short_listing"}, below_K (usable listings with fewer than K values))."""
    values, fail, below = [], dict.fromkeys(FAIL_KINDS, 0), 0
    cache = {}
    for p in phrases:
        if p == "":
            values.append(None)
            fail["empty_phrase"] += 1
            continue
        if p not in cache:
            key = (p, int(K))
            _require(key in listings, f"no listing for {key}")  # guard:missing_listing
            cache[p] = parse_listing(listings[key], K)
        v = cache[p]
        if len(v) < MIN_VALUES:
            values.append(None)
            fail["short_listing"] += 1
            continue
        below += len(v) < int(K)
        values.append(tuple(v))
    return SimpleNamespace(values=values, fail=fail, below_K=int(below))


def value_index(values_by_cond, K, embedder):
    """({condition: (n, K) int64 indices into emb, -1 padding (a failure is all -1)}, emb (m, 512) float32)."""
    strings = sorted({v for vals in values_by_cond.values() for x in vals if x is not None for v in x})
    emb = embedder.embed(strings)
    pos = {s: i for i, s in enumerate(strings)}
    out = {}
    for c, vals in values_by_cond.items():
        idx = np.full((len(vals), int(K)), -1, dtype=np.int64)
        for i, x in enumerate(vals):
            if x is not None:
                _require(len(x) <= int(K), f"{len(x)} values for K {K}")
                idx[i, :len(x)] = [pos[v] for v in x]
        out[c] = idx
    return out, emb


def _unit64(x, what) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    _require(bool(np.isfinite(x).all()), f"{what}: non-finite features (an item outside the rows)")  # guard:finite_items
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    _require(bool((n != 0).all()), f"{what}: a zero feature vector")
    return x / n


def crl_term(img, txt, pooled, vidx, emb, chunk=CHUNK) -> dict:
    """{condition: {direction: (n, 13) float32}}: T^c(query, candidate) = cos(V q, V k) with V the episode's basis
    for condition c (rows of ``emb`` at ``vidx[c]``, -1 a zero row); 0 where either vector is zero."""
    anchor, cand = np.asarray(pooled.anchor), np.asarray(pooled.candidates)
    n = len(anchor)
    _require(cand.shape == (n, N_CANDIDATES), f"candidates {cand.shape}, not ({n}, {N_CANDIDATES})")
    emb = np.asarray(emb)
    _require(emb.ndim == 2 and emb.shape[1] == DIM and emb.dtype == np.float32, f"emb {emb.shape} {emb.dtype}")
    E = np.concatenate([emb.astype(np.float64), np.zeros((1, DIM))])          # index -1 -> the zero row
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        idx = np.asarray(vidx[c])
        _require(idx.ndim == 2 and idx.shape[0] == n and idx.dtype == np.int64
                 and bool(((idx >= -1) & (idx < len(emb))).all()), f"basis indices of {c}: bad shape or range")
        for d in DIRECTIONS:
            Q, C = (img, txt) if d == "i2t" else (txt, img)
            t = np.empty((n, N_CANDIDATES), dtype=np.float32)
            for s in range(0, n, chunk):
                sl = slice(s, min(n, s + chunk))
                V = E[idx[sl]]                                                  # (m, K, 512)
                q = _unit64(Q[anchor[sl]], f"{d} queries")                     # (m, 512)
                k = _unit64(C[cand[sl]], f"{d} candidates")                    # (m, 13, 512)
                rq = np.matmul(V, q[:, :, None])[..., 0]                       # (m, K)
                rc = np.matmul(k, V.transpose(0, 2, 1))                        # (m, 13, K)
                num = np.matmul(rc, rq[:, :, None])[..., 0]                    # (m, 13)
                den = np.linalg.norm(rc, axis=-1) * np.linalg.norm(rq, axis=-1)[:, None]
                ok = den > 0
                t[sl] = np.where(ok, num / np.where(ok, den, 1.0), 0.0).astype(np.float32)
            out[c][d] = t
    return out


def dts_term(ctx, phrases_by_cond, K, listings, embedder) -> SimpleNamespace:
    """T^c of ``ctx``'s episodes for the given phrases (per condition, one per episode). -> SimpleNamespace(T,
    fail {c: {...}}, below_K {c: n}, n_values)."""
    b = {c: bases(phrases_by_cond[c], K, listings) for c in CONDITIONS}
    for c in CONDITIONS:
        _require(len(b[c].values) == ctx.n, f"{len(b[c].values)} phrases for {ctx.n} episodes")
    vidx, emb = value_index({c: b[c].values for c in CONDITIONS}, K, embedder)
    T = crl_term(ctx.img, ctx.txt, ctx.pooled, vidx, emb)
    return SimpleNamespace(T=T, fail={c: b[c].fail for c in CONDITIONS}, below_K={c: b[c].below_K for c in CONDITIONS},
                           n_values=int(len(emb)))


# ---------------------------------------------------------------- fusion, controls, frozen picks

def cf_term(T) -> dict:
    """DTS-CF's term: (z(T^a) + z(T^b)) / 2 per ranking row, z in float32, the mean in float64, cast to float32."""
    z = {c: {d: zscore_rows(torch.as_tensor(np.asarray(T[c][d]), dtype=torch.float32)).numpy() for d in DIRECTIONS}
         for c in CONDITIONS}
    m = {d: (0.5 * (z["a"][d].astype(np.float64) + z["b"][d].astype(np.float64))).astype(np.float32)
         for d in DIRECTIONS}
    return {c: {d: m[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def require_cf(scores, what):
    """Identical under both conditions (ValueError otherwise)."""
    _require_condition_free(scores, what)


def scored(scores, picks) -> SimpleNamespace:
    return SimpleNamespace(scores=scores, picks=picks, pa=per_anchor(scores))


def score_crossfit(cos, term, parity) -> SimpleNamespace:
    """DTS or DTS-N: crossfit_lambda's scores and picks {tune half: lambda}, and per_anchor."""
    return scored(*crossfit_lambda(cos, term, parity))


def score_cf(cos, T, parity, picks=None) -> SimpleNamespace:
    """DTS-CF from T: its term and its fused scores asserted condition-free and its gain 0 on every episode;
    cross-fitted, or with frozen ``picks``."""
    term = cf_term(T)
    require_cf(term, "DTS-CF term")  # guard:cf
    if picks is None:
        s = score_crossfit(cos, term, parity)
    else:
        s = scored(frozen_fused(cos, term, picks, parity), dict(picks))
    require_cf(s.scores, "DTS-CF scores")  # guard:cf
    _require(bool((s.pa["gain"] == 0).all()), "DTS-CF has a non-zero gain on some episode")  # guard:cf
    return s


def frozen_fused(cos, term, picks, parity) -> dict:
    """The pick of tune half h scores the episodes of parity 1 - h (crossfit_lambda's own convention)."""
    parity = np.asarray(parity)
    _require(sorted(picks) == [0, 1] and all(picks[h] in LAMBDAS for h in (0, 1)), f"picks {picks}")
    out = {c: {d: np.empty_like(np.asarray(cos[c][d])) for d in DIRECTIONS} for c in CONDITIONS}
    for h in (0, 1):
        apply = parity == 1 - h
        f = fused_scores(cos, term, picks[h])
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = f[c][d][apply]
    return out


def picks_to_json(picks) -> dict:
    return {str(h): ("inf" if picks[h] == float("inf") else float(picks[h])) for h in (0, 1)}


def picks_from_json(rec) -> dict:
    _require(sorted(rec) == ["0", "1"], f"picks keys {sorted(rec)}")
    out = {int(h): (float("inf") if v == "inf" else float(v)) for h, v in rec.items()}
    _require(all(v in LAMBDAS for v in out.values()), f"picks {rec} outside the lambda grid")
    return out


# ---------------------------------------------------------------- integers: tuning, sanity, the stop

def int4_sum(x, what) -> int:
    """sum of 4x over episodes, as int64 (every per-episode R@1 and gain is a multiple of 0.25)."""
    return int(as_int4(x, what).astype(np.int64).sum(dtype=np.int64))


def tune_score(pa) -> int:
    """(R@1 + gain) / 2 on n episodes is tune_score / (8 n): the same order, exact."""
    return int4_sum(pa["r1"], "R@1") + int4_sum(pa["gain"], "gain")


def choose_setting(scores) -> tuple:
    """The highest score; ties to the first in SETTINGS_ORDER."""
    _require(set(scores) == set(SETTINGS_ORDER), f"scores for {sorted(scores)}, not the 8 settings")
    best = SETTINGS_ORDER[0]
    for s in SETTINGS_ORDER[1:]:
        if scores[s] > scores[best]:
            best = s
    return best


def stop_decision(r1, n_expected, aff_hits=R.AFF_HITS_SEED42) -> dict:
    """The stop of rule section 7 item 6 on DTS's per-episode R@1 (cross-fitted fused scores, n_expected
    episodes)."""
    r1 = np.asarray(r1)
    _require(r1.shape == (int(n_expected),), f"R@1 of {r1.shape} episodes, not {n_expected}")
    hits = int4_sum(r1, "DTS R@1")
    return {"hits": hits, "aff_hits": int(aff_hits), "n_rankings": 4 * int(n_expected),
            "dts_above_aff": bool(hits > aff_hits)}


def parse_amsterdam(s) -> datetime:
    """'YYYY-MM-DD HH:MM[:SS]' in Amsterdam time -> an aware datetime."""
    from zoneinfo import ZoneInfo
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(s), fmt).replace(tzinfo=ZoneInfo(AMSTERDAM))
        except ValueError:
            pass
    raise ValueError(f"{s!r} is not 'YYYY-MM-DD HH:MM[:SS]' (Amsterdam time)")


def budget(clock_start, built_time, hours=BUDGET_HOURS) -> dict:
    """Rule section 7 item 7: built within ``hours`` of elapsed time from the clock start (both Amsterdam times)."""
    start, built = parse_amsterdam(clock_start), parse_amsterdam(built_time)
    deadline = start.astimezone(timezone.utc) + timedelta(hours=hours)       # elapsed time, safe across DST
    _require(built >= start, f"built at {built_time}, before the clock start {clock_start}")
    from zoneinfo import ZoneInfo
    return {"clock_start": str(clock_start), "built_time": str(built_time), "hours": hours,
            "deadline": deadline.astimezone(ZoneInfo(AMSTERDAM)).strftime("%Y-%m-%d %H:%M:%S"),
            "within_budget": bool(built.astimezone(timezone.utc) <= deadline)}


# ---------------------------------------------------------------- one setting, and the held scoring

def subset_context(ctx, positions) -> SimpleNamespace:
    """The episodes at ``positions`` of a context (their global episode indices kept for the answer keys)."""
    pos = np.asarray(positions, dtype=np.int64)
    p = ctx.pooled
    pooled = AspectEpisodes(p.aspect_a, p.aspect_b, *(np.asarray(getattr(p, f))[pos] for f in
                                                      ("anchor", "candidates", "pairs_a_img", "pairs_a_txt",
                                                       "pairs_b_img", "pairs_b_txt")))
    return SimpleNamespace(seed=ctx.seed, n=int(len(pos)), pooled=pooled, img=ctx.img, txt=ctx.txt,
                           cos={c: {d: np.asarray(ctx.cos[c][d])[pos] for d in DIRECTIONS} for c in CONDITIONS},
                           parity=np.asarray(ctx.parity)[pos], pair_index=np.asarray(ctx.pair_index)[pos],
                           episode_index=np.asarray(episode_index(ctx))[pos])


def episode_index(ctx) -> np.ndarray:
    """The episodes' indices in the seed's concatenated episodes (the answer keys): all of them unless a subset."""
    idx = getattr(ctx, "episode_index", None)
    return np.arange(ctx.n, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)


def phrases_for(ctx, answers, wording) -> dict:
    """{condition: [phrase per episode]} of ``ctx``'s episodes from the merged answers (every key present)."""
    raw = select_answers(answers, int(ctx.seed), wording, episode_index(ctx), total_episodes(ctx))
    return {c: [phrase_of(a) for a in raw[c]] for c in CONDITIONS}


def total_episodes(ctx) -> int:
    return int(getattr(ctx, "n_total", ctx.n))


def run_setting(ctx, answers, wording, K, listings, embedder) -> SimpleNamespace:
    """DTS of one setting on ``ctx``'s episodes, cross-fitted. -> SimpleNamespace(s (scored), term, fail, below_K,
    score (tune_score))."""
    t = dts_term(ctx, phrases_for(ctx, answers, wording), K, listings, embedder)
    s = score_crossfit(ctx.cos, t.T, ctx.parity)
    return SimpleNamespace(s=s, term=t, fail=t.fail, below_K=t.below_K, score=tune_score(s.pa))


def run_names(ctx, K, listings, embedder) -> SimpleNamespace:
    """DTS-N at K, cross-fitted."""
    t = dts_term(ctx, name_phrases(ctx.pair_index), K, listings, embedder)
    s = score_crossfit(ctx.cos, t.T, ctx.parity)
    return SimpleNamespace(s=s, term=t, fail=t.fail, below_K=t.below_K, gain_int4=int4_sum(s.pa["gain"], "gain"))


def held_dts_scores(ctx, answers, listings, record, embedder) -> dict:
    """DTS, DTS-CF and DTS-N on ``ctx``'s episodes (a held seed's RowContext, or any context with seed, n, pooled,
    img, txt (NaN outside the rows), cos, parity, pair_index) with the chosen setting and the frozen picks of
    ``record`` (results/dts_seed42.json). ``answers``: merge_verbaliser(...).answers (keys of ctx.seed and the chosen
    wording, all present); ``listings``: merge_listings(...).answers. The pick of seed-42 tune half h scores the
    episodes of parity 1 - h. -> {"dts", "dts_cf", "dts_n": per_anchor dict, "failures": {"dts", "dts_n": {c:
    {...}}}, "below_K": {...}, "setting", "picks"}. Computes no summary; ticket 14 calls it only after the verdict."""
    w, K = record["wording_id"], int(record["K"])
    _require((w, K) in SETTINGS_ORDER and record["setting"] == setting_name(w, K), f"setting {record['setting']!r}")
    picks = {s: picks_from_json(record["picks"][s]) for s in SCORERS}
    _require(ctx.cos["a"]["i2t"].shape == (ctx.n, N_CANDIDATES), "cos does not match the episodes")
    t = dts_term(ctx, phrases_for(ctx, answers, w), K, listings, embedder)
    tn = dts_term(ctx, name_phrases(ctx.pair_index), K, listings, embedder)
    dts = per_anchor(frozen_fused(ctx.cos, t.T, picks["dts"], ctx.parity))
    cf = score_cf(ctx.cos, t.T, ctx.parity, picks["dts_cf"]).pa
    dn = per_anchor(frozen_fused(ctx.cos, tn.T, picks["dts_n"], ctx.parity))
    return {"dts": dts, "dts_cf": cf, "dts_n": dn, "failures": {"dts": t.fail, "dts_n": tn.fail},
            "below_K": {"dts": t.below_K, "dts_n": tn.below_K}, "setting": record["setting"],
            "picks": {s: record["picks"][s] for s in SCORERS}}
