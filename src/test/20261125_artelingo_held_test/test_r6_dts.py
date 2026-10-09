"""Tests of r6_dts and run_r6_dts (ticket 11): describe-then-score parsing, the joins of the GPU outputs (C6), the CRL
projection, DTS and its controls, the tuning order, the stop and its budget, the held scoring with frozen picks, and
an end-to-end run of every stage on the real smoke episodes of seed 9001 (selection rows) with crafted GPU outputs.
No held row is loaded: the held path runs on synthetic data of the real shapes (308,723 rows, 512-d, 12,288 episodes,
13 candidates, 4 pairs a side) and, on seed 9001, as the same code in selection mode. Guards are deleted on copies.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_dts.py
"""
import ast
import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first: it puts MAIN's src in front and checks it)
import r6_dts as D  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import r6_gpu_listing as L  # noqa: E402
import run_r6_dts as RD  # noqa: E402

import torch  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_scorers import EDGE_EXTENSION, LAMBDA_GRID, EvalInputs, cosine_scores, crossfit_lambda  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

SETTINGS, SETTINGS_SHA = D.load_settings()
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
PY = sys.executable
N, N_PER, NC, NP = 3 * R.N_PER_PAIR, R.N_PER_PAIR, 13, 4
DECIMAL = re.compile(r"\d*\.\d+")
SEED = 9001


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, fname, guard=None, replace=None):
    """A copy of ``fname`` with the `# guard:<guard>` statements replaced by `pass` (or one text replaced)."""
    src = (HERE / fname).read_text()
    assert src.count(HERE_LINE) == 1
    if guard is not None:
        lines = src.splitlines(keepends=True)
        hits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Expr, ast.Return, ast.Assign))
                and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, guard
        for n in hits:
            indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
            lines[n.lineno - 1] = f"{indent}pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
        src = "".join(lines)
    if replace is not None:
        old, new = replace
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = tmp_path / f"{Path(fname).stem}_copy{next(_COUNT)}.py"
    path.write_text(src.replace(HERE_LINE, f"HERE = Path({str(HERE)!r})\n"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def test_every_marked_guard_has_a_mutation_test():
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    for f in ("r6_dts.py", "run_r6_dts.py"):
        marked = set(re.findall(r"# guard:(\w+)", (HERE / f).read_text()))
        assert marked <= tested, (f, marked - tested)


class FakeEncoder:
    """Deterministic unit vectors per string; records every call."""
    meta = {"class": "fake", "batch_size": 1}

    def __init__(self):
        self.calls = []

    def encode_texts(self, texts, batch_size=256):
        self.calls.append((list(texts), batch_size))
        out = []
        for t in texts:
            v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:12], 16)).standard_normal(512)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


def fake_embedder(path=None):
    enc = FakeEncoder()
    return D.ValueEmbedder(path, encoder_factory=lambda: enc), enc


def fingerprint(job, inputs=None, settings_sha=SETTINGS_SHA):
    fp = {"job": job, "model_id": SETTINGS["model"]["id"], "snapshot": SETTINGS["model"]["snapshot"],
          "settings_sha256": settings_sha, "scripts_sha256": {"x.py": "0" * 64}}
    if inputs is not None:
        fp["inputs_sha256"] = inputs
    return fp


def write_verbaliser_out(d, records, inputs, settings_sha=SETTINGS_SHA):
    d = Path(d)
    d.mkdir(parents=True)
    by_w = defaultdict(list)
    for r in records:
        by_w[r["wording"]].append(r)
    for w, recs in by_w.items():
        st = G.KeyedJsonl(d / f"phrases_{w}.jsonl", D.VERBALISE_FIELDS, D.VERBALISE_FIELDS[:4])
        for r in recs:
            st.add({f: r[f] for f in D.VERBALISE_FIELDS})
        st.save()
    G.write_json(d / "provenance.json", {"fingerprint": fingerprint("r6_gpu_verbalise", inputs, settings_sha),
                                         "runs": []})
    return d


def write_listing_out(d, answers, settings_sha=SETTINGS_SHA):
    d = Path(d)
    d.mkdir(parents=True)
    st = G.KeyedJsonl(d / "listings.jsonl", D.LISTING_FIELDS, D.LISTING_FIELDS[:2])
    for (p, K), a in answers.items():
        st.add({"phrase": p, "K": int(K), "answer": a})
    st.save()
    G.write_json(d / "provenance.json", {"fingerprint": fingerprint("r6_gpu_listing", None, settings_sha),
                                         "runs": []})
    return d


def write_job(d, pooled, seed, positions):
    """A verbaliser job folder as r6_gpu_inputs writes it (the two npz the fingerprint names)."""
    d = Path(d)
    d.mkdir(parents=True)
    positions = np.asarray(positions, dtype=np.int64)
    arrays = {"seed": np.int64(seed), "episode_index": positions,
              **{f: np.ascontiguousarray(getattr(pooled, f)[positions], dtype=np.int64) for f in G.PAIR_FIELDS}}
    np.savez(d / "verbalise_input.npz", **arrays)
    rows = np.unique(np.concatenate([arrays[f].ravel() for f in G.PAIR_FIELDS]))
    np.savez(d / "rows_manifest.npz", rows=rows)
    return {f: G.sha256_file(d / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}


def synth_pooled(rows, n_per_pair, seed=0):
    rng = np.random.default_rng(seed)
    n = 3 * n_per_pair
    pick = lambda *shape: rng.choice(rows, size=shape).astype(np.int64)  # noqa: E731
    return AspectEpisodes("mixed", "mixed", pick(n), pick(n, NC), pick(n, NP), pick(n, NP), pick(n, NP), pick(n, NP))


def records_for(seed, positions, wordings, answer_fn):
    return [{"seed": seed, "episode_index": int(i), "condition": c, "wording": w, "answer": answer_fn(int(i), c, w)}
            for i in positions for w in wordings for c in CONDITIONS]


def as_scores(x):
    return {c: {d: np.asarray(x[c][d], dtype=np.float32) for d in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- settings and parsing

def test_settings_match_the_rule_and_a_changed_copy_is_refused(tmp_path):
    s, sha = D.load_settings()
    assert sha == G.sha256_file(HERE / "dts_settings.json") and s["listing"]["marker_regex"] == D.RULE_MARKER
    for key, value in (("marker_regex", r"^\s*(\d+[.)])\s*"), ("min_values", 1)):
        bad = json.loads((HERE / "dts_settings.json").read_text())
        bad["listing"][key] = value
        p = tmp_path / f"bad_{key}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            D.load_settings(p)
    bad = json.loads((HERE / "dts_settings.json").read_text())
    bad["seed42_order"]["tuning_settings"] = bad["seed42_order"]["tuning_settings"][::-1]
    (tmp_path / "order.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="tuning order"):
        D.load_settings(tmp_path / "order.json")


@pytest.mark.parametrize("answer, phrase", [
    ("Melancholy", "melancholy"),
    ("  The   shared\tEMOTION:  awe!  ", "the shared emotion: awe"),
    ("\n\n  “Baroque style.”\nSecond line ignored", "baroque style"),
    ("‘Impressionism’ — light", "impressionism’ — light"),
    ("...!?", ""),
    ("", ""),
    ("   \n \t\n", ""),
    ("first second", "first"),
    ("(landscape)", "landscape"),
    ("- portrait genre", "portrait genre"),
    ("$5 colour", "$5 colour"),
    ("café  CRÈME", "café crème"),
])
def test_phrase_is_the_normalised_first_line(answer, phrase):
    assert D.phrase_of(answer) == phrase
    assert D.normalise(D.phrase_of(answer)) == D.phrase_of(answer)       # idempotent


def test_listing_parsing_markers_blank_lines_duplicates_punctuation_and_K():
    answer = ("1. Joy\n2) Sadness\n\n- joy\n* Awe.\n• Fear\n   \n10. “Contentment”\n"
              "3.5 stars\n-- dashes\nAmusement\nanger\nExcitement\nDisgust")
    vals = D.parse_listing(answer, 16)
    assert vals == ["joy", "sadness", "awe", "fear", "contentment", "5 stars", "dashes", "amusement", "anger",
                    "excitement", "disgust"]
    assert D.parse_listing(answer, 8) == vals[:8]
    assert D.parse_listing("1. 1. nested", 8) == ["1. nested"]                  # one marker removed only
    assert D.parse_listing("Joy\r\nJOY\rjoy!\n", 8) == ["joy"]                 # repeated after normalising
    assert D.parse_listing("", 8) == [] and D.parse_listing("\n\n- \n", 8) == []


def test_parsing_failures_are_counted_and_need_no_listing_for_an_empty_phrase():
    listings = {("one value", 8): "only this", ("two", 8): "a\nb", ("many", 8): "\n".join("abcdefghij")}
    b = D.bases(["", "one value", "two", "many", "", "two"], 8, listings)
    assert b.fail == {"empty_phrase": 2, "short_listing": 1}
    assert b.values == [None, None, ("a", "b"), tuple("abcdefgh"), None, ("a", "b")]
    assert b.below_K == 2


def test_missing_listing_raises(tmp_path):
    with pytest.raises(AssertionError, match="no listing"):
        D.bases(["two"], 16, {("two", 8): "a\nb"})
    mut = mutant(tmp_path, "r6_dts.py", guard="missing_listing")
    with pytest.raises(KeyError):                                       # without the guard: no clear refusal
        mut.bases(["two"], 16, {("two", 8): "a\nb"})


def test_name_phrases_target_the_pairs_first_aspect_under_a():
    pi = np.array([0, 1, 2, 2], dtype=np.int64)
    got = D.name_phrases(pi)
    assert got == {"a": ["emotion", "emotion", "style", "style"], "b": ["style", "genre", "genre", "genre"]}
    for k, (a, b, _) in enumerate(R.PAIRS):
        assert D.name_phrases(np.array([k]))["a"] == [a] and D.name_phrases(np.array([k]))["b"] == [b]


# ---------------------------------------------------------------- joins (C6)

def test_merge_dedupes_identical_keys_across_folders_and_refuses_a_conflict(tmp_path):
    inputs = {"rows_manifest.npz": "1" * 64, "verbalise_input.npz": "2" * 64}
    recs = records_for(42, range(4), ["W1"], lambda i, c, w: f"answer {i}{c}")
    a = write_verbaliser_out(tmp_path / "a", recs[:5], inputs)
    b = write_verbaliser_out(tmp_path / "b", recs[4:], inputs)                 # key 4 in both, same answer
    m = D.merge_verbaliser([a, b], ["W1"], SETTINGS_SHA, seeds={42})
    assert len(m.answers) == 8
    got = D.select_answers(m.answers, 42, "W1", np.arange(4), 4)
    assert got == {c: [f"answer {i}{c}" for i in range(4)] for c in CONDITIONS}
    bad = dict(recs[4], answer="something else")
    c = write_verbaliser_out(tmp_path / "c", [bad], inputs)
    with pytest.raises(AssertionError, match="different answers"):
        D.merge_verbaliser([a, b, c], ["W1"], SETTINGS_SHA)
    mut = mutant(tmp_path, "r6_dts.py", guard="same_answer")
    mut.merge_verbaliser([a, b, c], ["W1"], SETTINGS_SHA)                  # silently takes one of the two


def test_a_missing_or_duplicated_key_raises(tmp_path):
    inputs = {"rows_manifest.npz": "1" * 64, "verbalise_input.npz": "2" * 64}
    recs = records_for(42, range(3), ["W2"], lambda i, c, w: "x")
    d = write_verbaliser_out(tmp_path / "miss", [r for r in recs if not (r["episode_index"] == 1
                                                                         and r["condition"] == "b")], inputs)
    m = D.merge_verbaliser([d], ["W2"], SETTINGS_SHA)
    with pytest.raises(AssertionError, match=r"no verbaliser answer for \(42, 1, 'b', 'W2'\)"):
        D.select_answers(m.answers, 42, "W2", np.arange(3), 3)
    mut = mutant(tmp_path, "r6_dts.py", guard="missing_key")
    with pytest.raises(KeyError):
        mut.select_answers(m.answers, 42, "W2", np.arange(3), 3)
    dup = tmp_path / "dup"                                                   # one key twice in one file
    write_verbaliser_out(dup, recs, inputs)
    line = (dup / "phrases_W2.jsonl").read_text().splitlines()[0]
    with open(dup / "phrases_W2.jsonl", "a") as f:
        f.write(line + "\n")
    with pytest.raises(AssertionError, match="duplicate record"):
        D.merge_verbaliser([dup], ["W2"], SETTINGS_SHA)


def test_foreign_seed_and_out_of_range_keys_are_refused(tmp_path):
    inputs = {"rows_manifest.npz": "1" * 64, "verbalise_input.npz": "2" * 64}
    d = write_verbaliser_out(tmp_path / "s", records_for(9001, range(2), ["W1"], lambda i, c, w: "x"), inputs)
    with pytest.raises(AssertionError, match="a record of seed 9001"):
        D.merge_verbaliser([d], ["W1"], SETTINGS_SHA, seeds={42})
    mutant(tmp_path, "r6_dts.py", guard="foreign_seed").merge_verbaliser([d], ["W1"], SETTINGS_SHA, seeds={42})
    e = write_verbaliser_out(tmp_path / "r", records_for(42, [0, 1, 5], ["W1"], lambda i, c, w: "x"), inputs)
    m = D.merge_verbaliser([e], ["W1"], SETTINGS_SHA)
    with pytest.raises(AssertionError, match="beyond its 5 episodes"):
        D.select_answers(m.answers, 42, "W1", np.arange(2), 5)
    mutant(tmp_path, "r6_dts.py", guard="key_range").select_answers(m.answers, 42, "W1", np.arange(2), 5)


def test_outputs_of_other_settings_are_refused(tmp_path):
    inputs = {"rows_manifest.npz": "1" * 64, "verbalise_input.npz": "2" * 64}
    d = write_verbaliser_out(tmp_path / "v", records_for(42, range(1), ["W1"], lambda i, c, w: "x"), inputs,
                             settings_sha="f" * 64)
    with pytest.raises(AssertionError, match="not this dts_settings.json"):
        D.merge_verbaliser([d], ["W1"], SETTINGS_SHA)
    lst = write_listing_out(tmp_path / "l", {("joy", 8): "a\nb"}, settings_sha="f" * 64)
    with pytest.raises(AssertionError, match="not this dts_settings.json"):
        D.merge_listings([lst], SETTINGS_SHA)
    mut = mutant(tmp_path, "r6_dts.py", guard="settings_sha")
    mut.merge_verbaliser([d], ["W1"], SETTINGS_SHA)
    mut.merge_listings([lst], SETTINGS_SHA)


def test_listing_merge_dedupes_and_refuses_a_conflict(tmp_path):
    a = write_listing_out(tmp_path / "a", {("joy", 8): "a\nb", ("awe", 16): "c\nd"})
    b = write_listing_out(tmp_path / "b", {("joy", 8): "a\nb"})
    assert D.merge_listings([a, b], SETTINGS_SHA).answers == {("joy", 8): "a\nb", ("awe", 16): "c\nd"}
    c = write_listing_out(tmp_path / "c", {("joy", 8): "a\nz"})
    with pytest.raises(AssertionError, match="different answers"):
        D.merge_listings([a, c], SETTINGS_SHA)


def test_verbaliser_outputs_must_come_from_a_job_of_these_episodes(tmp_path):
    rows = np.arange(1000, 2000, dtype=np.int64)
    pooled = synth_pooled(rows, 8)
    eps = SimpleNamespace(seed=42, pooled=pooled, n=24)
    pos = np.array([0, 3, 9], dtype=np.int64)
    shas = write_job(tmp_path / "job", pooled, 42, pos)
    out = write_verbaliser_out(tmp_path / "out", records_for(42, pos, ["W1"], lambda i, c, w: "x"), shas)
    m = D.merge_verbaliser([out], ["W1"], SETTINGS_SHA)
    assert D.check_verbaliser_jobs(m, [tmp_path / "job"], eps) == {"job": shas}
    other = synth_pooled(rows, 8, seed=1)                                    # a job built from other episodes
    shas2 = write_job(tmp_path / "job2", other, 42, pos)
    with pytest.raises(AssertionError, match="not the episodes'"):
        D.check_verbaliser_jobs(m, [tmp_path / "job2"], eps)
    mutant(tmp_path, "r6_dts.py", guard="job_pairs").check_verbaliser_jobs(
        D.merge_verbaliser([write_verbaliser_out(tmp_path / "out2", records_for(42, pos, ["W1"], lambda i, c, w: "x"),
                                                 shas2)], ["W1"], SETTINGS_SHA), [tmp_path / "job2"], eps)
    with pytest.raises(AssertionError, match="not given"):                  # output of a job not passed
        D.check_verbaliser_jobs(m, [tmp_path / "job2"], SimpleNamespace(seed=42, pooled=other, n=24))
    with pytest.raises(KeyError):                                          # without the guard: no clear refusal
        mutant(tmp_path, "r6_dts.py", guard="job_of_output").check_verbaliser_jobs(
            m, [tmp_path / "job2"], SimpleNamespace(seed=42, pooled=other, n=24))
    stray = write_verbaliser_out(tmp_path / "stray", records_for(42, [5], ["W1"], lambda i, c, w: "x"), shas)
    with pytest.raises(AssertionError, match="records outside its job"):
        D.check_verbaliser_jobs(D.merge_verbaliser([stray], ["W1"], SETTINGS_SHA), [tmp_path / "job"], eps)


# ---------------------------------------------------------------- embeddings

def test_embedder_encodes_each_string_alone_once_and_caches_it(tmp_path):
    emb, enc = fake_embedder(tmp_path / "e.npz")
    x = emb.embed(["joy", "awe", "joy"])
    assert x.shape == (3, 512) and x.dtype == np.float32 and np.array_equal(x[0], x[2])
    assert enc.calls == [(["joy"], 1), (["awe"], 1)]
    sha = emb.save()
    again, enc2 = fake_embedder(tmp_path / "e.npz")
    assert np.array_equal(again.embed(["awe", "joy"]), x[[1, 0]]) and enc2.calls == []
    assert again.save() == sha and again.meta == FakeEncoder.meta
    assert D.ValueEmbedder(tmp_path / "e.npz").embed([]).shape == (0, 512)


def test_a_cache_of_another_encoder_is_refused(tmp_path):
    emb, _ = fake_embedder(tmp_path / "e.npz")
    emb.embed(["joy"])
    emb.save()

    class Other(FakeEncoder):
        meta = {"class": "other", "batch_size": 1}
    cached = D.ValueEmbedder(tmp_path / "e.npz", encoder_factory=Other)
    assert np.array_equal(cached.embed(["joy"]), emb.embed(["joy"]))       # cached strings need no encoder
    with pytest.raises(AssertionError, match="another encoder"):
        cached.embed(["new"])
    mut = mutant(tmp_path, "r6_dts.py", guard="embed_meta")
    mut.ValueEmbedder(tmp_path / "e.npz", encoder_factory=Other).embed(["new"])


@pytest.fixture(scope="module")
def clip():
    return D.clip_cpu()


def test_real_clip_value_embeddings_are_unit_float32_and_batch_independent(tmp_path, clip):
    emb = D.ValueEmbedder(tmp_path / "c.npz", encoder_factory=lambda: clip)
    words = ["joy", "baroque", "a long phrase about quiet impressionist landscapes"]
    x = emb.embed(words)
    assert x.dtype == np.float32 and np.allclose(np.linalg.norm(x.astype(np.float64), axis=1), 1, atol=1e-5)
    assert all(np.array_equal(x[i], clip.encode_texts([w], batch_size=1)[0]) for i, w in enumerate(words))
    assert emb.meta["commit"] == "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268" and emb.meta["template"] is None
    emb.save()
    back = D.ValueEmbedder(tmp_path / "c.npz", encoder_factory=lambda: clip)
    assert np.array_equal(back.embed(words[::-1]), x[::-1])


# ---------------------------------------------------------------- the CRL projection

def tiny_case():
    """One episode: rows 0 (anchor), 1 to 13 (candidates); features in the span of e0, e1, e2."""
    img = np.zeros((20, 512), np.float32)
    txt = np.zeros((20, 512), np.float32)
    img[0, 0] = 1                                           # anchor image along e0
    txt[0, :2] = [3, 4]                                     # anchor caption (0.6, 0.8) after unit norm
    for j in range(13):
        img[1 + j, :3] = [1, j % 3, 0.5]
        txt[1 + j, :3] = [j % 2, 1, (j % 4) - 1.5]
    pooled = AspectEpisodes("mixed", "mixed", np.array([0]), np.arange(1, 14)[None], np.zeros((1, 4), int),
                            np.zeros((1, 4), int), np.zeros((1, 4), int), np.zeros((1, 4), int))
    emb = np.zeros((3, 512), np.float32)
    emb[0, 0] = emb[1, 1] = 1
    emb[2, :2] = [np.sqrt(0.5), np.sqrt(0.5)]
    return img, txt, pooled, emb


def test_crl_projection_matches_a_hand_computation():
    img, txt, pooled, emb = tiny_case()
    vidx = {"a": np.array([[0, 1, -1]], dtype=np.int64), "b": np.array([[2, 1, 0]], dtype=np.int64)}
    T = D.crl_term(img, txt, pooled, vidx, emb)
    for c in CONDITIONS:
        V = emb[[i for i in vidx[c][0] if i >= 0]].astype(np.float64)
        for d in DIRECTIONS:
            q = (img if d == "i2t" else txt)[0].astype(np.float64)
            want = []
            for j in range(13):
                k = (txt if d == "i2t" else img)[1 + j].astype(np.float64)
                rq, rk = V @ (q / np.linalg.norm(q)), V @ (k / np.linalg.norm(k))
                want.append(rq @ rk / (np.linalg.norm(rq) * np.linalg.norm(rk)))
            assert T[c][d].dtype == np.float32 and T[c][d].shape == (1, 13)
            np.testing.assert_allclose(T[c][d][0], np.array(want), rtol=0, atol=1e-7)
    # exact values: the anchor image along e0 against basis {e0, e1}: R_q = (1, 0); a candidate caption (0, 1, -1.5)
    # (j = 0) maps to (0, 1/|k|) -> cos 0; (1, 1, 0.5) (j = 1) maps to (1, 1)/|k| -> cos 1/sqrt(2)
    assert T["a"]["i2t"][0, 0] == 0 and T["a"]["i2t"][0, 1] == np.float32(1 / np.sqrt(2))


def test_an_empty_basis_or_an_orthogonal_item_gives_T_zero():
    img, txt, pooled, emb = tiny_case()
    T = D.crl_term(img, txt, pooled, {"a": np.full((1, 8), -1, np.int64), "b": np.array([[2] + [-1] * 7])}, emb)
    assert all(np.array_equal(T["a"][d], np.zeros((1, 13), np.float32)) for d in DIRECTIONS)
    img2 = img.copy()
    img2[0] = 0
    img2[0, 2] = 1                                          # anchor image along e2: orthogonal to every value
    T2 = D.crl_term(img2, txt, pooled, {c: np.array([[0, 1, 2]]) for c in CONDITIONS}, emb)
    assert np.array_equal(T2["a"]["i2t"], np.zeros((1, 13), np.float32))
    assert np.isfinite(T2["a"]["t2i"]).all()


def test_items_outside_the_rows_are_refused(tmp_path):
    img, txt, pooled, emb = tiny_case()
    img[5] = np.nan                                         # candidate 4's image: outside the evaluated rows
    vidx = {c: np.array([[0, 1, 2]]) for c in CONDITIONS}
    with pytest.raises(AssertionError, match="non-finite"):
        D.crl_term(img, txt, pooled, vidx, emb)
    T = mutant(tmp_path, "r6_dts.py", guard="finite_items").crl_term(img, txt, pooled, vidx, emb)
    assert T["a"]["t2i"][0, 4] == 0                          # without the guard: the NaN item silently scores 0


def test_T_zero_fallback_gives_exactly_cosine_after_fusion():
    rng = np.random.default_rng(3)
    cos = as_scores({c: {d: np.tile(rng.standard_normal((N, NC)).astype(np.float32), 1) for d in DIRECTIONS}
                     for c in CONDITIONS})
    cos["b"] = cos["a"]
    T = as_scores({c: {d: rng.standard_normal((N, NC)) for d in DIRECTIONS} for c in CONDITIONS})
    fail = rng.random(N) < 0.1
    for c in CONDITIONS:
        for d in DIRECTIONS:
            T[c][d][fail] = 0                                # a parsing failure: T = 0 on that episode
    zc = {d: zscore_rows(torch.as_tensor(cos["a"][d])).numpy() for d in DIRECTIONS}
    for lam in [x for x in list(LAMBDA_GRID) + list(EDGE_EXTENSION) if np.isfinite(x)]:
        f = fused_scores(cos, T, lam)
        assert all(np.array_equal(f[c][d][fail], zc[d][fail]) for c in CONDITIONS for d in DIRECTIONS), lam
    s, picks = crossfit_lambda(cos, T, np.arange(N) % 2)
    if all(np.isfinite(v) for v in picks.values()):
        assert all(np.array_equal(s[c][d][fail], zc[d][fail]) for c in CONDITIONS for d in DIRECTIONS)


# ---------------------------------------------------------------- fusion, controls, frozen picks

def synth_terms(seed=5, n=N):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, NC)).astype(np.float32)
    cos = as_scores({c: {d: base + (0.1 * (d == "t2i")) for d in DIRECTIONS} for c in CONDITIONS})
    tgt = np.zeros((n, NC), np.float32)
    T = {}
    for j, c in enumerate(CONDITIONS):
        t = rng.standard_normal((n, NC)).astype(np.float32)
        t[:, j] += 1.5                                      # condition c favours its own target column
        T[c] = {d: t + tgt for d in DIRECTIONS}
    return cos, as_scores(T)


def test_dts_cf_is_condition_free_with_zero_gain_and_a_differing_term_raises(tmp_path, monkeypatch):
    cos, T = synth_terms()
    cf = D.cf_term(T)
    for d in DIRECTIONS:
        za = zscore_rows(torch.as_tensor(T["a"][d])).numpy().astype(np.float64)
        zb = zscore_rows(torch.as_tensor(T["b"][d])).numpy().astype(np.float64)
        assert np.array_equal(cf["a"][d], ((za + zb) * 0.5).astype(np.float32))
        assert np.array_equal(cf["a"][d], cf["b"][d])
        assert cf["a"][d].dtype == np.float32
    s = D.score_cf(cos, T, np.arange(N) % 2)
    assert (s.pa["gain"] == 0).all()
    s2 = D.score_cf(cos, T, np.arange(N) % 2, picks=s.picks)
    assert all(np.array_equal(s2.scores[c][d], s.scores[c][d]) for c in CONDITIONS for d in DIRECTIONS)

    def per_condition(T):                                   # the wiring slip: each condition its own z(T^c)
        return {c: {d: zscore_rows(torch.as_tensor(T[c][d])).numpy() for d in DIRECTIONS} for c in CONDITIONS}
    monkeypatch.setattr(D, "cf_term", per_condition)
    with pytest.raises(ValueError, match="condition-free"):
        D.score_cf(cos, T, np.arange(N) % 2)
    mut = mutant(tmp_path, "r6_dts.py", guard="cf")
    monkeypatch.setattr(mut, "cf_term", per_condition)
    assert not (mut.score_cf(cos, T, np.arange(N) % 2).pa["gain"] == 0).all()


def test_frozen_picks_reproduce_crossfit_lambda_on_the_tuning_seed():
    cos, T = synth_terms(seed=11)
    parity = np.arange(N) % 2
    s, picks = crossfit_lambda(cos, T, parity)
    fz = D.frozen_fused(cos, T, picks, parity)
    assert all(np.array_equal(fz[c][d], s[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    if picks[0] != picks[1]:                                # the other convention (h -> parity h) differs
        wrong = D.frozen_fused(cos, T, {0: picks[1], 1: picks[0]}, parity)
        assert not all(np.array_equal(wrong[c][d], s[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    assert D.picks_from_json(D.picks_to_json(picks)) == picks
    assert D.picks_to_json({0: float("inf"), 1: 64.0}) == {"0": "inf", "1": 64.0}
    assert D.picks_from_json({"0": "inf", "1": 0.25}) == {0: float("inf"), 1: 0.25}
    with pytest.raises(AssertionError):
        D.picks_from_json({"0": 3.0, "1": 0.25})


def failure_masks(n, seed=21, rate=0.1):
    rng = np.random.default_rng(seed)
    return {c: rng.random(n) < rate for c in CONDITIONS}


def test_failed_rows_take_the_lambda_zero_rows_at_every_lambda(tmp_path):
    cos, T = synth_terms(seed=13)
    failed = failure_masks(N)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            T[c][d][failed[c]] = 0                           # a parsing failure: T = 0 on that episode and condition
    zero = fused_scores(cos, T, 0.0)
    for lam in D.LAMBDAS:
        f, plain = D.fused_r6(cos, T, lam, failed), fused_scores(cos, T, lam)
        for c in CONDITIONS:
            for d in DIRECTIONS:
                assert np.array_equal(f[c][d][failed[c]], zero[c][d][failed[c]]), lam
                assert np.array_equal(f[c][d][~failed[c]], plain[c][d][~failed[c]]), lam
    inf = fused_scores(cos, T, float("inf"))
    assert (inf["a"]["i2t"][failed["a"]] == 0).all()                          # what crossfit_lambda would score
    mut = mutant(tmp_path, "r6_dts.py", guard="failed_rows")
    assert (mut.fused_r6(cos, T, float("inf"), failed)["a"]["i2t"][failed["a"]] == 0).all()


def test_the_r6_crossfit_equals_crossfit_lambda_bit_for_bit_when_nothing_failed():
    for seed in (14, 15):
        cos, T = synth_terms(seed=seed)
        parity = np.arange(N) % 2
        s0, p0 = crossfit_lambda(cos, T, parity)
        for failed in (None, {c: np.zeros(N, bool) for c in CONDITIONS}):
            s1, p1 = D.crossfit_lambda_r6(cos, T, parity, failed)
            assert p1 == p0 and all(np.array_equal(s1[c][d], s0[c][d]) and s1[c][d].dtype == s0[c][d].dtype
                                    for c in CONDITIONS for d in DIRECTIONS)
            fz = D.frozen_fused(cos, T, p0, parity, failed)
            assert all(np.array_equal(fz[c][d], s0[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    with pytest.raises(ValueError):
        D.crossfit_lambda_r6(cos, T, np.zeros(N, int))


def inf_world(n=N, seed=16):
    """Every row: cosine prefers a competitor by a wide margin, T prefers the target by a hair, so only lambda = inf
    (z(T) alone) ranks the target first; condition c's target is column c."""
    rng = np.random.default_rng(seed)
    cosr = rng.random((n, NC)).astype(np.float32) * 0.1
    cosr[:, 5] = 1.0
    cos = as_scores({c: {d: cosr for d in DIRECTIONS} for c in CONDITIONS})
    T = {}
    for j, c in enumerate(CONDITIONS):
        t = rng.random((n, NC)).astype(np.float32) * 0.01
        t[:, 5] = 0.999
        t[:, j] = 1.0
        T[c] = {d: t.copy() for d in DIRECTIONS}
    return cos, as_scores(T)


def test_lambda_inf_with_failures_scores_failed_rows_by_cosine():
    cos, T = inf_world()
    parity = np.arange(N) % 2
    failed = failure_masks(N, seed=22)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            T[c][d][failed[c]] = 0
    s_old, p_old = crossfit_lambda(cos, T, parity)
    s, picks = D.crossfit_lambda_r6(cos, T, parity, failed)
    assert picks == p_old == {0: float("inf"), 1: float("inf")}
    zc = fused_scores(cos, T, 0.0)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.array_equal(s[c][d][failed[c]], zc[c][d][failed[c]])
            assert (s_old[c][d][failed[c]] == 0).all()                         # the bug: an all-zero row, a miss
            assert np.array_equal(s[c][d][~failed[c]], s_old[c][d][~failed[c]])
    both = failed["a"] & failed["b"]
    pa, pc = per_anchor(s), per_anchor(cos)
    assert np.array_equal(pa["r1"][both], pc["r1"][both]) and both.any()      # both failed: cosine's own R@1
    fz = D.frozen_fused(cos, T, picks, parity, failed)
    assert all(np.array_equal(fz[c][d], s[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    # held path: the frozen inf picks fall back to cosine on failed rows too
    fz_inf = D.frozen_fused(cos, T, {0: float("inf"), 1: float("inf")}, parity, failed)
    assert all(np.array_equal(fz_inf[c][d][failed[c]], zc[c][d][failed[c]]) for c in CONDITIONS for d in DIRECTIONS)


def test_dts_cf_rows_where_both_conditions_failed_take_cosine():
    cos, T = inf_world(seed=17)
    parity = np.arange(N) % 2
    failed = failure_masks(N, seed=23, rate=0.3)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            T[c][d][failed[c]] = 0
    both, one = failed["a"] & failed["b"], failed["a"] ^ failed["b"]
    s = D.score_cf(cos, T, parity, picks={0: float("inf"), 1: float("inf")}, failed=failed)
    zc = fused_scores(cos, T, 0.0)
    plain = fused_scores(cos, D.cf_term(T), float("inf"))
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.array_equal(s.scores[c][d][both], zc[c][d][both])
            assert np.array_equal(s.scores[c][d][one], plain[c][d][one])        # one condition left: its z(T)
    assert (s.pa["gain"] == 0).all()
    s2 = D.score_cf(cos, T, parity, failed=failed)                              # cross-fitted, same masks
    assert all(np.array_equal(s2.scores[c][d][both], fused_scores(cos, T, 0.0)[c][d][both])
               for c in CONDITIONS for d in DIRECTIONS)


def test_frozen_picks_tie_tune_half_h_to_parity_1_minus_h():
    cos, T = synth_terms(seed=12, n=8)
    parity = np.arange(8) % 2
    fz = D.frozen_fused(cos, T, {0: 0.0, 1: float("inf")}, parity)
    z0, zinf = fused_scores(cos, T, 0.0), fused_scores(cos, T, float("inf"))
    assert np.array_equal(fz["a"]["i2t"][parity == 1], z0["a"]["i2t"][parity == 1])   # tuned on 0 -> parity 1
    assert np.array_equal(fz["a"]["i2t"][parity == 0], zinf["a"]["i2t"][parity == 0])


# ---------------------------------------------------------------- tuning order, the stop, the budget

def test_tuning_ties_go_to_the_first_setting_in_the_rule_order(tmp_path):
    order = D.SETTINGS_ORDER
    assert [D.setting_name(*s) for s in order] == ["W1 K8", "W1 K16", "W2 K8", "W2 K16", "W3 K8", "W3 K16",
                                                     "W4 K8", "W4 K16"]
    scores = dict.fromkeys(order, 100)
    assert D.choose_setting(scores) == ("W1", 8)
    scores[("W2", 16)] = scores[("W4", 8)] = 101
    assert D.choose_setting(scores) == ("W2", 16)
    scores[("W4", 16)] = 102
    assert D.choose_setting(scores) == ("W4", 16)
    mut = mutant(tmp_path, "r6_dts.py", replace=("if scores[s] > scores[best]:", "if scores[s] >= scores[best]:"))
    assert mut.choose_setting({**dict.fromkeys(order, 7), ("W2", 16): 9, ("W4", 8): 9}) == ("W4", 8)


def test_tune_score_orders_as_the_mean_of_r1_and_gain():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = {"r1": rng.integers(0, 5, 3072) / 4, "gain": rng.integers(-4, 5, 3072) / 4}
        b = {"r1": rng.integers(0, 5, 3072) / 4, "gain": rng.integers(-4, 5, 3072) / 4}
        crit = lambda m: 0.5 * (m["r1"].mean() + m["gain"].mean())  # noqa: E731
        assert (D.tune_score(a) > D.tune_score(b)) == (crit(a) > crit(b))
        assert D.tune_score(a) == int(round(8 * 3072 * crit(a)))


def r1_with_hits(h, n=N):
    """Per-episode R@1 (multiples of 0.25) summing to h / 4."""
    r1 = np.zeros(n)
    full, rest = divmod(h, 4)
    r1[:full] = 1.0
    r1[full] = rest / 4
    return r1


def test_the_stop_is_above_9406_hits_exactly_with_int64_sums(tmp_path):
    assert D.stop_decision(r1_with_hits(9406), N) == {"hits": 9406, "aff_hits": 9406, "n_rankings": 49152,
                                                       "dts_above_aff": False}
    assert D.stop_decision(r1_with_hits(9407), N)["dts_above_aff"] is True
    assert D.stop_decision(np.ones(N), N)["hits"] == 49152                  # far beyond int8 and int16
    assert isinstance(D.stop_decision(np.ones(N), N)["hits"], int)
    with pytest.raises(AssertionError, match="multiple of 0.25"):
        D.stop_decision(np.full(N, 0.3), N)
    with pytest.raises(AssertionError):
        D.stop_decision(np.ones(N - 1), N)
    mut = mutant(tmp_path, "r6_dts.py", replace=("\"dts_above_aff\": bool(hits > aff_hits)",
                                                  "\"dts_above_aff\": bool(hits >= aff_hits)"))
    assert mut.stop_decision(r1_with_hits(9406), N)["dts_above_aff"] is True       # the mutation is caught above
    mut8 = mutant(tmp_path, "r6_dts.py", replace=("as_int4(x, what).astype(np.int64).sum(dtype=np.int64)",
                                                   "as_int4(x, what).sum(dtype=np.int8)"))
    assert mut8.stop_decision(np.ones(N), N)["hits"] != 49152                      # int8 sums wrap


def test_budget_is_24_hours_of_elapsed_time_from_the_clock_start():
    b = D.budget("2026-10-09 12:29", "2026-10-10 12:29:00")
    assert b["within_budget"] is True and b["deadline"] == "2026-10-10 12:29:00"
    assert D.budget("2026-10-09 12:29", "2026-10-10 12:29:01")["within_budget"] is False
    assert D.budget("2026-10-09 12:29", "2026-10-09 18:00:00")["within_budget"] is True
    across = D.budget("2026-10-24 12:00", "2026-10-25 11:30:00")    # DST ends 25 Oct: 24.5 h elapsed
    assert across["within_budget"] is False and across["deadline"] == "2026-10-25 11:00:00"
    with pytest.raises(AssertionError, match="before the clock start"):
        D.budget("2026-10-09 12:29", "2026-10-09 12:00:00")
    with pytest.raises(ValueError):
        D.parse_amsterdam("2026-10-09T12:29+02:00")


# ---------------------------------------------------------------- the held scoring on synthetic data of the real shapes

@pytest.fixture(scope="module")
def synth_world():
    """Synthetic features of the real shape (NaN outside a synthetic row set), 12,288 episodes of held seed 52."""
    rng = np.random.default_rng(52)
    rows = np.sort(rng.choice(R.N_ROWS, 61_000, replace=False)).astype(np.int64)
    img = np.full((R.N_ROWS, 512), np.nan, np.float32)
    txt = np.full((R.N_ROWS, 512), np.nan, np.float32)
    img[rows] = rng.standard_normal((len(rows), 512), dtype=np.float32)
    txt[rows] = 0.5 * img[rows] + rng.standard_normal((len(rows), 512), dtype=np.float32)
    pooled = synth_pooled(rows, N_PER, seed=52)
    ctx = SimpleNamespace(seed=52, n=N, pooled=pooled, img=img, txt=txt, parity=np.arange(N) % 2,
                          pair_index=np.repeat(np.arange(3, dtype=np.int64), N_PER),
                          cos=cosine_scores(EvalInputs(img, txt), pooled))
    vocab = [f"value {i}" for i in range(60)]

    def answer(i, c, w):
        k = (i * 7 + (c == "b") * 3) % 23
        return "" if k == 0 else ("singleton" if k == 1 else f"Phrase {k}.\nignored")

    def listing(p, K):
        if p == "singleton":
            return "1. only"
        k = sum(map(ord, p)) % 40
        return "\n".join(f"{j + 1}. {vocab[(k + j) % 60]}" for j in range(K + 2))
    answers = {(52, i, c, "W3"): answer(i, c, "W3") for i in range(N) for c in CONDITIONS}
    phrases = sorted({D.phrase_of(a) for a in answers.values()} | set(D.TARGET_NAME.values()))
    listings = {(p, K): listing(p, K) for p in phrases if p for K in D.KS}
    record = {"setting": "W3 K16", "wording_id": "W3", "K": 16,
              "picks": {"dts": {"0": 1.0, "1": 0.5}, "dts_cf": {"0": 0.25, "1": "inf"}, "dts_n": {"0": 2.0, "1": 4.0}}}
    return SimpleNamespace(ctx=ctx, answers=answers, listings=listings, record=record, rows=rows)


def test_held_dts_scores_on_synthetic_held_shapes(synth_world, tmp_path):
    w = synth_world
    emb, enc = fake_embedder(tmp_path / "e.npz")
    out = D.held_dts_scores(w.ctx, w.answers, w.listings, w.record, emb)
    assert set(out) >= {"dts", "dts_cf", "dts_n", "failures", "below_K", "setting", "picks"}
    for s in D.SCORERS:
        assert set(out[s]) == set(D.METRICS) and all(out[s][m].shape == (N,) for m in D.METRICS)
        assert all(np.isfinite(out[s][m]).all() for m in D.METRICS)
    assert (out["dts_cf"]["gain"] == 0).all()
    n_empty = {c: sum(w.answers[(52, i, c, "W3")] == "" for i in range(N)) for c in CONDITIONS}
    n_single = {c: sum(w.answers[(52, i, c, "W3")] == "singleton" for i in range(N)) for c in CONDITIONS}
    assert out["failures"]["dts"] == {c: {"empty_phrase": n_empty[c], "short_listing": n_single[c]}
                                      for c in CONDITIONS}
    assert out["failures"]["dts_n"] == {c: {"empty_phrase": 0, "short_listing": 0} for c in CONDITIONS}
    assert enc.calls and all(len(texts) == 1 and bs == 1 for texts, bs in enc.calls)   # one string per call
    # the frozen picks: the same per-anchor arrays as fusing each parity with the other half's pick by hand
    t = D.dts_term(w.ctx, D.phrases_for(w.ctx, w.answers, "W3"), 16, w.listings, emb)
    by_hand = {c: {d: np.where((w.ctx.parity == 1)[:, None], fused_scores(w.ctx.cos, t.T, 1.0)[c][d],
                               fused_scores(w.ctx.cos, t.T, 0.5)[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    assert all(np.array_equal(out["dts"][m], per_anchor(by_hand)[m]) for m in D.METRICS)
    # frozen inf picks: the failed (episode, condition) rows are scored by cosine (rule section 7 item 2)
    rec_inf = dict(w.record, picks=dict(w.record["picks"], dts={"0": "inf", "1": "inf"}))
    got = D.held_dts_scores(w.ctx, w.answers, w.listings, rec_inf, emb)["dts"]
    inf, zc = fused_scores(w.ctx.cos, t.T, float("inf")), fused_scores(w.ctx.cos, t.T, 0.0)
    want = {c: {d: np.where(t.failed[c][:, None], zc[c][d], inf[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    assert all(t.failed[c].sum() == n_empty[c] + n_single[c] for c in CONDITIONS)
    assert all(np.array_equal(got[m], per_anchor(want)[m]) for m in D.METRICS)
    # a missing key refuses, as does a listing missing for a phrase the episodes need
    gone = dict(w.answers)
    del gone[(52, 4321, "b", "W3")]
    with pytest.raises(AssertionError, match=r"\(52, 4321, 'b', 'W3'\)"):
        D.held_dts_scores(w.ctx, gone, w.listings, w.record, emb)
    p = D.phrase_of(w.answers[(52, 5, "a", "W3")])
    short = {k: v for k, v in w.listings.items() if k != (p, 16)}
    with pytest.raises(AssertionError, match="no listing"):
        D.held_dts_scores(w.ctx, w.answers, short, w.record, emb)


def test_a_member_outside_the_rows_is_refused_on_held_shapes(synth_world, tmp_path):
    w = synth_world
    outside = np.setdiff1d(np.arange(200_000, 200_500), w.rows)[0]
    cand = w.ctx.pooled.candidates.copy()
    cand[17, 5] = outside
    p = w.ctx.pooled
    ctx = SimpleNamespace(**{**vars(w.ctx), "pooled": AspectEpisodes(p.aspect_a, p.aspect_b, p.anchor, cand,
                                                                      p.pairs_a_img, p.pairs_a_txt, p.pairs_b_img,
                                                                      p.pairs_b_txt)})
    emb, _ = fake_embedder()
    with pytest.raises(AssertionError, match="non-finite features"):
        D.held_dts_scores(ctx, w.answers, w.listings, w.record, emb)


# ---------------------------------------------------------------- runner stages on a synthetic smoke-sized world

BLOCK = {"emotion": 0, "style": 16, "genre": 32}              # the value words of each aspect name: dims 0 to 47


class OneHotEncoder(FakeEncoder):
    """'word i' -> the unit vector e_i (so a name's listing spans its aspect's block); anything else as FakeEncoder."""
    meta = {"class": "onehot", "batch_size": 1}

    def encode_texts(self, texts, batch_size=256):
        out = super().encode_texts(texts, batch_size)
        for j, t in enumerate(texts):
            m = re.fullmatch(r"word (\d+)", t)
            if m:
                out[j] = 0
                out[j, int(m.group(1))] = 1
        return out


def make_stage_world(tmp_path, sanity_listing_short=False):
    """A seed-9001-shaped context (192 episodes) whose anchors share one aspect block with the candidate of each
    condition, so DTS-N with the true names has a positive gain; its jobs and crafted GPU outputs."""
    rng = np.random.default_rng(9001)
    n_per = R.N_SMOKE
    n = 3 * n_per
    pair_index = np.repeat(np.arange(3, dtype=np.int64), n_per)
    n_rows = 14 * n + 100
    feats = 0.05 * rng.standard_normal((n_rows, 512))

    def vec(aspect):
        v = np.zeros(512)
        v[BLOCK[aspect]:BLOCK[aspect] + 8] = rng.random(8) + 0.25
        return v
    rows = rng.permutation(14 * n).reshape(n, 14)
    for e in range(n):
        a, b, _ = R.PAIRS[pair_index[e]]
        va, vb = vec(a), vec(b)
        feats[rows[e, 0]] += va + vb                                       # the anchor
        feats[rows[e, 1]] += va + vec(b)                                   # p_a: shares the anchor's a value
        feats[rows[e, 2]] += vec(a) + vb                                   # p_b: shares its b value
        for j in range(3, 14):
            feats[rows[e, j]] += vec(a) + vec(b)
    img = (feats + 0.02 * rng.standard_normal(feats.shape)).astype(np.float32)
    txt = (feats + 0.02 * rng.standard_normal(feats.shape)).astype(np.float32)
    ex = lambda: rng.integers(14 * n, n_rows, size=(n, NP)).astype(np.int64)  # noqa: E731
    pooled = AspectEpisodes("mixed", "mixed", rows[:, 0].astype(np.int64), rows[:, 1:].astype(np.int64), ex(), ex(),
                            ex(), ex())
    eps = SimpleNamespace(seed=SEED, n=n, pooled=pooled, pair_index=pair_index, parity=np.arange(n) % 2)
    ctx = SimpleNamespace(mode="selection", seed=SEED, n=n, n_total=n, eps=eps, pooled=pooled, parity=eps.parity,
                          pair_index=pair_index, img=img, txt=txt, cos=cosine_scores(EvalInputs(img, txt), pooled),
                          anchor_group=np.arange(n, dtype=np.int64), episodes_file_sha256="e" * 64)
    tune_pos = I.select_episodes(eps, RD.tune_per_pair(SEED))
    jt = write_job(tmp_path / "jobs/tune", pooled, SEED, tune_pos)
    jf = write_job(tmp_path / "jobs/full", pooled, SEED, np.arange(n))

    def answer(i, c, wd):
        k = (i * 5 + (c == "b") * 2 + int(wd[1])) % 17
        return "" if k == 0 else f"Quality {k}"
    vt = write_verbaliser_out(tmp_path / "out/vt", records_for(SEED, tune_pos, D.WORDINGS, answer), jt)
    vf = write_verbaliser_out(tmp_path / "out/vf", records_for(SEED, range(n), D.WORDINGS, answer), jf)
    lst = {}
    for p in sorted({D.phrase_of(answer(i, c, wd)) for i in range(n) for c in CONDITIONS for wd in D.WORDINGS}
                    | set(D.TARGET_NAME.values())):
        for K in D.KS:
            if p in D.TARGET_NAME:
                lst[(p, K)] = "just one" if sanity_listing_short else "\n".join(
                    f"{j + 1}. word {BLOCK[p] + j}" for j in range(K))
            elif p:
                k = sum(map(ord, p))
                lst[(p, K)] = "\n".join(f"- word {48 + (k + j) % 50}" for j in range(K))
    lo = write_listing_out(tmp_path / "out/l", lst)
    args = RD.parse_args(["--stage", "sanity", "--seed", str(SEED), "--out", str(tmp_path / "res"),
                          "--listing-out", str(lo), "--verbalise-out", str(vt), "--verbalise-job",
                          str(tmp_path / "jobs/tune")])
    full = dict(verbalise_out=[vf], verbalise_job=[tmp_path / "jobs/full"])
    return SimpleNamespace(ctx=ctx, args=args, full=full)


def run_stage(world, stage, **over):
    args = SimpleNamespace(**{**vars(world.args), "stage": stage, **over})
    fn = {"sanity": RD.stage_sanity, "tune": RD.stage_tune, "chosen": RD.stage_chosen}[stage]
    return fn(args, world.ctx, SETTINGS, SETTINGS_SHA)


@pytest.fixture
def fake_clip(monkeypatch):
    enc = OneHotEncoder()
    monkeypatch.setattr(D, "clip_cpu", lambda: enc)
    return enc


def test_stages_in_order_and_their_records(tmp_path, fake_clip, capsys):
    w = make_stage_world(tmp_path)
    with pytest.raises(RD.Refused, match="sanity has not passed"):
        run_stage(w, "tune")
    assert run_stage(w, "sanity") == 0
    assert run_stage(w, "tune") == 0
    assert run_stage(w, "chosen", **w.full) == 0
    res = tmp_path / "res"
    tune = json.loads((res / RD.TUNE).read_text())
    chosen = json.loads((res / RD.chosen_name(SEED)).read_text())
    assert tune["order"] == list(SETTINGS["seed42_order"]["tuning_settings"]) and tune["n_episodes"] == 48
    best = max(tune["order"], key=lambda s: tune["settings"][s]["score_int"])     # max keeps the first of ties
    assert tune["chosen"]["setting"] == best == chosen["setting"]
    for rec in (tune, chosen, json.loads((res / RD.SANITY).read_text())):
        assert rec["module_sha256"] == R.r6_module_shas() and rec["settings_sha256"] == SETTINGS_SHA
        assert re.fullmatch(r"[0-9a-f]{64}", rec["input_sha256"]["value_embeddings"]) and rec["time"]
    assert chosen["input_sha256"]["value_embeddings"] == G.sha256_file(res / RD.EMBEDDINGS)     # the cache grows
    assert set(chosen["picks"]) == set(D.SCORERS) and chosen["K"] == tune["chosen"]["K"]
    assert chosen["per_anchor_sha256"] == G.sha256_file(res / RD.per_anchor_name(SEED))
    assert chosen["failures"]["dts"]["a"]["empty_phrase"] > 0
    assert chosen["wording"] == SETTINGS["verbaliser"]["wordings"][chosen["wording_id"]]
    out = capsys.readouterr().out
    assert not DECIMAL.search(out) and all(line.startswith("dts ") for line in out.splitlines())
    # the held scoring, in selection mode on the same episodes, reproduces the chosen stage's per-anchor arrays
    ver = D.merge_verbaliser(w.full["verbalise_out"], D.WORDINGS, SETTINGS_SHA)
    held = D.held_dts_scores(w.ctx, ver.answers, D.merge_listings(w.args.listing_out, SETTINGS_SHA).answers, chosen,
                             D.ValueEmbedder(res / RD.EMBEDDINGS))
    with np.load(res / RD.per_anchor_name(SEED)) as z:
        for s in D.SCORERS:
            assert all(np.array_equal(held[s][m], z[f"{s}__{m}"]) for m in D.METRICS), s
    # outputs are never overwritten
    with pytest.raises(RD.Refused, match="never overwritten"):
        run_stage(w, "tune")


def test_sanity_fails_with_exit_3_when_the_ceiling_has_no_gain(tmp_path, fake_clip):
    w = make_stage_world(tmp_path, sanity_listing_short=True)       # T = 0: fused = cosine, gain 0
    assert run_stage(w, "sanity") == RD.EXIT_FAIL
    rec = json.loads((tmp_path / "res" / RD.SANITY).read_text())
    assert rec["passed"] is False and all(v["gain_int4_sum"] == 0 for v in rec["per_K"].values())
    assert rec["per_K"]["8"]["failures"]["a"]["short_listing"] == w.ctx.n
    with pytest.raises(RD.Refused, match="sanity has not passed"):
        run_stage(w, "tune")
    mut = mutant(tmp_path, "run_r6_dts.py", guard="sanity_exit")
    (tmp_path / "res" / RD.SANITY).unlink()
    assert mut.stage_sanity(w.args, w.ctx, SETTINGS, SETTINGS_SHA) == 0
    mut2 = mutant(tmp_path, "run_r6_dts.py", guard="sanity_first")
    mut2.stage_tune(SimpleNamespace(**{**vars(w.args), "stage": "tune"}), w.ctx, SETTINGS, SETTINGS_SHA)
    assert (tmp_path / "res" / RD.TUNE).is_file()


def test_the_frozen_pick_convention_is_checked_in_the_chosen_stage(tmp_path, fake_clip, monkeypatch):
    w = make_stage_world(tmp_path)
    assert run_stage(w, "sanity") == 0 and run_stage(w, "tune") == 0
    real = D.frozen_fused

    def wrong(cos, term, picks, parity, failed=None):       # a slip in the frozen assembly
        f = real(cos, term, picks, parity, failed)
        return {c: {d: -f[c][d] for d in DIRECTIONS} for c in CONDITIONS}
    monkeypatch.setattr(D, "frozen_fused", wrong)
    with pytest.raises(AssertionError, match="do not reproduce"):
        run_stage(w, "chosen", **w.full)
    mut = mutant(tmp_path, "run_r6_dts.py", guard="frozen_convention")
    mut.stage_chosen(SimpleNamespace(**{**vars(w.args), "stage": "chosen", **w.full}), w.ctx, SETTINGS, SETTINGS_SHA)


def test_overwrites_are_refused(tmp_path):
    p = tmp_path / "x.json"
    RD.write_new(p, {"a": 1})
    with pytest.raises(RD.Refused):
        RD.write_new(p, {"a": 2})
    mutant(tmp_path, "run_r6_dts.py", guard="no_overwrite").write_new(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}


# ---------------------------------------------------------------- the stop stage on crafted seed-42 records

def stop_world(tmp_path, hits, chosen_time="2026-10-10 09:00:00", modules=None):
    res = tmp_path / "res"
    res.mkdir(parents=True)
    mods = modules or R.r6_module_shas()
    np.savez(res / RD.per_anchor_name(42), dts__r1=r1_with_hits(hits))
    base = {"seed": 42, "module_sha256": mods, "settings_sha256": SETTINGS_SHA}
    G.write_json(res / RD.SANITY, {**base, "passed": True, "time": "2026-10-09 15:00:00"})
    G.write_json(res / RD.TUNE, {**base, "chosen": {"wording_id": "W1", "K": 8}, "time": "2026-10-09 20:00:00"})
    G.write_json(res / RD.chosen_name(42), {**base, "n_episodes": N, "setting": "W1 K8", "time": chosen_time,
                                            "per_anchor_file": RD.per_anchor_name(42),
                                            "per_anchor_sha256": G.sha256_file(res / RD.per_anchor_name(42))})
    return res


def stop_args(res, clock="2026-10-09 12:29"):
    return RD.parse_args(["--stage", "stop", "--out", str(res), "--clock-start", clock])


def test_stop_stage_9406_continues_and_9407_stops(tmp_path, capsys):
    res = stop_world(tmp_path / "a", 9406)
    assert RD.stage_stop(stop_args(res)) == 0
    rec = json.loads((res / RD.STOP).read_text())
    assert rec["hits"] == 9406 and rec["stop"] is False and rec["budget"]["within_budget"] is True
    res = stop_world(tmp_path / "b", 9407)
    assert RD.stage_stop(stop_args(res)) == RD.EXIT_FAIL
    rec = json.loads((res / RD.STOP).read_text())
    assert rec["stop"] is True and rec["reason"] == "DTS's hit count is above AFF's"
    out = re.sub(r"record \S+", "", capsys.readouterr().out)               # the paths hold the test's name
    assert not DECIMAL.search(out) and "9406" not in out and "9407" not in out
    mut = mutant(tmp_path, "run_r6_dts.py", replace=('stop = bool(not b["within_budget"] or dec["dts_above_aff"])',
                                                      'stop = bool(not b["within_budget"])'))
    res = stop_world(tmp_path / "c", 9407)
    assert mut.stage_stop(stop_args(res)) == 0                              # the mutation is caught above


def test_stop_stage_budget(tmp_path, monkeypatch):
    res = stop_world(tmp_path / "late", 100, chosen_time="2026-10-10 12:30:00")
    assert RD.stage_stop(stop_args(res)) == RD.EXIT_FAIL
    rec = json.loads((res / RD.STOP).read_text())
    assert rec["reason"] == "not built within the 24-hour budget" and rec["budget"]["within_budget"] is False
    assert rec["built"] is False and rec["stop"] is True
    late = stop_world(tmp_path / "late_above", 9407, chosen_time="2026-10-10 12:30:00")
    assert RD.stage_stop(stop_args(late)) == RD.EXIT_FAIL                  # both: the budget's reason comes first
    rec = json.loads((late / RD.STOP).read_text())
    assert rec["reason"] == "not built within the 24-hour budget" and rec["built"] is False
    ok = stop_world(tmp_path / "in_time", 100)
    assert RD.stage_stop(stop_args(ok)) == 0 and json.loads((ok / RD.STOP).read_text())["built"] is True
    empty = tmp_path / "empty"
    empty.mkdir()
    now = R.amsterdam_now()[:16]
    monkeypatch.setattr(RD, "DTS_CLOCK_START", now)                         # seed 42's clock start is pinned
    with pytest.raises(RD.Refused, match="not built yet"):                 # before the deadline: not evaluable
        RD.stage_stop(stop_args(empty, clock=now))
    monkeypatch.setattr(RD, "DTS_CLOCK_START", "2026-01-01 00:00")
    assert RD.stage_stop(stop_args(empty, clock="2026-01-01 00:00")) == RD.EXIT_FAIL
    assert json.loads((empty / RD.STOP).read_text())["built"] is False
    with pytest.raises(RD.Refused, match="clock-start"):
        RD.stage_stop(RD.parse_args(["--stage", "stop", "--out", str(res)]))


def first_built_world(tmp_path, chosen_time="2026-10-10 09:00:00"):
    """A seed-42 build in time whose chosen stage recorded dts_first_built.json (record_first_built, as the chosen
    stage calls it), with GPU output fingerprints in the sanity and chosen records."""
    res = stop_world(tmp_path, 100, chosen_time=chosen_time)
    fps = {"verbaliser": {"v": {"job": "r6_gpu_verbalise", "inputs_sha256": {"a": "1" * 64}}},
           "listing": {"l": {"job": "r6_gpu_listing"}}}
    for name in (RD.SANITY, RD.chosen_name(42)):
        rec = json.loads((res / name).read_text())
        rec["gpu_fingerprints"] = fps
        G.write_json(res / name, rec)
    chosen = json.loads((res / RD.chosen_name(42)).read_text())
    return res, chosen


def rerun_chosen(res, **changes):
    """The chosen record rewritten as a rerun after a module change writes it: a later time, the rest unchanged unless
    ``changes`` say otherwise."""
    rec = json.loads((res / RD.chosen_name(42)).read_text())
    rec.update({"time": "2026-10-12 09:00:00", **changes})
    G.write_json(res / RD.chosen_name(42), rec)


def test_a_rerun_past_the_deadline_keeps_the_first_build_time(tmp_path):
    """Rule section 7 items 5 and 7 (final review C): "built" is a one-time event; a rerun forced by a module change
    (rule section 6 item 7) re-evaluates the stop from the first build's time when the setting, the settings SHA-256
    and the GPU outputs are unchanged, and from its own time otherwise."""
    res, chosen = first_built_world(tmp_path / "a")
    assert RD.record_first_built(res, 42, chosen) == "written"
    assert RD.record_first_built(res, 42, chosen) == "kept"                  # once, never overwritten
    assert RD.record_first_built(res, 9001, chosen) is None                  # seed 42 only
    fb = json.loads((res / RD.FIRST_BUILT).read_text())
    assert fb["time"] == "2026-10-10 09:00:00" and fb["setting"] == "W1 K8" and fb["clock_start"] == R.DTS_CLOCK_START
    rerun_chosen(res)
    assert RD.stage_stop(stop_args(res)) == 0
    rec = json.loads((res / RD.STOP).read_text())
    assert rec["built"] is True and rec["stop"] is False and rec["budget_time_source"] == RD.FIRST_BUILT
    assert rec["budget"]["built_time"] == "2026-10-10 09:00:00"
    assert rec["first_built_sha256"] == G.sha256_file(res / RD.FIRST_BUILT)
    for name, change in (("setting", {"setting": "W2 K8"}), ("settings", {"settings_sha256": "0" * 64}),
                         ("outputs", {"gpu_fingerprints": {"verbaliser": {}, "listing": {}}})):
        res2, chosen2 = first_built_world(tmp_path / name)
        RD.record_first_built(res2, 42, chosen2)
        rerun_chosen(res2, **change)
        if name == "setting":
            tune = json.loads((res2 / RD.TUNE).read_text())
            tune["chosen"] = {"wording_id": "W2", "K": 8}
            G.write_json(res2 / RD.TUNE, tune)
        assert RD.stage_stop(stop_args(res2)) == RD.EXIT_FAIL, name
        rec = json.loads((res2 / RD.STOP).read_text())
        assert rec["reason"] == "not built within the 24-hour budget", name
        assert rec["budget_time_source"] == "this run's sanity and chosen records", name
    res3, chosen3 = first_built_world(tmp_path / "mut")
    RD.record_first_built(res3, 42, chosen3)
    rerun_chosen(res3, gpu_fingerprints={"verbaliser": {}, "listing": {}})
    mut = mutant(tmp_path, "run_r6_dts.py", guard="first_built_match")
    assert mut.stage_stop(stop_args(res3)) == 0                              # without the match: other outputs pass


def test_no_first_build_record_outside_the_budget(tmp_path):
    res, chosen = first_built_world(tmp_path / "late", chosen_time="2026-10-10 12:30:00")
    assert RD.record_first_built(res, 42, chosen) is None and not (res / RD.FIRST_BUILT).exists()


def test_the_seed42_stop_requires_the_pinned_clock_start(tmp_path):
    res = stop_world(tmp_path / "a", 100)
    with pytest.raises(RD.Refused, match="not the first DTS commit's"):
        RD.stage_stop(stop_args(res, clock="2026-10-09 18:00"))
    assert not (res / RD.STOP).exists()
    assert RD.DTS_CLOCK_START == R.DTS_CLOCK_START == "2026-10-09 12:29"
    assert mutant(tmp_path, "run_r6_dts.py", guard="clock_start").stage_stop(stop_args(res, clock="2026-10-09 "
                                                                                             "18:00")) == 0


def test_stop_refuses_records_of_other_dts_code_and_existing_outputs(tmp_path):
    mods = dict(R.r6_module_shas())
    mods["src/test/20261125_artelingo_held_test/r6_dts.py"] = "0" * 64
    res = stop_world(tmp_path / "a", 9000, modules=mods)
    with pytest.raises(RD.Refused, match="other DTS code bytes"):
        RD.stage_stop(stop_args(res))
    assert mutant(tmp_path, "run_r6_dts.py", guard="same_modules").stage_stop(stop_args(res)) == 0
    with pytest.raises(RD.Refused, match="never overwritten"):
        RD.stage_stop(stop_args(res))


def test_stop_refuses_another_setting_than_the_tunings(tmp_path):
    res = stop_world(tmp_path / "a", 9000)
    tune = json.loads((res / RD.TUNE).read_text())
    tune["chosen"] = {"wording_id": "W2", "K": 16}
    (res / RD.TUNE).write_text(json.dumps(tune))
    with pytest.raises(AssertionError, match="another setting"):
        RD.stage_stop(stop_args(res))
    assert mutant(tmp_path, "run_r6_dts.py", guard="stop_setting").stage_stop(stop_args(res)) == 0


def test_stop_refuses_a_per_anchor_file_other_than_the_chosen_stages(tmp_path):
    res = stop_world(tmp_path / "a", 9000)
    np.savez(res / RD.per_anchor_name(42), dts__r1=r1_with_hits(9407))        # replaced after the chosen stage
    with pytest.raises(AssertionError, match="not the chosen stage's file"):
        RD.stage_stop(stop_args(res))
    assert mutant(tmp_path, "run_r6_dts.py", guard="npz_sha").stage_stop(stop_args(res)) == RD.EXIT_FAIL


def test_chosen_refuses_dts_n_picks_other_than_the_sanitys(tmp_path, fake_clip):
    w = make_stage_world(tmp_path)
    assert run_stage(w, "sanity") == 0 and run_stage(w, "tune") == 0
    path = tmp_path / "res" / RD.SANITY
    rec = json.loads(path.read_text())
    for k in rec["per_K"]:
        rec["per_K"][k]["picks"] = {"0": 64.0, "1": 64.0} if rec["per_K"][k]["picks"] != {"0": 64.0, "1": 64.0} \
            else {"0": 0.0, "1": 0.0}
    path.write_text(json.dumps(rec))
    with pytest.raises(AssertionError, match="DTS-N's picks differ"):
        run_stage(w, "chosen", **w.full)
    mut = mutant(tmp_path, "run_r6_dts.py", guard="names_picks")
    assert mut.stage_chosen(SimpleNamespace(**{**vars(w.args), "stage": "chosen", **w.full}), w.ctx, SETTINGS,
                            SETTINGS_SHA) == 0


# ---------------------------------------------------------------- listing input

def test_listing_items_are_distinct_sorted_and_skip_empty_phrases():
    items = D.listing_items(["awe", "", "joy", "awe", "emotion"], [16, 8])
    assert items == [("awe", 8), ("emotion", 8), ("joy", 8), ("awe", 16), ("emotion", 16), ("joy", 16)]
    with pytest.raises(AssertionError, match="not normalised"):
        D.listing_items(["Awe"], [8])


def test_list_input_for_sanity_writes_a_job_the_listing_job_reads(tmp_path):
    job = tmp_path / "jobs/list_sanity"
    assert RD.main(["--stage", "list-input", "--for", "sanity", "--job-out", str(job), "--out",
                    str(tmp_path / "res")]) == 0
    assert sorted(p.name for p in job.iterdir()) == [RD.JOB_RECORD, RD.LISTING_INPUT]
    items = L.load_listing_input(job / RD.LISTING_INPUT, SETTINGS)
    assert items == [(p, K) for K in D.KS for p in ("emotion", "genre", "style")]
    rec = json.loads((job / RD.JOB_RECORD).read_text())
    assert rec["listing_input_sha256"] == G.sha256_file(job / RD.LISTING_INPUT) and rec["n_items"] == 6
    assert RD.main(["--stage", "list-input", "--for", "sanity", "--job-out", str(job)]) == RD.EXIT_REFUSED


# ---------------------------------------------------------------- end to end on the real smoke episodes of seed 9001

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    import src.eval.aspect_episodes as AE
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    names = R.value_names(data)
    del data
    index = AE.PaintingValueIndex(labels, split.groups)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    eps = E.build_seed(labels, split.groups, split.selection, index, vs, SEED, R.N_SMOKE)
    assert eps.sha == E.identity_targets()[SEED]["episodes_sha256"]          # AB's recorded smoke episodes
    base = tmp_path_factory.mktemp("real")
    path = E.save_episodes(base / f"episodes_seed{SEED}.npz", eps)
    return SimpleNamespace(eps=eps, path=path, base=base, names=names)


def craft_answer(i, c, w):
    k = (i * 31 + (c == "b") * 7 + int(w[1]) * 3) % 11
    return ["The shared emotion of awe", "1. Impressionist style\nsecond line", "", "   \n\n ",
            "“Melancholy.”", "zzz unlistable", "Bright   COLOUR palette", "Portrait genre!",
            "religious painting", "calm mood", "landscape"][k]


def craft_listing(p, K, names):
    if p == "zzz unlistable":
        return "1. just one value"
    if p in D.TARGET_NAME:
        vals = names[p]
    else:
        pool = names["emotion"] + names["style"] + names["genre"]
        k = sum(map(ord, p)) % len(pool)
        vals = pool[k:] + pool[:k]
    lines = [f"{j + 1}. {v.replace('_', ' ')}" for j, v in enumerate(vals[:K])]
    return "\n".join(lines[:1] + [lines[0], ""] + lines[1:])          # a repeated and a blank line


def run_cli(*argv):
    p = subprocess.run([PY, str(HERE / "run_r6_dts.py"), *map(str, argv)], env=ENV, capture_output=True, text=True,
                       timeout=900)
    return p


def craft_listing_out(d, job, names):
    items = L.load_listing_input(job / RD.LISTING_INPUT, SETTINGS)
    return write_listing_out(d, {(p, K): craft_listing(p, K, names) for p, K in items})


def test_end_to_end_on_seed_9001_with_crafted_gpu_outputs(real, tmp_path):
    base, eps = tmp_path, real.eps
    res, emb = base / "res", base / "res" / "emb.npz"
    common = ["--seed", SEED, "--episodes", real.path, "--out", res, "--embeddings", emb]
    logs = []

    def ok(p, code=0):
        logs.append((p.stdout, p.stderr))
        assert p.returncode == code, p.stdout + p.stderr[-3000:]
        return p

    # 1, 2: DTS-N's listing and the sanity
    ok(run_cli("--stage", "list-input", "--for", "sanity", "--job-out", base / "jobs/list_sanity", *common))
    l_s = craft_listing_out(base / "out/list_sanity", base / "jobs/list_sanity", real.names)
    ok(run_cli("--stage", "sanity", "--listing-out", l_s, *common))
    # 3, 4: the tuning subset's verbaliser outputs (two shards, one key in both), its listing, the tuning
    pos = I.select_episodes(eps, RD.tune_per_pair(SEED))
    jt = base / "jobs/verb_tune"
    jt.parent.mkdir(exist_ok=True)
    I_arrays = I.verbalise_arrays(eps, pos)
    jt.mkdir()
    np.savez(jt / "verbalise_input.npz", **I_arrays)
    np.savez(jt / "rows_manifest.npz", rows=np.unique(np.concatenate([I_arrays[f].ravel() for f in G.PAIR_FIELDS])))
    shas = {f: G.sha256_file(jt / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}
    recs = records_for(SEED, pos, D.WORDINGS, craft_answer)
    half = len(recs) // 2
    v1 = write_verbaliser_out(base / "out/vt1", recs[:half + 1], shas)
    v2 = write_verbaliser_out(base / "out/vt2", recs[half:], shas)
    vt = ["--verbalise-out", v1, "--verbalise-out", v2, "--verbalise-job", jt]
    ok(run_cli("--stage", "list-input", "--for", "tune", "--job-out", base / "jobs/list_tune", *vt, *common))
    l_t = craft_listing_out(base / "out/list_tune", base / "jobs/list_tune", real.names)
    ok(run_cli("--stage", "tune", *vt, "--listing-out", l_t, *common))
    chosen = json.loads((res / RD.TUNE).read_text())["chosen"]
    # 5, 6: the chosen wording on all 192 episodes, its listing, the chosen stage
    jf = base / "jobs/verb_full"
    full = I.verbalise_arrays(eps, np.arange(eps.n))
    jf.mkdir()
    np.savez(jf / "verbalise_input.npz", **full)
    np.savez(jf / "rows_manifest.npz", rows=np.unique(np.concatenate([full[f].ravel() for f in G.PAIR_FIELDS])))
    shas_f = {f: G.sha256_file(jf / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}
    vf = write_verbaliser_out(base / "out/vf", records_for(SEED, range(eps.n), [chosen["wording_id"]], craft_answer),
                              shas_f)
    vfa = ["--verbalise-out", vf, "--verbalise-job", jf]
    ok(run_cli("--stage", "list-input", "--for", "chosen", "--job-out", base / "jobs/list_chosen", *vfa, *common))
    l_c = craft_listing_out(base / "out/list_chosen", base / "jobs/list_chosen", real.names)
    ok(run_cli("--stage", "chosen", *vfa, "--listing-out", l_s, "--listing-out", l_c, *common))
    # 7: the stop
    start = (D.parse_amsterdam(R.amsterdam_now()) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    ok(run_cli("--stage", "stop", "--seed", SEED, "--out", res, "--clock-start", start))
    for f in (RD.SANITY, RD.TUNE, RD.chosen_name(SEED), RD.per_anchor_name(SEED), RD.STOP, "emb.npz"):
        assert (res / f).is_file(), f
    for j in ("list_sanity", "list_tune", "list_chosen"):
        assert (base / "jobs" / j / RD.LISTING_INPUT).is_file()
    for out, err in logs:                                     # stdout: pass/fail lines and paths only; no decimal
        assert all(line.startswith("dts ") for line in out.splitlines()), out
        assert not DECIMAL.search(re.sub(r"(record|folder|arrays) \S+", "", out + err)), out + err
    rec = json.loads((res / RD.chosen_name(SEED)).read_text())
    assert rec["setting"] == chosen["setting"] and rec["n_episodes"] == 192
    assert rec["failures"]["dts"]["a"]["empty_phrase"] + rec["failures"]["dts"]["b"]["empty_phrase"] > 0
    assert rec["failures"]["dts"]["a"]["short_listing"] + rec["failures"]["dts"]["b"]["short_listing"] > 0
    assert rec["below_K"]["dts_n"]["a"] > 0 or rec["K"] == 8                # emotion has 8 names: below 16
    stop = json.loads((res / RD.STOP).read_text())
    with np.load(res / RD.per_anchor_name(SEED)) as z:
        assert stop["hits"] == int((4 * z["dts__r1"]).sum()) and stop["stop"] is False
    # the held scoring, as the same code in selection mode, reproduces the chosen stage's arrays
    ctx = RD.load_context(SEED, real.path)
    ver = D.merge_verbaliser([vf], D.WORDINGS, SETTINGS_SHA, seeds={SEED})
    held = D.held_dts_scores(ctx, ver.answers, D.merge_listings([l_s, l_c], SETTINGS_SHA).answers, rec,
                             D.ValueEmbedder(emb))
    with np.load(res / RD.per_anchor_name(SEED)) as z:
        for s in D.SCORERS:
            assert all(np.array_equal(held[s][m], z[f"{s}__{m}"]) for m in D.METRICS), s
    assert held["failures"]["dts"] == rec["failures"]["dts"]


def test_episodes_other_than_the_recorded_ones_are_refused(real, tmp_path):
    with pytest.raises(AssertionError, match="not the recorded episodes"):
        RD.load_episodes_checked(9002, real.path)
    mutant(tmp_path, "run_r6_dts.py", guard="episodes_identity").load_episodes_checked(9002, real.path)


def test_list_input_for_a_held_seed_lists_the_chosen_wordings_phrases_and_the_names(tmp_path, synth_world):
    """Synthetic episodes of held seed 52 at the real shape (12,288), saved as the held runner saves them."""
    w = synth_world
    p = w.ctx.pooled
    per_pair = []
    for k, (a, b, _) in enumerate(R.PAIRS):
        sl = slice(k * N_PER, (k + 1) * N_PER)
        per_pair.append(AspectEpisodes(a, b, *(np.ascontiguousarray(getattr(p, f)[sl]) for f in E.FIELDS)))
    eps = SimpleNamespace(seed=52, per_pair=per_pair, pooled=p, n=N, pair_index=w.ctx.pair_index,
                          sha={name: E.episodes_sha256(ep) for name, ep in zip(R.PAIR_NAMES, per_pair)})
    path = E.save_episodes(tmp_path / "held_episodes_seed52.npz", eps)
    shas = write_job(tmp_path / "jobs/v52", p, 52, np.arange(N))
    recs = [{"seed": 52, "episode_index": i, "condition": c, "wording": "W3", "answer": a}
            for (s, i, c, wd), a in w.answers.items()]
    out = write_verbaliser_out(tmp_path / "out/v52", recs, shas)
    res = tmp_path / "res"
    G.write_json(res / RD.chosen_name(42), {"seed": 42, "wording_id": "W3", "K": 16})
    job = tmp_path / "jobs/list_held52"
    argv = ["--stage", "list-input", "--for", "held", "--seed", "52", "--episodes", str(path), "--verbalise-out",
            str(out), "--verbalise-job", str(tmp_path / "jobs/v52"), "--job-out", str(job), "--out", str(res)]
    assert RD.main(argv) == 0
    items = L.load_listing_input(job / RD.LISTING_INPUT, SETTINGS)
    want = {D.phrase_of(a) for a in w.answers.values()} - {""} | set(D.TARGET_NAME.values())
    assert items == sorted((q, 16) for q in want)
    with pytest.raises(SystemExit):                                         # a held seed only lists phrases
        RD.parse_args(["--stage", "sanity", "--seed", "52"])
    with pytest.raises(SystemExit):
        RD.parse_args(["--stage", "list-input", "--for", "held", "--seed", "42"])
