"""Tests of the descriptive pass's external rows (r6_external.py and run_r6_descriptive.py's two hooks; ticket 14; rule
section 10 item 5, section 8 item 3, section 7 items 2 to 4): DTS, DTS-CF, DTS-N, FT-LP, FT-LB, FT-LoRA and MLLM.

Worlds (no GPU, no DAS6, no held row): ticket 13's results folder (test_r6_descriptive.build_dir: synthetic episodes,
bundles, pass, verdict) with synthetic CLIP features of the real shape (308,723 x 512 float32, NaN outside the member
rows) and crafted GPU outputs in the jobs' real formats (verbaliser phrases_<W>.jsonl, listings.jsonl, provenance
fingerprints, rows_manifest.npz, ft_rows.npz, features_<variant>.npz, rerank_input.npz, scores.npz, <job>.perms.npz,
job_record.json): the smoke mode (seeds 9001 to 9003, 64 per pair) end to end through run_r6_descriptive.run, and the
held path on the real shapes (seed 52, 12,288 episodes; members drawn from a 61,744-row set). The FT-LP reproduction
reads the clipft LP run's checkpoint and stored features (val and selection rows only) and the cached features of the
same rows. Guards are removed on copies in tmp_path, never in place.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_external.py
"""
import ast
import hashlib
import importlib.util
import inspect
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_descriptive as D  # noqa: E402
import r6_dts as DT  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_external as XT  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_ft_features as FF  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import run_r6_dts as RDTS  # noqa: E402
import test_r6_descriptive as TD  # noqa: E402  (ticket 13's results-folder builder)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402
from src.eval.mllm_reranker import INSTRUCTION, unpermute  # noqa: E402

SETTINGS, SETTINGS_SHA = DT.load_settings()
W, K = "W2", 8
PICKS = {"dts": {"0": 1.0, "1": 0.5}, "dts_cf": {"0": 0.25, "1": "inf"}, "dts_n": {"0": 2.0, "1": 4.0}}
DECIMAL = re.compile(r"\d*\.\d+")
VOCAB = [f"value {i}" for i in range(60)]
N_HELD_LIKE = 61_744
_COUNT = itertools.count()


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, guard=None, replace=None):
    """A copy of r6_external.py with its `# guard:<guard>` statements replaced by `pass` (or one text replaced)."""
    src = (HERE / "r6_external.py").read_text()
    if guard is not None:
        lines = src.splitlines(keepends=True)
        hits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Expr, ast.Assign))
                and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, guard
        for n in hits:
            first = lines[n.lineno - 1]
            lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
        src = "".join(lines)
    if replace is not None:
        old, new = replace
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = tmp_path / f"r6_external_mut{next(_COUNT)}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    mod.encoder_factory = FakeEncoder
    return mod


class FakeEncoder:
    """Deterministic unit vectors per string (the CLIP text encoder's stand-in)."""
    meta = {"class": "fake", "batch_size": 1}

    def encode_texts(self, texts, batch_size=256):
        out = []
        for t in texts:
            v = np.random.default_rng(int(R.sha256_bytes(t.encode())[:12], 16)).standard_normal(512)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_clip(monkeypatch):
    monkeypatch.setattr(XT, "encoder_factory", FakeEncoder)


def write_json(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1))


def edit_json(path, fn):
    rec = json.loads(Path(path).read_text())
    fn(rec)
    write_json(path, rec)


def shas(folder, files) -> dict:
    return {f: G.sha256_file(Path(folder) / f) for f in files}


def write_manifest(path, rows):
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    np.savez_compressed(path, rows=rows, image_name=np.asarray([f"{int(r):020x}.jpg" for r in rows], dtype=str),
                        caption=np.asarray([f"caption {r}" for r in rows], dtype=str))


def answer(seed, i, c):
    """A crafted verbaliser answer: some empty (a parsing failure), some whose listing is too short."""
    k = (i * 7 + (c == "b") * 3 + seed) % 23
    return "" if k == 0 else ("Singleton" if k == 1 else f"Phrase {k}.\nignored line")


def listing(p, k):
    if p == "singleton":
        return "1. only"
    j0 = sum(map(ord, p)) % 40
    return "\n".join(f"{j + 1}. {VOCAB[(j0 + j) % 60]}" for j in range(k + 2))


def failures_of(seed, n):
    """{condition: {"empty_phrase", "short_listing"}} of the crafted answers of a seed."""
    out = {c: {"empty_phrase": 0, "short_listing": 0} for c in CONDITIONS}
    for i in range(n):
        for c in CONDITIONS:
            a = answer(seed, i, c)
            if a == "":
                out[c]["empty_phrase"] += 1
            elif a == "Singleton":
                out[c]["short_listing"] += 1
    return out


def write_record(d, seed, n_episodes):
    """A chosen-stage record (run_r6_dts.stage_chosen's fields the descriptive pass reads) and its passed stop."""
    rec = {"stage": "chosen", "seed": seed, "rule_sha256": R.RULE_SHA256, "settings_sha256": SETTINGS_SHA,
           "setting": f"{W} K{K}", "wording_id": W, "K": K, "n_episodes": n_episodes, "picks": PICKS,
           "embedder": dict(FakeEncoder.meta)}
    path = Path(d) / RDTS.chosen_name(seed)
    write_json(path, rec)
    write_json(Path(d) / RDTS.STOP, {"stage": "stop", "seed": seed, "built": True, "stop": False,
                                     "setting": rec["setting"], "input_sha256": {path.name: R.sha256_file(path)}})
    return path


def write_vjob(d, eps) -> dict:
    """A verbaliser job folder of all the seed's episodes (r6_gpu_inputs' arrays)."""
    d.mkdir(parents=True)
    arrays = I.verbalise_arrays(eps, np.arange(eps.n, dtype=np.int64))
    np.savez(d / "verbalise_input.npz", **arrays)
    write_manifest(d / "rows_manifest.npz", np.concatenate([arrays[f].ravel() for f in G.PAIR_FIELDS]))
    return shas(d, ("rows_manifest.npz", "verbalise_input.npz"))


def vrecords(seed, n, fn=answer):
    return [{"seed": seed, "episode_index": i, "condition": c, "wording": W, "answer": fn(seed, i, c)}
            for i in range(n) for c in CONDITIONS]


def write_vout(d, job_shas, recs, settings_sha=SETTINGS_SHA):
    d.mkdir(parents=True)
    st = G.KeyedJsonl(d / f"phrases_{W}.jsonl", DT.VERBALISE_FIELDS, DT.VERBALISE_FIELDS[:4])
    for r in recs:
        st.add(dict(r))
    st.save()
    G.write_json(d / "provenance.json", {"fingerprint": {"job": "r6_gpu_verbalise", "settings_sha256": settings_sha,
                                                         "inputs_sha256": job_shas}, "runs": []})
    return d


def write_lout(d, phrases):
    d.mkdir(parents=True)
    st = G.KeyedJsonl(d / "listings.jsonl", DT.LISTING_FIELDS, DT.LISTING_FIELDS[:2])
    for p in sorted(x for x in phrases if x):
        for k in DT.KS:
            st.add({"phrase": p, "K": int(k), "answer": listing(p, k)})
    st.save()
    G.write_json(d / "provenance.json", {"fingerprint": {"job": "r6_gpu_listing", "settings_sha256": SETTINGS_SHA},
                                         "runs": []})
    return d


def write_ft_job(d, rows) -> dict:
    d.mkdir(parents=True)
    write_manifest(d / "rows_manifest.npz", rows)
    np.savez(d / "ft_rows.npz", rows=np.asarray(rows, dtype=np.int64))
    return shas(d, ("rows_manifest.npz", "ft_rows.npz"))


def write_ft_out(d, job_shas, feats, ckpt=None):
    """features_<variant>.npz files and provenance.json as r6_gpu_ft_features writes them."""
    d.mkdir(parents=True)
    done = {}
    for v, (rows, img, txt) in feats.items():
        with open(d / f"features_{v}.npz", "wb") as f:
            np.savez(f, rows=rows, img=img, txt=txt)
        done[v] = G.sha256_file(d / f"features_{v}.npz")
    fp = {"job": "r6_gpu_ft_features", "variants": list(feats), "selected": FF.SELECTED,
          "ckpt_sha256": ckpt or {v: XT.FT_CKPTS[v]["sha256"] for v in XT.FT_VARIANTS},
          "scripts_sha256": {"x.py": "0" * 64}, "inputs_sha256": job_shas, "image_source": "r6_ft_cache"}
    G.write_json(d / "provenance.json", {"fingerprint": fp, "runs": [{"status": "complete",
                                                                       "features_sha256": done}]})
    return d


def ft_features(rows, rng):
    img = rng.standard_normal((len(rows), 512), dtype=np.float32)
    return rows, img, (0.5 * img + rng.standard_normal((len(rows), 512), dtype=np.float32)).astype(np.float32)


def write_rr_job(d, eps) -> SimpleNamespace:
    """A reranker job folder (r6_gpu_inputs.rerank_arrays) with job_record.json and the CPU-only permutations."""
    d.mkdir(parents=True)
    pos = np.arange(eps.n, dtype=np.int64)
    arrays, perms = I.rerank_arrays(eps, pos)
    write_manifest(d / "rows_manifest.npz", np.concatenate([arrays[k].ravel() for k in ("query_row", "cand_shown",
                                                                                         *G.PAIR_FIELDS)]))
    np.savez(d / "rerank_input.npz", **arrays)
    js = shas(d, ("rows_manifest.npz", "rerank_input.npz"))
    G.write_json(d / "job_record.json", {"seed": int(eps.seed), "n_episodes": int(eps.n), "files_sha256": js,
                                         "perms_sha256": G.sha256_bytes(np.ascontiguousarray(perms).tobytes())})
    with open(I.perms_path(d), "wb") as f:
        np.savez(f, seed=np.int64(eps.seed), episode_index=pos, perms=perms)
    return SimpleNamespace(path=d, shas=js, perms=perms, arrays=arrays)


RR_MODEL = {"model_id": SETTINGS["model"]["id"], "snapshot": SETTINGS["model"]["snapshot"],
            "max_pixels": 256 * 28 * 28, "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()}


def write_rr_out(d, job_shas, index, shown, **model):
    """scores.npz and provenance.json as r6_gpu_rerank writes them (its fingerprint's fields)."""
    d.mkdir(parents=True)
    with open(d / "scores.npz", "wb") as f:
        np.savez(f, episode_index=np.asarray(index, dtype=np.int64), scores_shown=np.asarray(shown, np.float32))
    fp = {"job": "r6_gpu_rerank", **RR_MODEL, **model, "scripts_sha256": {"x.py": "0" * 64},
          "inputs_sha256": job_shas}
    G.write_json(d / "provenance.json", {"fingerprint": fp, "runs": []})
    return d


def planted(perms, rng):
    """scores_shown in which, for every (episode, condition, direction), the letter showing the target column (p_a
    under a, p_b under b) holds the highest score."""
    shown = rng.standard_normal(perms.shape).astype(np.float32)
    for ci, c in enumerate(XT.RERANK_CONDITIONS):
        target = 0 if c == "a" else 1
        j = np.argmax(perms[:, ci] == target, axis=-1)                          # (n, 2): the letter of the target
        for di in range(2):
            shown[np.arange(len(perms)), ci, di, j[:, di]] = 10.0
    return shown


def feature_world(rows, rng):
    """CLIP-like features of the real shape: finite on ``rows``, NaN elsewhere (as a RowContext's masked img, txt)."""
    img = np.full((R.N_ROWS, 512), np.nan, np.float32)
    txt = np.full((R.N_ROWS, 512), np.nan, np.float32)
    img[rows] = rng.standard_normal((len(rows), 512), dtype=np.float32)
    txt[rows] = 0.5 * img[rows] + rng.standard_normal((len(rows), 512), dtype=np.float32)
    return img, txt


def make_ctx(eps, img, txt, ev):
    return SimpleNamespace(seed=eps.seed, n=eps.n, pooled=eps.pooled, img=img, txt=txt,
                           cos=cosine_scores(ev, eps.pooled), parity=np.asarray(eps.parity),
                           pair_index=np.asarray(eps.pair_index))


def craft(base, out, eps_by_seed, smoke, rng, dts_seeds=None, shards=2):
    """Every GPU output of the mode in base, and the sources entries naming them -> namespace."""
    base = Path(base)
    seeds = XT.mode_seeds(smoke)
    rec_seed = R.SMOKE_SEEDS[0] if smoke else R.DEV_SEED
    record = write_record(base / "dts", rec_seed, len(R.PAIRS) * (R.N_SMOKE if smoke else R.N_PER_PAIR))
    vjobs, vouts, answers = {}, {}, {}
    for s in (dts_seeds or list(eps_by_seed)):
        eps = eps_by_seed[s]
        js = write_vjob(base / f"vjob_{s}", eps)
        recs = vrecords(s, eps.n)
        vjobs[s], vouts[s] = base / f"vjob_{s}", write_vout(base / f"vout_{s}", js, recs)
        answers.update({(r["seed"], r["episode_index"], r["condition"], r["wording"]): r["answer"] for r in recs})
    phrases = {DT.phrase_of(a) for a in answers.values()} | set(DT.TARGET_NAME.values())
    lout = write_lout(base / "lout", phrases)
    members = XT.member_union(out, seeds)
    ft_js = write_ft_job(base / "ft_job", members)
    feats = {v: ft_features(members, rng) for v in XT.FT_VARIANTS}
    ft_out = write_ft_out(base / "ft_out", ft_js, feats)
    first = eps_by_seed[seeds[0]]
    rr = write_rr_job(base / "rr_job", first)
    shown = planted(rr.perms, rng)
    shown[1::3] = rng.standard_normal(shown[1::3].shape).astype(np.float32)   # not every episode a sure hit
    cut = np.linspace(0, first.n, shards + 1).astype(int)
    rr_outs = [write_rr_out(base / f"rr_out_{k}", rr.shas, np.arange(cut[k], cut[k + 1]), shown[cut[k]:cut[k + 1]])
               for k in range(shards)]
    sources = {"dts": {"record": str(record), "verbalise_job": [str(p) for p in vjobs.values()],
                       "verbalise_out": [str(p) for p in vouts.values()], "listing_out": [str(lout)],
                       "embeddings": str(base / "value_embeddings.npz")},
               "ft_lp": {}, "ft": {"job": str(base / "ft_job"), "out": [str(ft_out)]},
               "mllm": {"job": str(rr.path), "out": [str(p) for p in rr_outs]}}
    return SimpleNamespace(base=base, record=record, vjobs=vjobs, vouts=vouts, answers=answers, lout=lout,
                           listings={(p, k): listing(p, k) for p in phrases if p for k in DT.KS}, members=members,
                           ft_job=base / "ft_job", ft_js=ft_js, feats=feats, ft_out=ft_out, rr=rr, shown=shown,
                           rr_outs=rr_outs, sources=sources)


def by_hand_cosine(rows, img, txt, pooled):
    """per_anchor of the plain cosine of unit features looked up by row id in a dict (no placement, no searchsorted)."""
    fi = {int(r): x for r, x in zip(rows.tolist(), img)}
    ft = {int(r): x for r, x in zip(rows.tolist(), txt)}

    def unit(x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    a, cand = np.asarray(pooled.anchor), np.asarray(pooled.candidates)
    qi, qt = unit([fi[int(r)] for r in a]), unit([ft[int(r)] for r in a])
    ci = unit(np.stack([[fi[int(r)] for r in row] for row in cand]))
    ct = unit(np.stack([[ft[int(r)] for r in row] for row in cand]))
    s = {"i2t": np.einsum("nd,nkd->nk", qi, ct), "t2i": np.einsum("nd,nkd->nk", qt, ci)}
    return per_anchor({c: {d: s[d] for d in DIRECTIONS} for c in CONDITIONS})


def same_hits(pa, want):
    """Equal R@1, gain, other and strict per episode (discrete), and swap."""
    return all(np.array_equal(pa[m], want[m]) for m in METRICS)


# ---------------------------------------------------------------- the smoke world (module fixture)

@pytest.fixture(scope="module")
def readers():
    return B.load_readers()


@pytest.fixture(scope="module")
def sm(tmp_path_factory, readers):
    root = tmp_path_factory.mktemp("xsmoke")
    fx = TD.build_dir(root / "smoke", True, readers, 31)
    eps = {s: E.load_episodes(fx.out / f"held_episodes_seed{s}.npz") for s in R.SMOKE_SEEDS}
    rng = np.random.default_rng(31)
    img, txt = feature_world(XT.member_union(fx.out, R.SMOKE_SEEDS), rng)
    ev = EvalInputs(img, txt)
    ctx = {s: make_ctx(eps[s], img, txt, ev) for s in R.SMOKE_SEEDS}
    del ev
    world = craft(root / "gpu", fx.out, eps, True, rng)
    return SimpleNamespace(fx=fx, eps=eps, img=img, txt=txt, ctx=ctx, world=world, root=root)


def run_smoke(sm, tmp_path, sources, capsys=None, name="res"):
    out = TD.copy_dir(sm.fx, tmp_path, name)
    if sources is not None:
        write_json(out / XT.SOURCES_NAME, sources)
    code, text = TD.run(sm.fx, out, capsys, bundle_fn=lambda s: (sm.ctx[s], sm.fx.bundles[s]))
    rec = json.loads((out / "descriptive.json").read_text()) if (out / "descriptive.json").exists() else None
    return code, text, rec, out


def fresh_sources(sm, tmp_path):
    src = json.loads(json.dumps(sm.world.sources))
    src["dts"]["embeddings"] = str(tmp_path / f"emb{next(_COUNT)}.npz")
    return src


def expected_row(sm, pa_by_seed):
    sc = {s: sm.fx.scored[i] for i, s in enumerate(R.SMOKE_SEEDS)}
    seeds = list(pa_by_seed)
    return json.loads(json.dumps(R.C.jsonable(D.scorer_row(
        pa_by_seed, {s: sc[s]["aff_fused"] for s in seeds}, {s: np.asarray(sc[s]["cl"]) for s in seeds},
        {s: np.asarray(sc[s]["pair_index"]) for s in seeds}))))


# ---------------------------------------------------------------- bookkeeping

GUARD_TESTS = {
    "listed_exists": "test_a_listed_path_that_does_not_exist_stops",
    "source_keys": "test_an_unknown_sources_key_stops",
    "dts_record": "test_dts_record_must_be_the_chosen_record_of_the_mode",
    "dts_stop": "test_dts_record_must_have_passed_the_stop",
    "dts_job_seed": "test_dts_jobs_must_be_of_the_modes_seeds",
    "dts_out_job": "test_dts_outputs_must_be_of_listed_jobs",
    "lp_ckpt": "test_lp_checkpoint_must_be_the_selected_one",
    "lp_rows": "test_lp_needs_every_anchor_and_candidate_in_the_context",
    "ft_out_job": "test_ft_outputs_must_be_of_the_listed_job",
    "ft_ckpt": "test_ft_features_must_come_from_the_selected_checkpoints",
    "ft_run": "test_ft_features_must_be_a_runs_bytes",
    "ft_unique": "test_ft_join_duplicated_row_raises",
    "ft_rows": "test_ft_join_missing_or_extra_row_raises",
    "ft_job_rows": "test_ft_job_rows_must_be_the_episode_members",
    "ft_one_file": "test_ft_variant_in_two_folders_stops",
    "ft_members": "test_ft_every_member_of_the_seed_present",
    "mllm_unique": "test_mllm_join_duplicated_index_raises",
    "mllm_conflict": "test_mllm_join_duplicated_index_raises",
    "mllm_extra": "test_mllm_join_extra_index_raises",
    "mllm_missing": "test_mllm_join_missing_episode_raises",
    "mllm_seed": "test_mllm_job_must_be_of_the_first_seed",
    "mllm_perms": "test_mllm_permutations_must_be_the_recorded_ones",
    "mllm_out_job": "test_mllm_outputs_must_be_of_the_job",
    "mllm_cover_job": "test_mllm_job_must_hold_every_episode",
    "mllm_episodes": "test_mllm_job_must_be_the_seeds_episodes",
    "ctx_episodes": "test_context_must_hold_the_episode_files_episodes",
    "sources_required": "test_real_mode_requires_the_sources_file_and_its_four_entries",
    "dts_tuning_cache": "test_dts_refuses_the_tuning_cache_as_its_cache",
    "dts_encoder_meta": "test_dts_encoder_must_be_the_tuning_records",
    "dts_cache_meta": "test_dts_cache_must_be_made_by_the_tuning_records_encoder",
    "mllm_same_model": "test_mllm_folders_must_agree_on_the_model",
    "mllm_model": "test_mllm_model_must_be_the_probes_recorded_setting",
}


def test_every_marked_guard_has_a_mutation_test():
    found = set(re.findall(r"# guard:([a-z0-9_]+)", (HERE / "r6_external.py").read_text()))
    assert found == set(GUARD_TESTS), (sorted(found - set(GUARD_TESTS)), sorted(set(GUARD_TESTS) - found))
    for g, t in GUARD_TESTS.items():
        assert t in globals() and f'"{g}"' in inspect.getsource(globals()[t]), (g, t)


def outcome(fn):
    """What fn() raises ("<type>: <message>"), or None when it goes through."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (a mutant may fail anywhere; only the guard's message matters)
        return f"{type(e).__name__}: {e}"
    return None


def caught_then_not(tmp_path, guard, scenario, text):
    """scenario(module) raises with ``text`` on r6_external, and not (goes through, or fails elsewhere) without the
    guard (a copy)."""
    got = outcome(lambda: scenario(XT))
    assert got is not None and got.startswith("AssertionError") and text in got, got
    mut = mutant(tmp_path, guard=guard)
    after = outcome(lambda: scenario(mut))
    assert after is None or text not in after, after


# ---------------------------------------------------------------- end to end (acceptance 3)

def test_smoke_end_to_end_every_external_row_present_no_decimal(sm, tmp_path, capsys):
    src = fresh_sources(sm, tmp_path)
    code, text, rec, out = run_smoke(sm, tmp_path, src, capsys)
    assert code == 0, text
    assert DECIMAL.search(text) is None, text
    assert text.count("external rows DTS, DTS-CF, DTS-N, FT-LP, FT-LB, FT-LoRA") == 3 and "missing none" in text
    assert rec["scorers"][-len(XT.NAMES):] == list(XT.NAMES) and list(rec["external"]) == list(XT.NAMES)
    seeds = list(R.SMOKE_SEEDS)
    for name in XT.NAMES:
        e, r = rec["external"][name], rec["rows"][name]
        assert "missing" not in e, (name, e.get("missing"))
        assert e["seeds"] == ([9001] if name == "MLLM" else seeds)
        assert set(r) == {"pooled", "per_seed", "per_pair"} and set(r["per_pair"]) == set(R.PAIR_NAMES)
        assert set(r["pooled"]) == {"n_episodes", "r1", "gain", "swap", "aff_minus"}
        assert rec["labels"][name] == XT.LABELS[name]
        assert e["sources"] == {"file": XT.SOURCES_NAME, "sha256": R.sha256_file(out / XT.SOURCES_NAME),
                                "entry": src[XT.FAMILY[name]]}
    # DTS family: the rows are r6_dts.held_dts_scores' arrays (frozen picks) under describe's rows
    w = sm.world
    got = {s: DT.held_dts_scores(sm.ctx[s], w.answers, w.listings, json.loads(w.record.read_text()),
                                 DT.ValueEmbedder(None, encoder_factory=FakeEncoder)) for s in seeds}
    for key, name in XT.DTS_ROWS.items():
        assert rec["rows"][name] == expected_row(sm, {s: got[s][key] for s in seeds}), name
    dts = rec["external"]["DTS"]
    want_fail = {str(s): failures_of(s, sm.eps[s].n) for s in seeds}
    assert dts["parsing_failures"] == want_fail and dts["per_seed"]["9002"]["parsing_failures"] == want_fail["9002"]
    assert dts["parsing_failures_total"] == {c: {k: sum(want_fail[str(s)][c][k] for s in seeds)
                                                 for k in DT.FAIL_KINDS} for c in CONDITIONS}
    assert sum(want_fail["9001"]["a"].values()) > 0                                # failures do occur
    assert (dts["setting"], dts["wording_id"], dts["K"], dts["picks"]) == ("W2 K8", W, K, PICKS["dts"])
    assert dts["wording"] == SETTINGS["verbaliser"]["wordings"][W]
    assert rec["external"]["DTS-N"]["picks"] == PICKS["dts_n"] and rec["external"]["DTS-CF"]["picks"] == \
        PICKS["dts_cf"]
    ph = dts["phrases"]
    assert ph["n_phrases"] == 2 * 9 * R.N_SMOKE and len(ph["top"]) == XT.TOP_N
    counts = {}
    for s in seeds:
        for i in range(sm.eps[s].n):
            for c in CONDITIONS:
                p = DT.phrase_of(answer(s, i, c))
                counts[p] = counts.get(p, 0) + 1
    top = sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:XT.TOP_N]
    assert ph["top"] == [[p, k] for p, k in top]
    assert set(ph["by_pair_condition"]) == set(R.PAIR_NAMES)
    assert dts["record"]["sha256"] == R.sha256_file(w.record) and dts["embeddings"]["sha256"] == \
        R.sha256_file(src["dts"]["embeddings"])
    # FT rows: per_anchor of the plain cosine by row-id lookup (no placement): the same hits per episode
    for v in XT.FT_VARIANTS:
        rows, img, txt = w.feats[v]
        pa = {s: by_hand_cosine(rows, img, txt, sm.eps[s].pooled) for s in seeds}
        assert rec["rows"][f"FT-{v}"] == expected_row(sm, pa), v
        assert rec["external"][f"FT-{v}"]["features"]["sha256"] == R.sha256_file(w.ft_out / f"features_{v}.npz")
        assert rec["external"][f"FT-{v}"]["image_source"] == "r6_ft_cache"                 # disclosed with the row
    lp = XT.lp_maps(XT.resolve(XT.FT_CKPTS["LP"]["path"]))
    pa = {}
    for s in seeds:
        rows = np.unique(np.concatenate([sm.eps[s].pooled.anchor, sm.eps[s].pooled.candidates.ravel()]))
        f32 = {m: torch.nn.functional.linear(torch.from_numpy(getattr(sm, m)[rows]),
                                             torch.from_numpy(lp.W[m].astype(np.float32))).numpy()
               for m in ("img", "txt")}                                              # the run's own fp32 nn.Linear
        pa[s] = by_hand_cosine(rows, f32["img"], f32["txt"], sm.eps[s].pooled)
    assert rec["rows"]["FT-LP"] == expected_row(sm, pa)
    assert rec["external"]["FT-LP"]["checkpoint"]["sha256"] == XT.FT_CKPTS["LP"]["sha256"]
    # MLLM: the probe's own unpermute per (episode, condition, direction), seed 9001 only
    perms, n = w.rr.perms, sm.eps[9001].n
    sc = {c: {d: np.stack([unpermute(w.shown[i, ci, di], perms[i, ci, di]) for i in range(n)])
              for di, d in enumerate(XT.RERANK_DIRECTIONS)} for ci, c in enumerate(XT.RERANK_CONDITIONS)}
    assert rec["rows"]["MLLM"] == expected_row(sm, {9001: per_anchor(sc)})
    assert rec["external"]["MLLM"]["per_seed"] == {"9001": {"n_episodes": n}}
    assert rec["external"]["MLLM"]["permutations_sha256"] == G.sha256_bytes(perms.tobytes())
    assert rec["external"]["MLLM"]["model"] == RR_MODEL
    assert set(rec["external"]["MLLM"]["fingerprints"]) == set(src["mllm"]["out"])
    assert all(fp["snapshot"] == RR_MODEL["snapshot"] for fp in rec["external"]["MLLM"]["fingerprints"].values())
    assert not hasattr(sm.fx.env, XT.STATE)                                     # the run's state is dropped


# ---------------------------------------------------------------- a job not run (acceptance 4)

def test_a_job_not_run_gives_a_missing_row(sm, tmp_path, capsys):
    """MLLM marked missing in the sources; FT-LP not listed; FT-LoRA's file in no folder; seed 9003's verbaliser job
    listed without output: those rows carry "missing" and their reason, no number; FT-LB is still reported."""
    w, src = sm.world, fresh_sources(sm, tmp_path)
    src["mllm"] = {"missing": "the reranker was not run"}
    del src["ft_lp"]
    rows, img, txt = w.feats["LB"]
    only_lb = write_ft_out(tmp_path / "ft_out_lb", w.ft_js, {"LB": (rows, img, txt)})
    src["ft"]["out"] = [str(only_lb)]
    src["dts"]["verbalise_out"] = [str(w.vouts[9001]), str(w.vouts[9002])]
    code, text, rec, _ = run_smoke(sm, tmp_path, src, capsys)
    assert code == 0, text
    assert DECIMAL.search(text) is None, text
    ext = rec["external"]
    assert ext["MLLM"]["missing"] == "the reranker was not run"
    assert "lists no ft_lp entry" in ext["FT-LP"]["missing"]
    assert "no features_LoRA.npz" in ext["FT-LoRA"]["missing"]
    for name in ("DTS", "DTS-CF", "DTS-N"):
        assert "no output of seed 9003's verbaliser job" in ext[name]["missing"]
        assert ext[name]["seeds_computed_but_dropped"] == [9001, 9002] and ext[name]["seeds"] == []
    for name in ("MLLM", "FT-LP", "FT-LoRA", "DTS", "DTS-CF", "DTS-N"):
        assert name not in rec["rows"] and name not in rec["scorers"], name
    assert "missing" not in ext["FT-LB"] and ext["FT-LB"]["seeds"] == list(R.SMOKE_SEEDS)
    assert rec["scorers"][-1] == "FT-LB"
    # no sources file at all: every row missing (ticket 13's smoke test checks that run too)
    st = XT.new_state(tmp_path / "nowhere", True)
    assert all(isinstance(v, str) and XT.SOURCES_NAME in v for v in st.fam.values())


# ---------------------------------------------------------------- DTS joins (r6_dts' guards, through this path)

def dts_entry(sm, tmp_path, **over):
    e = fresh_sources(sm, tmp_path)["dts"]
    e.update(over)
    return e


def dts_one(mod, sm, entry, seed=9001):
    st = mod.load_dts(entry, sm.fx.out, True)
    return mod.dts_seed(st, seed, sm.ctx[seed], sm.eps[seed])


def test_dts_join_missing_duplicated_extra_keys_raise(sm, tmp_path):
    """(seed, episode_index, condition, wording) present exactly once: a key missing, twice in one file, twice across
    folders with different answers, beyond the seed's episodes, of another seed; a (phrase, K) listing missing."""
    w, n = sm.world, sm.eps[9001].n
    js = shas(w.vjobs[9001], ("rows_manifest.npz", "verbalise_input.npz"))
    recs = vrecords(9001, n)

    def with_out(recs_, extra_dirs=()):
        d = write_vout(tmp_path / f"v{next(_COUNT)}", js, recs_)
        return dts_entry(sm, tmp_path, verbalise_out=[str(d), *map(str, extra_dirs)])
    assert set(dts_one(XT, sm, dts_entry(sm, tmp_path))) == {"DTS", "DTS-CF", "DTS-N"}    # the crafted world joins
    gone = [r for r in recs if not (r["episode_index"] == 17 and r["condition"] == "b")]
    with pytest.raises(AssertionError, match=r"no verbaliser answer for \(9001, 17, 'b', 'W2'\)"):
        dts_one(XT, sm, with_out(gone))
    d = write_vout(tmp_path / f"v{next(_COUNT)}", js, recs)
    with open(d / f"phrases_{W}.jsonl", "a") as f:
        f.write(json.dumps(recs[5]) + "\n")
    with pytest.raises(AssertionError, match="duplicate record"):
        dts_one(XT, sm, dts_entry(sm, tmp_path, verbalise_out=[str(d)]))
    other = write_vout(tmp_path / f"v{next(_COUNT)}", js, [dict(recs[8], answer="a different answer")])
    with pytest.raises(AssertionError, match="seen twice with different answers"):
        dts_one(XT, sm, with_out(recs, [other]))
    same = write_vout(tmp_path / f"v{next(_COUNT)}", js, [dict(recs[8])])
    assert set(dts_one(XT, sm, with_out(recs, [same]))) == {"DTS", "DTS-CF", "DTS-N"}   # identical: merged once
    with pytest.raises(AssertionError, match="outside its job|beyond"):
        dts_one(XT, sm, with_out(recs + [dict(recs[0], episode_index=n)]))
    with pytest.raises(AssertionError, match="a record of seed 9002"):
        dts_one(XT, sm, with_out(recs + [dict(recs[0], seed=9002)]))
    p = DT.phrase_of(answer(9001, 3, "a"))
    short = write_lout(tmp_path / f"l{next(_COUNT)}", {DT.phrase_of(a) for a in w.answers.values()
                                                       if DT.phrase_of(a) != p} | set(DT.TARGET_NAME.values()))
    with pytest.raises(AssertionError, match="no listing"):
        dts_one(XT, sm, dts_entry(sm, tmp_path, listing_out=[str(short)]))


def test_dts_record_must_be_the_chosen_record_of_the_mode(sm, tmp_path):
    """Smoke mode reads a smoke seed's record of 192 episodes; held mode only seed 42's of 12,288."""
    w = sm.world
    d = tmp_path / "rec"
    shutil.copytree(w.record.parent, d)
    edit_json(d / w.record.name, lambda r: r.update(n_episodes=12288))
    edit_json(d / RDTS.STOP, lambda s: s["input_sha256"].update({w.record.name: R.sha256_file(d / w.record.name)}))
    caught_then_not(tmp_path, "dts_record", lambda m: m.load_dts(dts_entry(sm, tmp_path, record=str(d / w.record.name)),
                                                                 sm.fx.out, True), "not the chosen stage's record")
    with pytest.raises(AssertionError, match="not the chosen stage's record"):        # a smoke record in held mode
        XT.load_dts(dts_entry(sm, tmp_path), sm.fx.out, False)


def test_dts_record_must_have_passed_the_stop(sm, tmp_path):
    w = sm.world
    d = tmp_path / "rec"
    shutil.copytree(w.record.parent, d)
    edit_json(d / RDTS.STOP, lambda s: s.update(stop=True))
    entry = dts_entry(sm, tmp_path, record=str(d / w.record.name))
    caught_then_not(tmp_path, "dts_stop", lambda m: m.load_dts(entry, sm.fx.out, True), "no passed stop")
    edit_json(d / RDTS.STOP, lambda s: s.update(stop=False))
    edit_json(d / w.record.name, lambda r: r.update(note="other bytes"))
    with pytest.raises(AssertionError, match="no passed stop"):                       # the stop saw other bytes
        XT.load_dts(entry, sm.fx.out, True)
    (d / RDTS.STOP).unlink()
    with pytest.raises(AssertionError, match="dts_stop.json is missing"):
        XT.load_dts(entry, sm.fx.out, True)
    assert XT.STOP_NAME == RDTS.STOP


def test_dts_jobs_must_be_of_the_modes_seeds(sm, tmp_path):
    e42 = SimpleNamespace(**{**vars(sm.eps[9001]), "seed": 42})
    write_vjob(tmp_path / "vjob42", e42)
    entry = dts_entry(sm, tmp_path)
    entry["verbalise_job"] = entry["verbalise_job"] + [str(tmp_path / "vjob42")]
    caught_then_not(tmp_path, "dts_job_seed", lambda m: m.load_dts(entry, sm.fx.out, True),
                    "a verbaliser job of seed 42")


def test_dts_outputs_must_be_of_listed_jobs(sm, tmp_path):
    entry = dts_entry(sm, tmp_path)
    entry["verbalise_job"] = [p for p in entry["verbalise_job"] if not p.endswith("vjob_9002")]
    caught_then_not(tmp_path, "dts_out_job", lambda m: m.load_dts(entry, sm.fx.out, True),
                    "the output of a verbaliser job that")


# ---------------------------------------------------------------- sources

def test_a_listed_path_that_does_not_exist_stops(sm, tmp_path):
    entry = dts_entry(sm, tmp_path, listing_out=[str(tmp_path / "no_such_folder")])
    caught_then_not(tmp_path, "listed_exists", lambda m: m.load_dts(entry, sm.fx.out, True), "does not exist")
    with pytest.raises(AssertionError, match="does not exist"):                       # relative: under MAIN
        XT.listed("res/no_such_r6_folder", "x")
    assert XT.resolve("res/x") == R.MAIN / "res/x" and XT.resolve(tmp_path) == tmp_path


def test_an_unknown_sources_key_stops(sm, tmp_path):
    d = tmp_path / "out"
    write_json(d / XT.SOURCES_NAME, {"dts_typo": {}, "mllm": {"missing": "not run"}})
    caught_then_not(tmp_path, "source_keys", lambda m: m.load_sources(d), "are not objects keyed by")
    write_json(d / XT.SOURCES_NAME, {"mllm": {"missing": "not run", "job": "x"}})
    with pytest.raises(AssertionError, match="holds only a non-empty reason"):
        XT.new_state(d, True)


# ---------------------------------------------------------------- FT joins

def ft_entry(sm, out_dirs=None, job=None):
    return {"job": str(job or sm.world.ft_job), "out": [str(p) for p in (out_dirs or [sm.world.ft_out])]}


def ft_with(sm, tmp_path, rows_fn):
    """An output folder whose features_LB.npz has rows_fn(job rows)."""
    rows = rows_fn(sm.world.members.copy())
    rng = np.random.default_rng(5)
    return write_ft_out(tmp_path / f"ft{next(_COUNT)}", sm.world.ft_js, {"LB": ft_features(rows, rng)})


def test_ft_join_duplicated_row_raises(sm, tmp_path):
    d = ft_with(sm, tmp_path, lambda r: np.sort(np.concatenate([r, r[10:11]])))
    caught_then_not(tmp_path, "ft_unique", lambda m: m.load_ft(ft_entry(sm, [d]), sm.fx.out, True), "a row twice")


def test_ft_join_missing_or_extra_row_raises(sm, tmp_path):
    gone = ft_with(sm, tmp_path, lambda r: np.delete(r, 7))
    caught_then_not(tmp_path, "ft_rows", lambda m: m.load_ft(ft_entry(sm, [gone]), sm.fx.out, True),
                    "1 missing, 0 extra")
    spare = np.setdiff1d(np.arange(R.N_ROWS), sm.world.members)[3]
    extra = ft_with(sm, tmp_path, lambda r: np.sort(np.append(r, spare)))
    with pytest.raises(AssertionError, match="0 missing, 1 extra"):
        XT.load_ft(ft_entry(sm, [extra]), sm.fx.out, True)


def test_ft_job_rows_must_be_the_episode_members(sm, tmp_path):
    rows = np.delete(sm.world.members, 11)
    js = write_ft_job(tmp_path / "job", rows)
    out = write_ft_out(tmp_path / "out", js, {"LB": ft_features(rows, np.random.default_rng(1))})
    caught_then_not(tmp_path, "ft_job_rows", lambda m: m.load_ft(ft_entry(sm, [out], tmp_path / "job"),
                                                                 sm.fx.out, True), "member rows of the episodes")


def test_ft_every_member_of_the_seed_present(sm, tmp_path):
    f = XT.load_ft(ft_entry(sm), sm.fx.out, True).variants["LB"]
    first = int(sm.eps[9002].pooled.pairs_b_txt[5, 2])
    keep = f.rows != first
    g = SimpleNamespace(**{**vars(f), "rows": f.rows[keep], "img": f.img[keep], "txt": f.txt[keep]})
    caught_then_not(tmp_path, "ft_members", lambda m: m.ft_seed(g, sm.eps[9002], 9002),
                    "member rows of seed 9002's episodes are missing")
    assert XT.ft_seed(f, sm.eps[9002], 9002)["pa"]["r1"].shape == (sm.eps[9002].n,)


def test_ft_outputs_must_be_of_the_listed_job(sm, tmp_path):
    d = tmp_path / "out"
    shutil.copytree(sm.world.ft_out, d)
    edit_json(d / "provenance.json", lambda p: p["fingerprint"]["inputs_sha256"].update({"ft_rows.npz": "0" * 64}))
    caught_then_not(tmp_path, "ft_out_job", lambda m: m.load_ft(ft_entry(sm, [d]), sm.fx.out, True),
                    "not an output of the FT job")


def test_ft_features_must_come_from_the_selected_checkpoints(sm, tmp_path):
    d = tmp_path / "out"
    shutil.copytree(sm.world.ft_out, d)
    edit_json(d / "provenance.json", lambda p: p["fingerprint"]["ckpt_sha256"].update({"LoRA": "1" * 64}))
    caught_then_not(tmp_path, "ft_ckpt", lambda m: m.load_ft(ft_entry(sm, [d]), sm.fx.out, True),
                    "not made from the selected LoRA checkpoint")


def test_ft_features_must_be_a_runs_bytes(sm, tmp_path):
    d = tmp_path / "out"
    shutil.copytree(sm.world.ft_out, d)
    edit_json(d / "provenance.json", lambda p: p["runs"][0]["features_sha256"].update({"LB": "2" * 64}))
    caught_then_not(tmp_path, "ft_run", lambda m: m.load_ft(ft_entry(sm, [d]), sm.fx.out, True),
                    "no run of its provenance wrote these bytes")


def test_ft_variant_in_two_folders_stops(sm, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(sm.world.ft_out, d)
    caught_then_not(tmp_path, "ft_one_file", lambda m: m.load_ft(ft_entry(sm, [sm.world.ft_out, d]), sm.fx.out, True),
                    "is in 2 listed FT output folders")


# ---------------------------------------------------------------- FT-LP

def test_lp_checkpoint_must_be_the_selected_one(sm, tmp_path):
    p = tmp_path / "lp.pt"
    eye = torch.eye(512)
    torch.save({"variant": "LP", "lr": 3e-4, "epoch": 9, "params": {"img_map.weight": eye, "txt_map.weight": eye,
                                                                      "logit_scale": torch.tensor(4.6)}}, p)
    caught_then_not(tmp_path, "lp_ckpt", lambda m: m.load_lp({"checkpoint": str(p)}, sm.fx.out, True),
                    "is not the selected LP checkpoint's")
    lp = XT.load_lp({}, sm.fx.out, True)
    assert lp.sha256 == XT.FT_CKPTS["LP"]["sha256"] and lp.path == XT.resolve(XT.FT_CKPTS["LP"]["path"])


def test_lp_needs_every_anchor_and_candidate_in_the_context(sm, tmp_path):
    lp = XT.load_lp({}, sm.fx.out, True)
    img = sm.img.copy()
    img[int(sm.eps[9003].pooled.candidates[4, 9])] = np.nan
    ctx = SimpleNamespace(**{**vars(sm.ctx[9003]), "img": img})
    caught_then_not(tmp_path, "lp_rows", lambda m: m.lp_seed(lp, ctx, sm.eps[9003]), "no cached feature")


def test_lp_maps_reproduce_the_clipft_runs_stored_features():
    """Acceptance 2: the LP lr 3e-4 maps applied on the CPU (apply_map: float64, rounded to float32) to the cached CLIP
    features of 64 rows of the clipft run's features.npz (its val and selection rows; none is held) give the run's
    stored LP features within float32 rounding: max |diff| <= 2e-5 (the GPU's fp32 matmul sums in another order;
    observed 4.3e-6 for values up to 8.5) and per-row cosine >= 1 - 1e-6. The identity map and the transposed map
    miss by more than 0.5, so the check would catch a wrong map or orientation."""
    run = XT.resolve(XT.FT_CKPTS["LP"]["path"]).parent
    lp = XT.lp_maps(run / "best_params.pt")
    with np.load(run / "features.npz", allow_pickle=False) as z:
        stored_rows = z["rows"]
        pick = stored_rows[np.linspace(0, len(stored_rows) - 1, 64).astype(int)]
        pos = np.searchsorted(stored_rows, pick)
        ref = {"img": z["img"][pos], "txt": z["txt"][pos]}
    with np.load(R.MAIN / "src/test" / R.HELD_CODES_REL, allow_pickle=False) as z:
        assert not np.isin(pick, z["held_rows"]).any()                                # no held row is read
    shards = Path("/data/SSD2/pre_extract/artelingo/features/shards")                 # the FeatureManager cache
    lens = [len(np.load(shards / f"shard_{i:05d}" / "sample_ids.npy", mmap_mode="r")) for i in range(4)]
    assert sum(lens) == R.N_ROWS
    off = np.concatenate([[0], np.cumsum(lens)])

    def cached(kind):
        out = []
        for r in pick.tolist():
            s = int(np.searchsorted(off, r, side="right") - 1)
            out.append(np.load(shards / f"shard_{s:05d}" / f"{kind}_features.npy", mmap_mode="r")[r - off[s]])
        return np.stack(out).astype(np.float32)
    x = {"img": cached("img"), "txt": cached("txt")}
    for m in ("img", "txt"):
        y = XT.apply_map(x[m], lp.W[m])
        assert y.dtype == np.float32 and y.shape == (64, 512)
        cos = (y * ref[m]).sum(1) / np.linalg.norm(y, axis=1) / np.linalg.norm(ref[m], axis=1)
        assert float(np.abs(y - ref[m]).max()) <= 2e-5 and float(cos.min()) >= 1 - 1e-6, m
        assert np.array_equal(y[5:9], XT.apply_map(x[m][5:9], lp.W[m]))               # batch-independent
        for wrong in (np.eye(512), lp.W[m].T):
            assert float(np.abs(XT.apply_map(x[m], wrong) - ref[m]).max()) > 0.5, m


# ---------------------------------------------------------------- MLLM joins and the permutation

def mllm_entry(sm, outs=None, job=None):
    return {"job": str(job or sm.world.rr.path), "out": [str(p) for p in (outs or sm.world.rr_outs)]}


def mllm_one(mod, sm, entry):
    m = mod.load_mllm(entry, sm.fx.out, True)
    return mod.mllm_seed(m, sm.eps[9001], 9001)


def rr_with(sm, tmp_path, index, shown):
    return write_rr_out(tmp_path / f"rr{next(_COUNT)}", sm.world.rr.shas, index, shown)


def test_unpermute_puts_a_planted_best_candidate_in_its_column(sm, tmp_path):
    """Acceptance 1: in every (episode, condition, direction) the letter that showed the target column holds the best
    score; un-permuted, the target column (p_a under a, p_b under b) is the argmax, so R@1 = 1 and gain = 1 on every
    episode, through the real job folder. Un-permuting by a gather (shown[perm]) instead of the scatter fails it."""
    perms, n = sm.world.rr.perms, sm.eps[9001].n
    shown = planted(perms, np.random.default_rng(3))
    back = XT.unpermute_scores(shown, perms)
    assert (np.argmax(back[:, 0], axis=-1) == 0).all() and (np.argmax(back[:, 1], axis=-1) == 1).all()
    assert all(np.array_equal(back[i, c, d], unpermute(shown[i, c, d], perms[i, c, d]))
               for i in range(n) for c in range(2) for d in range(2))                 # the probe's own function
    d = rr_with(sm, tmp_path, np.arange(n), shown)
    pa = mllm_one(XT, sm, mllm_entry(sm, [d]))["pa"]
    assert (pa["r1"] == 1).all() and (pa["gain"] == 1).all()
    wrong = mutant(tmp_path, replace=("    np.put_along_axis(out, perms, shown, axis=-1)\n",
                                      "    out[...] = np.take_along_axis(shown, perms, axis=-1)\n"))
    pa_wrong = mllm_one(wrong, sm, mllm_entry(sm, [d]))["pa"]
    assert not (pa_wrong["r1"] == 1).all()                                             # the planted test catches it
    assert (np.argmax(wrong.unpermute_scores(shown, perms)[:, 0], axis=-1) != 0).any()


def test_mllm_join_missing_episode_raises(sm, tmp_path):
    w, n = sm.world, sm.eps[9001].n
    keep = np.delete(np.arange(n), 40)
    d = rr_with(sm, tmp_path, keep, w.shown[keep])
    caught_then_not(tmp_path, "mllm_missing", lambda m: mllm_one(m, sm, mllm_entry(sm, [d])),
                    "1 episodes of seed 9001 have no reranker scores, e.g. [40]")


def test_mllm_join_duplicated_index_raises(sm, tmp_path):
    """Twice in one scores.npz raises; twice across shard folders raises when the scores differ, and is merged once
    when they are identical (a relaunch copies its --prior scores)."""
    w, n = sm.world, sm.eps[9001].n
    idx = np.sort(np.append(np.arange(n), 12))
    d = rr_with(sm, tmp_path, idx, w.shown[idx])
    caught_then_not(tmp_path, "mllm_unique", lambda m: m.load_mllm(mllm_entry(sm, [d]), sm.fx.out, True),
                    "an episode index is held twice")
    sh = w.shown[:30].copy()
    sh[4, 1, 0, 2] += 1.0
    other = rr_with(sm, tmp_path, np.arange(30), sh)
    caught_then_not(tmp_path, "mllm_conflict",
                    lambda m: m.load_mllm(mllm_entry(sm, [*w.rr_outs, other]), sm.fx.out, True),
                    "episode 4 is held twice with different scores")
    same = rr_with(sm, tmp_path, np.arange(30), w.shown[:30])
    assert len(XT.load_mllm(mllm_entry(sm, [*w.rr_outs, same]), sm.fx.out, True).merged) == n


def test_mllm_join_extra_index_raises(sm, tmp_path):
    w, n = sm.world, sm.eps[9001].n
    d = rr_with(sm, tmp_path, np.arange(n + 1), np.concatenate([w.shown, w.shown[:1]]))
    caught_then_not(tmp_path, "mllm_extra", lambda m: m.load_mllm(mllm_entry(sm, [d]), sm.fx.out, True),
                    "episodes the job does not hold")


def rr_job_copy(sm, tmp_path, edit_npz=None, edit_record=None, edit_perms=None):
    """A copy of the reranker job (and its permutations file) with edits; outputs re-fingerprinted to it."""
    w = sm.world
    d = tmp_path / f"rrjob{next(_COUNT)}"
    shutil.copytree(w.rr.path, d)
    shutil.copy(I.perms_path(w.rr.path), I.perms_path(d))
    if edit_npz:
        with np.load(d / "rerank_input.npz") as z:
            a = {k: z[k] for k in z.files}
        edit_npz(a)
        np.savez(d / "rerank_input.npz", **a)
    if edit_perms:
        with np.load(I.perms_path(d)) as z:
            p = {k: z[k] for k in z.files}
        edit_perms(p)
        with open(I.perms_path(d), "wb") as f:
            np.savez(f, **p)
    js = shas(d, ("rows_manifest.npz", "rerank_input.npz"))
    edit_json(d / "job_record.json", lambda r: r.update(files_sha256=js))
    if edit_record:
        edit_json(d / "job_record.json", edit_record)
    n = sm.eps[9001].n
    out = write_rr_out(tmp_path / f"rro{next(_COUNT)}", js, np.arange(n), w.shown)
    return {"job": str(d), "out": [str(out)]}


def test_mllm_job_must_be_of_the_first_seed(sm, tmp_path):
    def to_9002(a):
        a["seed"] = np.int64(9002)
    entry = rr_job_copy(sm, tmp_path, edit_npz=to_9002, edit_record=lambda r: r.update(seed=9002),
                        edit_perms=lambda p: p.update(seed=np.int64(9002)))
    caught_then_not(tmp_path, "mllm_seed", lambda m: m.load_mllm(entry, sm.fx.out, True),
                    "a reranker job of seed 9002, not of 9001")


def test_mllm_permutations_must_be_the_recorded_ones(sm, tmp_path):
    def swap(p):
        p["perms"] = p["perms"].copy()
        p["perms"][3, 0, 1, [0, 1]] = p["perms"][3, 0, 1, [1, 0]]
    entry = rr_job_copy(sm, tmp_path, edit_perms=swap)
    caught_then_not(tmp_path, "mllm_perms", lambda m: m.load_mllm(entry, sm.fx.out, True),
                    "not the job's recorded permutations")
    entry = rr_job_copy(sm, tmp_path, edit_record=lambda r: r.update(perms_sha256="3" * 64))
    with pytest.raises(AssertionError, match="not the job's recorded permutations"):
        XT.load_mllm(entry, sm.fx.out, True)


def test_mllm_outputs_must_be_of_the_job(sm, tmp_path):
    d = rr_with(sm, tmp_path, np.arange(sm.eps[9001].n), sm.world.shown)
    edit_json(d / "provenance.json", lambda p: p["fingerprint"]["inputs_sha256"].update({"rerank_input.npz": "4" * 64}))
    caught_then_not(tmp_path, "mllm_out_job", lambda m: m.load_mllm(mllm_entry(sm, [d]), sm.fx.out, True),
                    "not an output of the reranker job")


def test_mllm_job_must_hold_every_episode(sm, tmp_path):
    """A job of the first 100 episodes only (r6_gpu_inputs --first-per-pair style subset): refused."""
    w = sm.world
    sub = SimpleNamespace(seed=9001, pooled=SimpleNamespace(**{f: getattr(sm.eps[9001].pooled, f)[:100]
                                                               for f in E.FIELDS}), n=100)
    rr = write_rr_job(tmp_path / "rrjob_sub", sub)
    out = write_rr_out(tmp_path / "rro_sub", rr.shas, np.arange(100), w.shown[:100])
    caught_then_not(tmp_path, "mllm_cover_job", lambda m: mllm_one(m, sm, mllm_entry(sm, [out], rr.path)),
                    "a job of seed 9001 holding 100 episodes, not the 192 episodes of seed 9001")
    m = XT.load_mllm(mllm_entry(sm), sm.fx.out, True)                               # the full job, another seed
    with pytest.raises(AssertionError, match="a job of seed 9001 holding 192 episodes, not the 192 episodes of "
                                             "seed 9002"):
        XT.mllm_seed(m, sm.eps[9002], 9002)


def test_mllm_job_must_be_the_seeds_episodes(sm, tmp_path):
    def shift(a):
        a["query_row"] = a["query_row"].copy()
        a["query_row"][7] = a["query_row"][8]
    entry = rr_job_copy(sm, tmp_path, edit_npz=shift)
    caught_then_not(tmp_path, "mllm_episodes", lambda m: mllm_one(m, sm, entry), "are not seed 9001's episodes")


# ---------------------------------------------------------------- the per-seed hook

def test_context_must_hold_the_episode_files_episodes(sm, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    p = sm.eps[9001].pooled
    cand = p.candidates.copy()
    cand[2, [3, 4]] = cand[2, [4, 3]]
    ctx = SimpleNamespace(**{**vars(sm.ctx[9001]), "pooled": AspectEpisodes(
        p.aspect_a, p.aspect_b, p.anchor, cand, p.pairs_a_img, p.pairs_a_txt, p.pairs_b_img, p.pairs_b_txt)})
    caught_then_not(tmp_path, "ctx_episodes", lambda m: m.seed_rows(SimpleNamespace(), 9001, ctx, sm.eps[9001], out,
                                                                    True),
                    "the context's episodes are not the episode file's")


def test_state_starts_at_the_first_seed_and_is_dropped_by_finish(sm, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    env = SimpleNamespace()
    rows = XT.seed_rows(env, 9001, sm.ctx[9001], sm.eps[9001], out, True)
    st = getattr(env, XT.STATE)
    assert list(rows) == list(XT.NAMES) and all(x["pa"] is None for x in rows.values())
    assert "MLLM" not in XT.seed_rows(env, 9002, sm.ctx[9002], sm.eps[9002], out, True)
    assert getattr(env, XT.STATE) is st
    XT.seed_rows(env, 9001, sm.ctx[9001], sm.eps[9001], out, True)                   # a new run: a new state
    assert getattr(env, XT.STATE) is not st
    for s in (9002, 9003):
        XT.seed_rows(env, s, sm.ctx[s], sm.eps[s], out, True)
    ext = XT.finish(env, out, True, {})
    assert list(ext) == list(XT.NAMES) and not hasattr(env, XT.STATE)
    assert all(x["pa"] == {} and XT.SOURCES_NAME in x["info"]["missing"] and x["info"]["sources"]["sha256"] is None
               for x in ext.values())
    # a state left by a run that stopped after its first seed (or made for another folder) is not used, and dropped
    XT.seed_rows(env, 9001, sm.ctx[9001], sm.eps[9001], out, True)
    stub = {"DTS": {"pa": {}, "label": "stub", "info": {"per_seed": {}}}}
    ext = XT.finish(env, out, True, stub)
    assert "sources" not in ext["DTS"]["info"] and not hasattr(env, XT.STATE)
    for s in R.SMOKE_SEEDS:
        XT.seed_rows(env, s, sm.ctx[s], sm.eps[s], out, True)
    assert "sources" not in XT.finish(env, tmp_path / "another_folder", True, {})["DTS"]["info"]


def test_an_unreadable_listed_input_stops_with_exit_5(sm, tmp_path, capsys):
    """A listed reranker job without its job_record.json: the pass stops (exit 5, nothing written), no crash, and the
    smoke message prints no decimal number."""
    src = fresh_sources(sm, tmp_path)
    job = tmp_path / "rrjob_no_record"
    shutil.copytree(sm.world.rr.path, job)
    shutil.copy(I.perms_path(sm.world.rr.path), I.perms_path(job))
    (job / "job_record.json").unlink()
    src["mllm"]["job"] = str(job)
    code, text, rec, out = run_smoke(sm, tmp_path, src, capsys)
    assert code == RD.EXIT_CONTRADICTION and rec is None, text
    assert "a listed input could not be read: FileNotFoundError" in text and DECIMAL.search(text) is None, text
    assert not (out / "descriptive.json").exists()


# ---------------------------------------------------------------- the held path on the real shapes

def synth_held_eps(seed, rng, pool):
    def rows(shape):
        return rng.choice(pool, size=shape).astype(np.int64)
    per_pair, sha = [], {}
    for (a, b, _), name in zip(R.PAIRS, R.PAIR_NAMES):
        n = R.N_PER_PAIR
        ep = AspectEpisodes(a, b, rows((n,)), rows((n, 13)), rows((n, 4)), rows((n, 4)), rows((n, 4)), rows((n, 4)))
        per_pair.append(ep)
        sha[name] = episodes_sha256(ep)
    n = len(R.PAIRS) * R.N_PER_PAIR
    return SimpleNamespace(seed=int(seed), per_pair=per_pair, pooled=concat_episodes(per_pair), sha=sha, n=n,
                           pair_index=np.repeat(np.arange(3, dtype=np.int64), R.N_PER_PAIR),
                           parity=np.arange(n, dtype=np.int64) % 2)


def test_held_path_real_shapes(tmp_path):
    """Held mode on the real shapes (308,723 x 512 features, 12,288 episodes per seed, 13 candidates, 4 pairs a side;
    members from a 61,744-row set; seed 42's record of 12,288 episodes): seed 52's hook gives all seven rows, each a
    per-anchor dict of 12,288 finite values; the planted reranker scores give R@1 = 1 on every episode; the parsing
    failures are the crafted ones; finish keeps MLLM (seed 52 is its seed) and drops the others, computed on one of
    the three seeds only."""
    rng = np.random.default_rng(52)
    pool = np.sort(rng.choice(R.N_ROWS, N_HELD_LIKE, replace=False)).astype(np.int64)
    out = tmp_path / "results"
    out.mkdir()
    eps = {}
    for s in R.HELD_SEEDS:
        e = synth_held_eps(s, rng, pool)
        E.save_episodes(out / f"held_episodes_seed{s}.npz", e)
        eps[s] = E.load_episodes(out / f"held_episodes_seed{s}.npz")
    img, txt = feature_world(pool, rng)
    ev = EvalInputs(img, txt)
    ctx = make_ctx(eps[52], img, txt, ev)
    del ev
    w = craft(tmp_path / "gpu", out, eps, False, rng, dts_seeds=[52], shards=3)
    w.shown[:] = planted(w.rr.perms, rng)
    for k, d in enumerate(w.rr_outs):
        with np.load(d / "scores.npz") as z:
            idx = z["episode_index"]
        with open(d / "scores.npz", "wb") as f:
            np.savez(f, episode_index=idx, scores_shown=w.shown[idx])
    write_json(out / XT.SOURCES_NAME, w.sources)
    env = SimpleNamespace()
    rows = XT.seed_rows(env, 52, ctx, eps[52], out, False)
    n = 3 * R.N_PER_PAIR
    assert list(rows) == list(XT.NAMES)
    for name, x in rows.items():
        assert x["pa"] is not None, (name, x["info"])
        assert all(x["pa"][m].shape == (n,) and np.isfinite(x["pa"][m]).all() for m in METRICS), name
    assert (rows["MLLM"]["pa"]["r1"] == 1).all() and (rows["DTS-CF"]["pa"]["gain"] == 0).all()
    assert rows["DTS"]["info"]["parsing_failures"] == failures_of(52, n)
    rows_lb, img_lb, txt_lb = w.feats["LB"]
    pa = rows["FT-LB"]["pa"]
    want = by_hand_cosine(rows_lb, img_lb, txt_lb, eps[52].pooled)
    assert same_hits(pa, want)
    info = XT.family_info(getattr(env, XT.STATE), "DTS")
    assert info["record"]["sha256"] == R.sha256_file(w.record)
    assert info["parsing_failures"] == {"52": failures_of(52, n)}
    assert info["setting"] == "W2 K8" and XT.family_info(getattr(env, XT.STATE), "MLLM")["seed"] == 52
    collected = RD.collect_external({}, 52, rows)
    ext = XT.finish(env, out, False, collected)                 # the hook saw seed 52 only: its state is not used
    assert sorted(ext["MLLM"]["pa"]) == [52] and "missing" not in ext["MLLM"]["info"]
    for name in ("DTS", "FT-LP", "FT-LB"):
        assert ext[name]["pa"] == {} and ext[name]["info"]["seeds_computed_but_dropped"] == [52], name
        assert ext[name]["info"]["missing"] == "not computed on every seed it is reported on"
    assert not hasattr(env, XT.STATE)


# ---------------------------------------------------------------- fix round (ticket 14's review)

@pytest.fixture(scope="module")
def held_res(tmp_path_factory, readers):
    """Ticket 13's held results folder (real shapes; its external_sources.json lists all four jobs as missing)."""
    return TD.build_dir(tmp_path_factory.mktemp("xheld") / "results", False, readers, 41)


def test_real_mode_requires_the_sources_file_and_its_four_entries(held_res, tmp_path, capsys, monkeypatch):
    """Item 2 (controller): in real (and reserve) mode an absent external_sources.json, or an absent entry, is refused
    with exit 4 before any bundle and nothing is written; a job not run is written {"missing": reason}. Smoke mode
    keeps an absent file or entry as a "missing" row."""
    full = TD.copy_dir(held_res, tmp_path, "full")
    with pytest.raises(TD.Reached):                                  # all four listed: past the refusals
        TD.run(held_res, full, bundle_fn=TD._boom)
    gone = TD.copy_dir(held_res, tmp_path, "gone")
    (gone / XT.SOURCES_NAME).unlink()
    code, text = TD.run(held_res, gone, capsys, bundle_fn=TD._boom)
    assert code == RD.EXIT_REFUSE and "external_sources.json is missing" in text, text
    assert sorted(p.name for p in gone.iterdir()) == sorted(p.name for p in full.iterdir() if p.name !=
                                                            XT.SOURCES_NAME)  # nothing written
    part = TD.copy_dir(held_res, tmp_path, "part")
    edit_json(part / XT.SOURCES_NAME, lambda s: s.pop("ft_lp"))
    code, text = TD.run(held_res, part, capsys, bundle_fn=TD._boom)
    assert code == RD.EXIT_REFUSE and "has no ft_lp entry" in text and not (part / "descriptive.json").exists()
    with pytest.raises(XT.SourcesRefused, match="is missing"):       # the hook refuses too (exit 4 in the runner)
        XT.seed_rows(SimpleNamespace(), 52, None, None, gone, False)
    with pytest.raises(RD.Refused, match="is missing"):
        RD.external_seed_rows(SimpleNamespace(), 52, None, None, None, gone, False)
    assert XT.require_sources(gone, True).present is False             # smoke: not refused
    mut = mutant(tmp_path, guard="sources_required")
    monkeypatch.setattr(RD, "XT", mut)
    with pytest.raises(TD.Reached):                                  # without the guard: on to the first bundle
        TD.run(held_res, gone, bundle_fn=TD._boom)
    with pytest.raises(TD.Reached):
        TD.run(held_res, part, bundle_fn=TD._boom)


class OtherEncoder(FakeEncoder):
    """The same vectors under another identity (another model commit or library version)."""
    meta = {"class": "fake", "batch_size": 1, "transformers": "another version"}


def test_dts_refuses_the_tuning_cache_as_its_cache(sm, tmp_path):
    """Item 1: the tuning stages' cache (beside the record, or results/dts_value_embeddings.npz), whose SHA-256 the
    record holds and which save() would rewrite, is not the descriptive cache; a copy elsewhere is."""
    tuning = sm.world.record.parent / XT.TUNING_EMB
    caught_then_not(tmp_path, "dts_tuning_cache",
                    lambda m: m.load_dts(dts_entry(sm, tmp_path, embeddings=str(tuning)), sm.fx.out, True),
                    "the tuning stages' value-embedding cache")
    with pytest.raises(AssertionError, match="the tuning stages' value-embedding cache"):
        XT.load_dts(dts_entry(sm, tmp_path, embeddings=str(R.RESULTS / XT.TUNING_EMB)), sm.fx.out, True)
    assert XT.TUNING_EMB == RDTS.EMBEDDINGS


def test_dts_encoder_must_be_the_tuning_records(sm, tmp_path, monkeypatch):
    """Item 1: before the first encode the encoder's identity must equal the record's "embedder"."""
    def scenario(m):
        m.encoder_factory = OtherEncoder
        return dts_one(m, sm, dts_entry(sm, tmp_path))
    monkeypatch.setattr(XT, "encoder_factory", OtherEncoder)
    caught_then_not(tmp_path, "dts_encoder_meta", scenario, "is not the tuning record's")


def test_dts_cache_must_be_made_by_the_tuning_records_encoder(sm, tmp_path):
    """Item 1: a cache whose meta names another encoder, even one holding every string the seed needs (so nothing
    would be encoded and its meta never compared), is refused when loaded."""
    path = tmp_path / "other_cache.npz"
    emb = DT.ValueEmbedder(path, encoder_factory=OtherEncoder)
    values = {v for (p, k), a in sm.world.listings.items() if k == K for v in DT.parse_listing(a, K)}
    emb.embed(sorted(values))
    emb.save()
    caught_then_not(tmp_path, "dts_cache_meta", lambda m: dts_one(m, sm, dts_entry(sm, tmp_path,
                                                                                   embeddings=str(path))),
                    "a cache made by the encoder")


def mllm_folders(sm, tmp_path, second_model):
    """Two shard folders of the job: the first with the probe's setting, the second with ``second_model``."""
    n = sm.eps[9001].n
    a = rr_with(sm, tmp_path, np.arange(0, 100), sm.world.shown[:100])
    b = write_rr_out(tmp_path / f"rr{next(_COUNT)}", sm.world.rr.shas, np.arange(100, n), sm.world.shown[100:],
                     **second_model)
    return mllm_entry(sm, [a, b])


def test_mllm_folders_must_agree_on_the_model(sm, tmp_path):
    """Item 3: the listed reranker folders agree on model_id, snapshot, max_pixels and instruction_sha256 (without the
    guard a second shard of another snapshot would pass, the first folder alone being compared with the setting)."""
    entry = mllm_folders(sm, tmp_path, {"snapshot": "f" * 40})
    caught_then_not(tmp_path, "mllm_same_model", lambda m: m.load_mllm(entry, sm.fx.out, True),
                    "the listed reranker folders differ in")
    entry = mllm_folders(sm, tmp_path, {"instruction_sha256": "5" * 64})
    with pytest.raises(AssertionError, match="the listed reranker folders differ in"):
        XT.load_mllm(entry, sm.fx.out, True)


def test_mllm_model_must_be_the_probes_recorded_setting(sm, tmp_path):
    """Item 3: and they equal the probe's recorded setting (dts_settings.json's model and max_pixels, the SHA-256 of
    mllm_reranker.INSTRUCTION)."""
    n = sm.eps[9001].n
    d = write_rr_out(tmp_path / "rr_px", sm.world.rr.shas, np.arange(n), sm.world.shown, max_pixels=512 * 28 * 28)
    caught_then_not(tmp_path, "mllm_model", lambda m: m.load_mllm(mllm_entry(sm, [d]), sm.fx.out, True),
                    "not the probe's recorded setting")
    assert XT.rerank_model() == RR_MODEL


def test_errors_while_computing_are_computation_errors(sm, tmp_path, monkeypatch):
    """Nit: a ValueError or TypeError while the rows are computed stops as a computation error (exit 5), not as an
    unreadable input (that one is shown by test_an_unreadable_listed_input_stops_with_exit_5)."""
    out = TD.copy_dir(sm.fx, tmp_path)
    write_json(out / XT.SOURCES_NAME, fresh_sources(sm, tmp_path))

    def boom(*a, **k):
        raise ValueError("planted")
    monkeypatch.setattr(XT, "lp_seed", boom)
    with pytest.raises(AssertionError, match="the computation failed: ValueError: planted"):
        XT.seed_rows(SimpleNamespace(), 9001, sm.ctx[9001], sm.eps[9001], out, True)
