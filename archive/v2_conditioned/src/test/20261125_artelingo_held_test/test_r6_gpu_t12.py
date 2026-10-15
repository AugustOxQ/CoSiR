"""Tests of ticket 12: the reported GPU baselines (FT-LB, FT-LoRA, in-context MLLM reranker): r6_gpu_inputs' rerank and FT
job builders, r6_gpu_t12 (their loaders), r6_ft_cache, r6_gpu_ft_features and r6_gpu_rerank (DECISION_RULE.md section 2,
section 8 item 3, section 10 item 2). Shown here:
- the candidate permutation equals the probe's formula (run_probe.py:157-158); un-permuting the scores of the shown
  order recovers candidate order; rerank_input.npz holds no unpermuted candidate array; the permutations stay in a
  CPU-only file;
- the reranker's prompts show pairs, query and candidates as the rule says, in the shown order (identity permutation
  inside the job), with a stubbed scorer; checkpoint every 50 episodes (a crash keeps the last checkpoint), resume,
  --prior, output files, --check-only --no-model; the 8B model is never loaded;
- r6_ft_cache on 20 real selection paintings gives bytes equal to the existing artelingo_clip224 cache;
- the FT feature script builds the model and loads LB and LoRA checkpoints on the CPU, encodes 4 real selection rows and
  matches the selection features stored in the clipft runs' features.npz;
- no label, aspect name, target or path in any job input (real smoke episodes, selection rows only), no metric and no
  src or r6_common import in the GPU scripts. No held row is read.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_gpu_t12.py
"""
import ast
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first: it puts MAIN's src in front and checks it)
import r6_episodes as E  # noqa: E402
import r6_ft_cache as C  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_ft_features as FF  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import r6_gpu_rerank as RR  # noqa: E402
import r6_gpu_t12 as T  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402

CHECKOUT = HERE.parents[2]
PY = sys.executable
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
MODULES = ("r6_gpu_t12.py", "r6_ft_cache.py", "r6_gpu_ft_features.py", "r6_gpu_rerank.py")
_COUNT = itertools.count()
SEED = 9001
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
SETTINGS, _ = G.load_settings()
SNAP = SETTINGS["model"]["snapshot"]
LB_JOB, LORA_JOB = "20261007-071151-65bb2f4", "20261007-071251-65bb2f4"
CLIPFT = R.MAIN / "res/cluster_jobs"
CKPT = {"LB": CLIPFT / LB_JOB / "code/outputs/clipft/LB_lr3e-5/best_params.pt",
        "LoRA": CLIPFT / LORA_JOB / "code/outputs/clipft/LoRA_lr1e-4/best_params.pt"}
REF = {"LB": CLIPFT / LB_JOB / "code/outputs/clipft/LB_lr3e-5/features.npz",
       "LoRA": CLIPFT / LORA_JOB / "code/outputs/clipft/LoRA_lr1e-4/features.npz"}
EXISTING_CACHE = Path("/data/SSD2/pre_extract/artelingo_clip224")
LABEL_KEYS = {"anchor", "candidates", "pair_index", "pair_names", "labels", "label", "emotion", "style", "genre",
              "art_style", "aspect", "painting", "paintings", "image_id", "target", "sample_id", "image_relpath"}


# ---------------------------------------------------------------- mutation helper (guard deleted on a copy)

def mutant(tmp_path, module, guard):
    """A copy of ``module`` with the `# guard:<guard>` statements replaced by `pass`."""
    src = (HERE / module).read_text()
    assert src.count(HERE_LINE) == 1, module
    lines = src.splitlines(keepends=True)
    hits = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Expr) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
    assert hits, f"no statement of {module} carries # guard:{guard}"
    for n in hits:
        indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
        lines[n.lineno - 1] = f"{indent}pass\n"
        for i in range(n.lineno, n.end_lineno):
            lines[i] = "\n"
    path = tmp_path / f"{Path(module).stem}_copy{next(_COUNT)}.py"
    path.write_text("".join(lines).replace(HERE_LINE, f"HERE = Path({str(HERE)!r})\n"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def test_every_marked_guard_has_a_mutation_test():
    marked = set()
    for m in MODULES:
        marked |= set(re.findall(r"# guard:(\w+)", (HERE / m).read_text()))
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    assert marked == tested, (marked - tested, tested - marked)


# ---------------------------------------------------------------- synthetic episodes and manifest

N_UNIVERSE = 600


def name_of(r):
    return f"{int(r):020x}.jpg"


def row_of_name(name):
    return int(Path(name).stem, 16)


def synthetic_eps(n=6, seed=52, rng_seed=1, n_rows=N_UNIVERSE):
    """Episodes of the real shapes: 13 distinct candidates, 4 pairs a side, distinct rows overall per episode."""
    rng = np.random.default_rng(rng_seed)
    anchor = np.empty(n, np.int64)
    cand = np.empty((n, 13), np.int64)
    pairs = {f: np.empty((n, 4), np.int64) for f in G.PAIR_FIELDS}
    for i in range(n):
        pick = rng.choice(n_rows, size=1 + 13 + 16, replace=False)
        anchor[i], cand[i] = pick[0], pick[1:14]
        for k, f in enumerate(G.PAIR_FIELDS):
            pairs[f][i] = pick[14 + 4 * k:18 + 4 * k]
    pooled = SimpleNamespace(anchor=anchor, candidates=cand, **pairs)
    return SimpleNamespace(seed=seed, pooled=pooled, n=n)


def write_manifest(path, rows):
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    np.savez_compressed(path, rows=rows, image_name=np.asarray([name_of(r) for r in rows], dtype=str),
                        caption=np.asarray([f"caption {r}" for r in rows], dtype=str))
    return G.Manifest(path)


def write_rerank_folder(base, eps, positions=None):
    """A rerank job folder from synthetic episodes (no annotations needed): -> (folder, arrays, perms)."""
    positions = np.arange(eps.n) if positions is None else np.asarray(positions)
    arrays, perms = I.rerank_arrays(eps, positions)
    rows = np.unique(np.concatenate([arrays[k].ravel() for k in ("query_row", "cand_shown", *G.PAIR_FIELDS)]))
    folder = base / f"job{next(_COUNT)}"
    folder.mkdir(parents=True)
    write_manifest(folder / "rows_manifest.npz", rows)
    np.savez(folder / "rerank_input.npz", **arrays)
    return folder, arrays, perms


def probe_perms(seed, n):
    """run_probe.py:157-158, verbatim."""
    return np.stack([np.random.default_rng([seed, i]).permuted(
        np.tile(np.arange(13), (2, 2, 1)), axis=-1) for i in range(n)])


def test_permutation_formula_equals_the_probes():
    for seed, n in ((52, 40), (9001, 7), (46, 3)):
        assert np.array_equal(I.rerank_permutations(seed, np.arange(n)), probe_perms(seed, n))
    pos = np.array([3, 17, 4095, 4096, 12287])           # a shard of the concatenated episodes keeps its global i
    full = probe_perms(52, 12288)
    assert np.array_equal(I.rerank_permutations(52, pos), full[pos])
    p = I.rerank_permutations(52, pos)
    assert p.shape == (5, 2, 2, 13) and (np.sort(p, axis=-1) == np.arange(13)).all()
    assert len({p[0, 0, 0].tobytes(), p[0, 0, 1].tobytes(), p[0, 1, 0].tobytes(), p[0, 1, 1].tobytes()}) == 4


def test_unpermuting_the_shown_scores_recovers_candidate_order():
    eps = synthetic_eps()
    arrays, perms = I.rerank_arrays(eps, np.arange(eps.n))
    rr = RR.load_reranker_module()
    cand = eps.pooled.candidates
    shown = arrays["cand_shown"]
    assert shown.shape == (6, 2, 2, 13) and shown.dtype == np.int64
    for i in range(eps.n):
        for c in range(2):
            for d in range(2):
                assert np.array_equal(shown[i, c, d], cand[i][perms[i, c, d]])          # letter j shows column perm[j]
                scores_shown = shown[i, c, d].astype(np.float32) * 0.5                   # a score that depends on the row
                back = rr.unpermute(scores_shown, perms[i, c, d])                        # the CPU un-permuting
                assert np.array_equal(back, cand[i].astype(np.float32) * 0.5)           # candidate (column) order


def test_rerank_input_holds_no_unpermuted_candidate_array(tmp_path):
    eps = synthetic_eps()
    arrays, _ = I.rerank_arrays(eps, np.arange(eps.n))
    assert set(arrays) == set(T.RERANK_KEYS)
    cand = eps.pooled.candidates
    shown = arrays["cand_shown"]
    assert not any(np.array_equal(shown[:, a, b], cand) for a in range(2) for b in range(2))
    assert all(v.shape != cand.shape or k == "cand_shown" or not np.array_equal(v, cand) for k, v in arrays.items())
    folder, _, _ = write_rerank_folder(tmp_path, eps)
    z = np.load(folder / "rerank_input.npz")
    assert set(z.files) == set(T.RERANK_KEYS) and not (set(z.files) & LABEL_KEYS)
    # the loader refuses an extra array, e.g. the candidates in column order, and a missing one
    np.savez(tmp_path / "extra.npz", **arrays, candidates=cand)
    with pytest.raises(AssertionError, match="keys"):
        T.load_rerank_input(tmp_path / "extra.npz")
    mut = mutant(tmp_path, "r6_gpu_t12.py", guard="rerank_keys")
    assert "candidates" not in mut.load_rerank_input(tmp_path / "extra.npz")          # guard deleted: it is let through
    np.savez(tmp_path / "bad.npz", **{**arrays, "cand_shown": arrays["cand_shown"][:, :, :, :12]})
    with pytest.raises(AssertionError, match="cand_shown"):
        T.load_rerank_input(tmp_path / "bad.npz")


def test_ft_rows_loader(tmp_path):
    rows = np.array([3, 9, 20], dtype=np.int64)
    man = write_manifest(tmp_path / "m.npz", rows)
    np.savez(tmp_path / "ok.npz", rows=rows)
    assert np.array_equal(T.load_ft_rows(tmp_path / "ok.npz", man), rows)
    np.savez(tmp_path / "extra.npz", rows=rows, labels=rows)
    with pytest.raises(AssertionError, match="keys"):
        T.load_ft_rows(tmp_path / "extra.npz")
    mut = mutant(tmp_path, "r6_gpu_t12.py", guard="ft_rows_keys")
    assert np.array_equal(mut.load_ft_rows(tmp_path / "extra.npz"), rows)             # guard deleted
    np.savez(tmp_path / "unsorted.npz", rows=rows[::-1].copy())
    with pytest.raises(AssertionError, match="sorted"):
        T.load_ft_rows(tmp_path / "unsorted.npz")
    np.savez(tmp_path / "foreign.npz", rows=np.array([3, 9, 21], dtype=np.int64))
    with pytest.raises(AssertionError, match="not in the manifest"):
        T.load_ft_rows(tmp_path / "foreign.npz", man)


# ---------------------------------------------------------------- the reranker's prompts, with a stubbed scorer

IMG_DIR = "/imgs"


def image_paths(messages):
    return [p["image"] for p in messages[0]["content"] if p["type"] == "image"]


def stub_score(messages):
    """The row id of each shown candidate, in the order shown (parsed from the prompt): a score that proves what was
    shown at letter j."""
    content, out, k = messages[0]["content"], [], 0
    texts = [p for p in content if p["type"] == "text"]
    start = [i for i, p in enumerate(content) if p["type"] == "text" and p["text"] == "Candidates:\n"][0]
    for p in content[start + 1:]:
        if p["type"] == "text" and (m := re.fullmatch(r"([A-M])\. caption (\d+)\n", p["text"])):
            out.append(float(m.group(2)))
        elif p["type"] == "image":
            out.append(float(row_of_name(Path(p["image"]).name)))
    assert len(out) == 13 and texts
    return np.asarray(out, dtype=np.float32)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("synth")
    eps = synthetic_eps(n=7)
    folder, arrays, perms = write_rerank_folder(base, eps)
    man = G.Manifest(folder / "rows_manifest.npz")
    inp = T.load_rerank_input(folder / "rerank_input.npz", man)
    return dict(base=base, eps=eps, folder=folder, arrays=arrays, perms=perms, man=man, inp=inp,
                rr=RR.load_reranker_module())


def test_prompts_follow_the_rule_in_the_shown_order(synth):
    inp, man, rr, eps = synth["inp"], synth["man"], synth["rr"], synth["eps"]
    for pos in (0, 3):
        for ci, cond in enumerate("ab"):
            own, other = ("a", "b") if cond == "a" else ("b", "a")
            for di, d in enumerate(("i2t", "t2i")):
                msgs = RR.episode_prompt(inp, man, pos, ci, di, IMG_DIR, rr.build_messages)
                text = "".join(p["text"] for p in msgs[0]["content"] if p["type"] == "text")
                imgs = [row_of_name(Path(x).name) for x in image_paths(msgs)]
                sup_i, con_i = inp[f"pairs_{own}_img"][pos], inp[f"pairs_{other}_img"][pos]
                sup_t, con_t = inp[f"pairs_{own}_txt"][pos], inp[f"pairs_{other}_txt"][pos]
                assert imgs[:4] == sup_i.tolist() and imgs[4:8] == con_i.tolist()
                for k in range(4):
                    assert f"caption: caption {sup_t[k]}\n" in text and f"caption: caption {con_t[k]}\n" in text
                shown = inp["cand_shown"][pos, ci, di].tolist()
                q = int(inp["query_row"][pos])
                assert "Example pairs:" in text and text.index("Example pairs:") < text.index("Counter-example pairs:")
                if d == "i2t":
                    assert imgs[8:] == [q] and f"{rr.LETTERS[0]}. caption {shown[0]}\n" in text
                    assert "Query:\n\n" in text                                     # an image, then a line break
                else:
                    assert imgs[8:] == shown and f"Query:\ncaption {q}\n" in text
                assert stub_score(msgs).tolist() == [float(x) for x in shown]      # letter j shows shown column j
                assert text.endswith("Answer:") and image_paths(msgs)[0].startswith(IMG_DIR)
    # same builder as the probe's QwenReranker: the instruction is its text
    assert "Pick the candidate that is alike" in rr.INSTRUCTION and eps.n == 7


def run_to_dir(synth, out, **kw):
    kw.setdefault("start", 0)
    kw.setdefault("stop", len(synth["inp"]["episode_index"]))
    return RR.run_episodes(synth["inp"], synth["man"], out, stub_score, synth["rr"].build_messages, IMG_DIR,
                           log=lambda *_: None, **kw)


def test_scores_are_the_shown_scores_and_unpermute_to_candidate_order(synth, tmp_path):
    res = run_to_dir(synth, tmp_path / "o")
    assert res["complete"] and res["n_scored"] == 7
    z = np.load(tmp_path / "o" / "scores.npz")
    assert sorted(z.files) == ["episode_index", "scores_shown"] and z["scores_shown"].dtype == np.float32
    assert z["scores_shown"].shape == (7, 2, 2, 13) and np.array_equal(z["episode_index"], np.arange(7))
    assert np.array_equal(z["scores_shown"], synth["arrays"]["cand_shown"].astype(np.float32))   # identity in the job
    rr, cand = synth["rr"], synth["eps"].pooled.candidates
    for i in range(7):
        for c in range(2):
            for d in range(2):
                back = rr.unpermute(z["scores_shown"][i, c, d], synth["perms"][i, c, d])
                assert np.array_equal(back, cand[i].astype(np.float32))


def test_checkpoint_every_50_episodes_resume_and_prior(synth, tmp_path):
    inp = synth["inp"]
    full = tmp_path / "full"
    run_to_dir(synth, full)
    want = np.load(full / "scores.npz")["scores_shown"]
    calls = {"n": 0}

    def crashing(messages):
        calls["n"] += 1
        if calls["n"] > 4 * 5:                                   # dies inside the sixth episode
            raise RuntimeError("node lost")
        return stub_score(messages)

    out = tmp_path / "crash"
    with pytest.raises(RuntimeError):
        RR.run_episodes(inp, synth["man"], out, crashing, synth["rr"].build_messages, IMG_DIR, 0, 7, every=2,
                        log=lambda *_: None)
    assert np.array_equal(np.load(out / "scores.npz")["episode_index"], np.arange(4))   # last checkpoint at episode 4
    mut = mutant(tmp_path, "r6_gpu_rerank.py", guard="rerank_checkpoint")
    calls["n"] = 0
    out2 = tmp_path / "crash_nockpt"
    with pytest.raises(RuntimeError):
        mut.run_episodes(inp, synth["man"], out2, crashing, synth["rr"].build_messages, IMG_DIR, 0, 7, every=2,
                         log=lambda *_: None)
    assert not (out2 / "scores.npz").exists()                  # guard deleted: nothing survives the crash
    seen = []

    def counting(messages):
        seen.append(1)
        return stub_score(messages)

    res = RR.run_episodes(inp, synth["man"], out, counting, synth["rr"].build_messages, IMG_DIR, 0, 7, every=2,
                          log=lambda *_: None)                  # resume: only the 3 missing episodes are scored
    assert res["complete"] and res["n_done_before"] == 4 and res["n_scored"] == 3 and len(seen) == 12
    assert np.array_equal(np.load(out / "scores.npz")["scores_shown"], want)
    # a rerun scores nothing; a shard writes only its episodes
    assert run_to_dir(synth, out)["n_scored"] == 0
    shard = tmp_path / "shard"
    run_to_dir(synth, shard, start=2, stop=5)
    assert np.load(shard / "scores.npz")["episode_index"].tolist() == [2, 3, 4]
    # --prior: a new folder copies the shard's episodes and runs the rest
    seen.clear()
    res = RR.run_episodes(inp, synth["man"], tmp_path / "next", counting, synth["rr"].build_messages, IMG_DIR, 0, 7,
                          prior=RR.load_scores(shard / "scores.npz"), log=lambda *_: None)
    assert res["n_prior"] == 3 and res["n_scored"] == 4 and len(seen) == 16
    assert np.array_equal(np.load(tmp_path / "next" / "scores.npz")["scores_shown"], want)


def test_a_folder_with_foreign_episodes_is_refused_and_the_guard_matters(synth, tmp_path):
    shard = tmp_path / "shard"
    run_to_dir(synth, shard, start=2, stop=5)
    with pytest.raises(AssertionError, match="does not plan"):
        run_to_dir(synth, shard, start=0, stop=2)
    mut = mutant(tmp_path, "r6_gpu_rerank.py", guard="planned")
    assert len(mut.load_scores(shard / "scores.npz", {0, 1})) == 3                    # guard deleted: accepted


def test_prior_needs_the_same_fingerprint(tmp_path):
    fp = {"job": "r6_gpu_rerank", "model_id": "m"}
    d = tmp_path / "earlier"
    G.begin_provenance(d, fp, {})
    save = RR.save_scores
    save(d, [3], np.zeros((1, 2, 2, 13), np.float32))
    assert set(RR.load_prior([d], fp, {3, 4})) == {3}
    with pytest.raises(AssertionError, match="not reusable"):
        RR.load_prior([d], dict(fp, model_id="other"), {3})
    mut = mutant(tmp_path, "r6_gpu_rerank.py", guard="prior_fingerprint")
    assert set(mut.load_prior([d], dict(fp, model_id="other"), {3})) == {3}           # guard deleted


def synthetic_cli_job(base, n=4, hub=True):
    eps = synthetic_eps(n=n, rng_seed=5)
    folder, arrays, _ = write_rerank_folder(base, eps)
    images = base / "images"
    images.mkdir(exist_ok=True)
    man = G.Manifest(folder / "rows_manifest.npz")
    for nm in RR.episode_image_names(T.load_rerank_input(folder / "rerank_input.npz", man), man, 0, n):
        (images / nm).write_bytes(b"")
    snap = G.snapshot_dir(base / "hub", SETTINGS["model"]["id"], SNAP)
    snap.mkdir(parents=True)
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for f in G.SNAPSHOT_FILES:
        (snap / f).write_text("{}")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": x
                                                                                  for i, x in enumerate(names)}}))
    for x in names:
        (snap / x).write_bytes(b"")
    (snap.parent.parent / "refs").mkdir()
    (snap.parent.parent / "refs/main").write_text(SNAP)
    return dict(folder=folder, images=images, hub=base / "hub")


def test_check_only_without_the_model_and_a_missing_image(tmp_path):
    j = synthetic_cli_job(tmp_path)
    args = ["--job-dir", str(j["folder"]), "--out", str(tmp_path / "out"), "--check-only", "--no-model",
            "--hub-cache", str(j["hub"]), "--image-dir", str(j["images"])]
    r = subprocess.run([PY, str(HERE / "r6_gpu_rerank.py"), *args], capture_output=True, text=True, env=ENV,
                       cwd=CHECKOUT, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "reranker inputs ok: seed 52, 4 episodes, range [0, 4)" in r.stdout
    assert "prompts built; stopping before the model" in r.stdout and not (tmp_path / "out").exists()
    assert not re.search(r"R@1|recall|accuracy", r.stdout, re.I)
    victim = sorted(j["images"].iterdir())[0]
    victim.unlink()
    bad = subprocess.run([PY, str(HERE / "r6_gpu_rerank.py"), *args], capture_output=True, text=True, env=ENV,
                         cwd=CHECKOUT, timeout=600)
    assert bad.returncode != 0 and "images missing under" in bad.stderr
    mut = mutant(tmp_path, "r6_gpu_rerank.py", guard="images")
    assert mut.main(args) == 0                                                         # guard deleted
    with pytest.raises(SystemExit):
        RR.parse_args(["--job-dir", "x", "--out", "y", "--no-model"])                  # --no-model needs --check-only


def test_the_8b_model_is_loaded_only_through_the_probes_reranker():
    src = (HERE / "r6_gpu_rerank.py").read_text()
    assert "QwenReranker(str(snap)" in src and "max_pixels" in src
    rr = RR.load_reranker_module()
    assert rr.QwenReranker.__init__.__defaults__[2] == 256 * 28 * 28 == SETTINGS["model"]["processor_kwargs"]["max_pixels"]


# ---------------------------------------------------------------- the FT cache and the FT feature script

def tiny_images(folder, rows, size=(61, 47)):
    from PIL import Image
    rng = np.random.default_rng(3)
    folder.mkdir(parents=True, exist_ok=True)
    for r in rows:
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(folder / name_of(r))


def test_cache_guards_on_synthetic_images(tmp_path):
    rows = [5, 9, 40]
    tiny_images(tmp_path / "img", rows)
    names = [name_of(r) for r in rows]
    rec = C.build_cache(tmp_path / "c", names, tmp_path / "img", workers=1, verbose=False)
    images, index, rec2 = C.load_cache(tmp_path / "c")
    assert images.shape == (3, 224, 224, 3) and images.dtype == np.uint8 and rec2 == rec
    assert index == {n: k for k, n in enumerate(names)} and rec["normalisation_check"]["max_abs_diff"] < 1e-5
    assert not any(re.search(r"/|wikiart", s, re.I) for s in index)                     # neutral names only
    assert np.array_equal(images[1], C.decode(tmp_path / "img" / names[1]))
    with pytest.raises(AssertionError, match="already exists"):
        C.build_cache(tmp_path / "c", names, tmp_path / "img", workers=1, verbose=False)
    mut = mutant(tmp_path, "r6_ft_cache.py", guard="no_overwrite")
    assert mut.build_cache(tmp_path / "c", names, tmp_path / "img", workers=1, verbose=False)["n_images"] == 3
    with pytest.raises(AssertionError, match="neutral"):
        C.build_cache(tmp_path / "c2", ["Impressionism_x.jpg"], tmp_path / "img", workers=1, verbose=False)
    mut = mutant(tmp_path, "r6_ft_cache.py", guard="neutral")
    with pytest.raises(AssertionError, match="missing"):                                # no longer stops at the name
        mut.build_cache(tmp_path / "c3", ["Impressionism_x.jpg"], tmp_path / "img", workers=1, verbose=False)
    # a damaged cache fails its SHA-256 check
    arr = np.load(tmp_path / "c" / C.IMAGES_FILE, mmap_mode="r+")
    arr[0, 0, 0, 0] ^= 1
    arr.flush()
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        C.load_cache(tmp_path / "c")
    mut = mutant(tmp_path, "r6_ft_cache.py", guard="verify")
    assert mut.load_cache(tmp_path / "c")[0].shape[0] == 3
    # images_for: from the cache, from the folder, and a name the cache lacks
    assert np.array_equal(FF.images_for([names[2]], None, tmp_path / "img")[0], C.decode(tmp_path / "img" / names[2]))
    with pytest.raises(AssertionError, match="not in the cache"):
        FF.images_for([name_of(7)], tmp_path / "c")
    mut = mutant(tmp_path, "r6_gpu_ft_features.py", guard="cache_names")
    with pytest.raises(KeyError):
        mut.images_for([name_of(7)], tmp_path / "c")


def test_checkpoint_identity_and_finite_guards(tmp_path):
    ok = {"variant": "LB", "lr": 3e-5, "epoch": 8, "params": {}}
    FF.check_checkpoint(ok, "LB")
    for bad, variant in ((dict(ok, epoch=7), "LB"), (dict(ok, lr=1e-5), "LB"), (ok, "LoRA")):
        with pytest.raises(AssertionError, match="expected"):
            FF.check_checkpoint(bad, variant)
    mut = mutant(tmp_path, "r6_gpu_ft_features.py", guard="ckpt_identity")
    mut.check_checkpoint(dict(ok, epoch=7), "LB")                                      # guard deleted
    rows = np.arange(3, dtype=np.int64)
    img = np.zeros((3, 512), np.float32)
    txt = img.copy()
    FF.write_features(tmp_path / "f.npz", rows, img, txt)
    z = np.load(tmp_path / "f.npz")
    assert z["rows"].dtype == np.int64 and z["img"].dtype == np.float32 and z["img"].shape == (3, 512)
    txt[1, 5] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        FF.write_features(tmp_path / "g.npz", rows, img, txt)
    mut = mutant(tmp_path, "r6_gpu_ft_features.py", guard="finite")
    mut.write_features(tmp_path / "g.npz", rows, img, txt)                             # guard deleted
    assert (tmp_path / "g.npz").exists() and not (tmp_path / "g.npz.part").exists()


# ---------------------------------------------------------------- real selection rows

class OnlyImageAndCaption(dict):
    """An annotation that refuses every field but image and caption."""

    def __getitem__(self, k):
        assert k in ("image", "caption"), f"annotation field {k!r} was read"
        return super().__getitem__(k)

    def get(self, k, default=None):
        assert k in ("image", "caption"), f"annotation field {k!r} was read"
        return super().get(k, default)

    def keys(self):
        raise AssertionError("annotation keys were listed")

    items = values = keys


class GuardedAnnotations(list):
    def __getitem__(self, i):
        return OnlyImageAndCaption(list.__getitem__(self, i))


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    keep = SimpleNamespace(sample_ids=data.sample_ids, paintings=data.paintings)
    del data
    index = AE.PaintingValueIndex(labels, split.groups)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    eps = E.build_seed(labels, split.groups, split.selection, index, vs, SEED, R.N_SMOKE)
    assert eps.sha == E.identity_targets()[SEED]["episodes_sha256"]
    base = tmp_path_factory.mktemp("real")
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    return dict(data=keep, selection=np.asarray(split.selection), eps=eps, base=base, annotations=annotations,
                staging=base / "images", labels_keys=None)


def selection_painting_rows(real, n):
    """One selection row for each of the first ``n`` distinct paintings (by position), ascending."""
    sel = np.sort(real["selection"])
    _, first = np.unique(np.asarray(real["data"].paintings)[sel], return_index=True)
    return np.sort(sel[np.sort(first)[:n]]).astype(np.int64)


def test_cache_bytes_equal_the_existing_cache_for_20_selection_paintings(real):
    if not EXISTING_CACHE.is_dir():
        pytest.skip("the existing artelingo_clip224 cache is not mounted")
    rows = selection_painting_rows(real, 20)
    job = real["base"] / "ft20"
    rec = I.write_ft_job(rows, real["data"].sample_ids, real["data"].paintings,
                         GuardedAnnotations(real["annotations"]), job, real["staging"])
    assert rec["n_rows"] == 20 and rec["n_images"] == 20
    man = G.Manifest(job / "rows_manifest.npz")
    names = C.image_names(man, T.load_ft_rows(job / "ft_rows.npz", man))
    assert len(names) == 20
    mine = real["base"] / "cache20"
    C.build_cache(mine, names, real["staging"], workers=2, verbose=False)
    images, index, _ = C.load_cache(mine)
    old_images = np.load(EXISTING_CACHE / "images_uint8.npy", mmap_mode="r")
    old = json.loads((EXISTING_CACHE / "paintings.json").read_text())["paintings"]
    sids = real["data"].sample_ids
    for r in rows.tolist():
        painting = str(real["data"].paintings[r])
        relpath = real["annotations"][int(sids[r])]["image"]
        assert old[painting]["image"] == relpath                                      # the same painting file
        want = np.asarray(old_images[old[painting]["index"]])
        got = np.asarray(images[index[I.image_name(relpath)]])
        assert got.dtype == np.uint8 and got.shape == (224, 224, 3) and got.tobytes() == want.tobytes()
    assert not any("/" in k for k in index)


def run_cli(args, timeout=1800):
    return subprocess.run([PY, str(HERE / "r6_gpu_ft_features.py"), *args], capture_output=True, text=True, env=ENV,
                          cwd=CHECKOUT, timeout=timeout)


def test_ft_features_build_and_load_on_the_cpu_and_match_the_clipft_features(real):
    missing = [str(p) for p in (*CKPT.values(), *REF.values()) if not p.is_file()]
    if missing:
        pytest.skip(f"clipft run files missing: {missing}")
    sel = np.sort(real["selection"])[:4]
    job = real["base"] / "ft4"
    I.write_ft_job(sel, real["data"].sample_ids, real["data"].paintings, GuardedAnnotations(real["annotations"]),
                   job, real["staging"])
    args = ["--job-dir", str(job), "--out", str(real["base"] / "ft4_out"), "--variants", "LB,LoRA", "--device", "cpu",
            "--image-dir", str(real["staging"]), "--check-only"]
    for v in ("LB", "LoRA"):
        args += ["--ckpt", f"{v}={CKPT[v]}", "--reference", f"{v}={REF[v]}"]
    r = run_cli(args)
    assert r.returncode == 0, r.stderr[-3000:]
    for v in ("LB", "LoRA"):
        line = next(x for x in r.stdout.splitlines() if x.startswith(f"check-only {v}:"))
        assert "4 rows encoded on cpu" in line
        nums = {k: float(x) for k, x in re.findall(r"(img_max_abs_diff|img_min_cos|txt_max_abs_diff|txt_min_cos) ([0-9.e+-]+)",
                                                    line)}
        assert len(nums) == 4, line
        # fp32 eval on this CPU against the node's GPU features: the same weights, the same cached pixels
        assert nums["img_min_cos"] > 0.9999 and nums["txt_min_cos"] > 0.9999, line
        assert nums["img_max_abs_diff"] < 5e-3 and nums["txt_max_abs_diff"] < 5e-3, line
    assert not (real["base"] / "ft4_out").exists()                                    # check-only writes nothing
    # a checkpoint of the wrong run is refused
    base = args[:args.index("--ckpt")]
    bad = run_cli(base + ["--ckpt", f"LB={CKPT['LoRA']}", "--ckpt", f"LoRA={CKPT['LoRA']}"])
    assert bad.returncode != 0 and "expected LB" in bad.stderr


def test_ft_features_full_path_writes_the_features_layout_on_the_cpu(real):
    missing = [str(p) for p in (*CKPT.values(), *REF.values()) if not p.is_file()]
    if missing:
        pytest.skip(f"clipft run files missing: {missing}")
    sel = np.sort(real["selection"])[:6]
    job = real["base"] / "ft6"
    I.write_ft_job(sel, real["data"].sample_ids, real["data"].paintings, GuardedAnnotations(real["annotations"]),
                   job, real["staging"])
    cache = real["base"] / "cache6"
    man = G.Manifest(job / "rows_manifest.npz")
    C.build_cache(cache, C.image_names(man), real["staging"], workers=2, verbose=False)
    out = real["base"] / "ft6_out"
    args = ["--job-dir", str(job), "--out", str(out), "--variants", "LB", "--device", "cpu", "--cache-dir", str(cache),
            "--ckpt", f"LB={CKPT['LB']}"]
    r = run_cli(args)
    assert r.returncode == 0, r.stderr[-3000:]
    z = np.load(out / "features_LB.npz")
    assert sorted(z.files) == ["img", "rows", "txt"] and np.array_equal(z["rows"], sel)
    assert z["img"].dtype == z["txt"].dtype == np.float32 and z["img"].shape == (6, 512)
    cmp = FF.compare_reference(REF["LB"], sel, z["img"], z["txt"])
    assert cmp["img_min_cos"] > 0.9999 and cmp["txt_min_cos"] > 0.9999
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["runs"][-1]["status"] == "complete" and "ckpt_sha256" in prov["fingerprint"]
    again = run_cli(args)                                                             # a rerun skips the finished variant
    assert again.returncode == 0 and "exists; skipped" in again.stdout
    # the cache path and the folder path give the same features
    out2 = real["base"] / "ft6_out2"
    r2 = run_cli(["--job-dir", str(job), "--out", str(out2), "--variants", "LB", "--device", "cpu",
                  "--image-dir", str(real["staging"]), "--ckpt", f"LB={CKPT['LB']}"])
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert np.array_equal(np.load(out2 / "features_LB.npz")["img"], z["img"])


def test_real_rerank_job_holds_row_ids_names_and_captions_only(real):
    eps = real["eps"]
    job = real["base"] / "rr_real"
    rec = I.write_rerank_job(eps, real["data"].sample_ids, real["data"].paintings,
                             GuardedAnnotations(real["annotations"]), job, first_per_pair=4, staging=real["staging"])
    assert rec["n_episodes"] == 12 and sorted(p.name for p in job.iterdir()) == [
        "images.txt", "job_record.json", "rerank_input.npz", "rows_manifest.npz"]
    man = G.Manifest(job / "rows_manifest.npz")
    inp = T.load_rerank_input(job / "rerank_input.npz", man)
    assert inp["seed"] == SEED and inp["cand_shown"].shape == (12, 2, 2, 13)
    z = np.load(job / "rerank_input.npz")
    assert set(z.files) == set(T.RERANK_KEYS) and not (set(z.files) & LABEL_KEYS)
    # the CPU-only permutations reproduce cand_shown from the episodes' candidates; they are not in the folder
    pp = I.perms_path(job)
    assert pp.is_file() and pp.parent == job.parent and not (job / pp.name).exists()
    p = np.load(pp)
    pos = inp["episode_index"]
    assert np.array_equal(p["episode_index"], pos) and int(p["seed"]) == SEED
    assert np.array_equal(p["perms"], probe_perms(SEED, eps.n)[pos])
    cand = eps.pooled.candidates[pos]
    for i in range(len(pos)):
        for c in range(2):
            for d in range(2):
                assert np.array_equal(inp["cand_shown"][i, c, d], cand[i][p["perms"][i, c, d]])
    assert np.array_equal(inp["query_row"], eps.pooled.anchor[pos])
    for f in G.PAIR_FIELDS:
        assert np.array_equal(inp[f], getattr(eps.pooled, f)[pos])
    # the folder is never overwritten (job, image map and permutations)
    with pytest.raises(AssertionError, match="exists"):
        I.write_rerank_job(eps, real["data"].sample_ids, real["data"].paintings, real["annotations"], job,
                           first_per_pair=4, staging=real["staging"])
    # nothing label-bearing in a shipped file: the sync script's own check
    spec = importlib.util.spec_from_file_location("das6_sync_r6", CHECKOUT / "scripts/das6_sync_r6.py")
    if Path("/root/.claude/skills/cluster-run").is_dir():
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        assert sync.forbidden_in_job(job, sync.style_folders()) == []
        assert sync.forbidden_in_job(real["base"] / "ft20", sync.style_folders()) == []
    sels = set(real["selection"].tolist())
    assert set(man.rows.tolist()) <= sels                                              # selection rows only, no held row
    # the sync plan accepts the job and its images
    r = subprocess.run(["/usr/bin/python3", "scripts/das6_sync_r6.py", "--node", "node402", "--job-dir", str(job),
                        "--images"], capture_output=True, text=True, env=ENV, cwd=CHECKOUT, timeout=600)
    if Path("/usr/bin/python3").is_file() and Path("/root/.claude/skills/cluster-run").is_dir():
        assert r.returncode == 0, r.stderr[-1500:]
        assert "rr_real" in r.stdout and "Plan only" in r.stdout


def test_real_rerank_prompts_have_the_real_shapes(real):
    job = real["base"] / "rr_real"
    man = G.Manifest(job / "rows_manifest.npz")
    inp = T.load_rerank_input(job / "rerank_input.npz", man)
    rr = RR.load_reranker_module()
    msgs = RR.episode_prompt(inp, man, 0, 0, 1, str(real["staging"]), rr.build_messages)      # t2i: 8 + 13 images
    assert len(image_paths(msgs)) == 21 and all(Path(x).is_symlink() for x in image_paths(msgs))
    msgs = RR.episode_prompt(inp, man, 0, 1, 0, str(real["staging"]), rr.build_messages)      # i2t: 8 + 1 images
    assert len(image_paths(msgs)) == 9
    assert RR.check_images(inp, man, 0, 12, real["staging"]) > 0


# ---------------------------------------------------------------- what the GPU scripts import and compute

def test_gpu_scripts_import_no_src_no_r6_common_and_compute_no_metric():
    for m in ("r6_gpu_rerank.py", "r6_ft_cache.py", "r6_gpu_t12.py", "r6_gpu_ft_features.py"):
        tree = ast.parse((HERE / m).read_text())
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods.append(n.module or "")
        assert not [x for x in mods if x.split(".")[0] in ("src", "r6_common", "r6_episodes", "r6_score", "r6_stats")], m
        text = (HERE / m).read_text()
        for bad in ("aspect_metrics", "run_probe", "recall", "argmax", "topk", "per_anchor", "summarize", "r_at_1"):
            assert bad not in text, (m, bad)
    ft = (HERE / "r6_gpu_ft_features.py").read_text()
    assert "import ft_train" in ft                                                     # allowed: the clipft training code
    for script in ("run_r6_ftfeat.sh", "run_r6_rerank.sh"):
        text = (CHECKOUT / "scripts" / script).read_text()
        assert "exec \"$PY\"" in text and "--check-only" in text and "set -euo pipefail" in text
        code = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
        assert not re.search(r"emotion|genre|art_style|label|wikiart|annotation", code, re.I), script
