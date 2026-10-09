"""Tests of the round-6 DTS GPU side (ticket 10) without a GPU and without real data: the settings file against the
rule's text, the verbaliser's message builder, the keyed jsonl outputs with checkpoint and resume, the listing job,
the greedy check, the snapshot check, the command lines and wrappers up to the model (--check-only --no-model), the
sync script's plan, and the absence of metric code.

Synthetic job inputs use the real shapes and dtypes: global row ids in [0, 308,723), int64 (n, 4) example pairs, a full
seed of 12,288 episodes or the tuning subset of 3,072 (the first 1,024 of each pair, episode_index 0..1023,
4096..5119, 8192..9215). Mutation tests load a copy of a module from tmp_path with one `# guard:<name>` statement
replaced by `pass`, and show that the guard's scenario then passes.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_gpu.py
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
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first: it puts MAIN's src in front and checks it)
import r6_gpu_common as G  # noqa: E402
import r6_gpu_listing as L  # noqa: E402
import r6_gpu_verbalise as V  # noqa: E402

CHECKOUT = HERE.parents[2]
PY = sys.executable
SYS_PY = "/usr/bin/python3"
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()
SETTINGS, SETTINGS_SHA = G.load_settings()
SNAP = SETTINGS["model"]["snapshot"]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
GPU_MODULES = ("r6_gpu_common.py", "r6_gpu_verbalise.py", "r6_gpu_listing.py")
GPU_SCRIPTS = ("scripts/run_r6_verbalise.sh", "scripts/run_r6_listing.sh", "scripts/das6_sync_r6.py")

# names a GPU job must never show the model (aspect names, ArtELingo's emotions, WikiArt's styles and genres)
ASPECT_WORDS = ["emotion", "emotions", "style", "styles", "art style", "art_style", "genre", "genres", "aspect"]
EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else"]
STYLES = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau_Modern", "Baroque",
          "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism", "Fauvism",
          "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
          "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo",
          "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e"]


def forbidden_words():
    from src.data.wikiart_genre import GENRE_NAMES
    names = ASPECT_WORDS + EMOTIONS + STYLES + list(GENRE_NAMES)
    return sorted({n.lower() for n in names} | {n.lower().replace("_", " ") for n in names})


FORBIDDEN = forbidden_words()


# ---------------------------------------------------------------- mutation helpers

def mutant_source(module, guard=None, replace=()):
    src = (HERE / module).read_text()
    assert src.count(HERE_LINE) == 1, module
    lines = src.splitlines(keepends=True)
    if guard is not None:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Expr) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, f"no statement of {module} carries # guard:{guard}"
        for n in hits:
            indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
            lines[n.lineno - 1] = f"{indent}pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
    src = "".join(lines).replace(HERE_LINE, f"HERE = Path({str(HERE)!r})\n")
    for old, new in replace:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def load_copy(tmp_path, module, guard=None, replace=()):
    """Import a copy of ``module`` with the `# guard:<guard>` statements replaced by `pass` and the ``replace``
    pairs (old, new) applied."""
    path = tmp_path / f"{Path(module).stem}_copy{next(_COUNT)}.py"
    path.write_text(mutant_source(module, guard, replace))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    return mod


def test_every_marked_guard_has_a_mutation_test():
    marked = set()
    for m in GPU_MODULES:
        marked |= set(re.findall(r"# guard:(\w+)", (HERE / m).read_text()))
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    assert marked == tested, (marked - tested, tested - marked)


# ---------------------------------------------------------------- the settings file against the rule (section 7)

def rule_text():
    return (HERE / "DECISION_RULE.md").read_text(encoding="utf-8")


def test_settings_copy_the_rule_verbatim():
    t = rule_text()
    flat = re.sub(r"\s+", " ", t)
    wordings = dict(re.findall(r'^\s*\| (W\d) \| "(.*)" \|$', t, re.M))
    assert sorted(wordings) == ["W1", "W2", "W3", "W4"]
    assert SETTINGS["verbaliser"]["wordings"] == wordings
    prompt = re.search(r'is asked: "(List K .*?)" \(K written as a number\)', flat).group(1)
    assert SETTINGS["listing"]["prompt"].format(K="K", phrase="<phrase>") == prompt
    assert L.listing_prompt("brushwork", 16, SETTINGS) == prompt.replace("List K", "List 16").replace("<phrase>",
                                                                                                    "brushwork")
    marker = re.search(r"A leading list marker \(`(.*?)`\)", flat).group(1)
    assert SETTINGS["listing"]["marker_regex"] == marker
    assert "Basis size K ∈ {8, 16}" in flat and SETTINGS["listing"]["K"] == [8, 16]
    assert "greedy, at most 32 new tokens" in flat and SETTINGS["generation"]["verbaliser"]["max_new_tokens"] == 32
    assert "greedy, at most 128 new tokens" in flat and SETTINGS["generation"]["listing"]["max_new_tokens"] == 128
    assert "Qwen3-VL-8B-Instruct" in flat and SETTINGS["model"]["id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert SETTINGS["model"]["processor_kwargs"] == {"max_pixels": 256 * 28 * 28}
    assert SETTINGS["rule_sha256"] == R.RULE_SHA256 == G.sha256_file(HERE / "DECISION_RULE.md")
    assert SETTINGS["stop"]["aff_hits_seed42"] == R.AFF_HITS_SEED42 and "greater than 9,406" in flat
    assert SETTINGS["stop"]["n_rankings_seed42"] == R.N_RANKINGS_SEED42
    assert SETTINGS["seed42_order"]["tuning_subset_first_per_pair"] == 1024 and "first 1,024 episodes" in flat
    assert SETTINGS["seed42_order"]["tuning_settings"] == [f"W{w} K{k}" for w in range(1, 5) for k in (8, 16)]
    assert SETTINGS["budget"]["hours"] == 24 and "24 hours of wall time from the first DTS commit" in flat
    assert SETTINGS["controls"]["DTS-N"]["phrase_of_target_aspect"] == {"emotion": "emotion", "style": "style",
                                                                       "genre": "genre"}
    assert SETTINGS["listing"]["min_values"] == 2 and "fewer than 2 values" in flat


def test_settings_pin_the_probes_snapshot_and_greedy_decoding():
    assert SNAP == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    probe_log = (R.TEST / "20261106_mllm_probe_8b/20261106_mllm_probe_8b_log.md")
    if probe_log.is_file():
        assert SNAP in probe_log.read_text()
    kw = SETTINGS["generation"]["generate_kwargs"]
    assert kw == {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None, "top_k": None}
    assert SETTINGS["generation"]["verbaliser"]["batch_size"] == 1
    assert SETTINGS["generation"]["listing"] == {"max_new_tokens": 128, "batch_size": 16, "padding_side": "left"}


def test_settings_file_is_in_the_module_shas():
    shas = R.r6_module_shas()
    rel = "src/test/20261125_artelingo_held_test/"
    for f in ("dts_settings.json", *GPU_MODULES, "r6_gpu_inputs.py"):
        assert shas[rel + f] == G.sha256_file(HERE / f)
    for f in GPU_SCRIPTS:
        assert shas[f] == G.sha256_file(CHECKOUT / f)


def settings_copy(tmp_path, edit):
    s = json.loads(G.SETTINGS_PATH.read_text())
    edit(s)
    p = tmp_path / f"settings{next(_COUNT)}.json"
    p.write_text(json.dumps(s))
    return p


def test_sampling_settings_fire_and_the_guard_matters(tmp_path):
    p = settings_copy(tmp_path, lambda s: s["generation"]["generate_kwargs"].update(do_sample=True, temperature=0.7))
    with pytest.raises(AssertionError, match="generation is not greedy"):
        G.load_settings(p)
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="settings_greedy")
    assert mut.load_settings(p)[0]["generation"]["generate_kwargs"]["do_sample"] is True


def test_unswapped_conditions_fire_and_the_guard_matters(tmp_path):
    p = settings_copy(tmp_path, lambda s: s["verbaliser"]["conditions"].update(b={"group_a": "pairs_a",
                                                                                  "group_b": "pairs_b"}))
    with pytest.raises(AssertionError, match="condition b must swap"):
        G.load_settings(p)
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="settings_swap")
    assert mut.load_settings(p)[0]["verbaliser"]["conditions"]["b"]["group_a"] == "pairs_a"


# ---------------------------------------------------------------- synthetic job inputs of the real shapes

def tuning_index():
    return np.concatenate([np.arange(k * R.N_PER_PAIR, k * R.N_PER_PAIR + 1024) for k in range(3)]).astype(np.int64)


def synthetic_input(episode_index, seed=52, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(episode_index)
    d = {"seed": np.int64(seed), "episode_index": np.asarray(episode_index, dtype=np.int64)}
    for f in G.PAIR_FIELDS:
        d[f] = rng.integers(0, R.N_ROWS, size=(n, G.NUM_PAIRS), dtype=np.int64)
    return d


def caption_of(r):
    return f"a caption written for row {r}"


def name_of(r):
    """A neutral image name per row (two rows of one painting would share it; here each row has its own)."""
    return hashlib.sha256(f"synthetic painting {r}".encode()).hexdigest()[:20] + ".jpg"


def synthetic_manifest(inp):
    rows = np.unique(np.concatenate([inp[f].ravel() for f in G.PAIR_FIELDS]))
    return {"rows": rows, "image_name": np.asarray([name_of(r) for r in rows.tolist()], dtype=str),
            "caption": np.asarray([caption_of(r) for r in rows.tolist()], dtype=str)}


def write_job(folder, inp, manifest=None, images=True):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = synthetic_manifest(inp) if manifest is None else manifest
    np.savez_compressed(folder / "rows_manifest.npz", **manifest)
    np.savez(folder / "verbalise_input.npz", **inp)
    if images:
        (folder / "images.txt").write_text("\n".join(sorted(set(manifest["image_name"].tolist()))) + "\n")
    return folder


@pytest.fixture(scope="module")
def tuning_job(tmp_path_factory):
    """The tuning subset's shape: 3,072 episodes (the first 1,024 of each pair), seed 52."""
    inp = synthetic_input(tuning_index())
    job = write_job(tmp_path_factory.mktemp("job") / "tune", inp)
    man = G.Manifest(job / "rows_manifest.npz")
    return dict(job=job, man=man, inp=G.load_verbalise_input(job / "verbalise_input.npz", man))


@pytest.fixture(scope="module")
def full_job(tmp_path_factory):
    """A full seed: 12,288 episodes, seed 52."""
    inp = synthetic_input(np.arange(3 * R.N_PER_PAIR), rng_seed=1)
    job = write_job(tmp_path_factory.mktemp("job") / "full", inp)
    man = G.Manifest(job / "rows_manifest.npz")
    return dict(job=job, man=man, inp=G.load_verbalise_input(job / "verbalise_input.npz", man))


ROOT = Path("/wikiart_root_for_tests")


# ---------------------------------------------------------------- the message builder (rule section 7 item 1)

def expected_content(inp, pos, condition, wording_id):
    first, second = ("pairs_a", "pairs_b") if condition == "a" else ("pairs_b", "pairs_a")
    parts = []
    for name, g in (("A", first), ("B", second)):
        parts.append({"type": "text", "text": f"Group {name}:\n"})
        for k in range(4):
            r_img, r_txt = int(inp[f"{g}_img"][pos, k]), int(inp[f"{g}_txt"][pos, k])
            parts += [{"type": "text", "text": f"Pair {k + 1}: image"},
                      {"type": "image", "image": str(ROOT / name_of(r_img))},
                      {"type": "text", "text": f"\ncaption: {caption_of(r_txt)}\n"}]
    parts.append({"type": "text", "text": "\n" + SETTINGS["verbaliser"]["wordings"][wording_id]})
    return parts


@pytest.mark.parametrize("pos", [0, 1, 1023, 1024, 3071])
def test_message_order_swap_and_wording_last(full_job, tuning_job, pos):
    for job in (full_job, tuning_job):
        inp, man = job["inp"], job["man"]
        for w in G.WORDING_IDS:
            for c in G.CONDITIONS:
                msgs = V.episode_messages(inp, man, pos, c, w, SETTINGS, ROOT)
                assert len(msgs) == 1 and msgs[0]["role"] == "user" and set(msgs[0]) == {"role", "content"}
                content = msgs[0]["content"]
                assert content == expected_content(inp, pos, c, w)
                assert len(content) == 2 * (1 + 4 * 3) + 1 and sum(p["type"] == "image" for p in content) == 8
                assert content[-1]["text"].endswith(SETTINGS["verbaliser"]["wordings"][w])
        a = V.episode_messages(inp, man, pos, "a", "W1", SETTINGS, ROOT)[0]["content"]
        b = V.episode_messages(inp, man, pos, "b", "W1", SETTINGS, ROOT)[0]["content"]
        assert a[1:13] == b[14:26] and a[14:26] == b[1:13] and a[-1] == b[-1]     # condition b swaps the groups


def test_pairs_keep_episode_column_order_and_cross_item_rows(tuning_job):
    inp, man = tuning_job["inp"], tuning_job["man"]
    content = V.episode_messages(inp, man, 7, "a", "W2", SETTINGS, ROOT)[0]["content"]
    imgs = [p["image"] for p in content if p["type"] == "image"]
    caps = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("\ncaption: ")]
    want_img = [str(ROOT / name_of(int(r))) for r in (*inp["pairs_a_img"][7], *inp["pairs_b_img"][7])]
    want_cap = [f"\ncaption: {caption_of(int(r))}\n" for r in (*inp["pairs_a_txt"][7], *inp["pairs_b_txt"][7])]
    assert imgs == want_img and caps == want_cap


def test_no_aspect_name_label_or_path_in_any_message_text(full_job):
    inp, man = full_job["inp"], full_job["man"]
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN) + r")\b", re.I)
    rng = np.random.default_rng(3)
    for pos in rng.choice(len(inp["episode_index"]), 64, replace=False).tolist():
        for w in G.WORDING_IDS:
            for c in G.CONDITIONS:
                for p in V.episode_messages(inp, man, pos, c, w, SETTINGS, ROOT)[0]["content"]:
                    if p["type"] == "text":
                        assert not pattern.search(p["text"]), (p["text"], pattern.search(p["text"]).group(0))
                        assert ".jpg" not in p["text"] and str(ROOT) not in p["text"]
                    else:                                     # images are loaded by their neutral name only
                        assert set(p) == {"type", "image"} and p["type"] == "image"
                        assert re.fullmatch(re.escape(str(ROOT)) + r"/[0-9a-f]{20}\.jpg", p["image"]), p["image"]
    for w in SETTINGS["verbaliser"]["wordings"].values():
        assert not pattern.search(w)


def test_bad_condition_and_wrong_pair_count_raise(tuning_job):
    with pytest.raises(ValueError, match="not a or b"):
        V.group_rows(tuning_job["inp"], 0, "c", SETTINGS)
    with pytest.raises(AssertionError, match="Group A has 3 pairs"):
        V.build_messages([("x", "y")] * 3, [("x", "y")] * 4, "w", SETTINGS)


# ---------------------------------------------------------------- job input files

def test_manifest_with_an_extra_key_fires_and_the_guard_matters(tuning_job, tmp_path):
    man = synthetic_manifest(synthetic_input(tuning_index()))
    np.savez(tmp_path / "m.npz", **man, emotion=np.zeros(len(man["rows"]), dtype=np.int64))
    with pytest.raises(AssertionError, match="keys"):
        G.Manifest(tmp_path / "m.npz")
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="manifest_keys")
    assert len(mut.Manifest(tmp_path / "m.npz").rows) == len(man["rows"])


def test_manifest_with_a_wikiart_path_fires_and_the_guard_matters(tmp_path):
    man = synthetic_manifest(synthetic_input(tuning_index()))
    man["image_name"] = man["image_name"].astype("<U64")
    man["image_name"][3] = "Impressionism/claude-monet_water-lilies.jpg"
    np.savez(tmp_path / "m.npz", **man)
    with pytest.raises(AssertionError, match="must be neutral"):
        G.Manifest(tmp_path / "m.npz")
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="neutral_names")
    assert mut.Manifest(tmp_path / "m.npz").image_name[3].startswith("Impressionism/")   # guard deleted: path kept


def test_row_missing_from_the_manifest_fires_and_the_guard_matters(tuning_job, tmp_path):
    man = tuning_job["man"]
    absent = int(np.setdiff1d(np.arange(R.N_ROWS), man.rows)[5])
    with pytest.raises(AssertionError, match="not in the manifest"):
        man.index([int(man.rows[0]), absent])
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="manifest_rows")
    got = mut.Manifest(tuning_job["job"] / "rows_manifest.npz").index([int(man.rows[0]), absent])
    assert int(man.rows[got[1]]) != absent                       # guard deleted: the wrong row's item is used


def test_input_with_episode_fields_fires_and_the_guard_matters(tmp_path):
    inp = synthetic_input(tuning_index())
    np.savez(tmp_path / "v.npz", **inp, candidates=np.zeros((len(inp["episode_index"]), 13), dtype=np.int64))
    with pytest.raises(AssertionError, match="keys"):
        G.load_verbalise_input(tmp_path / "v.npz")
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="input_keys")
    assert mut.load_verbalise_input(tmp_path / "v.npz")["seed"] == 52


@pytest.mark.parametrize("bad, match", [
    (lambda d: d.update(episode_index=d["episode_index"][::-1].copy()), "increasing"),
    (lambda d: d.update(pairs_a_img=d["pairs_a_img"].astype(np.int32)), "int64"),
    (lambda d: d.update(pairs_b_txt=d["pairs_b_txt"][:, :3].copy()), r"\(n, 4\)"),
])
def test_malformed_input_raises(tmp_path, bad, match):
    inp = synthetic_input(tuning_index())
    bad(inp)
    np.savez(tmp_path / "v.npz", **inp)
    with pytest.raises(AssertionError, match=match):
        G.load_verbalise_input(tmp_path / "v.npz")


# ---------------------------------------------------------------- keyed jsonl, checkpoints and resume

def fake_answer(messages) -> str:
    return "phrase " + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16] + "\nmore"


class FakeModel:
    def __init__(self, crash_at=None):
        self.crash_at, self.n = crash_at, 0

    def __call__(self, messages):
        self.n += 1
        if self.crash_at is not None and self.n == self.crash_at:
            raise RuntimeError("simulated crash")
        return fake_answer(messages)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run_v(job, out, model, start, stop, wordings=("W1", "W3"), **kw):
    return V.run_calls(job["inp"], job["man"], SETTINGS, out, list(wordings), start, stop, model, ROOT,
                       log=lambda *_: None, **kw)


def check_outputs(job, out, start, stop, wordings=("W1", "W3")):
    inp = job["inp"]
    for w in wordings:
        recs = read_jsonl(out / f"phrases_{w}.jsonl")
        keys = [(r["seed"], r["episode_index"], r["condition"], r["wording"]) for r in recs]
        want = {(52, int(inp["episode_index"][p]), c, w) for p in range(start, stop) for c in "ab"}
        assert len(keys) == len(set(keys)) == len(want) and set(keys) == want     # nothing repeated, nothing skipped
        pos = {int(e): p for p, e in enumerate(inp["episode_index"].tolist())}
        for r in recs:
            assert list(r) == ["seed", "episode_index", "condition", "wording", "answer"]
            assert type(r["seed"]) is int and type(r["episode_index"]) is int and isinstance(r["answer"], str)
            msgs = V.episode_messages(inp, job["man"], pos[r["episode_index"]], r["condition"], w, SETTINGS, ROOT)
            assert r["answer"] == fake_answer(msgs)                              # the answer sits under its own key


def test_stopped_after_50_calls_resumes_without_repeating_or_skipping(tuning_job, tmp_path):
    start, stop = 1000, 1040                    # crosses the pair boundary (episode 1023 -> 4096)
    m1 = FakeModel()
    res1 = run_v(tuning_job, tmp_path, m1, start, stop, max_calls=50)
    assert m1.n == 50 and res1["n_called"] == 50 and not res1["complete"]
    assert sum(len(read_jsonl(tmp_path / f"phrases_{w}.jsonl")) for w in ("W1", "W3")) == 50
    m2 = FakeModel()
    res2 = run_v(tuning_job, tmp_path, m2, start, stop)
    assert res2["n_done_before"] == 50 and m2.n == 160 - 50 and res2["complete"]
    check_outputs(tuning_job, tmp_path, start, stop)
    m3 = FakeModel()
    assert run_v(tuning_job, tmp_path, m3, start, stop)["n_called"] == 0 and m3.n == 0


def test_a_crash_between_checkpoints_loses_only_the_unsaved_calls(tuning_job, tmp_path):
    start, stop = 1000, 1040
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_v(tuning_job, tmp_path, FakeModel(crash_at=76), start, stop)             # calls 1..75 returned
    assert sum(len(read_jsonl(tmp_path / f"phrases_{w}.jsonl")) for w in ("W1", "W3")) == 50
    m2 = FakeModel()
    assert run_v(tuning_job, tmp_path, m2, start, stop)["complete"] and m2.n == 110
    check_outputs(tuning_job, tmp_path, start, stop)


def test_without_checkpoints_a_crash_loses_everything(tuning_job, tmp_path):
    mut = load_copy(tmp_path, "r6_gpu_verbalise.py", guard="verbalise_checkpoint")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="simulated crash"):
        mut.run_calls(tuning_job["inp"], tuning_job["man"], SETTINGS, out, ["W1", "W3"], 1000, 1040,
                      FakeModel(crash_at=76), ROOT, log=lambda *_: None)
    assert not (out / "phrases_W1.jsonl").exists()         # guard deleted: the 75 answers are lost, 75 calls repeat


def test_jsonl_with_a_duplicate_fires_and_the_guard_matters(tuning_job, tmp_path):
    run_v(tuning_job, tmp_path, FakeModel(), 0, 3, wordings=("W2",))
    p = tmp_path / "phrases_W2.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + lines[:1]) + "\n")
    with pytest.raises(AssertionError, match="duplicate record"):
        run_v(tuning_job, tmp_path, FakeModel(), 0, 3, wordings=("W2",))
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="jsonl_unique")
    st = mut.KeyedJsonl(p, V.FIELDS, V.KEY_FIELDS)
    assert st.load() == 7                                          # guard deleted: 6 calls stored as 7 records


def test_record_of_another_shard_fires_and_the_guard_matters(tuning_job, tmp_path):
    run_v(tuning_job, tmp_path, FakeModel(), 10, 12, wordings=("W2",))
    with pytest.raises(AssertionError, match="not one this run plans"):
        run_v(tuning_job, tmp_path, FakeModel(), 0, 3, wordings=("W2",))
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="jsonl_planned")
    st = mut.KeyedJsonl(tmp_path / "phrases_W2.jsonl", V.FIELDS, V.KEY_FIELDS)
    assert st.load(allowed={(52, 0, "a", "W2")}) == 4               # guard deleted: foreign records accepted


def test_incomplete_or_malformed_records_raise(tuning_job, tmp_path):
    run_v(tuning_job, tmp_path, FakeModel(), 0, 2, wordings=("W4",))
    p = tmp_path / "phrases_W4.jsonl"
    good = p.read_text()
    p.write_text(good[:-5])
    with pytest.raises(AssertionError, match="incomplete"):
        G.KeyedJsonl(p, V.FIELDS, V.KEY_FIELDS).load()
    rec = json.loads(good.splitlines()[0])
    rec["label"] = "x"
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(AssertionError, match="record fields"):
        G.KeyedJsonl(p, V.FIELDS, V.KEY_FIELDS).load()


def test_provenance_of_other_inputs_fires_and_the_guard_matters(tmp_path):
    fp = {"job": "r6_gpu_verbalise", "settings_sha256": SETTINGS_SHA, "inputs_sha256": {"a": "1"}}
    G.begin_provenance(tmp_path, fp, {"args": {}})
    G.begin_provenance(tmp_path, fp, {"args": {}})                                 # the same fingerprint resumes
    assert len(json.loads((tmp_path / "provenance.json").read_text())["runs"]) == 2
    other = dict(fp, inputs_sha256={"a": "2"})
    with pytest.raises(AssertionError, match="inputs_sha256"):
        G.begin_provenance(tmp_path, other, {"args": {}})
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="fingerprint")
    assert len(mut.begin_provenance(tmp_path, other, {"args": {}})["runs"]) == 3


# ---------------------------------------------------------------- answers with Unicode line separators

def test_answers_with_unicode_line_separators_resume(tuning_job, tmp_path):
    odd = ["a\u2028b", "c\u2029d", "e\x85f", "g\x0bh", "i\x1cj", "k\x0cl", "\u00e9t\u00e9"]

    class OddModel(FakeModel):
        def __call__(self, messages):
            return super().__call__(messages) + odd[self.n % len(odd)]

    run_v(tuning_job, tmp_path, OddModel(), 0, 10, max_calls=20)
    m2 = OddModel()
    res = run_v(tuning_job, tmp_path, m2, 0, 10)
    assert res["n_done_before"] == 20 and m2.n == 40 - 20 and res["complete"]
    for w in ("W1", "W3"):
        raw = (tmp_path / f"phrases_{w}.jsonl").read_bytes()
        assert raw.isascii() and raw.count(b"\n") == 20
        st = G.KeyedJsonl(tmp_path / f"phrases_{w}.jsonl", V.FIELDS, V.KEY_FIELDS)
        assert st.load() == 20 and sum(any(o in r["answer"] for o in odd) for r in st.records()) == 20


def test_splitlines_and_raw_unicode_would_break_resume(tuning_job, tmp_path):
    mut = load_copy(tmp_path, "r6_gpu_common.py", replace=(
        ('json.dumps(rec, ensure_ascii=True)', 'json.dumps(rec, ensure_ascii=False)'),
        ('for line in text.split("\\n")[:-1]:', 'for line in text.splitlines():')))
    st = mut.KeyedJsonl(tmp_path / "x.jsonl", V.FIELDS, V.KEY_FIELDS)
    st.add({"seed": 52, "episode_index": 0, "condition": "a", "wording": "W1", "answer": "light\u2028and shade"})
    st.save()
    with pytest.raises(json.JSONDecodeError):
        mut.KeyedJsonl(tmp_path / "x.jsonl", V.FIELDS, V.KEY_FIELDS).load()


# ---------------------------------------------------------------- --prior: an earlier launch's outputs (DAS6 resume)

def verbaliser_fingerprint(job, **change):
    fp = {"job": "r6_gpu_verbalise", "model_id": SETTINGS["model"]["id"], "snapshot": SNAP,
          "settings_sha256": SETTINGS_SHA,
          "scripts_sha256": G.script_shas(HERE / "r6_gpu_verbalise.py", HERE / "r6_gpu_common.py"),
          "inputs_sha256": {f: G.sha256_file(job["job"] / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}}
    fp.update(change)
    return fp


def test_prior_outputs_are_copied_not_recomputed_and_left_unchanged(tuning_job, tmp_path):
    fp = verbaliser_fingerprint(tuning_job)
    first = tmp_path / "launch1"
    G.begin_provenance(first, fp, {"args": {}})
    with pytest.raises(RuntimeError, match="simulated crash"):            # the first launch died after 75 calls
        run_v(tuning_job, first, FakeModel(crash_at=76), 100, 140)
    before = {f.name: G.sha256_file(f) for f in first.iterdir()}
    prior = V.load_prior([first], fp, ["W1", "W3"])
    m = FakeModel()
    res = run_v(tuning_job, tmp_path / "launch2", m, 100, 140, prior=prior)
    assert res["n_prior"] == 50 and m.n == 160 - 50 and res["complete"]  # 50 were checkpointed by the first launch
    check_outputs(tuning_job, tmp_path / "launch2", 100, 140)
    assert {f.name: G.sha256_file(f) for f in first.iterdir()} == before   # read only


def test_prior_of_another_fingerprint_fires_and_the_guard_matters(tuning_job, tmp_path):
    first = tmp_path / "launch1"
    G.begin_provenance(first, verbaliser_fingerprint(tuning_job, settings_sha256="0" * 64), {"args": {}})
    run_v(tuning_job, first, FakeModel(), 0, 4)
    with pytest.raises(AssertionError, match="not reusable"):
        V.load_prior([first], verbaliser_fingerprint(tuning_job), ["W1", "W3"])
    mut = load_copy(tmp_path, "r6_gpu_verbalise.py", guard="prior_fingerprint")
    assert len(mut.load_prior([first], verbaliser_fingerprint(tuning_job), ["W1", "W3"])["W1"]) == 8


def test_priors_that_disagree_raise(tuning_job, tmp_path):
    fp = verbaliser_fingerprint(tuning_job)
    for name, model in (("a", FakeModel()), ("b", lambda msgs: "another answer")):
        G.begin_provenance(tmp_path / name, fp, {"args": {}})
        run_v(tuning_job, tmp_path / name, model, 0, 2)
    with pytest.raises(AssertionError, match="disagree"):
        V.load_prior([tmp_path / "a", tmp_path / "b"], fp, ["W1", "W3"])


# ---------------------------------------------------------------- generation through stub model and processor

EOS, PAD, VOCAB = (2, 3), 3, 50


class BatchLike(dict):
    def to(self, device):
        return self


class StubTokenizer:
    """Prompt ids 10..39 (length from the text), pads on its padding_side; new tokens decode as w<id>."""

    def __init__(self):
        self.padding_side, self.calls = "right", []

    @staticmethod
    def encode(text):
        return [10 + ord(ch) % 30 for ch in text[:4 + len(text) % 11]]

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
        import torch
        self.calls.append({"padding_side": self.padding_side, "padding": padding,
                           "add_special_tokens": add_special_tokens, "return_tensors": return_tensors})
        seqs = [self.encode(t) for t in texts]
        n = max(len(x) for x in seqs)
        ids, mask = [], []
        for x in seqs:
            fill = [PAD] * (n - len(x))
            ids.append(fill + x if self.padding_side == "left" else x + fill)
            ones = [1] * len(x)
            mask.append([0] * len(fill) + ones if self.padding_side == "left" else ones + [0] * len(fill))
        return BatchLike(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in (*EOS, PAD)))


class StubProcessor:
    def __init__(self):
        self.tokenizer, self.calls = StubTokenizer(), []

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, return_dict=False,
                            return_tensors=None):
        import torch
        self.calls.append({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt,
                           "return_dict": return_dict, "return_tensors": return_tensors})
        text = "|".join(p.get("text", "<image>") for m in conv for p in m["content"])
        if not tokenize:
            return text
        ids = self.tokenizer.encode(text)
        return BatchLike(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))


class StubModel:
    """generate(): greedy decoding of fixed logits (row b, step t: token 40 + (t + b) % 9, end-of-sequence at step
    stops[b]), through the given logits processors; records its arguments. With ``off_step`` row 0 takes a token
    that is not the argmax at that step (what sampling would do)."""
    device = "cpu"

    def __init__(self, stops=(10_000,), off_step=None):
        self.generation_config = SimpleNamespace(eos_token_id=list(EOS), pad_token_id=PAD)
        self.stops, self.off_step, self.calls = stops, off_step, []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, logits_processor=None, **kwargs):
        import torch
        self.calls.append(dict(kwargs, max_new_tokens=max_new_tokens, input_ids=input_ids.clone(),
                               attention_mask=attention_mask.clone()))
        b_n = input_ids.shape[0]
        seq, done = input_ids.clone(), torch.zeros(b_n, dtype=torch.bool)
        for t in range(max_new_tokens):
            logits = torch.zeros(b_n, VOCAB)
            for b in range(b_n):
                logits[b, EOS[0] if t == self.stops[b % len(self.stops)] else 40 + (t + b) % 9] = 1.0
            for proc in logits_processor or []:
                logits = proc(seq, logits)
            nxt = logits.argmax(-1)
            if self.off_step == t:
                nxt[0] = 49
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= torch.isin(nxt, torch.tensor(EOS))
            if bool(done.all()):
                break
        return seq


def words(b, n):
    return " ".join(f"w{40 + (t + b) % 9}" for t in range(n))


def greedy_kwargs_ok(call):
    return (call.get("do_sample") is False and call.get("num_beams") == 1
            and all(k in call and call[k] is None for k in ("temperature", "top_p", "top_k")))


def check_verbaliser_generation(vmod):
    """make_generate on a stub: greedy kwargs, 32 new tokens at most, only the new tokens decoded."""
    model, proc = StubModel(), StubProcessor()
    msgs = V.build_messages([("i.jpg", "c")] * 4, [("j.jpg", "d")] * 4, "a wording", SETTINGS)
    text = vmod.make_generate(proc, model, SETTINGS)(msgs)
    call = model.calls[0]
    assert greedy_kwargs_ok(call), call
    assert call["max_new_tokens"] == 32 and text == words(0, 32)            # the limit applied; prompt not decoded
    assert proc.calls == [{"tokenize": True, "add_generation_prompt": True, "return_dict": True,
                           "return_tensors": "pt"}]
    model, proc = StubModel(stops=(5,)), StubProcessor()
    assert vmod.make_generate(proc, model, SETTINGS)(msgs) == words(0, 5)    # stops at end-of-sequence


def check_listing_generation(lmod):
    """make_generate_batch on a stub: greedy kwargs, 128 new tokens at most, left padding, one text per row."""
    model, proc = StubModel(stops=(5, 9, 10_000)), StubProcessor()
    convs = [L.listing_messages(p, k, SETTINGS) for p, k in (("light", 8), ("brushwork and texture", 16), ("x", 8))]
    texts = lmod.make_generate_batch(proc, model, SETTINGS)(convs)
    call = model.calls[0]
    assert greedy_kwargs_ok(call), call
    assert call["max_new_tokens"] == 128
    tc = proc.tokenizer.calls[0]
    assert tc["padding_side"] == "left" and tc["padding"] is True and tc["add_special_tokens"] is False
    mask = call["attention_mask"]
    assert bool((mask[:, -1] == 1).all()) and bool((mask[:, 0] == 0).any())   # left padding reached generate
    assert texts == [words(0, 5), words(1, 9), words(2, 128)]
    assert all(c["tokenize"] is False and c["add_generation_prompt"] is True for c in proc.calls)


def test_generation_settings_reach_generate():
    check_verbaliser_generation(V)
    check_listing_generation(L)


def test_a_non_greedy_token_from_generate_raises():
    model, proc = StubModel(off_step=2), StubProcessor()
    msgs = V.build_messages([("i.jpg", "c")] * 4, [("j.jpg", "d")] * 4, "a wording", SETTINGS)
    with pytest.raises(RuntimeError, match="not the greedy choice"):
        V.make_generate(proc, model, SETTINGS)(msgs)
    with pytest.raises(RuntimeError, match="not the greedy choice"):
        L.make_generate_batch(StubProcessor(), StubModel(off_step=0), SETTINGS)([L.listing_messages("x", 8, SETTINGS)])


@pytest.mark.parametrize("module, old, new, fails", [
    ("r6_gpu_common.py", "logits_processor=LogitsProcessorList([rec]),\n                             **kwargs)",
     "logits_processor=LogitsProcessorList([rec]))", {"verbaliser", "listing"}),            # no greedy kwargs
    ("r6_gpu_verbalise.py", 'max_new = settings["generation"]["verbaliser"]["max_new_tokens"]', "max_new = 128",
     {"verbaliser"}),
    ("r6_gpu_listing.py", 'tok.padding_side = settings["generation"]["listing"]["padding_side"]', "pass",
     {"listing"}),
    ("r6_gpu_listing.py", 'max_new = settings["generation"]["listing"]["max_new_tokens"]', "max_new = 32",
     {"listing"}),
    ("r6_gpu_common.py", "new = seq[:, inputs[\"input_ids\"].shape[1]:].cpu().numpy()", "new = seq.cpu().numpy()",
     {"verbaliser", "listing"}),                                                            # prompt kept
])
def test_generation_mutations_fail_the_checks(tmp_path, module, old, new, fails):
    mut = load_copy(tmp_path, module, replace=((old, new),))
    if module == "r6_gpu_common.py":                        # the verbaliser and listing copies use the mutant
        vmod, lmod = load_copy(tmp_path, "r6_gpu_verbalise.py"), load_copy(tmp_path, "r6_gpu_listing.py")
        vmod.G = lmod.G = mut
    else:
        vmod = mut if module == "r6_gpu_verbalise.py" else V
        lmod = mut if module == "r6_gpu_listing.py" else L
    failed = set()
    for name, check, mod in (("verbaliser", check_verbaliser_generation, vmod),
                             ("listing", check_listing_generation, lmod)):
        try:
            check(mod)
        except (AssertionError, RuntimeError):
            failed.add(name)
    assert failed == fails


# ---------------------------------------------------------------- greedy check and snapshot

def test_greedy_check():
    am = np.array([[5, 7], [6, 9], [2, 9]])                     # (T=3 steps, B=2 rows); eos 2, pad 0
    assert G.check_greedy(np.array([[5, 6, 2], [7, 9, 9]]), am, [2, 3], 0)
    assert G.check_greedy(np.array([[5, 2, 0], [7, 9, 9]]), np.array([[5, 7], [2, 9], [4, 9]]), [2, 3], 0)
    with pytest.raises(RuntimeError, match="after|greedy"):
        G.check_greedy(np.array([[5, 2, 4], [7, 9, 9]]), np.array([[5, 7], [2, 9], [4, 9]]), [2, 3], 0)
    with pytest.raises(AssertionError, match="do not match"):
        G.check_greedy(np.array([[5, 6]]), am, [2], 0)


def test_non_greedy_token_fires_and_the_guard_matters(tmp_path):
    tokens, am = np.array([[5, 8, 2]]), np.array([[5], [6], [2]])           # step 1 sampled 8, argmax was 6
    with pytest.raises(RuntimeError, match="not the greedy choice"):
        G.check_greedy(tokens, am, [2], 0)
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="greedy")
    assert mut.check_greedy(tokens, am, [2], 0)


def fake_hub(root, shards=2, drop_shard=False, dangling=False):
    snap = G.snapshot_dir(root, SETTINGS["model"]["id"], SNAP)
    snap.mkdir(parents=True)
    names = [f"model-0000{i + 1}-of-0000{shards}.safetensors" for i in range(shards)]
    for f in G.SNAPSHOT_FILES:
        (snap / f).write_text("{}")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": n
                                                                                  for i, n in enumerate(names)}}))
    for n in names[:-1] if drop_shard else names:
        (snap / n).write_bytes(b"")
    if dangling:
        os.symlink(root / "blobs/missing", snap / "vocab.json")
    (snap.parent.parent / "refs").mkdir()
    (snap.parent.parent / "refs/main").write_text(SNAP)
    return snap


def test_snapshot_check(tmp_path):
    info = G.check_snapshot(fake_hub(tmp_path / "ok"))
    assert info["shards"] == 2 and info["refs_main"] == SNAP
    with pytest.raises(AssertionError, match="missing blobs"):
        G.check_snapshot(fake_hub(tmp_path / "dangling", dangling=True))
    with pytest.raises(AssertionError, match="model snapshot missing"):
        G.check_snapshot(tmp_path / "nothing")


def test_missing_shard_fires_and_the_guard_matters(tmp_path):
    snap = fake_hub(tmp_path / "hub", drop_shard=True)
    with pytest.raises(AssertionError, match="weight shards missing"):
        G.check_snapshot(snap)
    mut = load_copy(tmp_path, "r6_gpu_common.py", guard="snapshot_shards")
    assert mut.check_snapshot(snap)["shards"] == 2


def test_local_snapshot_is_the_pinned_one():
    snap = G.snapshot_dir("/data/SSD2/HF_home/hub", SETTINGS["model"]["id"], SNAP)
    if not snap.is_dir():
        pytest.skip("no local copy of the 8B model")
    info = G.check_snapshot(snap)
    assert info["refs_main"] == SNAP and info["shards"] == 4


# ---------------------------------------------------------------- the listing job (rule section 7 item 2)

PHRASES = [f"synthetic criterion {i}" for i in range(200)]


def listing_job(folder, items):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / L.INPUT_NAME).write_text("".join(json.dumps({"phrase": p, "K": k}) + "\n" for p, k in items))
    return folder


class FakeLister:
    def __init__(self, crash_at_batch=None):
        self.batches, self.crash_at_batch = [], crash_at_batch

    def __call__(self, conversations):
        if self.crash_at_batch is not None and len(self.batches) + 1 == self.crash_at_batch:
            raise RuntimeError("simulated crash")
        self.batches.append(len(conversations))
        return [fake_answer(c) for c in conversations]


def run_l(items, out, model, cache=None, **kw):
    return L.run_listing(items, SETTINGS, out, model, cache or {}, 16, log=lambda *_: None, **kw)


def check_listing(items, out, cache=None):
    recs = read_jsonl(out / "listings.jsonl")
    keys = [(r["phrase"], r["K"]) for r in recs]
    assert len(keys) == len(set(keys)) and set(keys) == set(items)
    for r in recs:
        assert list(r) == ["phrase", "K", "answer"]
        want = (cache or {}).get((r["phrase"], r["K"])) or fake_answer(L.listing_messages(r["phrase"], r["K"],
                                                                                          SETTINGS))
        assert r["answer"] == want


def test_listing_prompt_and_message():
    msgs = L.listing_messages("brushwork and texture", 8, SETTINGS)
    assert msgs == [{"role": "user", "content": [{"type": "text", "text": (
        "List 8 distinct values of the following criterion for describing paintings: brushwork and texture. "
        "One value per line, no numbering.")}]}]


def test_listing_resumes_without_repeating_or_skipping(tmp_path):
    items = [(p, k) for p in PHRASES for k in (8, 16)]
    m1 = FakeLister()
    res1 = run_l(items, tmp_path, m1, max_calls=50)
    assert m1.batches == [16] * 4 and len(read_jsonl(tmp_path / "listings.jsonl")) == 64 and not res1["complete"]
    m2 = FakeLister()
    res2 = run_l(items, tmp_path, m2)
    assert res2["n_done_before"] == 64 and sum(m2.batches) == 400 - 64 and res2["complete"]
    check_listing(items, tmp_path)


def test_listing_crash_keeps_the_checkpoint_and_the_guard_matters(tmp_path):
    items = [(p, k) for p in PHRASES[:60] for k in (8, 16)]
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_l(items, tmp_path / "a", FakeLister(crash_at_batch=6))               # batches 1..5 returned (80)
    assert len(read_jsonl(tmp_path / "a/listings.jsonl")) == 64                  # saved after batch 4 (>= 50)
    mut = load_copy(tmp_path, "r6_gpu_listing.py", guard="listing_checkpoint")
    with pytest.raises(RuntimeError, match="simulated crash"):
        mut.run_listing(items, SETTINGS, tmp_path / "b", FakeLister(crash_at_batch=6), {}, 16, log=lambda *_: None)
    assert not (tmp_path / "b/listings.jsonl").exists()                          # guard deleted: all 80 lost


def test_cached_listings_are_copied_not_recomputed(tmp_path):
    items = [(p, k) for p in PHRASES[:40] for k in (8, 16)]
    cache = {k: f"cached answer for {k}" for k in items[::3]}
    m = FakeLister()
    res = run_l(items, tmp_path, m, cache=cache)
    assert res["n_cached"] == len(cache) and sum(m.batches) == len(items) - len(cache)
    check_listing(items, tmp_path, cache)


def listing_fingerprint():
    return {"job": "r6_gpu_listing", "model_id": SETTINGS["model"]["id"], "snapshot": SNAP,
            "settings_sha256": SETTINGS_SHA,
            "scripts_sha256": G.script_shas(HERE / "r6_gpu_listing.py", HERE / "r6_gpu_common.py")}


def test_cache_of_another_fingerprint_fires_and_the_guard_matters(tmp_path):
    items = [(p, 8) for p in PHRASES[:5]]
    run_l(items, tmp_path / "old", FakeLister())
    G.begin_provenance(tmp_path / "old", dict(listing_fingerprint(), settings_sha256="0" * 64), {"args": {}})
    with pytest.raises(AssertionError, match="cannot be reused"):
        L.load_caches([tmp_path / "old"], listing_fingerprint())
    mut = load_copy(tmp_path, "r6_gpu_listing.py", guard="cache_fingerprint")
    assert len(mut.load_caches([tmp_path / "old"], listing_fingerprint())) == 5


def test_cache_of_the_same_fingerprint_is_read(tmp_path):
    items = [(p, 16) for p in PHRASES[:5]]
    run_l(items, tmp_path / "old", FakeLister())
    G.begin_provenance(tmp_path / "old", listing_fingerprint(), {"args": {}})
    got = L.load_caches([tmp_path / "old"], listing_fingerprint())
    assert set(got) == set(items)


def test_repeated_listing_input_fires_and_the_guard_matters(tmp_path):
    job = listing_job(tmp_path / "j", [("brushwork", 8), ("light", 8), ("brushwork", 8)])
    with pytest.raises(AssertionError, match="repeated"):
        L.load_listing_input(job / L.INPUT_NAME, SETTINGS)
    mut = load_copy(tmp_path, "r6_gpu_listing.py", guard="input_unique")
    assert len(mut.load_listing_input(job / L.INPUT_NAME, SETTINGS)) == 3


@pytest.mark.parametrize("line, match", [
    ({"phrase": "light", "K": 12}, "is not 8 or 16"),
    ({"phrase": "light", "K": "8"}, "is not 8 or 16"),
    ({"phrase": "Light", "K": 8}, "normalised"),
    ({"phrase": " light", "K": 8}, "normalised"),
    ({"phrase": "", "K": 8}, "normalised"),
    ({"phrase": "light", "K": 8, "aspect": "style"}, "keys"),
])
def test_malformed_listing_input_raises(tmp_path, line, match):
    p = tmp_path / L.INPUT_NAME
    p.write_text(json.dumps(line) + "\n")
    with pytest.raises(AssertionError, match=match):
        L.load_listing_input(p, SETTINGS)


# ---------------------------------------------------------------- command lines and wrappers, no GPU

def run(cmd, env=None, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, env=env or ENV, cwd=cwd or CHECKOUT, timeout=600)


@pytest.fixture(scope="module")
def cli_job(tmp_path_factory):
    """A tuning-shaped job folder whose first two episodes' images exist (empty files), a fake hub."""
    base = tmp_path_factory.mktemp("cli")
    inp = synthetic_input(tuning_index(), rng_seed=4)
    man = synthetic_manifest(inp)
    needed = np.unique(np.concatenate([inp[f][:2].ravel() for f in ("pairs_a_img", "pairs_b_img")]))
    images = base / "images"
    images.mkdir()
    for r in needed.tolist():
        (images / name_of(r)).write_bytes(b"")
    job = write_job(base / "jobs" / "s52_tune", inp, man, images=False)
    (job / "images.txt").write_text("\n".join(sorted(name_of(r) for r in needed.tolist())) + "\n")
    hub = base / "hub"
    fake_hub(hub)
    listing_job(base / "jobs" / "list1", [(p, k) for p in PHRASES[:20] for k in (8, 16)])
    return dict(base=base, job=job, images=images, hub=hub)


def test_verbaliser_check_only_without_a_gpu(cli_job):
    out = cli_job["base"] / "out_v"
    r = run([PY, str(HERE / "r6_gpu_verbalise.py"), "--job-dir", str(cli_job["job"]), "--out", str(out),
             "--wordings", "W1,W2", "--stop", "2", "--check-only", "--no-model", "--hub-cache", str(cli_job["hub"]),
             "--image-dir", str(cli_job["images"])])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "verbaliser inputs ok: seed 52, 3072 episodes, range [0, 2)" in r.stdout
    assert "imports ok: torch" in r.stdout and "stopping before the model" in r.stdout and not out.exists()


def test_verbaliser_check_only_finds_a_missing_image_and_the_guard_matters(cli_job, tmp_path):
    args = ["--job-dir", str(cli_job["job"]), "--out", str(tmp_path / "o"), "--wordings", "W1", "--start", "2",
            "--stop", "3", "--check-only", "--no-model", "--hub-cache", str(cli_job["hub"]),
            "--image-dir", str(cli_job["images"])]
    r = run([PY, str(HERE / "r6_gpu_verbalise.py"), *args])
    assert r.returncode != 0 and "images missing under" in r.stderr
    mut = load_copy(tmp_path, "r6_gpu_verbalise.py", guard="images")
    assert mut.main(args) == 0                                     # guard deleted: missing images go unnoticed


@pytest.mark.parametrize("extra, match", [
    (["--wordings", "W5"], "--wordings must be distinct ids"),
    (["--wordings", "W1,W1"], "--wordings must be distinct ids"),
    (["--wordings", "W1", "--no-model"], "--no-model needs --check-only"),
    (["--wordings", "W1", "--start", "4", "--stop", "4"], "need 0 <= --start < --stop"),
    ([], "required: --wordings"),
])
def test_verbaliser_argument_errors(cli_job, extra, match):
    r = run([PY, str(HERE / "r6_gpu_verbalise.py"), "--job-dir", str(cli_job["job"]), "--out", "x", *extra])
    assert r.returncode == 2 and match in r.stderr, r.stderr[-500:]


def test_listing_check_only_without_a_gpu(cli_job):
    out = cli_job["base"] / "out_l"
    r = run([PY, str(HERE / "r6_gpu_listing.py"), "--job-dir", str(cli_job["base"] / "jobs/list1"), "--out",
             str(out), "--check-only", "--no-model", "--hub-cache", str(cli_job["hub"])])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "listing inputs ok: 40 (phrase, K) pairs, 0 cached in 0 folders, batch size 16" in r.stdout
    assert "stopping before the model" in r.stdout and not out.exists()
    r = run([PY, str(HERE / "r6_gpu_listing.py"), "--job-dir", "x", "--out", "y", "--no-model"])
    assert r.returncode == 2 and "--no-model needs --check-only" in r.stderr


def wrapper_env(cli_job, **extra):
    env = dict(ENV, R6_JOB_ROOT=str(cli_job["base"] / "jobs"), R6_IMAGE_DIR=str(cli_job["images"]),
               HF_HUB_CACHE_OVERRIDE=str(cli_job["hub"]), R6_PYTHON=PY)
    env.update(extra)
    return env


def test_wrappers_check_only_no_model(cli_job):
    r = run(["bash", "scripts/run_r6_verbalise.sh", "s52_tune", "--wordings", "W1", "--stop", "2", "--check-only",
             "--no-model"], env=wrapper_env(cli_job, R6_OUT=str(cli_job["base"] / "wo")))
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    for line in ("model snapshot ok", "images: ", "inputs ok", "stopping before the model"):
        assert line in r.stdout
    r = run(["bash", "scripts/run_r6_listing.sh", "list1", "--check-only", "--no-model"],
            env=wrapper_env(cli_job, R6_OUT=str(cli_job["base"] / "lo")))
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    assert "listing input: 40 lines" in r.stdout and "stopping before the model" in r.stdout


def test_wrappers_refuse_missing_inputs(cli_job, tmp_path):
    r = run(["bash", "scripts/run_r6_verbalise.sh", "no_such_job", "--wordings", "W1"], env=wrapper_env(cli_job))
    assert r.returncode == 2 and "job input missing" in r.stderr
    r = run(["bash", "scripts/run_r6_verbalise.sh", "../s52_tune", "--wordings", "W1"], env=wrapper_env(cli_job))
    assert r.returncode == 2 and "bad job name" in r.stderr
    env = wrapper_env(cli_job, HF_HUB_CACHE_OVERRIDE=str(tmp_path / "empty_hub"))
    r = run(["bash", "scripts/run_r6_listing.sh", "list1", "--check-only", "--no-model"], env=env)
    assert r.returncode == 2 and "pinned snapshot missing" in r.stderr
    env = wrapper_env(cli_job, R6_IMAGE_DIR=str(tmp_path / "no_images"))
    r = run(["bash", "scripts/run_r6_verbalise.sh", "s52_tune", "--wordings", "W1", "--check-only", "--no-model"],
            env=env)
    assert r.returncode == 2 and "job images missing" in r.stderr


def test_wrappers_follow_the_probe_template():
    for f in GPU_SCRIPTS[:2]:
        text = (CHECKOUT / f).read_text()
        for needle in ("set -euo pipefail", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
                       "OMP_NUM_THREADS=8", "/local/wding/r6_jobs", 'exec "$PY"'):
            assert needle in text, (f, needle)
        assert "/tmp" not in text


# ---------------------------------------------------------------- the sync script (plan only, nothing copied)

needs_sys_py = pytest.mark.skipif(not Path(SYS_PY).is_file() or not Path("/root/.claude/skills/cluster-run").is_dir(),
                                  reason="system python3 or the cluster-run skill is missing")


@needs_sys_py
def test_sync_plan_for_a_job_folder(cli_job):
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node401", "--job-dir", str(cli_job["job"])])
    assert r.returncode == 0, r.stderr[-1500:]
    assert f"r6_job_s52_tune: dir, " in r.stdout and "-> /local/wding/r6_jobs/s52_tune" in r.stdout
    assert "Plan only; pass --run to copy." in r.stdout


@needs_sys_py
def test_sync_refuses_episode_or_label_data(cli_job, tmp_path):
    job = tmp_path / "bad_job"
    write_job(job, synthetic_input(tuning_index()))
    np.savez(job / "extra.npz", candidates=np.zeros((2, 13), dtype=np.int64))
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node401", "--job-dir", str(job)])
    assert r.returncode != 0 and "REFUSING" in r.stderr and "candidates" in r.stderr
    (job / "extra.npz").unlink()
    (job / "episodes_seed52.npz").write_bytes(b"")
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node401", "--job-dir", str(job)])
    assert r.returncode != 0 and "an episodes file" in r.stderr
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node999x", "--job-dir", str(cli_job["job"])])
    assert r.returncode != 0


@needs_sys_py
@pytest.mark.skipif(not Path("/data/PDD/wikiart_proj/wikiart").is_dir(), reason="no local WikiArt tree")
@pytest.mark.parametrize("plant, match", [
    ("npz", "a WikiArt path"), ("json", "a WikiArt path"), ("images", "not neutral"), ("map", "an image map"),
])
def test_sync_refuses_wikiart_paths_in_a_job_folder(tmp_path, plant, match):
    job = tmp_path / "job_with_path"
    inp = synthetic_input(tuning_index())
    man = synthetic_manifest(inp)
    neutral = sorted(set(man["image_name"].tolist()))
    if plant == "npz":                                          # a path inside a compressed unicode array
        man["image_name"] = man["image_name"].astype("<U64")
        man["image_name"][7] = "Baroque/rembrandt_the-night-watch.jpg"
    write_job(job, inp, man)
    if plant == "npz":                                          # images.txt stays neutral: only the npz holds it
        (job / "images.txt").write_text("\n".join(neutral) + "\n")
        match = "rows_manifest.npz (image_name.npy): a WikiArt path"
    if plant == "json":
        (job / "job_record.json").write_text(json.dumps({"note": "Ukiyo_e/hokusai_the-great-wave.jpg"}))
    if plant == "images":
        with open(job / "images.txt", "a") as f:
            f.write("Realism/ivan-shishkin_morning.jpg\n")
    if plant == "map":
        (job / "s.image_map.json").write_text("{}")
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node401", "--job-dir", str(job)])
    assert r.returncode != 0 and "REFUSING" in r.stderr and match in r.stderr, r.stderr[-800:]


@needs_sys_py
@pytest.mark.skipif(not Path("/data/PDD/wikiart_proj/wikiart").is_dir(), reason="no local WikiArt tree")
def test_utf32_search_finds_a_path_in_a_compressed_unicode_array(tmp_path):
    """The npz case above passes only through the UTF-32-LE search: without it a copy of the sync script ships the
    folder."""
    job = tmp_path / "job_with_path"
    inp = synthetic_input(tuning_index())
    man = synthetic_manifest(inp)
    neutral = sorted(set(man["image_name"].tolist()))
    man["image_name"] = man["image_name"].astype("<U64")
    man["image_name"][7] = "Baroque/rembrandt_the-night-watch.jpg"
    write_job(job, inp, man)
    (job / "images.txt").write_text("\n".join(neutral) + "\n")
    src = (CHECKOUT / "scripts/das6_sync_r6.py").read_text()
    line = '    alts += [re.escape((f + "/").encode("utf-32-le")) for f in folders]\n'
    assert src.count(line) == 1
    copy = tmp_path / "das6_sync_r6_no_utf32.py"
    copy.write_text(src.replace(line, ""))
    r = run([SYS_PY, str(copy), "--node", "node401", "--job-dir", str(job)])
    assert r.returncode == 0 and "Plan only" in r.stdout, r.stderr[-800:]       # the path would be shipped
    r = run([SYS_PY, "scripts/das6_sync_r6.py", "--node", "node401", "--job-dir", str(job)])
    assert r.returncode != 0 and "rows_manifest.npz (image_name.npy): a WikiArt path" in r.stderr


def test_wrapper_refuses_a_name_that_is_not_neutral(cli_job, tmp_path):
    job = cli_job["base"] / "jobs" / "s52_bad_name"
    write_job(job, synthetic_input(tuning_index()), images=False)
    (job / "images.txt").write_text("Romanticism/caspar-david-friedrich_wanderer.jpg\n")
    r = run(["bash", "scripts/run_r6_verbalise.sh", "s52_bad_name", "--wordings", "W1", "--check-only", "--no-model"],
            env=wrapper_env(cli_job))
    assert r.returncode == 2 and "not neutral" in r.stderr


def test_sync_script_runs_on_python_3_10():
    ast.parse((CHECKOUT / "scripts/das6_sync_r6.py").read_text(), feature_version=(3, 10))


# ---------------------------------------------------------------- no metric, no labels in the GPU code

def test_gpu_code_imports_neither_src_nor_r6_common():
    allowed = {"argparse", "hashlib", "json", "os", "platform", "re", "socket", "sys", "time", "datetime",
               "pathlib", "zoneinfo", "numpy", "torch", "transformers", "r6_gpu_common", "zipfile", "cluster", "tempfile"}
    for f in [HERE / m for m in GPU_MODULES] + [CHECKOUT / "scripts/das6_sync_r6.py"]:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert set(names) <= allowed, (f.name, names)


def test_no_metric_code_in_the_gpu_scripts():
    pattern = re.compile(r"R@1|\br1\b|recall|accuracy|per_anchor|summarize|cosine_scores|as_int4|aspect_metrics|"
                         r"\bgain\b|hits?\b|rank\b|argsort|np\.mean\(|\.mean\(\)|(?<!then-)score", re.I)
    for f in [HERE / m for m in GPU_MODULES] + [CHECKOUT / s for s in GPU_SCRIPTS]:
        found = sorted({m.group(0) for m in pattern.finditer(f.read_text())})
        assert not found, (f.name, found)
