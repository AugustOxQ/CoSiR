"""Tests of r6_gpu_inputs (ticket 10) on real selection rows: the smoke seed 9001's episodes (64 per pair, built by
r6_episodes on selection rows and checked against their recorded hashes) become a verbaliser job folder. Shown here:
the C6 join (row -> data.sample_ids[row] -> annotation), that only the image path and caption of an annotation are
read, that no label, aspect name, anchor or candidate reaches the folder, the tuning subset (--first-per-pair), the GPU
job's own loaders and --check-only --no-model on the real folder, and the sync script's plan for it. No held row is
read: the episodes lie in the selection split (asserted), and the data's held features are dropped at load.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_gpu_inputs.py
"""
import ast
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
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402

CHECKOUT = HERE.parents[2]
WIKIART = Path("/data/PDD/wikiart_proj/wikiart")
HUB = Path("/data/SSD2/HF_home/hub")
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()
SEED = 9001
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
LABEL_KEYS = {"anchor", "candidates", "pair_index", "pair_names", "labels", "label", "emotion", "style", "genre",
              "art_style", "aspect", "painting", "paintings", "image_id", "target"}


def mutant(tmp_path, guard):
    """A copy of r6_gpu_inputs with the `# guard:<guard>` statements replaced by `pass`."""
    src = (HERE / "r6_gpu_inputs.py").read_text()
    assert src.count(HERE_LINE) == 1
    lines = src.splitlines(keepends=True)
    hits = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Expr) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
    assert hits, guard
    for n in hits:
        indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
        lines[n.lineno - 1] = f"{indent}pass\n"
        for i in range(n.lineno, n.end_lineno):
            lines[i] = "\n"
    path = tmp_path / f"r6_gpu_inputs_copy{next(_COUNT)}.py"
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
    marked = set(re.findall(r"# guard:(\w+)", (HERE / "r6_gpu_inputs.py").read_text()))
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    assert marked == tested, (marked - tested, tested - marked)


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

    def items(self):
        raise AssertionError("annotation items were listed")

    def values(self):
        raise AssertionError("annotation values were listed")


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
    names = R.value_names(data)
    keep = SimpleNamespace(sample_ids=data.sample_ids, paintings=data.paintings)
    del data
    index = AE.PaintingValueIndex(labels, split.groups)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    eps = E.build_seed(labels, split.groups, split.selection, index, vs, SEED, R.N_SMOKE)
    assert eps.sha == E.identity_targets()[SEED]["episodes_sha256"]          # AB's recorded smoke episodes
    base = tmp_path_factory.mktemp("real")
    path = E.save_episodes(base / f"episodes_seed{SEED}.npz", eps)
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    job = base / "s9001_full"
    rec = I.write_job(eps, keep.sample_ids, keep.paintings, GuardedAnnotations(annotations), job, None,
                      G.sha256_file(path))                     # only image and caption may be read (else it raises)
    return dict(data=keep, selection=split.selection, eps=eps, path=path, base=base, annotations=annotations,
                names=names, job=job, rec=rec)


def npz_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_job_folder_holds_exactly_the_contracts_files_and_keys(real):
    job = real["job"]
    assert sorted(p.name for p in job.iterdir()) == sorted(I.JOB_FILES)
    man, vin = npz_arrays(job / "rows_manifest.npz"), npz_arrays(job / "verbalise_input.npz")
    assert sorted(man) == sorted(G.MANIFEST_KEYS) and sorted(vin) == sorted(G.VERBALISE_KEYS)
    assert not (set(man) | set(vin)) & LABEL_KEYS
    strings = {k for d in (man, vin) for k, v in d.items() if v.dtype.kind in "USO"}
    assert strings == {"image_relpath", "caption"}                       # captions and paths: the only strings
    assert all(v.dtype == np.int64 for d in (man, vin) for k, v in d.items() if k not in strings)
    rec = json.loads((job / "job_record.json").read_text())
    assert rec == real["rec"] and rec["seed"] == SEED and rec["n_episodes"] == 3 * R.N_SMOKE
    assert rec["files_sha256"] == {f: G.sha256_file(job / f) for f in I.JOB_FILES[:3]}
    assert rec["episodes_file_sha256"] == G.sha256_file(real["path"])
    assert rec["episodes_sha256_in_pair_order"] == [real["eps"].sha[p] for p in R.PAIR_NAMES]


def test_verbalise_input_is_the_episodes_example_pairs(real):
    vin, eps = npz_arrays(real["job"] / "verbalise_input.npz"), real["eps"]
    assert int(vin["seed"]) == SEED and np.array_equal(vin["episode_index"], np.arange(eps.n))
    for f in G.PAIR_FIELDS:
        assert np.array_equal(vin[f], getattr(eps.pooled, f))


def test_manifest_c6_join_on_real_selection_rows(real):
    man, vin, d, ann = (npz_arrays(real["job"] / "rows_manifest.npz"), npz_arrays(real["job"] / "verbalise_input.npz"),
                        real["data"], real["annotations"])
    rows = man["rows"]
    assert np.array_equal(rows, np.unique(np.concatenate([vin[f].ravel() for f in G.PAIR_FIELDS])))
    assert np.isin(rows, real["selection"]).all()                                # selection rows only
    assert np.array_equal(man["sample_id"], d.sample_ids[rows])                  # C6: data.sample_ids[row]
    assert not np.array_equal(man["sample_id"], rows)                            # the join is not positional
    sid = man["sample_id"].tolist()
    assert man["caption"].tolist() == [ann[s]["caption"] for s in sid]
    assert man["image_relpath"].tolist() == [ann[s]["image"] for s in sid]
    assert [Path(p).stem for p in man["image_relpath"].tolist()] == d.paintings[rows].tolist()
    assert man["caption"].tolist() != [ann[r]["caption"] for r in rows.tolist()]   # annotations[row] would differ


def test_images_list_and_the_gpu_loaders(real):
    job = real["job"]
    man = G.Manifest(job / "rows_manifest.npz")
    inp = G.load_verbalise_input(job / "verbalise_input.npz", man)
    images = (job / "images.txt").read_text().splitlines()
    assert images == sorted(set(man.image_relpath.tolist())) and all((WIKIART / p).is_file() for p in images)
    s, _ = G.load_settings()
    import r6_gpu_verbalise as V
    eps, ann, d = real["eps"], real["annotations"], real["data"]
    for pos in (0, 64, 191):
        content = V.episode_messages(inp, man, pos, "b", "W3", s, WIKIART)[0]["content"]
        shown = [p["image"] for p in content if p["type"] == "image"]
        caps = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("\ncaption")]
        img_rows = [*eps.pooled.pairs_b_img[pos], *eps.pooled.pairs_a_img[pos]]      # condition b: Group A = pairs_b
        txt_rows = [*eps.pooled.pairs_b_txt[pos], *eps.pooled.pairs_a_txt[pos]]
        assert shown == [str(WIKIART / ann[int(d.sample_ids[r])]["image"]) for r in img_rows]
        assert caps == [f"\ncaption: {ann[int(d.sample_ids[r])]['caption']}\n" for r in txt_rows]


def walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_no_aspect_or_value_name_in_the_record_or_keys(real):
    names = {"emotion", "style", "genre", "aspect"} | {str(n).lower() for v in real["names"].values() for n in v}
    names |= {n.replace("_", " ") for n in names}
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b", re.I)
    rec = json.loads((real["job"] / "job_record.json").read_text())
    hits = [s for s in walk_strings(rec) if pattern.search(s)]
    assert not hits, hits
    for f in ("rows_manifest.npz", "verbalise_input.npz"):
        with zipfile.ZipFile(real["job"] / f) as z:
            assert not [n for n in z.namelist() if pattern.search(Path(n).stem)]


def test_first_per_pair_through_the_command_line(real):
    out = real["base"] / "s9001_first16"
    assert I.main(["--episodes", str(real["path"]), "--out", str(out), "--first-per-pair", "16"]) == 0
    vin = npz_arrays(out / "verbalise_input.npz")
    want = np.concatenate([np.arange(k * R.N_SMOKE, k * R.N_SMOKE + 16) for k in range(3)])
    assert np.array_equal(vin["episode_index"], want)
    for f in G.PAIR_FIELDS:
        assert np.array_equal(vin[f], getattr(real["eps"].pooled, f)[want])
    assert json.loads((out / "job_record.json").read_text())["first_per_pair"] == 16
    with pytest.raises(AssertionError, match="has 64 episodes"):
        I.select_episodes(real["eps"], 65)


def test_existing_job_folder_fires_and_the_guard_matters(real, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    args = (real["eps"], real["data"].sample_ids, real["data"].paintings, real["annotations"], out)
    with pytest.raises(AssertionError, match="never overwritten"):
        I.write_job(*args)
    mut = mutant(tmp_path, guard="no_overwrite")
    assert mut.write_job(*args)["n_episodes"] == 192                     # guard deleted: the folder is replaced


def test_positional_join_fires_and_the_guard_matters(real, tmp_path):
    rows = npz_arrays(real["job"] / "rows_manifest.npz")["rows"]
    wrong = np.arange(R.N_ROWS, dtype=np.int64)                          # annotations[row]: the C6 mistake
    with pytest.raises(AssertionError, match="C6 join"):
        I.build_manifest(rows, wrong, real["data"].paintings, real["annotations"])
    mut = mutant(tmp_path, guard="c6_join")
    got = mut.build_manifest(rows, wrong, real["data"].paintings, real["annotations"])
    assert got["caption"].tolist() == [real["annotations"][r]["caption"] for r in rows.tolist()]   # wrong captions


def test_non_unique_sample_ids_fire_and_the_guard_matters(real, tmp_path):
    rows = npz_arrays(real["job"] / "rows_manifest.npz")["rows"]
    sids = real["data"].sample_ids.copy()
    out_rows = np.setdiff1d(np.arange(R.N_ROWS), rows)[:2]
    sids[out_rows[0]] = sids[out_rows[1]]                                # a repeated id on rows the job does not use
    with pytest.raises(AssertionError, match="uniquely index"):
        I.build_manifest(rows, sids, real["data"].paintings, real["annotations"])
    mut = mutant(tmp_path, guard="sample_ids")
    assert len(mut.build_manifest(rows, sids, real["data"].paintings, real["annotations"])["rows"]) == len(rows)


def test_verbaliser_check_only_on_the_real_job(real):
    if not (G.snapshot_dir(HUB, "Qwen/Qwen3-VL-8B-Instruct", G.load_settings()[0]["model"]["snapshot"])).is_dir():
        pytest.skip("no local copy of the 8B model")
    r = subprocess.run([sys.executable, str(HERE / "r6_gpu_verbalise.py"), "--job-dir", str(real["job"]), "--out",
                        str(real["base"] / "v_out"), "--wordings", "W1,W2,W3,W4", "--check-only", "--no-model",
                        "--hub-cache", str(HUB), "--image-root", str(WIKIART)],
                       capture_output=True, text=True, env=ENV, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    assert f"seed {SEED}, 192 episodes, range [0, 192)" in r.stdout and "stopping before the model" in r.stdout


def test_processor_renders_the_real_messages(real):
    """The snapshot's processor (CPU; no model weights) renders a real verbaliser message with its 8 images and no
    path in the text, and the listing's batched tokenisation (render, then the tokenizer with left padding) gives the
    ids of the processor's own tokenisation."""
    s, _ = G.load_settings()
    snap = G.snapshot_dir(HUB, s["model"]["id"], s["model"]["snapshot"])
    if not snap.is_dir():
        pytest.skip("no local copy of the 8B model")
    from transformers import AutoProcessor
    import r6_gpu_listing as L
    import r6_gpu_verbalise as V
    proc = AutoProcessor.from_pretrained(str(snap), **s["model"]["processor_kwargs"])
    probe = R.TEST / "20261106_mllm_probe_8b/results/render_check_local.json"   # the 8B probe's processor record
    if probe.is_file():
        rec = json.loads(probe.read_text())["processor_image_settings"]
        ip = proc.image_processor
        assert str(ip.size) == rec["size"] and type(ip).__name__ == rec["image_processor_class"]
        for k in ("patch_size", "temporal_patch_size", "merge_size", "resample", "image_mean", "image_std"):
            assert json.loads(json.dumps(getattr(ip, k))) == rec[k], k
    man = G.Manifest(real["job"] / "rows_manifest.npz")
    inp = G.load_verbalise_input(real["job"] / "verbalise_input.npz", man)
    msgs = V.episode_messages(inp, man, 5, "a", "W2", s, WIKIART)
    enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt")
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vision_start = proc.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    assert int((enc["input_ids"] == vision_start).sum()) == 8 and enc["image_grid_thw"].shape[0] == 8
    assert ".jpg" not in text and str(WIKIART) not in text
    assert text.index("Group A:") < text.index("Group B:") < text.index(s["verbaliser"]["wordings"]["W2"])
    assert text.rstrip().endswith("<|im_start|>assistant")
    tok = proc.tokenizer
    tok.padding_side = s["generation"]["listing"]["padding_side"]
    convs = [L.listing_messages("brushwork and texture", 8, s), L.listing_messages("light", 16, s)]
    rendered = [proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    batch = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    for i, c in enumerate(convs):
        own = proc.apply_chat_template(c, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt")["input_ids"][0]
        mask = batch["attention_mask"][i].bool()
        assert batch["input_ids"][i][mask].tolist() == own.tolist()
    assert batch["attention_mask"][1, 0] == 0 and batch["attention_mask"][:, -1].tolist() == [1, 1]   # left padding


def test_sync_plan_for_the_real_job_and_its_images(real):
    if not Path("/usr/bin/python3").is_file() or not Path("/root/.claude/skills/cluster-run").is_dir():
        pytest.skip("system python3 or the cluster-run skill is missing")
    r = subprocess.run(["/usr/bin/python3", "scripts/das6_sync_r6.py", "--node", "node402", "--job-dir",
                        str(real["job"]), "--images"], capture_output=True, text=True, env=ENV, cwd=CHECKOUT,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    n = len((real["job"] / "images.txt").read_text().splitlines())
    assert f"wikiart_r6_images_chunk0: selected, {n} files" in r.stdout
    assert "-> /local/wding/Dataset/wikiart_proj/wikiart" in r.stdout and "-> /local/wding/r6_jobs/s9001_full" in r.stdout
    assert "Plan only; pass --run to copy." in r.stdout
