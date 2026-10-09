"""Tests of r6_gpu_inputs (ticket 10) on real selection rows: the smoke seed 9001's episodes (64 per pair, built by
r6_episodes on selection rows and checked against their recorded hashes) become a verbaliser job folder. Shown here:
the C6 join (row -> data.sample_ids[row] -> annotation), that only the image path and caption of an annotation are
read, that no label, aspect name, WikiArt path, anchor or candidate reaches the shipped folder (images travel under
neutral names; the name -> path map stays beside the folder), the staged image links, the tuning subset
(--first-per-pair), the GPU job's own loaders and --check-only --no-model on the real folder, and the sync script's
plan for it. No held row is read: the episodes lie in the selection split (asserted), and the data's held features
are dropped at load.

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
              "art_style", "aspect", "painting", "paintings", "image_id", "target", "sample_id", "image_relpath"}


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
    job, staging = base / "jobs" / "s9001_full", base / "images"
    rec = I.write_job(eps, keep.sample_ids, keep.paintings, GuardedAnnotations(annotations), job, None,
                      G.sha256_file(path), staging)            # only image and caption may be read (else it raises)
    return dict(data=keep, selection=split.selection, eps=eps, path=path, base=base, annotations=annotations,
                names=names, job=job, rec=rec, staging=staging)


def npz_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ann_of(real, r):
    """The annotation of global row r through the C6 join."""
    return real["annotations"][int(real["data"].sample_ids[r])]


def test_job_folder_holds_exactly_the_contracts_files_and_keys(real):
    job = real["job"]
    assert sorted(p.name for p in job.iterdir()) == sorted(I.JOB_FILES)
    assert I.image_map_path(job).is_file() and I.image_map_path(job).parent == job.parent      # beside, not inside
    man, vin = npz_arrays(job / "rows_manifest.npz"), npz_arrays(job / "verbalise_input.npz")
    assert sorted(man) == ["caption", "image_name", "rows"] and sorted(vin) == sorted(G.VERBALISE_KEYS)
    assert not (set(man) | set(vin)) & LABEL_KEYS
    strings = {k for d in (man, vin) for k, v in d.items() if v.dtype.kind in "USO"}
    assert strings == {"image_name", "caption"}                    # neutral names and captions: the only strings
    assert all(re.fullmatch(r"[0-9a-f]{20}\.jpg", n) for n in man["image_name"].tolist())
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
    assert man["caption"].tolist() == [ann_of(real, r)["caption"] for r in rows.tolist()]   # C6: data.sample_ids
    assert man["image_name"].tolist() == [I.image_name(ann_of(real, r)["image"]) for r in rows.tolist()]
    assert man["caption"].tolist() != [ann[r]["caption"] for r in rows.tolist()]   # annotations[row] would differ
    assert man["image_name"].tolist() != [I.image_name(ann[r]["image"]) for r in rows.tolist()]
    m = json.loads(I.image_map_path(real["job"]).read_text())
    assert all(I.image_name(rel) == name for name, rel in m["images"].items())  # the CPU-only map: name -> path
    by_name = {}
    for r, n in zip(rows.tolist(), man["image_name"].tolist()):                 # one name per painting
        by_name.setdefault(n, set()).add(str(d.paintings[r]))
    assert all(len(v) == 1 for v in by_name.values()) and len(by_name) < len(rows)


def test_images_staged_and_loaded_by_neutral_name_only(real):
    job, staging = real["job"], real["staging"]
    man = G.Manifest(job / "rows_manifest.npz")
    inp = G.load_verbalise_input(job / "verbalise_input.npz", man)
    images = (job / "images.txt").read_text().splitlines()
    img_rows = np.unique(np.concatenate([inp[f].ravel() for f in ("pairs_a_img", "pairs_b_img")]))
    assert images == sorted({I.image_name(ann_of(real, r)["image"]) for r in img_rows.tolist()})
    assert len(images) < len(set(man.image_name.tolist()))          # caption rows' paintings are not shipped
    m = json.loads(I.image_map_path(job).read_text())
    assert sorted(m["images"]) == images and m["staging"] == str(staging.resolve())
    for n in images:                                                # each staged link opens the mapped file
        assert (staging / n).is_symlink() and os.path.realpath(staging / n) == str(WIKIART / m["images"][n])
    s, _ = G.load_settings()
    import r6_gpu_verbalise as V
    eps = real["eps"]
    for pos in (0, 64, 191):
        content = V.episode_messages(inp, man, pos, "b", "W3", s, staging)[0]["content"]
        shown = [p["image"] for p in content if p["type"] == "image"]
        caps = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("\ncaption")]
        img = [*eps.pooled.pairs_b_img[pos], *eps.pooled.pairs_a_img[pos]]          # condition b: Group A = pairs_b
        txt = [*eps.pooled.pairs_b_txt[pos], *eps.pooled.pairs_a_txt[pos]]
        assert shown == [str(staging / I.image_name(ann_of(real, r)["image"])) for r in img]
        assert [os.path.realpath(x) for x in shown] == [str(WIKIART / ann_of(real, r)["image"]) for r in img]
        assert caps == [f"\ncaption: {ann_of(real, r)['caption']}\n" for r in txt]


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


def label_pattern(real):
    from src.data.wikiart_genre import GENRE_NAMES
    styles = [p.name for p in WIKIART.iterdir() if p.is_dir()]
    names = {"emotion", "style", "genre", "aspect"} | {str(n).lower() for v in real["names"].values() for n in v}
    names |= {n.lower() for n in styles} | {n.lower() for n in GENRE_NAMES}
    names |= {n.replace("_", " ") for n in names}
    assert len(styles) == 27 and "impressionism" in names and "sadness" in names and "landscape" in names
    return re.compile(r"(?<![0-9A-Za-z])(" + "|".join(re.escape(n) for n in sorted(names)) + r")(?![0-9A-Za-z])",
                      re.I)


def test_no_style_genre_or_emotion_name_in_the_shipped_folder(real):
    """Every string of the shipped folder (file names, npz member names, image names, images.txt, the record) is free
    of style, genre, emotion and aspect names; captions are exempt (the rule ships them: "row IDs, images and
    captions"; they are free text that may say "awe" or "portrait") but carry no WikiArt path either."""
    pattern = label_pattern(real)
    job = real["job"]
    strings = [p.name for p in job.rglob("*")]
    for f in ("rows_manifest.npz", "verbalise_input.npz"):
        with zipfile.ZipFile(job / f) as z:
            strings += z.namelist()
    man = npz_arrays(job / "rows_manifest.npz")
    strings += man["image_name"].tolist() + (job / "images.txt").read_text().splitlines()
    strings += list(walk_strings(json.loads((job / "job_record.json").read_text())))
    hits = sorted({x for x in strings if pattern.search(x)})
    assert not hits, hits[:5]
    styles = [p.name for p in WIKIART.iterdir() if p.is_dir()]
    assert not [c for c in man["caption"].tolist() if any(f"{st}/" in c for st in styles)]
    m = json.loads(I.image_map_path(job).read_text())                   # the CPU-only map does hold the paths
    assert any(pattern.search(rel) for rel in m["images"].values())


def test_first_per_pair_through_the_command_line(real):
    out = real["base"] / "jobs" / "s9001_first16"
    assert I.main(["--episodes", str(real["path"]), "--out", str(out), "--first-per-pair", "16",
                   "--image-staging", str(real["staging"])]) == 0
    vin = npz_arrays(out / "verbalise_input.npz")
    want = np.concatenate([np.arange(k * R.N_SMOKE, k * R.N_SMOKE + 16) for k in range(3)])
    assert np.array_equal(vin["episode_index"], want)
    for f in G.PAIR_FIELDS:
        assert np.array_equal(vin[f], getattr(real["eps"].pooled, f)[want])
    assert json.loads((out / "job_record.json").read_text())["first_per_pair"] == 16
    assert set((out / "images.txt").read_text().split()) <= set((real["job"] / "images.txt").read_text().split())
    with pytest.raises(AssertionError, match="has 64 episodes"):
        I.select_episodes(real["eps"], 65)


def test_existing_job_folder_fires_and_the_guard_matters(real, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    args = (real["eps"], real["data"].sample_ids, real["data"].paintings, real["annotations"], out, None, None,
            tmp_path / "images")
    with pytest.raises(AssertionError, match="never overwritten"):
        I.write_job(*args)
    mut = mutant(tmp_path, guard="no_overwrite")
    assert mut.write_job(*args)["n_episodes"] == 192                     # guard deleted: the folder is replaced


def test_staged_link_to_another_file_fires_and_the_guard_matters(real, tmp_path):
    m = json.loads(I.image_map_path(real["job"]).read_text())
    name, rel = sorted(m["images"].items())[0]
    other = sorted(set(m["images"].values()) - {rel})[0]
    staging = tmp_path / "images"
    staging.mkdir()
    os.symlink(WIKIART / other, staging / name)                         # a stale link under the name
    with pytest.raises(AssertionError, match="is not the link to"):
        I.stage_images({name: rel}, staging)
    mut = mutant(tmp_path, guard="staging_target")
    assert mut.stage_images({name: rel}, staging) == 1                  # guard deleted: the wrong image stays
    assert os.path.realpath(staging / name) == str(WIKIART / other)


def test_positional_join_fires_and_the_guard_matters(real, tmp_path):
    rows = npz_arrays(real["job"] / "rows_manifest.npz")["rows"]
    wrong = np.arange(R.N_ROWS, dtype=np.int64)                          # annotations[row]: the C6 mistake
    with pytest.raises(AssertionError, match="C6 join"):
        I.build_manifest(rows, wrong, real["data"].paintings, real["annotations"])
    mut = mutant(tmp_path, guard="c6_join")
    got, _ = mut.build_manifest(rows, wrong, real["data"].paintings, real["annotations"])
    assert got["caption"].tolist() == [real["annotations"][r]["caption"] for r in rows.tolist()]   # wrong captions


def test_non_unique_sample_ids_fire_and_the_guard_matters(real, tmp_path):
    rows = npz_arrays(real["job"] / "rows_manifest.npz")["rows"]
    sids = real["data"].sample_ids.copy()
    out_rows = np.setdiff1d(np.arange(R.N_ROWS), rows)[:2]
    sids[out_rows[0]] = sids[out_rows[1]]                                # a repeated id on rows the job does not use
    with pytest.raises(AssertionError, match="uniquely index"):
        I.build_manifest(rows, sids, real["data"].paintings, real["annotations"])
    mut = mutant(tmp_path, guard="sample_ids")
    assert len(mut.build_manifest(rows, sids, real["data"].paintings, real["annotations"])[0]["rows"]) == len(rows)


def test_verbaliser_check_only_on_the_real_job(real):
    if not (G.snapshot_dir(HUB, "Qwen/Qwen3-VL-8B-Instruct", G.load_settings()[0]["model"]["snapshot"])).is_dir():
        pytest.skip("no local copy of the 8B model")
    r = subprocess.run([sys.executable, str(HERE / "r6_gpu_verbalise.py"), "--job-dir", str(real["job"]), "--out",
                        str(real["base"] / "v_out"), "--wordings", "W1,W2,W3,W4", "--check-only", "--no-model",
                        "--hub-cache", str(HUB), "--image-dir", str(real["staging"])],
                       capture_output=True, text=True, env=ENV, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    assert f"seed {SEED}, 192 episodes, range [0, 192)" in r.stdout and "stopping before the model" in r.stdout


def test_processor_renders_the_real_messages(real):
    """The snapshot's processor (CPU; no model weights) renders a real verbaliser message with its 8 images, read
    through the neutral links, and no path in the text; the listing's batched tokenisation (render, then the
    tokenizer with left padding) gives the ids of the processor's own tokenisation."""
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
    msgs = V.episode_messages(inp, man, 5, "a", "W2", s, real["staging"])
    enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt")
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vision_start = proc.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    assert int((enc["input_ids"] == vision_start).sum()) == 8 and enc["image_grid_thw"].shape[0] == 8
    assert ".jpg" not in text and str(real["staging"]) not in text
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
    assert f"r6_images_chunk0: selected, {n} files" in r.stdout
    assert f"{real['staging'].resolve()} -> /local/wding/r6_jobs/images" in r.stdout
    assert "-> /local/wding/r6_jobs/s9001_full" in r.stdout and "Plan only; pass --run to copy." in r.stdout
    assert "wikiart" not in r.stdout.split("Plan for")[1].lower()       # no WikiArt path in the plan's transfers
