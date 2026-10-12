"""Encoder extraction and ESC1 (DECISION_RULE M10, section 5 stage E; spec B: ED4, ED5, ED7, ESC1, 3.4).

    python r8_enc_extract.py --encoder {aladin,dinov2l,siglip2} [--smoke N] [--device cuda|cpu] [--batch-size B]

Embeds the 49,121 non-held images (r8_common.nonheld_images(), ascending CSD order) from the original files, each
encoder with its own preprocessing and pinned checkpoint, and stores them raw (not normalised) in
/data/SSD2/pre_extract/artelingo/r8_enc_<name>/ (smoke: results/smoke/enc/r8_enc_<name>/):

  embeddings.npy   float32 [n, d], row j = image image_index[j]
  image_index.npy  int64 [n]; its SHA-256 is that of the contiguous int64 bytes (ESC1 d)
  meta.json        keys listed in META_KEYS below; written last (its existence = the extraction finished)

then ESC1 (a) to (e) and results/stageE_<AL|DI|SG>.json (smoke: results/smoke/enc/stageE_<..>.json), top-level
`pass` and `smoke`; `evidence` = {name: {path, sha256}} of every file ESC1 (b) to (d) (and (e)) is computed from, for
the C13 re-derivation (rule section 7 d):
  (b) shape, (c) finite : embeddings (reread from disk after writing)
  (d) index hash        : image_index (stored) and row_to_image (the source the start hash is computed from)
  (e) repeat            : repeat_embeddings + repeat_cosines (beside the record, in stageE_<..>_evidence/) vs embeddings
A failed stage records why and lists only the files that exist. A failure exits non-zero (the encoder drops, ED4).

No held image is opened (ED7): every dataset item asserts its CSD index is a non-held one before opening the file, and
the main process checks the indices that came back equal the expected ones.
GPU work runs under flock -n -o -E 75 /tmp/gpu0.lock (rule section 9); this file does not lock by itself.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

import r8_common as C

import torch
from PIL import Image

IMAGE_ROOT = Path("/data/PDD/wikiart_proj/wikiart")        # M10; CSD meta.json images.root
ALADIN_DIR = "/data/SSD2/third_party/ALADIN"
PYDEPS = "/data/SSD2/third_party/pydeps"
PRE = Path("/data/SSD2/pre_extract/artelingo")
HF = Path("/root/.cache/huggingface/hub")
COS_MIN = 0.9999                                             # ESC1 (e)
REPEAT_N = 200

ENCODERS = {
    "aladin": dict(code="AL", dim=256, batch=128,
                   checkpoint="/data/SSD2/third_party/ALADIN_weights/aladinVIT_bamfg_64.47.pt",
                   revision="ALADIN repo commit 3ddd1124bf4a009e649621c680adb03d74a27953",
                   inputs=("/data/SSD2/third_party/ALADIN/main.py", "/data/SSD2/third_party/ALADIN/vit_networks.py",
                           "/data/SSD2/third_party/ALADIN_weights/aladinVIT_bamfg_64.47.pt"),
                   preprocess="main.load_img (alpha pasted on white; 1-2 channel to RGB), Resize((224,224)) bilinear "
                              "no crop, ToTensor, Normalize(0.5,0.5); main.AViT.encode (fc2 style code), raw 256-d"),
    "dinov2l": dict(code="DI", dim=1024, batch=128, hf="facebook/dinov2-large",
                    revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                    preprocess="PIL.Image.open(p).convert('RGB'); AutoImageProcessor (resize shortest edge 256 "
                               "bicubic, centre crop 224, ImageNet mean/std); pooler_output = CLS after final norm"),
    "siglip2": dict(code="SG", dim=1152, batch=64, hf="google/siglip2-so400m-patch14-384",
                    revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
                    preprocess="PIL.Image.open(p).convert('RGB'); AutoProcessor image part (384x384); "
                               "AutoModel.get_image_features"),
}
for _n, _e in ENCODERS.items():
    if "hf" in _e:
        _snap = HF / f"models--{_e['hf'].replace('/', '--')}/snapshots/{_e['revision']}"
        _e["checkpoint"] = str(_snap / "model.safetensors")
        _e["inputs"] = (str(_snap / "model.safetensors"),)

META_KEYS = ("encoder", "code", "checkpoint", "checkpoint_sha256", "revision", "hf_model", "image_index_sha256",
             "image_index_file", "image_order", "n", "dim", "dtype", "embeddings_file", "embeddings_sha256",
             "preprocess", "batch_size", "device", "tf32", "smoke", "esc1", "evidence", "versions", "started",
             "finished", "seconds_full_pass", "images_per_second")


def sha_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def index_sha(idx):
    """ESC1 (d): SHA-256 of the contiguous int64 array."""
    return sha_bytes(np.asarray(idx).astype(np.int64))


# ---------------------------------------------------------------- data

class Images(torch.utils.data.Dataset):
    """Item i = (i, CSD index, preprocessed tensor). Opens only non-held images (B:ED7)."""
    def __init__(self, images, paths, allowed, load):
        self.images, self.paths, self.allowed, self.load = np.asarray(images), paths, allowed, load

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        j = int(self.images[i])
        assert self.allowed[j], f"held image {j} would be opened"          # B:ED7
        return j, self.load(IMAGE_ROOT / self.paths[j])


def worker_init(_):
    torch.set_num_threads(1)


# ---------------------------------------------------------------- encoders

def build_aladin(spec, device):
    sys.path.append(PYDEPS)                    # ml_collections (the env itself is untouched)
    sys.path.insert(0, ALADIN_DIR)
    import main as M                           # read-only clone
    import torchvision.transforms as T
    state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    model = M.AViT()
    res = model.load_state_dict(state, strict=True)                  # ESC1 (a): strict
    assert not res.missing_keys and not res.unexpected_keys, res
    model.eval().to(device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def load(p):
        return tf(M.load_img(p))

    @torch.no_grad()
    def embed(x):
        return model.encode(x.to(device))
    return load, embed


def _hf_model(spec, auto, device):
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(spec["hf"], revision=spec["revision"], local_files_only=True,
                                            output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}                         # ESC1 (a): strict
    assert not bad, f"loading info not empty: { {k: list(v)[:5] for k, v in bad.items()} }"
    return model.eval().to(device)


def build_dinov2l(spec, device):
    from transformers import AutoImageProcessor
    model = _hf_model(spec, None, device)
    proc = AutoImageProcessor.from_pretrained(spec["hf"], revision=spec["revision"], local_files_only=True)

    def load(p):
        im = Image.open(p).convert("RGB")
        return proc(images=im, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(x):
        return model(pixel_values=x.to(device)).pooler_output
    return load, embed


def build_siglip2(spec, device):
    from transformers import AutoProcessor
    model = _hf_model(spec, None, device)
    proc = AutoProcessor.from_pretrained(spec["hf"], revision=spec["revision"], local_files_only=True)

    def load(p):
        im = Image.open(p).convert("RGB")
        return proc(images=im, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(x):
        e = model.get_image_features(pixel_values=x.to(device))
        return getattr(e, "pooler_output", e)
    return load, embed


BUILD = dict(aladin=build_aladin, dinov2l=build_dinov2l, siglip2=build_siglip2)


def run_pass(images, paths, allowed, load, embed, batch, workers, dim, label):
    """Embeds ``images`` in order; returns float32 [n, dim]. Checks every index that comes back (B:ED7)."""
    ds = Images(images, paths, allowed, load)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                                     worker_init_fn=worker_init, persistent_workers=False)
    out = np.empty((len(images), dim), np.float32)
    pos, t0 = 0, time.time()
    for k, (idx, x) in enumerate(dl):
        n = len(idx)
        assert np.array_equal(idx.numpy(), np.asarray(images[pos:pos + n])), "batch order differs from the index"
        e = embed(x).float().cpu().numpy()
        assert e.shape == (n, dim), (e.shape, dim)
        out[pos:pos + n] = e
        pos += n
        if k % 50 == 0:
            print(f"[{label}] {pos}/{len(images)} {pos / (time.time() - t0):.1f} img/s", flush=True)
    assert pos == len(images)
    return out


# ---------------------------------------------------------------- files

def evid(path):
    return dict(path=str(path), sha256=C.sha_file(path))


def save_npy_once(path, a):
    path = C._prepare(path)
    with C._open_once(path) as f:
        np.save(f, a)
    return path


def versions():
    import PIL
    import torchvision
    v = dict(torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, PIL=PIL.__version__,
             python=sys.version.split()[0])
    try:
        import transformers
        v["transformers"] = transformers.__version__
    except ImportError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--smoke", type=int, default=0, help="first N non-held images only; outputs under results/smoke/enc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert a.workers <= 8
    spec, name = ENCODERS[a.encoder], a.encoder
    smoke = a.smoke > 0
    out_dir = (C.SMOKE / "enc" / f"r8_enc_{name}") if smoke else PRE / f"r8_enc_{name}"
    rec_dir = (C.SMOKE / "enc") if smoke else C.RESULTS
    stage_path = rec_dir / f"stageE_{spec['code']}.json"
    evid_dir = rec_dir / f"stageE_{spec['code']}_evidence"
    if not smoke:
        assert a.device == "cuda", "full runs on the GPU"                # rule gap: M10 says local GPU under the lock
        for p in (stage_path, out_dir / "embeddings.npy"):
            if p.exists():
                raise SystemExit(f"{p} exists; refusing to overwrite")
    elif a.device == "cuda":
        assert torch.cuda.is_available()
    batch = a.batch_size or spec["batch"]
    started = C.now_ams()
    state = dict(files={}, items={}, reason=None)
    t_start = time.time()
    try:
        C.assert_rule()
        torch.backends.cuda.matmul.allow_tf32 = False                    # M1: GPU passes
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        # ESC1 (a): the weight SHA-256 of section 10 (assert_inputs stops on a difference), then the strict load
        sha = C.assert_inputs(keys=spec["inputs"], r7=False)
        weight_sha = sha[spec["checkpoint"]]
        state["items"]["a_weights_sha256"] = dict(path=spec["checkpoint"], sha256=weight_sha, ok=True)
        # the images: only non-held ones, in CSD ascending order
        nh = C.nonheld_images()
        paths = json.loads((C.STYLE / "meta.json").read_text())["images"]["image_paths"]
        assert len(paths) == C.N_CSD_IMAGES
        allowed = np.zeros(C.N_CSD_IMAGES, bool)
        allowed[nh] = True
        assert allowed.sum() == C.N_NONHELD_IMAGES and (C.N_CSD_IMAGES - allowed.sum()) == 12_281, "held image count"
        n = a.smoke if smoke else len(nh)
        assert n <= len(nh)
        images = nh[:n].astype(np.int64)
        assert not (~allowed[images]).any()
        assert (np.diff(images) > 0).all()
        # ESC1 (d): the hash computed from row_to_image.npy at the start (unique image indices of non-held rows)
        from_r2i = np.unique(C.row_to_image()[C.ctx().nonheld]).astype(np.int64)
        assert np.array_equal(from_r2i[:n], images)
        start_hash = index_sha(from_r2i[:n])
        dev = torch.device(a.device)
        load, embed = BUILD[name](spec, dev)
        state["items"]["a_strict_load"] = dict(ok=True)
        print(f"[{name}] loaded; n={n} batch={batch} device={a.device}", flush=True)

        t0 = time.time()
        emb = run_pass(images, paths, allowed, load, embed, batch, a.workers, spec["dim"], name)
        full_seconds = time.time() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        files = state["files"]
        emb_path = save_npy_once(out_dir / "embeddings.npy", emb)
        idx_path = save_npy_once(out_dir / "image_index.npy", images)
        files["embeddings"], files["image_index"] = evid(emb_path), evid(idx_path)
        files["row_to_image"] = evid(C.STYLE / "row_to_image.npy")

        # ESC1 (e): the first min(200, n) images again, in a separate pass after the full pass
        m = min(REPEAT_N, n)
        rep = run_pass(images[:m], paths, allowed, load, embed, min(batch, m), a.workers, spec["dim"], name + " repeat")
        evid_dir.mkdir(parents=True, exist_ok=True)
        rep_path = save_npy_once(evid_dir / "repeat_embeddings.npy", rep)
        files["repeat_embeddings"] = evid(rep_path)

        # ESC1 (b) to (d), (e) are computed from the stored files, reread from disk
        e_disk = np.load(emb_path, mmap_mode="r")
        i_disk = np.load(idx_path)
        r_disk = np.load(rep_path)
        shape_ok = e_disk.shape == (n, spec["dim"]) and (smoke or e_disk.shape == (C.N_NONHELD_IMAGES, spec["dim"]))
        finite_ok = bool(np.isfinite(e_disk).all())
        stored_hash = index_sha(i_disk)
        hash_ok = i_disk.dtype == np.int64 and stored_hash == start_hash
        a_ = np.asarray(e_disk[:m], np.float64)
        b_ = np.asarray(r_disk, np.float64)
        cos = (a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1))
        cos_path = save_npy_once(evid_dir / "repeat_cosines.npy", cos)
        files["repeat_cosines"] = evid(cos_path)
        rep_ok = bool(len(cos) == m and np.isfinite(cos).all() and (cos >= COS_MIN).all())
        state["items"].update(
            b_shape=dict(ok=bool(shape_ok), shape=list(e_disk.shape), expected_rows=n if smoke else C.N_NONHELD_IMAGES),
            c_finite=dict(ok=finite_ok),
            d_index_hash=dict(ok=bool(hash_ok), stored=stored_hash, from_row_to_image_at_start=start_hash,
                              dtype=str(i_disk.dtype)),
            e_repeat=dict(ok=rep_ok, n=m, min_cosine=float(cos.min()), threshold=COS_MIN))
        esc1_ok = all(v["ok"] for v in state["items"].values())
        emb_sha = files["embeddings"]["sha256"]
        meta = dict(
            encoder=name, code=spec["code"], checkpoint=spec["checkpoint"], checkpoint_sha256=weight_sha,
            revision=spec["revision"], hf_model=spec.get("hf"), image_index_sha256=stored_hash,
            image_index_file=str(idx_path),
            image_order="r8_common.nonheld_images()[:n]: ascending CSD image index of non-held rows (row_to_image.npy)",
            n=int(n), dim=spec["dim"], dtype="float32", embeddings_file=str(emb_path), embeddings_sha256=emb_sha,
            preprocess=spec["preprocess"], batch_size=batch, device=a.device, tf32=False, smoke=smoke,
            esc1=dict(pass_=esc1_ok, items=state["items"]), evidence=files, versions=versions(), started=started,
            finished=C.now_ams(), seconds_full_pass=full_seconds, images_per_second=n / full_seconds)
        meta["esc1"]["pass"] = meta["esc1"].pop("pass_")
        assert set(META_KEYS) <= set(meta)
        C.write_json_once(out_dir / "meta.json", meta)
        files["meta"] = evid(out_dir / "meta.json")
        if not esc1_ok:
            state["reason"] = "ESC1 failed: " + ", ".join(k for k, v in state["items"].items() if not v["ok"])
        ok = esc1_ok
        extra = dict(n=int(n), dim=spec["dim"], embeddings_sha256=emb_sha, image_index_sha256=stored_hash,
                     seconds_full_pass=full_seconds, images_per_second=n / full_seconds)
    except BaseException as ex:                                          # B:ED4: a failed load or item drops the encoder
        traceback.print_exc()
        ok, extra = False, {}
        state["reason"] = f"{type(ex).__name__}: {ex}"
    rec = dict(encoder=name, code=spec["code"], smoke=smoke, **{"pass": bool(ok)}, reason=state["reason"],
               checkpoint=spec["checkpoint"], revision=spec["revision"], batch_size=batch, device=a.device,
               esc1=state["items"], evidence=state["files"], started=started, finished=C.now_ams(),
               seconds=time.time() - t_start, **extra)
    C.write_json_once(stage_path, rec)
    print(f"[{name}] ESC1 {'PASS' if ok else 'FAIL'}; record {stage_path}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
