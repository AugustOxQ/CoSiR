"""Round 6 fine-tuned CLIP features for the reported baselines FT-LB and FT-LoRA (DECISION_RULE.md of this folder:
section 2, section 8 item 3; contracts section 9; gpu_path section 4, job J4): the raw projections (no normalisation)
of the images and captions of the rows of ft_rows.npz, from the selected checkpoints (LB lr 3e-5 epoch 8, LoRA lr 1e-4
epoch 10; best_params.pt of the clipft runs).

Inputs. The job folder (rows_manifest.npz and ft_rows.npz, written by r6_gpu_inputs.py): row ids, neutral image names
and captions. The images come from a uint8 224 cache (r6_ft_cache.py, --cache-dir; entry per neutral name) or, without
one, are decoded from <image dir>/<name> with the same decode. The job reads no label, aspect name, path or candidate
order and computes no metric.

Imports. Unlike the other GPU modules this one imports the clipft training code (ft_train.build_model / load_trainable /
encode_rowset, from src/test/20261124_clip_lightweight_ft in the code checkout): the model, the LoRA configuration and
the encoder calls are then the ones that trained and selected the checkpoints. ft_train imports torch, ft_data and (lazily,
not here) src. r6_common is not imported.

Output (<out>, i.e. outputs/r6_ft/<job>/): features_<variant>.npz {rows int64 ascending, img float32 (n, 512), txt float32
(n, 512)} (the layout of ft_train.write_features) and provenance.json (checkpoints' SHA-256, scripts, inputs, versions).
A variant whose file exists is skipped on a rerun.

Run (DAS6 through scripts/run_r6_ftfeat.sh):
    python r6_gpu_ft_features.py --job-dir <job> --out <dir> --variants LB,LoRA --ckpt LB=<best_params.pt> --ckpt LoRA=<..>
        [--cache-dir <r6_ft_cache folder>] [--image-dir <neutral images>] [--device cuda]
    ... --check-only [--reference LB=<clipft features.npz>]
        inputs, images, checkpoints, then one batch of 4 rows through the full path (CPU with --device cpu); with a
        reference file the 4 rows are compared with its stored features (max abs difference and minimum cosine are
        printed: they are not metrics).
Guards carry a `# guard:<name>` marker; the tests delete each on a copy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_ft_cache as C  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_t12 as T  # noqa: E402

FT_DIR = HERE.parents[1] / "test" / "20261124_clip_lightweight_ft"
SELECTED = {"LB": {"lr": 3e-5, "epoch": 8}, "LoRA": {"lr": 1e-4, "epoch": 10}}
CHECK_ROWS = 4


def import_ft():
    """The clipft training module (ft_train), imported from this code checkout."""
    if str(FT_DIR) not in sys.path:
        sys.path.insert(0, str(FT_DIR))
    import ft_train
    return ft_train


def check_checkpoint(ckpt, variant):
    """The loaded best_params.pt is the selected run of ``variant`` (variant, lr and epoch)."""
    want = SELECTED[variant]
    G._require(ckpt.get("variant") == variant and float(ckpt.get("lr")) == want["lr"]
               and int(ckpt.get("epoch")) == want["epoch"] and "params" in ckpt,
               f"checkpoint is {ckpt.get('variant')} lr {ckpt.get('lr')} epoch {ckpt.get('epoch')}, "
               f"expected {variant} lr {want['lr']} epoch {want['epoch']}")  # guard:ckpt_identity


def load_variant(ft, variant, ckpt_path, device):
    """Pretrained CLIP B/32 with the variant's trainable parameters from the checkpoint, in eval mode."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    check_checkpoint(ckpt, variant)
    model = ft.build_model(variant, ft.load_clip())
    ft.load_trainable(model, ckpt["params"])
    return model.to(device).eval()


def images_for(names, cache_dir=None, image_dir=None):
    """(len(names), 224, 224, 3) uint8 for the neutral ``names``: rows of the cache, else decoded from image_dir."""
    if cache_dir is not None:
        images, index, _ = C.load_cache(cache_dir, verify=False)
        miss = [x for x in names if x not in index]
        G._require(not miss, f"{len(miss)} images are not in the cache {cache_dir}, e.g. {miss[:3]}")  # guard:cache_names
        return np.ascontiguousarray(images[np.asarray([index[x] for x in names], dtype=np.int64)])
    G._require(image_dir is not None, "need --cache-dir or --image-dir")
    return np.stack([C.decode(Path(image_dir) / x) for x in names])


def make_rowset(ft, manifest, rows, tokenizer):
    """-> (RowSet for ft.encode_rowset, uint8 images in its cache order are fetched by the caller from .paintings).
    Painting = neutral image name; position = rank among the distinct names of the rows."""
    rows = np.asarray(rows, dtype=np.int64)
    pos_in = manifest.index(rows)
    names = manifest.image_name[pos_in]
    uniq, pos = np.unique(names, return_inverse=True)
    _, first = np.unique(pos, return_index=True)
    ids, mask = ft.tokenize(tokenizer, manifest.caption[pos_in].tolist())
    return ft.RowSet(rows, pos.astype(np.int64), uniq, first.astype(np.int64),
                     cache_rows=np.arange(len(uniq), dtype=np.int64), ids=ids, mask=mask)


def encode(ft, model, variant, manifest, rows, tokenizer, device, cache_dir=None, image_dir=None, batch=None):
    """-> (img, txt) float32 numpy (len(rows), 512): the raw projections of the rows (fp32, eval mode)."""
    rs = make_rowset(ft, manifest, rows, tokenizer)
    images = images_for(rs.paintings.tolist(), cache_dir, image_dir)
    img, txt = ft.encode_rowset(model, variant, rs, device, images=images, batch=batch or ft.EVAL_BATCH)
    return img.numpy().astype(np.float32), txt.numpy().astype(np.float32)


def write_features(path, rows, img, txt):
    """features.npz layout: rows int64 ascending, img and txt float32 (n, 512); finite; atomic."""
    rows = np.asarray(rows, dtype=np.int64)
    G._require(bool((np.diff(rows) > 0).all()) and img.shape == txt.shape == (len(rows), 512)
               and img.dtype == txt.dtype == np.float32, f"{path}: rows, img and txt do not have the features layout")
    G._require(bool(np.isfinite(img).all() and np.isfinite(txt).all()), f"{path}: non-finite features")  # guard:finite
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, rows=rows, img=img, txt=txt)
    os.replace(tmp, path)


def compare_reference(path, rows, img, txt) -> dict:
    """Max abs difference and minimum cosine of (img, txt) of ``rows`` against a clipft features.npz (same layout)."""
    with np.load(path, allow_pickle=False) as z:
        ref_rows, ri, rt = z["rows"], z["img"], z["txt"]
    pos = np.searchsorted(ref_rows, rows)
    pos = np.minimum(pos, len(ref_rows) - 1)
    G._require(bool((ref_rows[pos] == rows).all()), f"{path}: not all check rows are in the reference file")
    out = {}
    for name, a, b in (("img", img, ri[pos]), ("txt", txt, rt[pos])):
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        out[f"{name}_max_abs_diff"] = float(np.abs(a - b).max())
        out[f"{name}_min_cos"] = float(cos.min())
    return out


def parse_pairs(items, what):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        G._require(k in SELECTED and v and k not in out, f"--{what} wants VARIANT=PATH with a new variant from LB, LoRA")
        out[k] = v
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Round 6 fine-tuned CLIP features (LB, LoRA); prints no metric")
    ap.add_argument("--job-dir", required=True, help="folder with rows_manifest.npz and ft_rows.npz")
    ap.add_argument("--out", required=True, help="output folder (features_<variant>.npz, provenance.json)")
    ap.add_argument("--variants", default="LB,LoRA", help="comma-separated, from LB, LoRA")
    ap.add_argument("--ckpt", action="append", default=[], help="VARIANT=best_params.pt (one per variant)")
    ap.add_argument("--cache-dir", default=None, help="r6_ft_cache folder (images by neutral name)")
    ap.add_argument("--image-dir", default=os.environ.get("R6_IMAGE_DIR") or G.DEFAULT_IMAGE_DIR,
                    help="folder of the neutral images, used when there is no --cache-dir")
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--batch", type=int, default=None, help="encode batch size (default ft_train.EVAL_BATCH)")
    ap.add_argument("--check-only", action="store_true", help="check inputs and checkpoints, encode 4 rows, write nothing")
    ap.add_argument("--reference", action="append", default=[], help="with --check-only: VARIANT=clipft features.npz")
    args = ap.parse_args(argv)
    args.variants = [v for v in args.variants.split(",") if v]
    if not args.variants or len(set(args.variants)) != len(args.variants) or not set(args.variants) <= set(SELECTED):
        ap.error(f"--variants must be distinct ids from {', '.join(SELECTED)}")
    args.ckpt = parse_pairs(args.ckpt, "ckpt")
    args.reference = parse_pairs(args.reference, "reference")
    if not set(args.variants) <= set(args.ckpt):
        ap.error("every variant needs a --ckpt VARIANT=PATH")
    if args.reference and not args.check_only:
        ap.error("--reference needs --check-only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    job = Path(args.job_dir)
    manifest = G.Manifest(job / "rows_manifest.npz")
    rows = T.load_ft_rows(job / "ft_rows.npz", manifest)
    use_cache = args.cache_dir is not None
    names = sorted(set(manifest.image_name[manifest.index(rows)].tolist()))
    if use_cache:
        _, index, _ = C.load_cache(args.cache_dir, verify=False)
        miss = [x for x in names if x not in index]
        G._require(not miss, f"{len(miss)} of {len(names)} images are not in the cache {args.cache_dir}")
    else:
        miss = [x for x in names if not (Path(args.image_dir) / x).is_file()]
        G._require(not miss, f"{len(miss)} of {len(names)} images missing under {args.image_dir}, e.g. {miss[:3]}")
    for v in args.variants:
        G._require(Path(args.ckpt[v]).is_file(), f"checkpoint missing: {args.ckpt[v]}")
    print(f"ft features inputs ok: {len(rows)} rows, {len(names)} images, variants {args.variants}", flush=True)
    ft = import_ft()
    import torch
    tokenizer = ft.load_tokenizer()
    ft_scripts = [FT_DIR / "ft_train.py", FT_DIR / "ft_data.py"]
    fingerprint = {"job": "r6_gpu_ft_features", "variants": args.variants, "selected": SELECTED,
                   "ckpt_sha256": {v: G.sha256_file(args.ckpt[v]) for v in args.variants},
                   "scripts_sha256": G.script_shas(__file__, G.__file__, C.__file__, *ft_scripts),
                   "inputs_sha256": {f: G.sha256_file(job / f) for f in ("rows_manifest.npz", "ft_rows.npz")},
                   "image_source": "r6_ft_cache" if use_cache else "decoded from the neutral images"}
    if args.check_only:
        sub = rows[:CHECK_ROWS]
        for v in args.variants:
            model = load_variant(ft, v, args.ckpt[v], args.device)
            img, txt = encode(ft, model, v, manifest, sub, tokenizer, args.device, args.cache_dir, args.image_dir)
            G._require(img.shape == txt.shape == (len(sub), 512) and bool(np.isfinite(img).all() and np.isfinite(txt).all()),
                       f"{v}: check batch did not give finite (n, 512) features")
            msg = f"check-only {v}: {len(sub)} rows encoded on {args.device}"
            if v in args.reference:
                cmp = compare_reference(args.reference[v], sub, img, txt)
                msg += ", vs reference " + ", ".join(f"{k} {x:.3g}" for k, x in cmp.items())
            print(msg, flush=True)
        return 0
    out = Path(args.out)
    prov = G.begin_provenance(out, fingerprint, {"args": {k: v for k, v in vars(args).items()}, "n_rows": len(rows)})
    done = {}
    for v in args.variants:
        target = out / f"features_{v}.npz"
        if target.exists():
            print(f"{target.name} exists; skipped", flush=True)
            continue
        model = load_variant(ft, v, args.ckpt[v], args.device)
        img, txt = encode(ft, model, v, manifest, rows, tokenizer, args.device, args.cache_dir, args.image_dir,
                          args.batch)
        write_features(target, rows, img, txt)
        done[v] = G.sha256_file(target)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{target.name}: {len(rows)} rows written", flush=True)
    G.end_provenance(out, prov, status="complete", versions=G.versions(), features_sha256=done)
    print(f"ft features: complete, outputs in {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
