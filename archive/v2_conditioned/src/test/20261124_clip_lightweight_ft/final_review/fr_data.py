"""Final review: data integrity on the real cache and data (CPU, sampled). Never indexes a held row's annotation,
image or feature; held painting ids are read only to test the cache's disjointness (as the builder's guard does).

Run (CPU): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  /root/miniconda3/envs/CoSiR/bin/python fr_data.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FT = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FT))
from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
import ft_train  # noqa: E402  (the trainer's own data path)

CACHE = Path("/data/SSD2/pre_extract/artelingo_clip224")
WIKI = Path("/data/PDD/wikiart_proj/wikiart")
CLIP = "openai/clip-vit-base-patch32"
out = {}


def say(k, v):
    out[k] = v
    print(k, ":", v, flush=True)


def main():
    torch.set_num_threads(8)
    rng = np.random.default_rng(20261007)
    data = load_artelingo()
    sp = artelingo_splits(data)
    paint = np.asarray(data.paintings)
    sids = np.asarray(data.sample_ids)
    ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
    split = {"scorer_train": np.sort(sp.scorer_train), "val": np.sort(sp.val), "selection": np.sort(sp.selection)}
    nonheld = np.concatenate(list(split.values()))
    held_p = set(paint[np.asarray(sp.held)].tolist())                     # ids only

    # --- splits: painting- and group-disjoint
    ps = {k: set(paint[v].tolist()) for k, v in split.items()}
    gs = {k: set(np.asarray(sp.groups)[v].tolist()) for k, v in split.items()}
    gh = set(np.asarray(sp.groups)[np.asarray(sp.held)].tolist())
    say("split_paintings", {k: len(v) for k, v in ps.items()})
    say("painting_overlaps", {f"{a}&{b}": len(ps[a] & ps[b]) for a in ps for b in ps if a < b}
        | {f"{a}&held": len(ps[a] & held_p) for a in ps})
    say("group_overlaps", {f"{a}&{b}": len(gs[a] & gs[b]) for a in gs for b in gs if a < b}
        | {f"{a}&held": len(gs[a] & gh) for a in gs})

    # --- cache index and integrity
    rec = json.loads((CACHE / "cache_record.json").read_text())
    h = hashlib.sha256()
    with open(CACHE / "images_uint8.npy", "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    say("cache_sha_ok", h.hexdigest() == rec["sha256"]["images_uint8.npy"])
    pj = json.loads((CACHE / "paintings.json").read_text())["paintings"]
    cache_p = set(pj)
    say("cache_n", len(pj))
    say("cache_eq_nonheld_paintings", cache_p == (ps["scorer_train"] | ps["val"] | ps["selection"]))
    say("cache_held_overlap", len(cache_p & held_p))
    order = sorted(pj)
    say("cache_index_is_sorted_order", all(pj[p]["index"] == k for k, p in enumerate(order)))
    # image path per painting equals every non-held row's annotation image
    bad = 0
    for r in nonheld:
        if ann[int(sids[r])]["image"] != pj[str(paint[r])]["image"]:
            bad += 1
    say("rows_whose_annotation_image_differs_from_cache_index", bad)
    # annotation painting key equals data.paintings on non-held rows (the positional join), if the key exists
    if "painting" in ann[int(sids[nonheld[0]])]:
        say("rows_whose_annotation_painting_differs", int(sum(ann[int(sids[r])]["painting"] != paint[r] for r in nonheld)))

    # --- cache images vs WikiArt files (fresh decode with the CLIP processor's resize + centre crop)
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
    proc = CLIPImageProcessor.from_pretrained(CLIP, do_normalize=False, do_rescale=False)
    images = np.load(CACHE / "images_uint8.npy", mmap_mode="r")

    def decode(path):
        with Image.open(path) as im:
            px = proc(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]
        return np.clip(np.rint(px), 0, 255).astype(np.uint8).transpose(1, 2, 0)

    samp = rng.choice(len(order), 48, replace=False)
    diffs = []
    for k in samp:
        p = order[k]
        diffs.append(int(np.abs(decode(WIKI / pj[p]["image"]).astype(int) - images[pj[p]["index"]].astype(int)).max()))
    say("cache_vs_wikiart_max_abs_uint8_diff_48_paintings", max(diffs))
    # negative control: neighbouring cache row differs
    say("cache_neighbour_control_max_diff", int(np.abs(decode(WIKI / pj[order[samp[0]]]["image"]).astype(int)
                                                      - images[(pj[order[samp[0]]]["index"] + 1) % len(order)].astype(int)).max()))

    # --- the trainer's own training-pair path on the full scorer-train set
    tok = CLIPTokenizer.from_pretrained(CLIP)
    idx = {k: v for k, v in split.items()}
    cache_index = {p: v["index"] for p, v in pj.items()}
    train = ft_train.make_rowset("LB", data, idx["scorer_train"], ann, tok, cache_index)
    ds = ft_train.train_dataset(train, CACHE)
    say("train_rows_paintings", [len(train.rows), len(train.paintings)])
    js = rng.choice(len(train.rows), 64, replace=False)
    img_bad = cap_bad = wiki_bad = 0
    for n_j, j in enumerate(js):
        jj, u8, ids, mask = ds[int(j)]
        row = int(train.rows[j])
        a = ann[int(sids[row])]
        if not np.array_equal(u8.numpy(), images[cache_index[str(paint[row])]]):
            img_bad += 1
        if n_j < 16 and not np.array_equal(u8.numpy(), decode(WIKI / a["image"])):
            wiki_bad += 1
        enc = tok([a["caption"]], return_tensors="np", max_length=77, padding="max_length", truncation=True)
        if not (np.array_equal(ids.numpy(), enc["input_ids"][0]) and np.array_equal(mask.numpy(), enc["attention_mask"][0])):
            cap_bad += 1
    say("train_items_image_mismatch_vs_painting_cache_row_64", img_bad)
    say("train_items_image_mismatch_vs_wikiart_file_16", wiki_bad)
    say("train_items_caption_token_mismatch_64", cap_bad)
    # sampler: every epoch one row per painting, no painting twice in any batch (epochs 1..10)
    worst = 0
    for e in range(1, 11):
        o = ft_train.epoch_order(train.pos, e)
        assert len(o) == len(train.paintings) and len(np.unique(train.pos[o])) == len(o)
        for b in ft_train.batches(o, 256):
            worst = max(worst, len(b) - len(np.unique(train.pos[b])))
    say("sampler_epochs_1_10_duplicate_paintings_in_a_batch", worst)
    # groups with more than one painting id in scorer_train (identical image vectors under different ids)
    g = np.asarray(sp.groups)[train.rows]
    pg = {}
    for gi, p in zip(g.tolist(), paint[train.rows].tolist()):
        pg.setdefault(gi, set()).add(p)
    multi = [len(v) for v in pg.values() if len(v) > 1]
    say("scorer_train_groups_with_several_paintings", [len(multi), int(sum(multi))])

    # --- caption and image alignment against the frozen cached features (an independent check of sample_ids)
    model = CLIPModel.from_pretrained(CLIP).float().eval()
    rows = np.concatenate([rng.choice(v, 24, replace=False) for v in split.values()])
    caps = [ann[int(sids[r])]["caption"] for r in rows]
    with torch.no_grad():
        enc = tok(caps, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        t = model.text_projection(model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output)
    ref = torch.from_numpy(data.txt_features[rows])
    cos = torch.nn.functional.cosine_similarity(t, ref, dim=-1)
    say("caption_vs_cached_txt_cos_min_72_rows", float(cos.min()))
    say("caption_vs_cached_txt_max_abs", float((t - ref).abs().max()))
    shift = torch.nn.functional.cosine_similarity(t, torch.roll(ref, 1, 0), dim=-1)
    say("caption_control_shifted_rows_cos_max", float(shift.max()))
    stock = CLIPImageProcessor.from_pretrained(CLIP)
    pix = []
    for r in rows[:24]:
        with Image.open(WIKI / ann[int(sids[r])]["image"]) as im:
            pix.append(stock(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0])
    with torch.no_grad():
        v = model.visual_projection(model.vision_model(pixel_values=torch.stack(pix)).pooler_output)
    vref = torch.from_numpy(data.img_features[rows[:24]])
    say("image_vs_cached_img_cos_min_24_rows", float(torch.nn.functional.cosine_similarity(v, vref, dim=-1).min()))
    # frozen image features within a painting: identical across its rows?
    nh_p = paint[nonheld]
    o = np.argsort(nh_p, kind="stable")
    rs, pp = nonheld[o], nh_p[o]
    firsts = np.r_[0, np.nonzero(pp[1:] != pp[:-1])[0] + 1]
    first_of = np.repeat(firsts, np.diff(np.r_[firsts, len(pp)]))
    dev = np.abs(data.img_features[rs] - data.img_features[rs[first_of]]).max(1)
    say("frozen_img_rows_differing_from_their_paintings_first_row", int((dev > 0).sum()))
    (HERE / "fr_data.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
