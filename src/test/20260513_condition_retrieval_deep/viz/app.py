"""
CoSiR Condition Retrieval Visualizer
Run: uvicorn app:app --reload --port 8765
"""

import os, sys, importlib.util, torch, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sklearn.decomposition import PCA
import torch.nn.functional as F

# Load combiner directly to bypass src/model/__init__.py (which imports cuml)
_combiner_path = Path(__file__).parent.parent.parent.parent.parent / "src/model/combiner.py"
_spec = importlib.util.spec_from_file_location("combiner", _combiner_path)
_combiner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_combiner_mod)
Combiner_new = _combiner_mod.Combiner_new

EXPERIMENT_DIR = Path(
    "/project/CoSiR/res/CoSiR_Experiment/impressions"
    "/20260519_195824_CoSiR_Experiment"
)
TYPE_NAMES = ["caption", "description", "impression", "aesthetic"]
TYPE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

app = FastAPI()

DATA: dict = {}


def _r_at_k(ranks: torch.Tensor, k: int) -> float:
    return (ranks < k).float().mean().item() * 100


@app.on_event("startup")
async def startup():
    global DATA
    print("Loading data…")

    fixed = torch.load(
        EXPERIMENT_DIR / "condition_viz/fixed_data.pt",
        map_location="cpu", weights_only=False,
    )
    img_emb  = fixed["all_img_emb"]           # [706, 512]
    txt_emb  = fixed["all_txt_emb"]           # [2824, 512]
    img_paths = fixed["image_paths"]          # list[706]
    raw_text  = fixed["all_raw_text"]         # list[2824]
    img2txt   = fixed["image_to_text_map"]    # [706, 4]
    txt2img   = fixed["text_to_image_map"]    # [2824]
    cap_types = fixed["test_caption_types"]   # [2824]
    n_img, n_txt = img_emb.shape[0], txt_emb.shape[0]

    epoch_data = torch.load(
        EXPERIMENT_DIR / "condition_viz/epoch_0999.pt",
        map_location="cpu", weights_only=False,
    )
    reps       = epoch_data["representatives"]         # [30, 16]
    lbl_all    = epoch_data["label_embeddings_all"]    # [12123, 16]
    tr_types   = epoch_data["train_sample_types"]      # [12123]
    cfg        = epoch_data["combiner_config"]
    K          = reps.shape[0]

    ca = torch.load(
        EXPERIMENT_DIR / "condition_analysis/epoch_0999.pt",
        map_location="cpu", weights_only=False,
    )

    # ── Combiner ──────────────────────────────────────────────────────────────
    combiner = Combiner_new(
        clip_feature_dim=cfg["clip_feature_dim"],
        label_dim=cfg["label_dim"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
    )
    combiner.load_state_dict(epoch_data["combiner_state_dict"])
    combiner.eval()

    # ── Other-side projection (txt is other side for img-combine) ─────────────
    import torch.nn as nn
    _fd = cfg["clip_feature_dim"]
    other_proj = nn.Linear(_fd, _fd)
    if "other_proj_state_dict" in epoch_data:
        other_proj.load_state_dict(epoch_data["other_proj_state_dict"])
    else:
        nn.init.eye_(other_proj.weight)
        nn.init.zeros_(other_proj.bias)
    other_proj.eval()
    with torch.no_grad():
        txt_emb_proj = other_proj(txt_emb)  # projected text for model-based sims

    # Conditioned image embeddings for all reps [K, n_img, 512]
    print("Computing conditioned embeddings…")
    cond_img = torch.zeros(K, n_img, 512)
    with torch.no_grad():
        for ri in range(K):
            c = reps[ri].unsqueeze(0).expand(n_img, -1)
            cond_img[ri] = combiner(img_emb, None, c)

    # ── Condition-space PCA ───────────────────────────────────────────────────
    all_conds = torch.cat([lbl_all, reps], dim=0).numpy()
    pca_c = PCA(n_components=2, random_state=42)
    all_c2d = pca_c.fit_transform(all_conds)
    train_c2d = all_c2d[:len(lbl_all)]
    rep_c2d   = all_c2d[len(lbl_all):]

    # ── Joint embedding PCA ───────────────────────────────────────────────────
    print("Computing joint PCA…")
    joint_all = torch.cat([img_emb, txt_emb, cond_img.reshape(-1, 512)], dim=0).numpy()
    pca_j = PCA(n_components=2, random_state=42)
    pca_j.fit(joint_all)

    img_2d       = pca_j.transform(img_emb.numpy())        # [n_img, 2]
    txt_2d       = pca_j.transform(txt_emb.numpy())        # [n_txt, 2]
    cond_img_2d  = np.zeros((K, n_img, 2))
    for ri in range(K):
        cond_img_2d[ri] = pca_j.transform(cond_img[ri].numpy())

    # ── Per-rep statistics ────────────────────────────────────────────────────
    clip_gt_rank     = ca["clip_gt_rank"]       # [n_txt]
    clip_i2t_gt_rank = ca["clip_i2t_gt_rank"]   # [n_img]

    # Clip sim matrices for rank dynamics
    clip_sims = txt_emb @ img_emb.T             # [n_txt, n_img]
    clip_gt_sims = clip_sims[torch.arange(n_txt), txt2img]  # [n_txt]
    clip_mask = torch.ones(n_txt, n_img, dtype=torch.bool)
    clip_mask[torch.arange(n_txt), txt2img] = False
    clip_selectivity = (clip_gt_sims.mean() - clip_sims[clip_mask].mean()).item()

    per_rep_stats = []
    for ri in range(K):
        t2i_ranks = ca["per_rep_gt_rank"][ri]       # [n_txt]
        i2t_ranks = ca["per_rep_i2t_gt_rank"][ri]   # [n_img]

        sims = txt_emb_proj @ cond_img[ri].T        # [n_txt, n_img]
        gt_sims = sims[torch.arange(n_txt), txt2img]
        ngt_mask = torch.ones(n_txt, n_img, dtype=torch.bool)
        ngt_mask[torch.arange(n_txt), txt2img] = False
        sel = (gt_sims.mean() - sims[ngt_mask].mean()).item()

        # Rank dynamics: displaced distractors and new distractors
        above_clip = clip_sims > clip_gt_sims.unsqueeze(1)  # [n_txt, n_img]
        above_cond = sims      > gt_sims.unsqueeze(1)
        displaced  = (above_clip & ~above_cond).sum(1).float().mean().item()
        new_dist   = (~above_clip & above_cond).sum(1).float().mean().item()
        frac_imp   = (t2i_ranks < clip_gt_rank).float().mean().item()

        # Dominant type: nearest 100 training embeddings to this rep
        dists = ((lbl_all - reps[ri].unsqueeze(0)) ** 2).sum(-1)
        nn_idx = dists.topk(100, largest=False).indices
        type_counts = [int((tr_types[nn_idx] == t).sum()) for t in range(4)]

        # Per-type T2I R@1
        per_type_t2i_r1 = [
            _r_at_k(t2i_ranks[cap_types == t], 1) for t in range(4)
        ]
        per_type_i2t_r1 = []
        for t in range(4):
            # For I2T type t: among images, what fraction have their type-t GT text at rank 0
            top1_txt = ca["per_rep_i2t_topk_indices"][ri, :, 0]   # [n_img]
            gt_t = img2txt[:, t]                                    # [n_img]
            per_type_i2t_r1.append(float((top1_txt == gt_t).float().mean().item() * 100))

        per_rep_stats.append({
            "t2i_r1":  _r_at_k(t2i_ranks, 1),
            "t2i_r5":  _r_at_k(t2i_ranks, 5),
            "t2i_r10": _r_at_k(t2i_ranks, 10),
            "i2t_r1":  _r_at_k(i2t_ranks, 1),
            "i2t_r5":  _r_at_k(i2t_ranks, 5),
            "i2t_r10": _r_at_k(i2t_ranks, 10),
            "selectivity":  round(sel, 4),
            "displaced":    round(displaced, 2),
            "new_distractors": round(new_dist, 2),
            "frac_improved": round(float(frac_imp) * 100, 1),
            "type_counts":   type_counts,
            "per_type_t2i_r1": [round(x, 1) for x in per_type_t2i_r1],
            "per_type_i2t_r1": [round(x, 1) for x in per_type_i2t_r1],
        })

    clip_top1_t2i = ca["per_rep_topk_indices"][0]  # just shape ref; use clip_gt_rank
    per_type_clip_t2i = [_r_at_k(clip_gt_rank[cap_types == t], 1) for t in range(4)]
    per_type_clip_i2t = []
    for t in range(4):
        top1 = (img_emb @ txt_emb.T).argmax(dim=1)  # [n_img] best txt per img
        gt_t = img2txt[:, t]
        per_type_clip_i2t.append(float((top1 == gt_t).float().mean().item() * 100))

    DATA = {
        "n_img": n_img, "n_txt": n_txt, "K": K,
        "image_paths": img_paths,
        "raw_text": raw_text,
        "cap_types": cap_types.tolist(),
        "img2txt": img2txt.tolist(),
        "txt2img": txt2img.tolist(),
        # Condition space
        "train_c2d": train_c2d.tolist(),
        "train_types": tr_types.tolist(),
        "rep_c2d": rep_c2d.tolist(),
        # Joint embedding PCA
        "img_2d": img_2d.tolist(),
        "txt_2d": txt_2d.tolist(),
        "cond_img_2d": cond_img_2d.tolist(),
        # Stats
        "per_rep_stats": per_rep_stats,
        "clip_stats": {
            "t2i_r1":  _r_at_k(clip_gt_rank, 1),
            "t2i_r5":  _r_at_k(clip_gt_rank, 5),
            "t2i_r10": _r_at_k(clip_gt_rank, 10),
            "i2t_r1":  _r_at_k(clip_i2t_gt_rank, 1),
            "i2t_r5":  _r_at_k(clip_i2t_gt_rank, 5),
            "i2t_r10": _r_at_k(clip_i2t_gt_rank, 10),
            "selectivity": round(clip_selectivity, 4),
            "per_type_t2i_r1": [round(x, 1) for x in per_type_clip_t2i],
            "per_type_i2t_r1": [round(x, 1) for x in per_type_clip_i2t],
        },
        # Retrieval cache (kept for /api/condition endpoint)
        "_ca": {
            "per_rep_gt_rank":         ca["per_rep_gt_rank"],
            "per_rep_topk_idx":        ca["per_rep_topk_indices"],
            "per_rep_topk_scores":     ca["per_rep_topk_scores"],
            "per_rep_i2t_gt_rank":     ca["per_rep_i2t_gt_rank"],
            "per_rep_i2t_topk_idx":    ca["per_rep_i2t_topk_indices"],
            "per_rep_i2t_topk_scores": ca["per_rep_i2t_topk_scores"],
            "clip_gt_rank":            clip_gt_rank,
            "clip_i2t_gt_rank":        clip_i2t_gt_rank,
        },
    }
    print("Ready.")


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/data")
async def get_data():
    """Return all metadata and stats (no heavy retrieval arrays)."""
    d = {k: v for k, v in DATA.items() if not k.startswith("_")}
    return JSONResponse(d)


@app.get("/api/condition/{rep_idx}")
async def get_condition(rep_idx: int):
    if rep_idx < 0 or rep_idx >= DATA["K"]:
        raise HTTPException(404, "rep_idx out of range")

    ca      = DATA["_ca"]
    n_txt   = DATA["n_txt"]
    n_img   = DATA["n_img"]
    txt2img = torch.tensor(DATA["txt2img"])
    img2txt = torch.tensor(DATA["img2txt"])
    cap_types = torch.tensor(DATA["cap_types"])
    clip_gt  = ca["clip_gt_rank"]
    clip_i2t = ca["clip_i2t_gt_rank"]

    t2i_ranks = ca["per_rep_gt_rank"][rep_idx]      # [n_txt]
    i2t_ranks = ca["per_rep_i2t_gt_rank"][rep_idx]  # [n_img]
    t2i_topk  = ca["per_rep_topk_idx"][rep_idx]     # [n_txt, 10]
    t2i_scores = ca["per_rep_topk_scores"][rep_idx] # [n_txt, 10]
    i2t_topk  = ca["per_rep_i2t_topk_idx"][rep_idx] # [n_img, 10]
    i2t_scores = ca["per_rep_i2t_topk_scores"][rep_idx]

    # Best T2I queries: largest improvement in rank
    t2i_improvement = (clip_gt - t2i_ranks).float()  # positive = better
    best_t2i_idx = t2i_improvement.argsort(descending=True)[:16].tolist()

    best_t2i_queries = []
    for qi in best_t2i_idx:
        best_t2i_queries.append({
            "txt_idx":    int(qi),
            "text":       DATA["raw_text"][qi],
            "type":       int(cap_types[qi]),
            "clip_rank":  int(clip_gt[qi]),
            "cond_rank":  int(t2i_ranks[qi]),
            "improvement": int(t2i_improvement[qi]),
            "top10_img_idx": t2i_topk[qi].tolist(),
            "top10_scores":  [round(float(s), 4) for s in t2i_scores[qi].tolist()],
            "gt_img_idx":   int(txt2img[qi]),
        })

    # Best I2T images: largest improvement
    i2t_improvement = (clip_i2t - i2t_ranks).float()
    best_i2t_idx = i2t_improvement.argsort(descending=True)[:16].tolist()

    best_i2t_images = []
    for ii in best_i2t_idx:
        best_i2t_images.append({
            "img_idx":    int(ii),
            "clip_rank":  int(clip_i2t[ii]),
            "cond_rank":  int(i2t_ranks[ii]),
            "improvement": int(i2t_improvement[ii]),
            "top10_txt_idx": i2t_topk[ii].tolist(),
            "top10_scores":  [round(float(s), 4) for s in i2t_scores[ii].tolist()],
            "gt_txt_idxs":   img2txt[ii].tolist(),
        })

    return JSONResponse({
        "rep_idx": rep_idx,
        "stats": DATA["per_rep_stats"][rep_idx],
        "best_t2i_queries": best_t2i_queries,
        "best_i2t_images":  best_i2t_images,
    })


@app.get("/img/{img_idx}")
async def serve_image(img_idx: int):
    if img_idx < 0 or img_idx >= DATA["n_img"]:
        raise HTTPException(404)
    p = DATA["image_paths"][img_idx]
    if not os.path.exists(p):
        raise HTTPException(404, f"File not found: {p}")
    return FileResponse(p, media_type="image/png")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
