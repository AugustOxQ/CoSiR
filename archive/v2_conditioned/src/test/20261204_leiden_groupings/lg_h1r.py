"""EXPLORATORY side check (decides nothing). Retrain the buddy line's Attention-h1 student on scorer-train paintings only.
Port of run_attention_h1_embedding_snapshot_pilot.py / run_learned_student_arch_sweep_pilot.py (buddy worktree, read-only).
Same hyperparameters and seed 42; only the data changes. Deviations (see results/h1r.json "deviations"):
  - plateau stopping used held-out paintings' recall; here the recall is measured on scorer-train (teacher graphs vs the
    student's own graph, 2,000 sampled train nodes), because no held painting may enter.
  - GoEmotions probabilities: the stored per-row affect_probs (src/test/20261018_affect_factor_learning/cache), mean-pooled
    per painting over the painting's scorer-train rows (the old code mean-pooled English caption rows).
  - painting nodes sorted by name; image = first scorer-train row, text = mean over the painting's scorer-train rows.
Run (from /project/CoSiR):
  flock -n -o -E 75 /tmp/gpu0.lock env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261204_leiden_groupings/lg_h1r.py
"""
import json
import sys
import time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import torch
from scipy.sparse import coo_matrix, csgraph
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.conditional_buddy.buddy_graph import ensure_connected, ensure_min_degree, mutual_knn  # noqa: E402
from src.conditional_buddy.compute_buddies import build_buddy_graphs  # noqa: E402
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

RES = HERE / "results"
AFFECT_NPZ = ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
D_SHARED, CONTENT_PCA_DIM, BATCH_SIZE, TEMPERATURE, LR = 32, 50, 1024, 0.1, 1e-3
MAX_EPOCHS, PLATEAU_WINDOW, PLATEAU_REL, CHECKPOINT_EVERY, EDGE_SAMPLE = 200, 5, 0.01, 5, 2000
SEED, K, ALPHA = 42, 20, 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class AttentionFusion(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=D_SHARED, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(D_SHARED)

    def forward(self, c, a):
        tokens = torch.stack((c, a), dim=1)
        attended, w = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        return F.normalize(self.norm(attended.mean(dim=1)), dim=1), w


class Student(nn.Module):  # "attn1"
    def __init__(self):
        super().__init__()
        self.proj_content = nn.Linear(CONTENT_PCA_DIM, D_SHARED)
        self.proj_affect = nn.Linear(28, D_SHARED)
        self.fusion = AttentionFusion(1)

    def forward(self, content, affect):
        return self.fusion(F.normalize(self.proj_content(content), dim=1), F.normalize(self.proj_affect(affect), dim=1))


def single_graph(x, device):
    x = l2(np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)
    adj = mutual_knn(x, K=K, device=device, use_half=True)
    g, _ = ensure_min_degree(adj, x, x, device=device, use_half=True)
    g, _ = ensure_connected(g, x, x, alpha=ALPHA, device=device, use_half=True)
    if csgraph.connected_components(g, directed=False, return_labels=False) != 1:
        raise RuntimeError("graph repair failed")
    return g


def detect_communities(E, seed=SEED):
    c = E.tocoo()
    m = c.row < c.col
    g = ig.Graph(n=E.shape[0], edges=list(zip(c.row[m].tolist(), c.col[m].tolist())))
    return np.array(leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed).membership, dtype=np.int64)


def upper_edges(g):
    g = coo_matrix(g)
    keep = g.row < g.col
    return np.column_stack((g.row[keep], g.col[keep])).astype(np.int64)


def sample_pairs(edges, rng):
    return edges[rng.choice(len(edges), size=BATCH_SIZE, replace=len(edges) < BATCH_SIZE)]


def infonce(emb, pairs, device):
    p = torch.as_tensor(pairs, dtype=torch.long, device=device)
    a, b = emb[p[:, 0]], emb[p[:, 1]]
    t = torch.arange(BATCH_SIZE, device=device)
    return 0.5 * (F.cross_entropy(a @ b.T / TEMPERATURE, t) + F.cross_entropy(b @ a.T / TEMPERATURE, t))


def overlap(ref, cmp_, nodes):
    o = []
    for n in nodes:
        r = ref[int(n)].indices
        if len(r):
            o.append(np.isin(r, cmp_[int(n)].indices).mean())
    return float(np.mean(o))


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = load_artelingo()
    scorer_train = np.asarray(artelingo_splits(data).scorer_train)
    names = np.asarray(data.paintings)[scorer_train].astype(str)
    paintings, g = np.unique(names, return_inverse=True)     # name-sorted painting nodes
    P = len(paintings)
    log(f"{len(scorer_train)} scorer-train rows, {P} paintings, device {device}")
    first = np.zeros(P, dtype=np.int64)
    first[g[::-1]] = np.arange(len(g))[::-1]
    img = np.asarray(data.img_features[scorer_train][first], dtype=np.float32)
    txt_rows = np.asarray(data.txt_features[scorer_train], dtype=np.float32)
    cnt = np.bincount(g, minlength=P).astype(np.float32)
    txt = np.zeros((P, txt_rows.shape[1]), np.float32)
    np.add.at(txt, g, txt_rows)
    txt /= cnt[:, None]
    z = np.load(AFFECT_NPZ)["affect_probs"]
    if z.shape != (len(scorer_train), 28):
        raise AssertionError("affect_probs rows differ from scorer_train")
    aff = np.zeros((P, 28), np.float32)
    np.add.at(aff, g, z)
    aff /= cnt[:, None]

    content = np.concatenate((l2(img), l2(txt)), axis=1).astype(np.float64)
    content = PCA(n_components=CONTENT_PCA_DIM, random_state=SEED).fit_transform(content).astype(np.float32)
    _, _, content_graph = build_buddy_graphs(img, txt, K=K, alpha=ALPHA, device=device, connect_components=True)
    affect_graph = single_graph(aff, device)
    c_edges, a_edges = upper_edges(content_graph), upper_edges(affect_graph)
    log(f"teacher edges: content {len(c_edges):,}, affect {len(a_edges):,}")
    content_t = torch.as_tensor(content, device=device)
    affect_t = torch.as_tensor(aff, device=device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Student().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    nodes = np.random.default_rng(SEED).choice(P, size=min(EDGE_SAMPLE, P), replace=False)

    def recalls():
        model.eval()
        with torch.no_grad():
            e, _ = model(content_t, affect_t)
        sg = single_graph(e.cpu().numpy(), device)
        return overlap(content_graph, sg, nodes), overlap(affect_graph, sg, nodes)

    traj = [recalls()]
    plateau, stop, last = 0, f"reached MAX_EPOCHS={MAX_EPOCHS}", {}
    t_train = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        cp, ap = sample_pairs(c_edges, rng), sample_pairs(a_edges, rng)
        ids, inv = np.unique(cp.reshape(-1), return_inverse=True)
        it = torch.as_tensor(ids, dtype=torch.long, device=device)
        ce, _ = model(content_t[it], affect_t[it])
        c_loss = infonce(ce, inv.reshape(-1, 2), device)
        ae, _ = model(content_t, affect_t)
        a_loss = infonce(ae, ap, device)
        loss = c_loss + a_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = {"epoch": epoch, "content_loss": float(c_loss), "affect_loss": float(a_loss)}
        if epoch % CHECKPOINT_EVERY:
            continue
        cur = recalls()
        prev = traj[-1]
        traj.append(cur)
        log(f"epoch {epoch} content_recall {cur[0]:.4f} affect_recall {cur[1]:.4f} closs {last['content_loss']:.3f} aloss {last['affect_loss']:.3f}")
        imp = lambda c, p: (c - p) / max(abs(p), 1e-12)  # noqa: E731
        plateau = plateau + 1 if imp(cur[0], prev[0]) < PLATEAU_REL and imp(cur[1], prev[1]) < PLATEAU_REL else 0
        if plateau >= PLATEAU_WINDOW:
            stop = f"both recalls plateaued for {PLATEAU_WINDOW} consecutive checkpoints"
            break
    train_s = time.time() - t_train
    model.eval()
    with torch.no_grad():
        e, _ = model(content_t, affect_t)
    comm = detect_communities(single_graph(e.cpu().numpy(), device))
    row_lab = np.unique(comm[g], return_inverse=True)[1].astype(np.int64)    # per scorer-train row, compact
    RES.mkdir(exist_ok=True)
    np.savez(RES / "h1r_labels.npz", h1r=row_lab, scorer_train=scorer_train)
    gz = np.load(RES / "lg_group.npz")
    if not np.array_equal(gz["scorer_train"], scorer_train):
        raise AssertionError("scorer_train differs from lg_group.npz")
    sizes = np.bincount(row_lab)
    out = {"n_communities": int(row_lab.max()) + 1, "n_paintings": int(P), "n_painting_communities": int(len(np.unique(comm))),
           "group_sizes_rows": sizes.tolist(), "min_group_rows": int(sizes.min()),
           "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1), "device": device,
           "stop_reason": stop, "epochs_run": last["epoch"], "final_losses": last,
           "recall_trajectory_train": traj, "ami_vs_old_h1": float(adjusted_mutual_info_score(gz["h1"], row_lab)),
           "old_h1_n_groups": int(gz["h1"].max()) + 1,
           "deviations": ["plateau recall on scorer-train, not held-out", "affect = stored per-row GoEmotions probs mean-pooled over scorer-train rows",
                          "txt/img per painting from scorer-train rows only", "GoEmotions probs are the stored ones, not re-extracted"]}
    (RES / "h1r.json").write_text(json.dumps(out, indent=1))
    log(f"{out['n_communities']} communities; AMI vs old h1 {out['ami_vs_old_h1']:.3f}; {out['total_seconds']}s")


if __name__ == "__main__":
    main()
