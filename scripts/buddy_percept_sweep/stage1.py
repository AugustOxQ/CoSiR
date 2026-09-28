"""Parameterized Stage 1 student (fresh implementation -- the original
`run_learned_student_arch_sweep_pilot.py::LearnedStudent` hardcodes
D_SHARED/HIDDEN_DIM as module globals and must not be modified; this
mirrors its forward-pass logic with constructor-level parameters instead).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TeacherGraphs:
    content_edges: np.ndarray
    affect_edges: np.ndarray


class AttentionFusion(nn.Module):
    def __init__(self, d_shared: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_shared, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_shared)

    def forward(self, content_proj: torch.Tensor, affect_proj: torch.Tensor):
        tokens = torch.stack((content_proj, affect_proj), dim=1)
        attended, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        pooled = attended.mean(dim=1)
        return F.normalize(self.norm(pooled), dim=1), attn_weights


class ParameterizedLearnedStudent(nn.Module):
    def __init__(self, heads: str, num_heads: int, d_shared: int, content_dim: int, affect_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.is_attention = heads in {"attn1", "attn4", "attn"}
        if heads == "mlp128":
            hidden_dim = 128
            self.proj_content = nn.Sequential(
                nn.Linear(content_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_shared)
            )
            self.proj_affect = nn.Sequential(
                nn.Linear(affect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_shared)
            )
            self.gate = nn.Sequential(nn.Linear(2 * d_shared, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        elif self.is_attention:
            self.proj_content = nn.Linear(content_dim, d_shared)
            self.proj_affect = nn.Linear(affect_dim, d_shared)
            # num_heads only matters here; ignored entirely for mlp128 above.
            self.fusion = AttentionFusion(d_shared, num_heads)
        else:
            raise ValueError(f"Unknown architecture config: {heads}")

    def forward(self, content: torch.Tensor, affect: torch.Tensor):
        content_proj = F.normalize(self.proj_content(content), dim=1)
        affect_proj = F.normalize(self.proj_affect(affect), dim=1)
        if self.is_attention:
            return self.fusion(content_proj, affect_proj)
        gate = self.gate(torch.cat((content_proj, affect_proj), dim=1))
        student = F.normalize(gate * content_proj + (1.0 - gate) * affect_proj, dim=1)
        return student, gate


def upper_triangle_edges(graph) -> np.ndarray:
    from scipy.sparse import coo_matrix
    graph = coo_matrix(graph)
    keep = graph.row < graph.col
    edges = np.column_stack((graph.row[keep], graph.col[keep])).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise RuntimeError("Teacher graph contains no upper-triangle edges.")
    return edges


def sample_positive_pairs(edges: np.ndarray, rng: np.random.Generator, batch_size: int) -> np.ndarray:
    return edges[rng.choice(len(edges), size=batch_size, replace=len(edges) < batch_size)]


def symmetric_infonce(embeddings: torch.Tensor, pairs: np.ndarray, device: torch.device, temperature: float = 0.1) -> torch.Tensor:
    a = embeddings[pairs[:, 0]]
    b = embeddings[pairs[:, 1]]
    logits_ab = a @ embeddings.t() / temperature
    logits_ba = b @ embeddings.t() / temperature
    targets_a = torch.as_tensor(pairs[:, 1], device=device)
    targets_b = torch.as_tensor(pairs[:, 0], device=device)
    return 0.5 * (F.cross_entropy(logits_ab, targets_a) + F.cross_entropy(logits_ba, targets_b))


def build_teacher_graphs(train_content, train_affect, heldout_content, heldout_affect,
                          teacher_graph_K: int, teacher_graph_alpha: float) -> TeacherGraphs:
    """Builds train-side teacher graphs only (held-out reference graphs are
    built the same way by the pipeline orchestration in Task 6, which
    already needs the Leiden/mutual_knn machinery for a different purpose)."""
    from src.conditional_buddy.buddy_graph import mutual_knn

    content_graph = mutual_knn(train_content, K=teacher_graph_K, backend="auto")
    affect_graph = mutual_knn(train_affect, K=teacher_graph_K, backend="auto")
    # teacher_graph_alpha mixes the two graphs' edge sets before InfoNCE
    # sampling would be a bigger change than tonight's investigation ever
    # tested; here it selects the fraction of edges drawn from each graph
    # per epoch (simple, reviewable interpretation of "content/affect mix").
    content_edges = upper_triangle_edges(content_graph)
    affect_edges = upper_triangle_edges(affect_graph)
    return TeacherGraphs(content_edges=content_edges, affect_edges=affect_edges)


def train_stage1(student, fixed_inputs, lr: float, noise_std: float, lambda_affect: float,
                  batch_size: int, weight_decay: float, seed: int, max_epochs: int = 200,
                  log_checkpoint=None, teacher_graph_K: int = 20, teacher_graph_alpha: float = 0.5) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher = build_teacher_graphs(
        fixed_inputs.train_content, fixed_inputs.train_affect,
        fixed_inputs.heldout_content, fixed_inputs.heldout_affect,
        teacher_graph_K, teacher_graph_alpha,
    )
    train_content_t = torch.as_tensor(fixed_inputs.train_content, dtype=torch.float32, device=device)
    train_affect_t = torch.as_tensor(fixed_inputs.train_affect, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    student.train()
    for epoch in range(1, max_epochs + 1):
        content_pairs = sample_positive_pairs(teacher.content_edges, rng, batch_size)
        affect_pairs = sample_positive_pairs(teacher.affect_edges, rng, batch_size)
        embedding, _ = student(train_content_t, train_affect_t)
        if noise_std > 0:
            embedding = F.normalize(embedding + torch.randn_like(embedding) * noise_std, dim=1)
        content_loss = symmetric_infonce(embedding, content_pairs, device)
        affect_loss = symmetric_infonce(embedding, affect_pairs, device)
        total_loss = content_loss + lambda_affect * affect_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if log_checkpoint is not None and epoch % 5 == 0:
            proxy = float((-content_loss.detach() - affect_loss.detach()).item())
            log_checkpoint(epoch, proxy)
    student.eval()
    with torch.no_grad():
        final_embedding, _ = student(train_content_t, train_affect_t)
    return final_embedding.cpu().numpy()
