# CoSiR v2 Block 1: Stage 1 topic-formation rebuild (content-only first)

> **For agentic workers:** implement task-by-task, TDD (failing test first), function/class-formal
> code (typed, documented, no ad hoc scripts). Steps use checkbox (`- [ ]`) syntax.

**Goal:** rebuild the validated Attention-h1 buddy-graph Stage 1 mechanism fresh in `cosir-v2`,
content-only first (no affect teacher), as clean typed modules — the foundation Candidate A's
factor-discovery stage (next plan) builds on.

**Spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md` — read §3 and the
"Candidate A architecture" section before starting. Reference implementation for understanding
the validated mechanism (read for understanding, **do not copy**):
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py` on branch
`experiment/percept_topic_pipeline` (`git show experiment/percept_topic_pipeline:<path>` from
this worktree to read it without switching branches).

## Global constraints

- Branch: `cosir-v2` (this worktree, `/project/CoSiR-v2`). Never work on `experiment/*` branches.
- Function/class-formal: typed signatures, docstrings on public functions/classes, no bare
  scripts with top-level side effects. Small, focused modules — not one giant file.
- `seed=42` for every stochastic step (project convention).
- No `cuml`/`cugraph` imports — this project's `libllvmlite.so` chain is documented broken
  locally; use `leidenalg`/`python-igraph` (CPU) for community detection, same as the
  buddy-percept sweep's already-validated approach.
- New standalone diagnostic/validation scripts go under `src/test/YYYYMMDD_<name>/` (dated-folder
  convention).
- Content-only for this plan — no affect teacher graph, no GoEmotions dependency. A
  self-supervised-pretrained affect arm is a later, separate task, not part of this plan.
- Reuse `src/conditional_buddy/buddy_graph.py` (`mutual_knn`, `union_graph`, `ensure_min_degree`,
  `ensure_connected`) as-is for graph construction — confirmed sufficient, do not reimplement.
- Reuse `src/utils/feature_manager.py` (`FeatureManager`) for cached CLIP feature loading —
  confirmed reusable infra.
- Implementation dispatched to Codex (direct `codex e --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check -c mcp_servers={} -C /project/CoSiR-v2 --json -` invocation, task piped
  via stdin — not `codeagent-wrapper`, confirmed broken). Claude reviews every diff before the
  next task starts.

---

### Task 1: `ContentTeacherGraph` — typed wrapper over buddy graph construction

**Files:**
- Create: `src/model/graph.py`
- Test: `src/test/test_graph.py`

**Interfaces:**
- `@dataclass GraphConfig`: `k: int = 30`, `alpha: float = 0.5`, `min_degree: int = 1`,
  `seed: int = 42` (mirrors `buddy_graph.py`'s existing knobs — do not invent new hyperparameter
  semantics, just give them a typed, documented home).
- `build_content_graph(img_features: np.ndarray, txt_features: np.ndarray, config: GraphConfig) -> scipy.sparse.csr_matrix`
  — calls `buddy_graph.mutual_knn` per modality, `buddy_graph.union_graph` to combine, then
  `ensure_min_degree`/`ensure_connected` from the same module. Returns the (N,N) symmetric binary
  adjacency. No new graph-construction logic — this task is a clean, tested, documented seam
  between Block 1's new code and the reused `buddy_graph.py`, not a reimplementation.

- [ ] Write failing tests: shape/symmetry of the returned graph on a small synthetic
  img/txt feature array; confirms `ensure_connected` guarantee holds (every node has ≥1 edge);
  confirms determinism under `seed=42` (same inputs → identical graph across two calls).
- [ ] Implement `src/model/graph.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: `feat(cosir-v2): ContentTeacherGraph wrapper over buddy graph construction`

---

### Task 2: `AttentionFusionStudent` — the Stage 1 encoder

**Files:**
- Create: `src/model/student.py`
- Test: `src/test/test_student.py`

**Interfaces:**
- `class AttentionFusionStudent(nn.Module)`: `__init__(feature_dim: int, output_dim: int = 32,
  dropout: float = 0.1)`. Single-head self-attention over the two per-sample input tokens
  (frozen CLIP image feature, frozen CLIP text feature — no affect input in this content-only
  build), fusing them into one output vector per sample: LayerNorm, then L2-normalize the final
  output (matches the validated "attn1" architecture's output convention — read the reference
  implementation for the exact attention-fusion shape, e.g. how the two tokens are stacked and
  attended over, before committing to a specific `nn.MultiheadAttention` call signature).
  `forward(img_feat: Tensor[B, feature_dim], txt_feat: Tensor[B, feature_dim]) -> Tensor[B, output_dim]`.
- Deterministic init under `seed=42` (a fixed seed produces identical initial weights across
  construction calls — write this as an explicit test, not an assumption).

- [ ] Write failing tests: output shape, L2-norm == 1 per row, gradient flows to all parameters
  from a dummy loss, determinism of initial weights under a fixed seed.
- [ ] Implement `src/model/student.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: `feat(cosir-v2): AttentionFusionStudent encoder`

---

### Task 3: Stage 1 training loop — symmetric InfoNCE against the teacher graph

**Files:**
- Create: `src/train/stage1.py`
- Test: `src/test/test_stage1_training.py`

**Interfaces:**
- `@dataclass Stage1Config`: `lr: float`, `epochs: int`, `batch_size: int`, `temperature: float`,
  `seed: int = 42` — sensible defaults matching the reference implementation's validated
  hyperparameters where known; flag any value you can't confirm from the reference as an
  explicit default choice in a docstring, not a silent guess.
- `train_stage1(img_features: np.ndarray, txt_features: np.ndarray, graph: scipy.sparse.csr_matrix, config: Stage1Config) -> tuple[AttentionFusionStudent, np.ndarray]`
  — trains the student with symmetric InfoNCE using the teacher graph's edges as the positive-pair
  supervision signal (graph-neighbor sampling; read the reference implementation for the exact
  symmetric-InfoNCE formulation before implementing — "symmetric" here has a specific meaning in
  that code, do not assume). Returns the trained student and the final (N, output_dim) embedding
  matrix.
- No wandb/logging dependency in this module — plain Python logging or print, keep it decoupled
  from any experiment-tracking choice (that's a later concern, not Block 1's).

- [ ] Write failing tests on a small synthetic graph + random features: loss decreases over a few
  epochs (sanity, not a tight bound); training is deterministic under `seed=42` (two runs with
  identical inputs produce identical final embeddings); a completely disconnected/degenerate
  graph raises a clear error rather than silently training on nothing.
- [ ] Implement `src/train/stage1.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: `feat(cosir-v2): Stage 1 symmetric InfoNCE training loop`

---

### Task 4: Community detection over the trained embedding space

**Files:**
- Create: `src/model/communities.py`
- Test: `src/test/test_communities.py`
- Modify: `requirements.txt` (append `python-igraph`, `leidenalg` if not already present — check
  first, the old prototype-seeding code that used these was removed in the foundation-stripping
  commit, but the dependency may or may not still be listed)

**Interfaces:**
- `detect_communities(embeddings: np.ndarray, k: int = 20, seed: int = 42) -> np.ndarray` —
  builds a fresh kNN graph over the trained student's **output embeddings** (not the original
  teacher graph — community/topic detection runs on the learned space, matching the validated
  buddy investigation's convention), then Leiden community detection (CPU, `leidenalg` +
  `python-igraph`, no `cuml`/`cugraph`). Returns (N,) int community labels, 0-indexed, no gaps.
- `community_stats(labels: np.ndarray) -> dict` — community count, size distribution, min/max
  size — a small diagnostic used by Task 5's validation, not a full evaluation harness.

- [ ] Write failing tests: two well-separated synthetic clusters produce 2 communities; labels
  are 0-indexed with no gaps; determinism under `seed=42`.
- [ ] Implement `src/model/communities.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: `feat(cosir-v2): Leiden community detection over trained embeddings`

---

### Task 5: End-to-end validation on ArtELingo (content-only baseline number)

**Files:**
- Create: `src/test/20260928_stage1_validation/run_validation.py`,
  `src/test/20260928_stage1_validation/README.md`
- Create: `docs/reports/auto/v2/2026-09-28_block1_stage1_validation.md`

**Interfaces:** none new — this task wires Tasks 1-4 together into one real run and reports what
comes out. Uses cached ArtELingo CLIP features (already extracted, per
`/data/SSD2/pre_extract/artelingo/features` — confirm this path or the equivalent still resolves
from `FeatureManager` before assuming it does) and real ArtELingo emotion/genre labels for
held-out AMI, matching the metric convention already established in
`docs/reports/auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md` (held-out emotion AMI,
genre AMI, occupancy/balance across communities).

- [ ] Write `run_validation.py`: load cached features, build the content graph (Task 1), train
  Stage 1 (Task 3), detect communities (Task 4), compute held-out AMI against real ArtELingo
  labels and occupancy stats, print a clear summary.
- [ ] Run it for real (local GPU) and record the actual numbers.
- [ ] Write `docs/reports/auto/v2/2026-09-28_block1_stage1_validation.md`: report the numbers
  plainly, and explicitly compare against the original (affect-included) Attention-h1 result from
  `docs/reports/auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md` — **expect a real
  difference, not a match**, since this build deliberately drops the affect teacher; the point of
  this task is establishing this rebuild's own content-only baseline number, not reproducing the
  old one. State plainly whether content-only alone clears the same held-out AMI Pareto bar
  (emotion > 0.1236 and genre > 0.1954) the original did, or not.
- [ ] Commit: `docs(cosir-v2): Block 1 content-only Stage 1 validation on ArtELingo`

---

## Self-review

**Placeholder scan:** no TBD/TODO; Task 3's hyperparameter-default uncertainty is explicitly
flagged as a documented default choice for the implementer to record, not a silent gap.
**Scope:** five tasks, each independently testable, ending in one real validated number — right
sized for one plan. Factor discovery (Candidate A's next stage) is deliberately a separate,
later plan, not folded in here, matching the block-by-block build order the spec commits to.
**Type consistency:** `build_content_graph`'s `csr_matrix` output feeds directly into
`train_stage1`'s `graph` parameter; `train_stage1`'s `np.ndarray` embedding output feeds
directly into `detect_communities`. No cross-task signature mismatches.
