# Brief: architecture sweep — linear vs. MLP vs. attention fusion

Read `run_learned_student_stage1_pilot.py` in full first (the current,
addendum-updated version). Copy it to a new file
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py`
and generalize it per below. Do NOT run it — execution happens separately,
on GPU, outside this task.

## Context and grid (chosen to isolate one question at a time)

Stage 1 (linear heads + scalar gate) and Stage 2 (MLP heads + scalar gate)
already exist and gave a clean result: MLP made things worse on both AMI
axes, both splits (`learned_student_stage2_pilot_report.md`). That
comparison changed the PROJECTION HEADS. This sweep asks a different,
cleanly separated question: holding the (already-shown-best) LINEAR
projection heads fixed, does a richer COMBINATION mechanism (multi-head
self-attention over the two projected views) beat the simple scalar gate?
And, as a capacity-robustness check on Stage 2's finding, does an even
bigger MLP (128 hidden units, vs. Stage 2's 64) also underperform, or was
64 specifically too small?

Grid (5 points total, reusing 2 existing results, running 3 new ones):

| label | projection heads | combination mechanism | new run? |
|---|---|---|---|
| Linear | linear | scalar gate | no — reuse Stage 1's numbers exactly |
| MLP-64 | 2-layer MLP, 64 hidden | scalar gate | no — reuse Stage 2's numbers exactly |
| MLP-128 | 2-layer MLP, 128 hidden | scalar gate | yes |
| Attention-h1 | linear | 1-head self-attention | yes |
| Attention-h4 | linear | 4-head self-attention | yes |

Everything else (PCA dimensionality, D_SHARED=32, loss, training data and
sampling, learning rate, epoch budget, checkpoint cadence, stopping rule,
Pareto bar) stays identical to Stage 1 for all three new runs — isolate one
variable per row.

## The attention combination mechanism (new — specify precisely)

```python
class AttentionFusion(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=D_SHARED, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(D_SHARED)

    def forward(self, content_proj: torch.Tensor, affect_proj: torch.Tensor):
        tokens = torch.stack((content_proj, affect_proj), dim=1)  # (N, 2, D_SHARED)
        attended, attn_weights = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )  # attended: (N, 2, D_SHARED); attn_weights: (N, 2, 2)
        pooled = attended.mean(dim=1)  # (N, D_SHARED)
        student = F.normalize(self.norm(pooled), dim=1)
        return student, attn_weights
```

For the `Attention-h1`/`Attention-h4` variants, `LearnedStudent` keeps
`proj_content`/`proj_affect` as plain `nn.Linear` (identical to Stage 1),
drops the `gate` submodule, and adds `self.fusion = AttentionFusion(num_heads)`.
Its `forward()` computes `content_proj`/`affect_proj` exactly as Stage 1
does (project, then `F.normalize`), then calls
`self.fusion(content_proj, affect_proj)` in place of the gate-and-weighted-
sum step, returning `(student, attn_weights)` in place of `(student, gate)`.

**Diagnostic adaptation** (`evaluate_checkpoint` and the final collapse
determination both currently read `gate`/`gate_mean`/`gate_std`/
`gate_saturated_fraction` — adapt these for the attention variants without
changing their meaning or the collapse thresholds):
- Define "self-attention weight" per node as the average of the two
  diagonal entries of that node's `(2, 2)` attention matrix (how much each
  token attends to itself, averaged over the content-token's and the
  affect-token's own diagonal entries) — this is the direct structural
  analogue of the gate's "how much this node favors content over affect"
  scalar, on the same `[0, 1]` scale (since attention weights sum to 1 over
  the 2 keys per query, a diagonal entry near 1 means near-total self-
  attention / near-total ignoring of the other view, analogous to gate
  saturation).
  attention).
- Report this value as `gate_mean`/`gate_std`/`gate_saturated_fraction` in
  the checkpoint dict and report tables — same field names, same
  `<0.1 or >0.9` saturation threshold, same collapse-rule usage — so the
  existing collapse-determination code and report-table structure need NO
  other changes. Just document in a comment that "gate" here means
  "self-attention weight" for the attention variants.
- The gradient-share diagnostic (`selected_parameter_gradient_norm`)
  currently reads `model.proj_content` and `model.gate`'s parameters — for
  the attention variants, read `model.proj_content` and `model.fusion`'s
  parameters instead (the structural analogue: the shared/mixing
  component, not the per-view encoder).

For `MLP-128`, this is simply Stage 2's exact architecture with
`HIDDEN_DIM=128` instead of 64 — no other changes needed, reuse the gate
mechanism exactly as Stage 2 has it.

## Sweep mechanics

Reuse the weight-sweep script's outer-loop pattern (read
`run_learned_student_weight_sweep_pilot.py` for the established, already-
reviewed pattern of re-seeding and re-initializing per sweep point, and the
`[[label]]`-prefixed logging convention — adapt the prefix to the
architecture label, e.g. `[MLP-128]`, `[Attention-h1]`). Loop over the 3
NEW configurations only (`MLP-128`, `Attention-h1`, `Attention-h4`); do not
re-run Linear or MLP-64 — hardcode their already-verified final numbers
(train emotion/genre AMI, held-out emotion/genre AMI, and their final
gradient share / gate mean / gate saturated fraction from their own
committed reports) as fixed reference rows in the results table, clearly
labeled as reused, not recomputed.

Use a single `LearnedStudent` class whose `__init__` takes a config
(`heads: str` selecting among `"mlp128"`, `"attn1"`, `"attn4"`) and builds
the right submodules per branch — this is cleaner than three separate
model classes and keeps the training loop, which is otherwise IDENTICAL
across all three, written once.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/learned_student_arch_sweep_pilot_report.md`:

- Context paragraph explaining the two questions this sweep answers (does
  attention beat the gate, holding linear heads fixed; is Stage 2's
  MLP-underperforms finding robust to more capacity).
- A 5-row x 2-split results table (10 rows), reusing the same columns as
  the weight-sweep report (architecture label, split, emotion AMI, genre
  AMI, collapse verdict, final gradient share, final gate/attention-weight
  mean, final gate/attention-weight saturated fraction), with the two reused
  rows (Linear, MLP-64) clearly marked "(reference, not rerun)".
- Apply the SAME held-out Pareto bar (`emotion AMI > 0.1236 AND genre AMI >
  0.1954`) to every new row; state which (if any) clear it, and whether any
  beats Stage 1's held-out numbers on both axes simultaneously (the
  standard this whole investigation has used to call something the "new
  best result").
- A short paragraph specifically comparing `Attention-h1` vs. `Attention-h4`
  (does more heads help, hurt, or make no material difference) and a
  separate short paragraph on `MLP-128` vs. `MLP-64` (does more MLP
  capacity change Stage 2's conclusion).
- An honest concluding paragraph naming the single best-performing
  configuration among all 5 by held-out summed AMI, and stating plainly
  whether it changes this investigation's standing conclusion (Stage 1 /
  Linear is the best mechanism found) or not.

Print clear timestamped progress logs matching the existing scripts' format
throughout, with the per-configuration label prefix on every line during
that configuration's run.
