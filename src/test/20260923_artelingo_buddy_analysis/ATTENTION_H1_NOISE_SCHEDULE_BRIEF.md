# Brief: Attention-h1 with embedding noise + cosine LR schedule (untested lever)

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_noise_schedule_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: closing a confirmed gap, not a guess

Read `run_attention_h1_embedding_snapshot_pilot.py` (the standing Attention-h1
re-fit script) and `run_learned_student_arch_sweep_pilot.py` (defines
`LearnedStudent`, `symmetric_infonce`, `sample_positive_pairs`,
`content_batch_embeddings`, `LEARNING_RATE`, `MAX_EPOCHS`,
`PLATEAU_REL_IMPROVEMENT`, `PLATEAU_WINDOW`, `CHECKPOINT_EVERY`,
`evaluate_checkpoint`, `detect_communities`) in full first.

A comprehensive status memo (independently produced by reading this
project's entire buddy/Attention-h1/learned-student history) found, by
grepping every `run_learned_student*` and `run_attention_h1*` script, that
**no pilot in this lineage has ever tried Gaussian embedding noise or a
cosine-annealed learning rate** — every one uses a fixed Adam learning rate.
Meanwhile, a separate PercepT-replication investigation found that adding
exactly these two mechanisms (latent noise + Adam 1e-3 cosine-annealed to
1e-5) to that paper's own autoencoder+DEC pipeline produced by far the
highest latent-space silhouette measured anywhere in this project (0.5120,
versus Attention-h1's own measured 0.0392). This pilot transplants the same
*spirit* — perturb the representation during training, and use a full
LR-decay schedule instead of a fixed rate — onto Attention-h1's own
architecture. It is **not** a literal one-line port: Attention-h1 has no
autoencoder or reconstruction decoder, and its 32-D embedding is
LayerNorm'd and L2-normalized (unit hypersphere), unlike PercepT's
unconstrained 128-D latent. State this adaptation explicitly in the
script's docstring and in the report: noise here is added to the embedding
immediately before each InfoNCE loss computes on it, then the perturbed
embedding is re-normalized to unit L2 norm before entering
`symmetric_infonce`, so the architecture's own unit-norm invariant (and
therefore the InfoNCE temperature's effective meaning) is preserved. There
is no reconstruction target to keep clean; the *evaluation* embeddings
(epoch-0 and final snapshots, and everything Leiden/AMI/silhouette is
computed from) must always be the **clean, noise-free** forward pass — only
the two training losses see the noised, re-normalized version.

## Implementation

Reuse everything from `run_attention_h1_embedding_snapshot_pilot.py`
unchanged except the two items below: teacher graph construction
(`pipeline.build_buddy_graphs`, `single_modality.build_single_modality_graph`),
`LearnedStudent("attn1")` architecture, `sample_positive_pairs`,
`content_batch_embeddings`, `symmetric_infonce`, `evaluate_checkpoint`,
the plateau-based stopping rule (`PLATEAU_REL_IMPROVEMENT`,
`PLATEAU_WINDOW`), and `detect_communities`/Leiden for both epoch-0 and
final embeddings on both splits. Reuse the sibling-import (`load_module`)
pattern from that script exactly, including its `sys.path` fix comment.

**Change 1 — embedding noise before each InfoNCE loss, at training time
only:**

```python
def noisy_unit_embedding(embedding: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0.0:
        return embedding
    noised = embedding + noise_std * torch.randn_like(embedding)
    return torch.nn.functional.normalize(noised, dim=-1)
```

Apply this to `content_embeddings` right before `arch.symmetric_infonce(...)`
for the content loss, and to `affect_embeddings` right before
`arch.symmetric_infonce(...)` for the affect loss, each training step. Do
NOT apply it to any embedding used for `evaluate_checkpoint` diagnostics, or
to the epoch-0/final embeddings saved for Leiden/AMI/silhouette evaluation
— those forward passes stay exactly as in the base script (`model.eval()`,
`torch.no_grad()`, clean embedding, no noise, no renormalization beyond
whatever the model's own LayerNorm/L2 output already does).

**Change 2 — cosine-annealed learning rate in place of the fixed
`arch.LEARNING_RATE`:**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=arch.MAX_EPOCHS, eta_min=1e-5
)
```

Step the scheduler once per epoch, guarded so it never steps past
`arch.MAX_EPOCHS` (plateau-based early stopping may end training well
before that; if it ever ran past, which it structurally cannot since the
epoch loop itself is bounded by `arch.MAX_EPOCHS`, hold at the floor rather
than let cosine rise again — mirror the guard style already used in
`run_percept_stage1_faithful_recipe_pilot.py`'s DEC phase for this exact
scenario). Log the learning rate at each logged checkpoint alongside the
existing recall diagnostics.

## Single-seed screen, then stress only the winner

**Screen, seed=42, `NOISE_STD` in `(0.0, 0.02, 0.05, 0.1)`** — all four
runs use the cosine LR schedule above; `0.0` isolates the LR-schedule
effect alone as a control, the other three isolate noise strength. Each
value gets a fresh model instance and fresh training run (do not reuse
weights across values). For each, report: held-out emotion AMI, held-out
genre AMI, held-out silhouette (compute this the same way the pseudo-
contrastive pilot's report describes doing it retrospectively — same
`silhouette_score` call with `sample_size=min(4000, len(idx))`,
`random_state=42`, on the final held-out embedding and its Leiden labels;
grep `run_attention_h1_leiden_pseudo_contrastive_pilot.py` for the exact
call if it computes this directly, and reuse that, rather than
reimplementing it slightly differently), and the same held-out Pareto bar
already used throughout this investigation (emotion AMI > 0.1236 AND genre
AMI > 0.1954). Also report train-split AMIs and silhouette, and the number
of Leiden communities found pre/post training on each split, exactly as
the base snapshot script already does.

**Winner selection:** among the noise_std values that clear the held-out
Pareto bar, pick the one with the highest held-out silhouette (not the
highest AMI margin) — silhouette is the axis this pilot exists to move,
and AMI-bar clearance is already the gate. If none clears the bar, do not
pick a winner on AMI grounds alone; instead pick the value with the
highest held-out silhouette among all four, and say explicitly in the
report that this is a "best available, does not clear the Pareto bar"
selection, not a validated winner.

**Stress test:** run the selected value at 3 additional seeds,
`SEEDS = (7, 123, 2024)`, using fresh `torch.manual_seed`/`np.random.seed`
per seed exactly as the base script's seeding pattern, keeping the winning
`noise_std` and the same cosine schedule fixed. Report the four-seed table
and summary statistics (mean/min/max for both held-out AMIs and held-out
silhouette, and how many of the 4 seeds clear the held-out Pareto bar) in
the same format this investigation's other 4-seed reports use.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_noise_schedule_pilot_report.md`
with: the screen table (4 noise_std values, both splits, all metrics above,
plus Leiden community counts), the training/LR trajectory for each
screened value (checkpoint cadence matching the base script's
`CHECKPOINT_EVERY`), the winner-selection rationale stated explicitly, the
four-seed stress table and summary statistics, and an explicit comparison
against three reference points already established in this investigation:
(a) the standing Attention-h1 baseline (held-out emotion AMI 0.1249, genre
AMI 0.2404, held-out silhouette 0.0392), (b) the PercepT-replication
standing balance-hack (held-out emotion AMI 0.1252, genre AMI 0.2486,
silhouette never measured), and (c) the PercepT-replication faithful-recipe
result (held-out emotion AMI 0.1092, genre AMI 0.3288, silhouette 0.5120).
State plainly, in a clearly headed final section, whether this pilot beats
the Attention-h1 baseline on silhouette, whether it beats either PercepT-
replication result on any axis, and whether it does so robustly (4/4
seeds) or only in the single-seed screen. Do not soften or oversell a
partial or negative result — this investigation's established convention is
blunt, numeric verdicts.
