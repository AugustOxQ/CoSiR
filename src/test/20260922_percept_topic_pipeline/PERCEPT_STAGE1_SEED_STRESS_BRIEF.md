# Brief: PercepT Stage 1 — seed stress test at lambda=1000

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_seed_stress_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: why this stress test, and what "seed" must vary

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_balance_sweep_v2_pilot.py`, and
`percept_stage1_balance_sweep_v2_pilot_report.md` in full first.

At `SEED=42`, `LAMBDA_BALANCE=1000` produced the first non-collapsed,
held-out-Pareto-bar-clearing PercepT Stage 1 result in this project
(held-out emotion AMI=0.1242, genre AMI=0.2466), but the margin over the
emotion bar (0.1236) is thin — about 0.5%. Before trusting this as a real,
reproducible result rather than a lucky roll, run 3 NEW seeds and report all
4 (including the already-established seed 42 result, cited from the v2
report, not recomputed) together.

**Unlike the lambda sweep (which deliberately shared ONE pretrained
autoencoder across all sweep points to isolate lambda as the only variable),
this stress test must vary every source of randomness per seed** — the
question here is whether the whole pipeline reliably succeeds under
different random initialization, not just whether DEC's own dynamics are
sensitive to lambda. For each of the 3 new seeds, independently:
- Seed both `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
  (if CUDA available) BEFORE building the autoencoder, so its random weight
  initialization differs per seed.
- Pretrain a FRESH autoencoder from that seed's initialization (100 epochs,
  same hyperparameters as the base pilot) — pretraining's minibatch shuffle
  order also depends on the torch RNG state set by this seed, so this is not
  shareable across seeds, unlike the lambda sweep's design.
- Run `initialize_cluster_centers()` with `sklearn.cluster.KMeans(...,
  random_state=seed)` (not the fixed `base.SEED=42` the base pilot
  hardcodes — modify a local copy of this function, or pass `random_state`
  as a parameter, so it uses the CURRENT seed under test).
- Run DEC to convergence with `LAMBDA_BALANCE=1000` fixed (the winning value
  from the sweep), same convergence criterion, same 67-of-100 pruning.

The GoEmotions-RoBERTa affect-embedding extraction and the fused-embedding
construction (`extract_affect_embedding_nodes()`, `fused_embeddings()`) have
no randomness of their own (deterministic given fixed model weights and
input) — compute these ONCE and reuse across all 3 seeds, exactly like the
lambda sweep already does, since re-extracting them per seed would be a
pure waste of ~3.5 minutes each with zero effect on the actual test.

## Seeds to test

Use `SEEDS = (7, 123, 2024)` — 3 new seeds, distinct from the already-run 42.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_seed_stress_pilot_report.md`
with:
- A results table across all 4 seeds (42 cited from the v2 sweep report,
  plus the 3 new runs) x 2 splits: seed, split, emotion AMI, genre AMI,
  collapse verdict, held-out Pareto bar clearance. State plainly in the
  table or a footnote which row is cited vs. newly measured.
- Summary statistics across the 4 held-out emotion AMI values and 4 held-out
  genre AMI values (mean, min, max, and how many of the 4 seeds clear each
  individual bar and both simultaneously).
- An explicit, honest verdict: if 4/4 seeds clear the held-out Pareto bar,
  say this is a robust result. If some seeds miss it, report exactly how
  many and by how much, and do not round up language like "mostly works" to
  "works" — state the actual fraction and margins plainly, since this
  stress test exists specifically to catch an unlucky-seed false positive
  in the original sweep.
- Use this investigation's established verdict language ("Collapsed" /
  "Merely a compromise" / "Real success") per seed/split, plus one overall
  closing sentence characterizing whether lambda=1000 should be treated as
  the reliable standing PercepT Stage 1 configuration going into Stage 2, or
  whether it should be treated as fragile/seed-dependent.
