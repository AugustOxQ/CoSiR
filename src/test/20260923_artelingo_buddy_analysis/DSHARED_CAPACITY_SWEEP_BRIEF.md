# Brief: does widening D_SHARED help, independent of any clustering loss? (brainstorm candidate 3)

Write one new script. Do NOT run it — execution happens separately, on
GPU, outside this task.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md`, candidate 3
  — "Widen `D_SHARED` without changing the objective." Its own caution:
  Attention-h1's logged effective rank at 95% already grew from 12 (init)
  to 20 of 32 in the arch sweep, and schedule trajectories reach ~17–19 —
  evidence *against* a hard 32-D bottleneck, so this is a cheap
  independent check, not an expected big win.
- `run_attention_h1_embedding_snapshot_pilot.py`, in full — the plain,
  unmodified baseline recipe this pilot must otherwise leave untouched
  (fixed Adam LR, no noise, no schedule, ordinary Leiden post-hoc). This is
  the pilot to structurally clone, at three widths.
- `run_learned_student_arch_sweep_pilot.py` — confirm (already checked)
  that `D_SHARED` is referenced **only** as a bare module-level global
  inside `AttentionFusion.__init__` and `LearnedStudent.__init__` (lines
  ~100–133), never hardcoded elsewhere as a literal 32. This means setting
  `arch.D_SHARED = <value>` on the imported sibling module, before calling
  `arch.LearnedStudent("attn1")`, correctly changes every layer width for
  that instantiation — confirm this yourself by reading the file, and do
  **not** edit the shared architecture file itself; the module-attribute
  override is the only change needed.
- `attention_h1_embedding_snapshot_pilot_report.md` and
  `attention_h1_baseline_seed_stress_pilot_report.md` — the existing,
  already-validated D_SHARED=32 seed-42 baseline (no rerun needed at 32;
  cite its numbers as this pilot's first row).

## What to build

Write
`src/test/20260923_artelingo_buddy_analysis/run_dshared_capacity_sweep_pilot.py`.

Reuse `run_attention_h1_embedding_snapshot_pilot.py`'s structure wholesale
(teacher graph construction, training loop, two-teacher symmetric InfoNCE
losses, fixed Adam LR, plateau stopping rule, post-hoc Leiden on train and
held-out, AMI/silhouette scoring) — do not add noise or a cosine schedule;
this pilot isolates capacity alone, per the memo's explicit framing
("independent of any clustering-loss mechanism"). For each of
`D_SHARED_VALUES = (64, 128)` at seed 42: set `arch.D_SHARED = value`
before constructing `arch.LearnedStudent("attn1")`, train exactly as the
baseline sibling does, and additionally log:
- effective rank at 95% variance and top eigenvalue fraction of the final
  held-out embedding (reuse whatever helper `evaluate_checkpoint` or the
  arch sweep module already computes this with — do not reimplement PCA/
  eigenvalue logic from scratch if a helper exists),
- final content/affect teacher-graph recall,
- held-out emotion AMI, genre AMI, both against the same Pareto bar
  (emotion > 0.1236, genre > 0.1954),
- Leiden community occupancy (count, min/max/median, below-1% count) for
  both train and held-out, and
- held-out silhouette using the same seed-42 two-stage sampling convention
  every other pilot in this directory uses (`np.random.default_rng(42)`
  draw of ≤6,000, `silhouette_score(sample_size=min(4000, len(idx)),
  random_state=42)`).

**Screen only, no stress unless a width clearly earns it**: if any
D_SHARED value in `{64, 128}` clears the AMI Pareto bar **and** shows a
meaningfully higher held-out silhouette than the cited D_SHARED=32
baseline (state a concrete threshold you use for "meaningfully higher" in
the report, e.g. a fixed absolute-difference bar consistent with this
investigation's other screen-then-stress pilots), stress that one winning
width at seeds 7, 123, 2024, same methodology as every other 4-seed stress
in this investigation. If nothing meaningfully improves, do not stress
anything further — report the negative result plainly, per this
investigation's established convention.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/dshared_capacity_sweep_pilot_report.md`
with: a Method section explaining the module-attribute-override approach
(state plainly that the shared architecture file was not modified); a
combined table with D_SHARED=32 (cited baseline), 64, and 128 side by
side (effective rank, top eigen fraction, teacher recalls, both AMIs,
occupancy, silhouette, Pareto-bar verdict); the stress table if a width
warranted one; and a final verdict section stating plainly whether
capacity alone explains any of the silhouette gap to PercepT, consistent
with this investigation's blunt, numeric-verdict convention.

Do not touch git, do not modify any other file in the repository.
