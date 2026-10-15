# Handoff: CoSiR v2 Candidate A, next step (d): condition-aware factor learning

**Written 2026-09-30, at the end of the stage-(d) session, for the next chat. Read this whole file
first.** The user chose **option (d): revisit factor learning.** The goal is to make the factors
themselves more condition-aware, carrying over stage (d)'s lessons. Nothing has been designed yet.
The next chat starts with brainstorming.

## 1. What to do first (in order)

1. **Invoke `superpowers:brainstorming`.** This is an **architectural** task: new training objectives
   for the factor encoders, a new evaluation protocol, and possibly new modules. Follow the full path:
   questions one at a time, then approaches, then design sections approved one by one, then a written
   spec, then `superpowers:writing-plans`, then execution.
2. **Read these before asking the user anything:**
   - Parent spec: `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md` (Candidate A
     §1-4, build order).
   - Stage (d) spec: `docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md`. §9
     says factor fine-tuning is revisited only if the ceiling shows the frozen factors lack the
     information. **That gate is now answered: they do lack it** (§2 below).
   - Stage (d) reports, which have been corrected:
     - `docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md`
     - `docs/reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md` (options (a)-(d) at the end)
   - Factor-repair background (how R3 was obtained, and why the old space collapsed):
     `docs/reports/auto/v2/2026-10-09_candidate_a_factor_collapse_diagnosis.md` and
     `docs/reports/auto/v2/2026-10-11_candidate_a_factor_repair.md`.
   - Stage (d) SDD ledger (all rulings, and deferred minors worth carrying forward):
     `.superpowers/sdd/2026-09-30-cosir-v2-candidate-a-stage-d/progress.md`
3. **Write a short understanding note back to the user,** separating what they said from your
   assumptions. Then ask the design questions in §4, one at a time, with a recommended option first.

## 2. Where things stand (numbers to carry)

- **R3**, the current factor recipe, has 32 non-negative factors over frozen CLIP ArtELingo features.
  It uses InfoNCE agreement plus a decorrelation penalty, is `R3_CONFIG` in
  `src/train/train_factors.py`, and was selected under user-amended gates. Checkpoint:
  `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt` (SHA-256 `1c299fc0…e453f`).
  - It fixed the old collapse (participation ratio 1.3 → ~21).
  - Its codes match image to caption better than raw CLIP does.
- **Stage (d)** trained a scorer on frozen R3 codes. It was a naive-initialized residual interface
  with learned β and τ, trained on self-generated conditions from three sources.
  - **Selection** (15% carve-out of train paintings):
    - CLIP-cluster conditions (G3/G4) and Block-1-community conditions (G5) beat naive on
      condition-use gain Δ, by +2.4 to +3.0 R@1 points.
    - Factor-combination conditions (G1/G2) did not transfer to human conditions.
    - The swap term never helped.
    - G3 was selected by the tie rule.
  - **Held test:**
    - Criterion 1 (Δ in both directions): NOT MET, and inconclusive, since power was about 0.38.
    - Criterion 2 (emotion-vs-style swap test): MET, +6.4 points.
  - **Final-review corrections** (post-hoc, selection rows):
    - Criterion 2's pass is reproduced by the zero-parameter naive rule at G3's learned β (0.05).
      G3 minus β-matched naive is +0.05 [−2.10, +2.20].
    - About 35% of Δ is the β drop.
    - The **β-collapse mechanism**: stage-(d) hard training negatives are CLIP-nearest, so CLIP
      misleads in training but helps in evaluation (CLIP beats hard negatives 27-32% of the time vs
      68-75% for random ones).
  - **The key finding: the cross-validated per-label oracle is about equal to naive.** It is one best
    weight vector per human label (emotion or art style) on the frozen R3 factors. At β 0.3, pooled,
    on selection rows, i2t / t2i R@1:

    | | i2t | t2i |
    |---|---:|---:|
    | Label oracle | 18.41 | 21.04 |
    | Naive rule | 18.36 | 20.36 |
    | Oracle's random-target null | 7.69 | 7.62 |

    **The frozen factors, not the weighting rule, limit emotion and style conditioning.** The earlier
    per-episode "ceiling" (~85%) was oracle flexibility, since a random target reaches ~77%; ignore
    it.
- **Reference held numbers** (R@1 i2t / t2i, 1,024 + 1,024 held label episodes):

  | Model | i2t | t2i |
  |---|---:|---:|
  | Naive on R3 (β 0.3) | 17.63 | 20.90 |
  | G3 | 16.70 | 21.34 |
  | CLIP-only | 11.52 | 14.89 |
  | Uniform | 15.43 | 16.26 |
  | Chance | 7.7 | 7.7 |

## 3. Hard constraints and lessons (carry into the new design)

- **Human labels (emotion, art style) are evaluation-only.** They never enter training, which is a
  parent-spec constraint. Self-generated conditions must come from CoSiR's own structure.
- **Circularity.** CLIP-cluster and community conditions are *external* to the factor space, so
  training factor encoders on them is not circular in the factor sense. Factor-mined conditions would
  be circular, and they also didn't transfer. The parent spec's lagging-snapshot concern applies only
  to factor-mined conditions.
- **Guard against re-collapse.** Keep R3's losses and gates:
  - `src/eval/factor_gates.py`, preset `AMENDED_2026_09_29_THRESHOLDS`;
  - `readout_reference` equal to the R0 readout;
  - effective dimensionality, redundancy and pair retrieval must stay healthy.
- **β.** Freeze β, or pre-register "naive at the learned β" as the baseline. Never compare against
  naive at a fixed β while training β.
- **Negatives.** Match training negatives to evaluation negatives. Evaluation uses random clean
  negatives; stage (d)'s CLIP-nearest hard negatives made training reward ignoring CLIP.
- **Power.** Pre-register sample sizes from selection-set variance. Stage (d)'s criterion 1 had
  power of about 0.38.
- **The key diagnostic for the new factors** is the cross-validated **label oracle**
  (`label_oracle_ranks`) and its null, compared with naive on R3. If new factors don't raise the
  label oracle above naive-on-R3, they carry no more condition information.
- **Evaluation data freshness (a user decision to raise).** The held split has been read in two
  final tests (repair-plan Task 7 and stage-(d) Task 7), and stage (d) was designed after seeing the
  first. Options:
  - new held episodes with a different seed on the same held rows, with disclosure;
  - a fresh held carve-out, which is impossible without re-splitting, since R3 was trained on train
    rows;
  - accepting reuse with disclosure;
  - a new dataset.

  Also note that if the new factor encoders train only on **scorer-train** rows, the 15% selection
  set becomes truly out-of-sample for the factors, which is cleaner than in stage (d).
- **No human-judged set exists yet.** That is stage (e).

## 4. Design questions to bring to the user (one at a time, recommendation first)

1. **What signal makes the factors condition-aware?** My candidate directions (not yet evaluated):
   - (i) Fine-tune R3's factor encoders jointly with the stage-(d) scorer on external conditions
     (CLIP clusters and communities, the ones that transferred), keeping the R3 losses and gates.
   - (ii) Re-learn the factors so that groups defined by external conditions separate along few
     factors (a factor-level supervised-contrastive objective on CLIP-cluster or community
     membership).
   - (iii) Check capacity first: more factors, or TopK sparsity. Is 32 enough?
   - (iv) Modality-specific factor subsets. Style is largely visual and emotion largely in the
     captions; Task 9 of an earlier plan showed the gains split by direction.

   A cheap probe before committing may be worth it: the label oracle on raw CLIP-PCA features, or
   on factors fitted to CLIP clusters, to see whether *any* 32-dimensional code carries more
   emotion or style information than R3.
2. **Frozen-β vs learned-β with a matched baseline.**
3. **Which condition sources:** G3 (CLIP clusters) and G5 (communities) are the candidates. G5
   alone beat its β-matched naive on the selection swap test (+2.59 [+0.73, +4.44]).
4. **Evaluation data freshness** (§3).
5. **Pre-registered criteria and stop points,** powered from selection variance.

## 5. Working conventions (the user's standing preferences)

- **Execution:**
  - SDD (`superpowers:subagent-driven-development`) with **Claude Code subagents** as implementers,
    the model sized to each task (no Codex unless the user asks).
  - Per-task reviewer subagents, then an **Opus** whole-branch final review, then ONE fix wave.
  - Claude is controller and reviewer, and keeps a ledger in `.superpowers/sdd/<plan>/progress.md`.
- **Style:** the user likes **plain-language, verdict-first** summaries, not dense statistics. They
  approve design section by section. Once a plan is approved they allow **overnight automation**,
  stopping only at pre-registered stop points or destructive actions.
- **Git:** commit **locally on `main`** (`/project/CoSiR`); **never push** unless asked. Commit
  trailers, after a blank line:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: <the current session URL from the system reminder>`
- **Reports:**
  - Location: `docs/reports/auto/v2/YYYY-MM-DD_<topic>.md`. The v2 date sequence is at 10-14, so
    continue from 10-15.
  - Index: one row in `docs/reports/reports_sum.md`, plus the "Current work" line. Then run
    `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py`, which must print OK.
- **Code layout:**
  - Real runs go in dated folders `src/test/yyyymmdd_<name>/`, each with a log and a local
    `.gitignore`.
  - `src/` changes follow TDD.
  - Change logs go in `.claude/yyyymmdd_log.md` (gitignored; write them, do not add them).
- **Environment:**
  - Python `/root/miniconda3/envs/CoSiR/bin/python`; seed 42; no `cuml`/`cugraph`.
  - Local RTX 3090 and 32 cores. CPU-bound runs can go in parallel processes.
  - **Another Claude session may be working in this repo** (a matched PercepT-vs-buddy H2H). Stage
    only your own files, and retry if you hit an index.lock.
- **Stale ccg state:** if a ccg "loop detected" warning appears, the user wants stale `.ccg/tasks/*`
  entries archived. The user also said never to engage `.ccg/tasks` for this work.

## 6. Code entry points (all on `main`)

| Area | Where |
|---|---|
| Data and splits | `src/data/artelingo.py` (`load_artelingo`, with `art_styles`); `src/data/splits.py` (`leakage_groups`, `grouped_split`, `grouped_subsplit`); `src/data/sampling.py` (`draw_distinct`) |
| Factors | `src/model/factors.py`; `src/train/factors.py` (losses); `src/train/train_factors.py` (`R3_CONFIG`, `train_factors`, `encode_rows`, checkpoints); `src/eval/factor_gates.py` |
| Conditions | `src/train/condition_sources.py` (`FactorComboSource`, `ClipClusterSource` with `.from_labels`, `CommunitySource`); `src/train/condition_episodes.py` |
| Scorer | `src/model/condition_interface.py`; `src/train/train_scorer.py` |
| Evaluation | `src/eval/label_episodes.py` (`standard_label_episodes`, `label_episodes_sha256`); `src/eval/condition_eval.py` (`label_ranks`, `condition_use_gain`, `build_human_swap_episodes`, `human_swap_success`, `ceiling_ranks(target_column=…)`, `label_oracle_ranks`) |
| Stage-(d) runs, reusable | `src/test/20261013_stage_d_selection/run_selection.py` (`--prepare` cache: split, R3 codes, CLIP k-means and community labels); `run_posthoc.py` (β-matched naive, Δ split, oracles, mechanism check); `src/test/20261014_stage_d_final/run_final.py`. Checkpoints for G1-G5 and the seeds are local and gitignored. |
