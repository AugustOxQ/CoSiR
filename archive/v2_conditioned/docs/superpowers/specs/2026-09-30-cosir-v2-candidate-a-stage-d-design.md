# CoSiR v2 Candidate A — Stage (d): trained conditional scorer (design)

**Status:** design approved section by section in conversation (2026-09-30); awaiting review of this
written spec. **Parent spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md`
("Candidate A architecture" §2-4, build order (d)). **Builds on:** the factor-repair plan
(`docs/superpowers/plans/2026-10-08-cosir-v2-candidate-a-factor-repair.md`) and its reports in
`docs/reports/auto/v2/` (collapse diagnosis, repair grid, held-out condition evaluation).

## 1. Intent

Build and test a **trained conditional scorer** `s(I,T|c)`. It reads a condition `c` from a few
support examples (and contrast examples), then ranks captions for an image (i2t) or images for a
caption (t2i) by that condition.

**What it must show:** that it *uses the condition better than the zero-parameter naive rule* on
the same factors, **in both retrieval directions**, and that its ranking flips correctly when only
the condition changes.

**Why this is the next step.** On held-out paintings, the repaired factor space (R3) improved on
the old recipe, but in image→text almost all of that improvement (83%) was condition-independent.
The naive rule did not use the condition better there. Two pieces of evidence point at the rule
rather than the factors:
- In i2t the space improved overall while the rule's condition use did not.
- An earlier probe found a ranking-trained head reversed 41/44 condition swaps, against the naive
  rule's 13-15.

## 2. Settled decisions (user-approved, 2026-09-30)

| Decision | Choice | Reason |
|---|---|---|
| What trains | **Only the scorer** (condition interface, β, temperature). R3's factor encoders stay **frozen** | Gains are attributable, and R3's validation stays intact. Mining is non-circular by construction, and there is no risk of re-collapse. |
| Training conditions | **Self-generated from CoSiR's own structure; compare three sources** | Human labels never enter training (parent spec). The source is the main experimental variable. |
| Selection data | **15% of the train paintings**, carved out as a selection set | Val was spent choosing R3 and flatters the naive rule. Held stays full-size and untouched for the final test. |
| Interface form | **Naive rule + learned per-factor correction**, zero-initialized | Step 0 equals the baseline. Tiny, hard to overfit, interpretable. |
| Objective | Ranking loss, **with and without** the swap term, as an experimental factor | Tests the parent spec's assumption that a swap term is needed. |

## 3. Data, conditions and episodes

**Frozen factors.**
- R3 checkpoint: `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt` (SHA-256
  `1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f`).
- Load with `load_factor_checkpoint`; encode all rows once with `encode_rows`. Codes are never
  updated.

**Splits.**
- The existing painting-grouped split (seed 42) is 216,107 train / 30,872 val / 61,744 held rows.
- Stage (d) splits **train** again, by leakage group (seed 42):
  - about 85% **scorer-train**;
  - about 15% **selection**.
- Val is not used. **Held is untouched until the final test** (§6).
- Every condition source, cluster model, graph and episode used for training is built from
  scorer-train rows only.

**The three condition sources.** Each defines groups of items that "share a condition":

| Source | Group ("inside") | Outside | Swap-capable |
|---|---|---|---|
| **Factor combinations** | Choose k ∈ {1,2,3} R3 factors uniformly, with weights ~ Dirichlet(1). The group is the top 10% of scorer-train rows on the weighted pair-code score. | Bottom 50% on that score | Yes: two disjoint factor sets whose top-10% groups both contain the anchor |
| **CLIP clusters** | k-means (k=64, seed 42, CPU; no cuML) fitted on L2-normalized raw CLIP **image** features (visual condition) and, separately, **caption** features (caption condition). A group is one cluster of one view. | Any other cluster of the same view | Yes: the anchor's image cluster vs its caption cluster |
| **Block 1 communities** | Stage-1 communities (graph, Stage 1 and detection via the existing `build_content_graph`, `train_stage1` and `detect_communities`) on scorer-train rows | Any other community | **No** (one group per item): trained with the ranking loss only |

**Training episode.** Its shape mirrors the held-out human-label episodes, so the skill transfers.
- The anchor, **4 supports** and **4 positives** come from the group; **4 contrasts** come from
  outside.
- **12 negatives** come from outside the group:
  - **6 hard:** the outside items nearest the anchor in raw CLIP (pair feature), so generic
    similarity cannot solve the episode;
  - **6 random**.
- No painting appears twice in an episode. Distinctness is keyed on leakage-group ids.
- Scored in both directions: i2t uses the anchor image vs candidate captions; t2i uses the anchor
  caption vs candidate images.

**Swap episode** (swap runs only).
- The anchor lies in two groups, A and B, from the same source. Each condition has its own 4
  supports and 4 contrasts.
- Shared pool:
  - 3 items from A\B;
  - 3 items from B\A;
  - 10 negatives outside both groups (5 hard, 5 random).
- Under condition A the A\B items are the positives; under condition B the B\A items are. No
  painting repeats.

**Known confound.** The sources offer very different numbers of distinct conditions: effectively
unlimited for factor combinations, 128 for CLIP (two views × 64), and about 20 for communities.
The report must state this when comparing sources.

## 4. Model and loss

**Per-factor evidence.** Computed from pair codes `0.5·(img+txt)` of the 4 supports and the 4
contrasts, for each factor `l`:
- the naive gap `g_l = mean_S − mean_C`;
- `mean_S`, `mean_C`, `std_S`;
- `act_S` and `act_C`: the fraction of supports and of contrasts with the factor active.

Code-valued features are standardized by fixed per-factor statistics from scorer-train pair codes.

**Interface.** A tied MLP 6 → 16 → 16 → 1 (ReLU), shared across all 32 factors (a few hundred
parameters). Its last layer is **zero-initialized**.
- `w_l = ReLU(g_l + correction_l)`, then L1-normalized; an all-zero vector stays zero.
- At step 0 this equals the naive rule exactly.

**Score.**
- `s = β·cos(CLIP_query, CLIP_cand) + Σ_l w_l·q_l·c_l`, via the existing `conditional_score`.
- `β = softplus(ρ)`, initialized to 0.3 (Task 7's val choice for R3 naive).
- Logits are `s/τ`, with a learnable `τ` initialized to the standard deviation of step-0 scores on
  the first training batch.

**Loss.**
- A **multi-positive ranking loss** per episode and per direction: `−log(Σ_pos e^{s/τ} / Σ_all e^{s/τ})`,
  averaged over directions and the batch.
- Swap runs add `λ_swap = 1.0 ×` the same loss on swap episodes, evaluated under both conditions.

**Training (fixed in advance, no tuning).**
- Adam at learning rate 1e-3, for 3,000 steps.
- Each batch is 64 freshly mined episodes (swap runs add 64 swap episodes), so no episode is
  reused.
- Seed 42. No early stopping.

## 5. Selection (on the selection set)

**Runs.** The only data-driven choice in stage (d) is among these five:

| Run | Source | Swap term |
|---|---|---|
| G1 | Factor combinations | no |
| G2 | Factor combinations | yes |
| G3 | CLIP clusters | no |
| G4 | CLIP clusters | yes |
| G5 | Block 1 communities | no |

**Baseline.** "Naive" is the step-0 model: the naive rule with β = 0.3.

**Selection episodes.**
- `standard_label_episodes` on selection rows: 2,048 emotion + 2,048 art style, seed 42.
- The Ruling 13 construction applies: leakage-group keys, clean negatives, and "something else"
  excluded as a target.

**Selection score.** The condition-use gain over naive:
`Δ(m) = [R@1(m) − R@1(m | wrong condition)] − [R@1(naive) − R@1(naive | wrong condition)]`
- "Wrong condition" swaps each episode's supports and contrasts with another episode's, using a
  seeded derangement.
- The score is averaged over {emotion, art style} × {i2t, t2i}.
- Ranks are tie-aware.

**Rule.**
1. Highest Δ wins.
2. Runs within 1.0 R@1 point of the best tie; a tie goes to the no-swap run, then the earlier run
   in the table.
3. **Stop point:** if the best Δ ≤ +0.5 R@1 points, no run beats naive. Report that and stop;
   there are no extra runs without user approval.

**Replication.** Retrain the selected recipe with seeds 43 and 44, for the scorer init and episode
sampling. Report Δ for each seed.

## 6. Final held-out test (pre-registered; the selected scorer vs naive, same episodes)

1. **Primary: uses the condition better than naive, in each direction separately.**
   - `Δ_held(i2t)` and `Δ_held(t2i)` on the held label episodes identical to Task 7's: 1,024
     emotion + 1,024 art style, pooled over label types.
   - The episodes are rebuilt with `standard_label_episodes` (seed 42) and asserted equal to
     Task 7's recorded SHA-256s.
   - Uncertainty: a paired bootstrap over episodes, 5,000 resamples, seed 42.
   - **Met iff the 95% CI is above 0 in i2t AND in t2i.** Judged on the seed-42 model; seeds 43
     and 44 are reported alongside.
2. **Human swap test** (new episodes; 1,024 from held rows, seed 42).
   - One held anchor `a` has two conditions:
     - `c_emo`: `a`'s emotion (never "something else"). Supports come from other paintings with
       that emotion. Contrasts come from paintings with no annotation of it.
     - `c_style`: `a`'s art style. Supports are other paintings of that style; contrasts are
       other styles.
   - One pool:
     - `p_emo`: same emotion, different style;
     - `p_style`: same style, from a painting with no annotation of `a`'s emotion;
     - 11 negatives: other emotion and other style, from paintings with no annotation of `a`'s
       emotion.
   - **Success** means `p_emo` ranks above `p_style` under `c_emo` AND `p_style` ranks above
     `p_emo` under `c_style`.
   - **Met iff the selected scorer's success rate minus naive's has a paired-bootstrap 95% CI above
     0**, pooled over both directions. Each direction is also reported.

**Diagnostics (reported, not criteria).**
- **Ceiling check** on the selection set: per-episode best weights on the frozen R3 codes.
  Projected gradient ascent on the simplex maximizes the positive's margin over the hardest
  negative, with β = 0.3. It answers whether fine-tuning the factors is ever worth revisiting.
- Results per source, and the effect of the swap term.
- The three seeds.
- CLIP-only and uniform-weight rows.
- R3 naive's Task 7 numbers, for continuity.

**Reaching stage (e)** (the human-judged evaluation set) is the user's decision after the report.

## 7. Constraints

- **Training data:** human labels (emotion, art style) are **evaluation-only**. No other external
  condition taxonomy is used.
- **Frozen factors:** R3 factors are never updated.
- **Held-out rows:** used only in §6, and never for any choice.
- **Mining scope:** mining, clustering, graphs and communities use scorer-train rows only. Episodes
  never repeat a painting, keyed on leakage groups.
- **Environment:** seed 42 unless stated; no `cuml`/`cugraph`; conda env `CoSiR`; repo
  `/project/CoSiR`, branch `main`.
- **Code layout:** function/class-formal code in `src/` with unit tests. Real runs go in dated
  `src/test/yyyymmdd_<name>/` folders, each with a log and a local `.gitignore`.
- **Change log:** each modified `src/` file gets an entry in `.claude/yyyymmdd_log.md`.
- **Reports:** go in `docs/reports/auto/v2/YYYY-MM-DD_<topic>.md`, with a row in
  `docs/reports/reports_sum.md` and a pass of `python scripts/check_reports_sum.py`. Each report
  puts its verdict first, and every number comes from a real run.
- **Execution:** SDD with Claude Code subagents as implementers (no Codex unless the user asks),
  Claude as controller and reviewer, and a whole-branch final review.

## 8. Code structure

| Module | Responsibility |
|---|---|
| `src/train/condition_sources.py` | The three sources. Each is fitted on given rows and exposes the groups containing an item, the inside and outside pools, and swap capability. |
| `src/train/condition_episodes.py` | Group-aware multi-positive episode miner, hard negatives from raw CLIP, the swap-episode miner, and painting-distinctness via leakage groups |
| `src/model/condition_interface.py` | Per-factor evidence features, the residual tied-MLP interface, and the scorer (β, τ, `conditional_score`) |
| `src/train/train_scorer.py` | `ScorerTrainingConfig`, `train_scorer`, and scorer checkpoint save/load |
| `src/eval/` additions | Human swap episodes, the wrong-condition control, the condition-use gain Δ with a paired bootstrap, and the ceiling check |

**Key tests:**
- The step-0 interface equals the naive rule exactly.
- Episodes: no painting repeats; positives are inside the group and negatives outside; hard
  negatives are the anchor's nearest outside items.
- Swap episodes: both conditions contain the anchor, and the A\B and B\A roles are disjoint.
- Human swap episodes: the `p_emo` / `p_style` definitions hold and negatives are clean.
- Training beats naive on a synthetic task where the naive rule is deliberately suboptimal.
- Δ and the bootstrap match hand-computed values.

**Build order** (the plan's tasks):
1. Sources and the miner.
2. Interface, scorer and trainer.
3. Evaluation additions.
4. Selection split, runs G1-G5 and selection (**stop point**).
5. Replication and the final held test.

## 9. Out of scope

- Fine-tuning the factor encoders. Revisit only if the ceiling check shows the frozen factors lack
  the needed information.
- Stage (e), the human-judged evaluation set.
- The free-text condition path.
- Candidate B.
- Hyperparameter sweeps.

## 10. Risks and caveats to carry into the reports

- **Selection set:** it contains paintings whose features R3's encoders saw (never their labels).
  This affects all candidates equally, so it is fine for choosing, but not for final numbers.
- **Source confound:** the sources differ in how many distinct conditions they offer (§3).
- **Emotion labels:** emotion is per annotation. Clean negatives mitigate, but do not remove, the
  ambiguity.
- **Label coverage:** only two human condition types are available, so success says nothing yet
  about arbitrary stated conditions (that is stage (e)'s job).
- **Held split:** a single held split. Task 7 already reported on it, though nothing was selected
  on it.

## Glossary

- **Naive rule:** condition weights = ReLU(support mean − contrast mean), L1-normalized.
- **Condition-use gain (Δ):** how much more a model gains from being told the right condition
  (versus a wrong one) than the naive rule gains.
- **Swap:** the same query and candidates, with a different condition; the right answer should
  change.
- **Ceiling check:** the best any weighting of the frozen factors could do on an episode.
