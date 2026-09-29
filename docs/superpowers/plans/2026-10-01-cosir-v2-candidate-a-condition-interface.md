# CoSiR v2 Candidate A, Stage (c): condition interface + episode mining

> **For agentic workers:** TDD, function/class-formal code, no ad hoc scripts. Checkbox steps.

**Goal:** build and validate, on its own, the mechanism that turns a handful of demonstrated
image-text pairs into a per-factor condition weight vector `w(c) ∈ R_+^L`, plus the pipeline that
automatically mines the support/contrast/candidate-pool episodes used to train and test it — both
**before** the scoring function or swap-loss training exist (those are stage (d), a separate,
later plan).

**Spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md`, "Candidate A
architecture" §1-2 and 4 ("Condition interface", "Training"). **Important context**: read
`docs/reports/2026-09-30_cosir_v2_candidate_a_usage_balance_no_whitening.md` in full — it is the
validated factor-discovery recipe this stage builds on (raw CLIP features + `usage_balance_penalty`,
**no whitening**; do not reintroduce whitening, it was tested twice and only hurt reconstruction).
Also read `/root/.claude/projects/-project-CoSiR/memory/project_candidate-a-factor-discovery-status.md`
if available for the short version of that history.

## Global constraints

- Branch: `cosir-v2` (`/project/CoSiR-v2`).
- Reuse from Block 1/factor-discovery, do not rebuild: `build_content_graph`/`GraphConfig`
  (`src/model/graph.py`); `detect_communities`/`community_stats` (`src/model/communities.py`);
  `SharedFactorEncoder` (`src/model/factors.py`) with `encode_image(img_feat) -> Tensor[B,L]`,
  `encode_text(txt_feat) -> Tensor[B,L]`; the full loss suite in `src/train/factors.py` including
  `usage_balance_penalty`; `train_factors`/`FactorTrainingConfig` (`src/train/train_factors.py`).
- Reuse `FeatureManager` for cached ArtELingo CLIP features, same `artelingo_train.json` positional
  join pattern as every prior real-validation script in this project.
- **No persisted factor-encoder checkpoint exists yet** — every prior validation script trained
  `SharedFactorEncoder` in-memory and discarded it (training is fast, ~24s per Task 4's timing
  table). Follow the same pattern here: each task's real-data script retrains the encoder inline
  using Task 6's exact recipe (raw features, `lambda_usage_balance` at Task 4's value, no
  whitening) rather than adding new checkpoint save/load infrastructure.
- No `cuml`/`cugraph`. `seed=42` for every stochastic step.
- Codex is the default implementer (direct `codex e --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check -c mcp_servers={} -C /project/CoSiR-v2 --json -` invocation). Never engage
  `.ccg/tasks/` scaffolding or a secondary review layer — Claude is the sole reviewer.
- Real validation runs on the local GPU are pre-authorized (per user direction, established
  2026-09-28, reconfirmed through Candidate A's factor-discovery tasks).
- This stage does **not** touch the scoring function `s(I,T|c)` or any swap-loss training loop —
  those are stage (d), a separate plan. Do not implement them here even partially; stay scoped to
  the condition encoder and the mining pipeline.

## Review Focus

- An episode where the "support" set accidentally contains the anchor itself, or support and
  contrast sets overlap — mining must explicitly exclude the anchor and de-duplicate indices
  across every role in an episode.
- A targeted factor `l*` that is effectively dead or near-dead in the current dataset (very few
  items have meaningful activation) — mining must skip or resample such factors rather than
  silently producing a degenerate episode with too few high/low candidates.
- `ConditionEncoder` given a **single** support pair and no contrast pairs (the spec's "a handful"
  can be as few as one) — the set-encoder (mean-pooling) must not crash or silently misbehave on
  a size-1 "set".
- Support and contrast sets containing the **same** underlying item (a pair can plausibly appear
  in both if selection logic has an off-by-one on the high/low threshold split) — must be
  prevented by construction, not just hoped to be rare.
- A real-data run where the "recovery" sanity check (Task 3) does no better than chance at
  identifying the true targeted factor — this must be reported as a real negative result, not
  smoothed over, matching every prior real-validation report in this project.

---

### Task 1: `ConditionEncoder` — set-encoder producing `w(c) ∈ R_+^L`

**Files:** Create `src/model/condition.py`; test `src/test/test_condition.py`.

**Interfaces:**
- `class ConditionEncoder(nn.Module)`: `__init__(self, hidden_dim: int = 16)`. Consumes per-pair
  factor codes (`Tensor[*, L]`, any `L`, the module does not hardcode a factor count — its scoring
  head is applied per-factor via broadcasting, tied across factors).
- `forward(self, support_img_codes: Tensor[S, L], support_txt_codes: Tensor[S, L], contrast_img_codes: Tensor[C, L] | None = None, contrast_txt_codes: Tensor[C, L] | None = None) -> Tensor[L]`
  (a single condition weight vector; batching over multiple conditions at once is not required by
  this task — a later stage can add it if needed):
  1. Per-pair signature: `pair_code = 0.5 * (img_code + txt_code)` for each support pair (elementwise
     mean of that pair's image and text factor activations — matches Task 2's paired-agreement
     assumption that genuine pairs share factor activation).
  2. Permutation-invariant pooling: `support_summary = pair_codes.mean(dim=0)` (`Tensor[L]`). Same
     for `contrast_summary` if contrast codes are given; if not given, use a `Tensor[L]` of zeros
     **and** a separate learned scalar flag (a single `nn.Parameter`) added to the head's input so
     the module can distinguish "no contrast supplied" from "contrast summary happens to be zero" —
     do not silently conflate these two cases.
  3. Per-factor scoring head: a small MLP shared across all `L` factors (e.g.
     `nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))` applied with the
     3 inputs being `[support_summary_l, contrast_summary_l, has_contrast_flag]` broadcast per
     factor `l`, i.e. reshape to `(L, 3)` and apply the MLP row-wise, squeeze to `(L,)`) — document
     this exact shape contract in a docstring since it is easy to get the broadcasting axis wrong.
  4. Non-negativity: `w(c) = F.softplus(raw_scores)` (document why softplus over ReLU: this head's
     output feeds a later scoring function that weights factor agreement multiplicatively — a hard
     zero from ReLU permanently kills a factor's gradient path through this head at initialization,
     softplus keeps a small positive gradient everywhere, a real property this task's own tests
     should check, not just assert by docstring).

- [ ] Write failing tests:
  - Output shape is `(L,)` for `L` in {1, 8, 32}, regardless of `S`/`C`.
  - Output is non-negative (softplus property, but test the module's actual output, not the
    formula).
  - **Permutation invariance**: shuffling the order of support pairs (and separately, contrast
    pairs) does not change the output (exact equality, since mean-pooling is exactly
    permutation-invariant).
  - **Size-1 support set, no contrast**: `S=1`, `contrast_img_codes=None` does not crash and
    produces a finite, non-negative `(L,)` output.
  - **No accidental support/contrast overlap sensitivity bug**: construct a fixture where
    `support_img_codes`/`support_txt_codes` encode a clear activation spike on factor index 3 (high
    `pair_code[3]`, near-zero elsewhere) and `contrast_*_codes` encode a flat/uniform profile with
    **no** spike on factor 3; after a short training loop (see below) the model's `w(c)[3]` should
    end up the largest entry — **this is the single most important test in this task**.
  - **Training signal test** (supports the fixture above): since this module has learnable
    parameters (the per-factor MLP head) but no loss function is defined elsewhere yet, write a
    small self-contained training loop *in the test* using a simple recoverability objective:
    `loss = F.cross_entropy(w(c).unsqueeze(0), torch.tensor([true_factor_idx]))` (treating "which
    factor is the condition" as an `L`-way classification problem over the raw pre-softplus scores,
    or over `w(c)` directly — your call, document which and why), run ~50-200 steps of Adam on a
    handful of synthetic support/contrast fixtures each with a different planted `true_factor_idx`,
    and assert accuracy (does `argmax(w(c))` match `true_factor_idx`) exceeds chance (`1/L`) by a
    wide, unambiguous margin on held-out fixtures not used for training. Document this explicitly as
    a **validation-only auxiliary objective** — it is not the real swap-loss training from stage
    (d), it only checks the architecture has the right inductive bias to be trainable for this job.
- [ ] Implement, run tests, self-review (does the `has_contrast_flag` actually change behavior when
  toggled with an all-zeros contrast summary vs. no contrast at all — check this explicitly, it's
  the seed of a silent bug where "no contrast" and "contrast summary is exactly zero" get confused).
- [ ] Commit: `feat(cosir-v2): ConditionEncoder set-encoder producing per-factor condition weights`

---

### Task 2: episode mining — support/contrast/candidate-pool construction from factor + graph structure

**Files:** Create `src/train/episodes.py`; test `src/test/test_episodes.py`.

**Interfaces:**
- `@dataclass EpisodeMiningConfig`: `num_support: int = 4`, `num_contrast: int = 4`,
  `num_hard_negatives: int = 4`, `num_condition_distractors: int = 4`,
  `num_anchor_distractors: int = 4`, `high_activation_percentile: float = 90.0`,
  `low_activation_percentile: float = 50.0`, `min_pool_size: int = 50` (a targeted factor whose
  high- or low-activation pool falls below this after percentile split is skipped and resampled —
  the guard against the "near-dead factor" failure mode in Review Focus), `seed: int = 42` —
  sensible defaults are your call to choose and document, there is no validated reference for these
  exact numbers.
- `@dataclass Episode`: `anchor_idx: int`, `targeted_factor: int`, `support_idxs: list[int]`,
  `contrast_idxs: list[int]`, `positive_idx: int`, `hard_negative_idxs: list[int]`,
  `condition_distractor_idxs: list[int]`, `anchor_distractor_idxs: list[int]`. All index lists and
  `anchor_idx`/`positive_idx` refer to positions in the same `(N, L)` code arrays passed to
  `mine_episodes`. `targeted_factor` is retained as ground truth for validation only — a real
  training loop (stage d) would not have access to it and must not use it as a training signal
  there, only here for testing the mining pipeline and Task 1's encoder honestly.
- `mine_episodes(img_codes: np.ndarray, txt_codes: np.ndarray, config: EpisodeMiningConfig, num_episodes: int) -> list[Episode]`

**Mining logic per episode** (document each rule exactly, this is the part a fresh reader most needs
spelled out):
1. Compute `pair_codes = 0.5 * (img_codes + txt_codes)` once, `(N, L)`.
2. Pick `targeted_factor = l*` uniformly at random from factors whose high/low pools (step 3) both
   satisfy `min_pool_size` (resample up to a bounded number of attempts, then raise a clear error if
   no valid factor is found — do not loop forever).
3. Rank all `N` items by `pair_codes[:, l*]`. `high_pool` = items at or above
   `high_activation_percentile`; `low_pool` = items at or below `low_activation_percentile`. These
   two pools are disjoint by construction (percentiles are on opposite ends).
4. `anchor_idx`: sample one item from `high_pool`.
5. `support_idxs`: sample `num_support` items from `high_pool`, excluding `anchor_idx`, without
   replacement.
6. `contrast_idxs`: sample `num_contrast` items from `low_pool`, without replacement.
7. `positive_idx`: sample one item from `high_pool`, excluding `anchor_idx` and every index already
   used in `support_idxs` — this is "another item that genuinely shares the condition with the
   anchor," distinct from the support set the condition encoder gets to see.
8. **Hard negatives** (matched on other axes, differ on the targeted factor): from `low_pool`,
   compute cosine similarity between each candidate's `pair_codes` **with index `l*` zeroed out**
   and the **positive's** `pair_codes` with index `l*` zeroed out; take the `num_hard_negatives`
   highest-similarity candidates, excluding indices already used. Document this exact definition:
   "similar to the positive on every factor except the condition, but low on the condition itself."
9. **Condition-only distractors** (share the condition, nothing else): from `high_pool`, compute
   cosine similarity between each candidate's non-`l*` sub-vector and the **anchor's** non-`l*`
   sub-vector; take the `num_condition_distractors` **lowest**-similarity candidates, excluding
   indices already used. Document: "high on the condition like the anchor, but otherwise
   dissimilar."
10. **Anchor-only distractors** (resemble the anchor overall, wrong on the condition): from
    `low_pool`, compute cosine similarity between each candidate's non-`l*` sub-vector and the
    **anchor's** non-`l*` sub-vector; take the `num_anchor_distractors` **highest**-similarity
    candidates, excluding indices already used. Document: "resembles the anchor on everything else,
    but does not satisfy the condition — distinct from a hard negative, which is matched against
    the *positive*, not the anchor."
11. If any pool runs out of unused candidates before reaching its requested count, reduce that
    role's count for this episode and record the shortfall (do not raise, do not silently pad with
    duplicates) — Task 3's real-data report must state how often this happened, if at all.

- [ ] Write failing tests on small synthetic `(N, L)` code arrays with planted structure (e.g.
  `N=200`, `L=8`, construct items with a clear, controllable activation on one factor):
  - No episode's `support_idxs`/`contrast_idxs`/`positive_idx`/every distractor list contains
    `anchor_idx`, and no index appears in two different roles within the same episode (the
    Review Focus overlap bug) — assert this directly by set-intersection checks across all role
    lists, for every mined episode in the test.
  - A factor engineered to have a near-empty high-activation pool (fewer than `min_pool_size` items
    above the percentile) is never chosen as `targeted_factor` across many mining calls.
  - Hard negatives are verifiably drawn from the **low** pool (low activation on `l*`) while having
    high non-`l*` similarity to the positive — construct a fixture where this is unambiguous and
    assert on the actual selected indices' properties, not just that the function returns
    something.
  - Condition-only distractors are verifiably drawn from the **high** pool with low non-`l*`
    similarity to the anchor; anchor-only distractors are verifiably drawn from the **low** pool
    with high non-`l*` similarity to the anchor — same fixture-based direct assertion style.
- [ ] Implement, run tests, self-review (re-check step 11's shortfall bookkeeping is exercised by
  at least one test with a deliberately small `N`).
- [ ] Commit: `feat(cosir-v2): episode mining pipeline (support/contrast/candidate-pool from factor + graph structure)`

---

### Task 3: real validation on ArtELingo (the actual point of this plan)

**Files:** Create `src/test/20261001_condition_interface_validation/run_validation.py`; create
`docs/reports/2026-10-01_cosir_v2_candidate_a_condition_interface_validation.md`.

Reuse Task 3-6's real-data loading pattern exactly (same `FeatureManager`/`artelingo_train.json`
positional join). Train `SharedFactorEncoder` inline using Task 6's exact validated recipe (raw
CLIP features, `usage_balance_penalty` at Task 4's `lambda_usage_balance` value, no whitening, 32
factors, seed 42) to get real `(N, 32)` image and text codes for all 308,723 samples — this is a
prerequisite input, not new modeling work.

**Part A — mining quality, at real-data scale**: run `mine_episodes` on the real codes for at least
500 episodes. Report, plainly:
1. How many distinct factors got chosen as `targeted_factor` across all episodes, and whether any
   factor was skipped for insufficient pool size (Review Focus item 2) — report the count, not just
   "it worked."
2. Mean and distribution of the shortfall count from mining logic step 11 (how often a role's
   requested count couldn't be fully met) — report plainly if this happened often, it would mean
   the percentile/pool-size defaults need revisiting.
3. A concrete sanity spot-check: for 10 sampled episodes, report the targeted factor and the actual
   `pair_codes[:, l*]` values for the anchor, one support item, one contrast item, the positive, one
   hard negative, one condition-only distractor, and one anchor-only distractor — enough for a human
   reader to visually confirm the high/low split is real, not a bug that happens to pass the unit
   tests' small fixtures but not real data's actual activation distribution.

**Part B — condition encoder recovery, at real-data scale**: using the mined episodes' known
`targeted_factor` as ground truth (validation-only, per Task 2's docstring), run the same
recoverability training from Task 1's test (train `ConditionEncoder` to predict `targeted_factor`
from each episode's `(support, contrast)` codes), but now on a real train/held-out split of the 500+
mined episodes (e.g. 80/20, by episode, seed 42 — document the split). Report:
1. Recovery accuracy on the held-out episodes (`argmax(w(c)) == targeted_factor`), against chance
   (`1/32 ≈ 3.1%`).
2. Whether accuracy is roughly uniform across the different targeted factors that appeared, or
   concentrated on a few "easy" factors — report the breakdown, not just the aggregate, since an
   aggregate can hide a mode where the encoder only ever learns to recognize 2-3 factors.
3. If accuracy is at or near chance, say so directly, as plainly as every prior real-validation
   report in this project — this is a real possible outcome (it would mean the architecture, or the
   mining pipeline's episodes, don't carry a learnable signal for condition recovery yet) and stage
   (d)'s planning would need to account for it, not paper over it.

State the plain verdict up front in the report: is this a usable foundation for stage (d) (the
scoring function and swap-loss training), or does either the mining pipeline or the condition
encoder show a real problem that needs a design decision before proceeding — matching this
project's established report convention (verdict first, then evidence, then caveats).

- [ ] Implement the real-data script (Part A + Part B) and write the report.
- [ ] Commit: `git add src/test/20261001_condition_interface_validation/ docs/reports/2026-10-01_cosir_v2_candidate_a_condition_interface_validation.md && git commit -m "docs(cosir-v2): validate condition interface + episode mining on real ArtELingo data"`

---

### Task 4: confusion-pattern diagnostic for the 19 dead factors

**Added 2026-10-01** after Task 3's real validation found held-out condition recovery well above
chance overall (21.95% vs. 3.125%) but sharply uneven: 19 of 32 factors had zero correct held-out
predictions while several others reached 67-100%. Read
`docs/reports/2026-10-01_cosir_v2_candidate_a_condition_interface_validation.md` in full for the
exact numbers this task must explain. This is a **diagnostic-only** task — no architecture or
training change yet; the point is to find out *why* the 19 factors are dead before choosing a fix.

**Files:** create `src/test/20261002_condition_confusion_diagnostic/run_diagnostic.py`; create
`docs/reports/2026-10-02_cosir_v2_candidate_a_condition_confusion_diagnostic.md`.

**Method:** reproduce Task 3's exact setup byte-for-byte where it matters for reproducibility (same
Task 6 factor-encoder recipe, same `mine_episodes` call with the same `EpisodeMiningConfig` and
`seed=42`, same 80/20 episode split with `seed=42`, same `ConditionEncoder` training) so the mined
episodes and train/held-out split are identical to Task 3's. Additionally:
1. For every held-out episode, record `(true_factor, predicted_factor)` where
   `predicted_factor = argmax(w(c))`. Build the full 32x32 confusion matrix (rows = true factor,
   columns = predicted factor).
2. For each of the 19 factors Task 3 reported as zero-correct, report the full distribution of what
   they were predicted as instead (which factor(s) absorbed their held-out episodes, and in what
   proportion) — not just the single most common wrong prediction.
3. **Global prediction skew check**: across *all* held-out predictions (not just the wrong ones),
   report how concentrated the predicted-factor distribution is (e.g. what fraction of all 205
   held-out predictions land on the 5 most-predicted factors) — this tests whether the encoder is
   collapsing toward a handful of "attractor" factors regardless of the true condition, a different
   failure mode from correlated-but-distinct factors being confused with each other.
4. **Real-data correlation check**: compute the `(32, 32)` pairwise cosine similarity matrix between
   factor columns of `pair_codes` across all 308,723 real samples (i.e., how correlated each pair of
   factors' real activation patterns is, independent of any episode or training). For each of the 19
   dead factors, report whether its most-common wrong prediction (from step 2) is among its top-3
   most-correlated factors by this real-data measure. Report the fraction of dead factors for which
   this holds.

**Report, plainly, verdict up front**: which story does the evidence support — (a) real-data factor
correlation genuinely makes some conditions ambiguous from a few examples (dead factors' wrong
predictions concentrate on their most-correlated real-data neighbors), (b) the encoder collapses
toward a small fixed set of attractor factors regardless of the true condition (a training/capacity
issue, not a factor-space issue), or (c) neither pattern is clear (wrong predictions look close to
uniformly spread across the other 31 factors, suggesting something else, e.g. too little training
data per factor). State which of these (or a mix) the numbers actually show, do not force-fit to
one story if the evidence is mixed.

- [ ] Implement the diagnostic script and write the report with the confusion matrix, per-dead-
  factor breakdown, global skew check, and real-data correlation check.
- [ ] Commit: `git add src/test/20261002_condition_confusion_diagnostic/ docs/reports/2026-10-02_cosir_v2_candidate_a_condition_confusion_diagnostic.md && git commit -m "docs(cosir-v2): diagnose condition-recovery confusion pattern on dead factors"`

---

### Task 5: class-balanced training fix + recovery re-test

**Added 2026-10-02** after Task 4's diagnostic found the condition-recovery failure is story (b):
the encoder collapses onto a handful of attractor factors (5 factors absorbed 68.78% of held-out
predictions; 16/32 factors were never predicted at all) rather than confusing genuinely correlated
factors (the real-data-correlation check ruled out story (a), 0/19). This points at the
recoverability-check's training procedure — 819 training episodes over 32 classes (~25/factor), 100
epochs, `lr=0.05`, no class-balanced sampling — as an imbalanced/undertrained-classifier artifact,
not evidence the factor space itself is unusable. Read
`docs/reports/2026-10-02_cosir_v2_candidate_a_condition_confusion_diagnostic.md` in full.

**Scope note**: this task changes the recoverability-check's *training procedure* (more episodes,
class-balanced batch sampling, a revised optimization schedule) to test whether the collapse is
fixable — it does **not** change `ConditionEncoder`'s architecture (from Task 1) or `mine_episodes`'
mining logic (from Task 2). If, after this fix, the collapse persists, that is real evidence the
architecture itself needs reconsideration — report that plainly rather than trying further training
tweaks past what's specified here.

**Files:** create `src/test/20261003_condition_recovery_balanced_retrain/run_retrain.py`; create
`docs/reports/2026-10-03_cosir_v2_candidate_a_condition_recovery_balanced_retrain.md`.

**Changes to test, in one combined re-run** (report if you find you need to isolate them
individually to explain a surprising result — but run the combined version as the primary result):
1. **More episodes**: `num_episodes=4096` (4x Task 3/4's 1024) via the same `mine_episodes` call,
   same `EpisodeMiningConfig` defaults, seed 42 — gives ~128 episodes/factor on average instead of
   ~32, reducing per-class sample-size noise. Use the same 80/20 episode-level split (seed 42).
2. **Class-balanced batch sampling**: during `ConditionEncoder` training, weight each training
   episode by `1 / (count of training episodes sharing its targeted_factor)` (inverse-frequency
   weighting — a standard fix for classifier collapse toward majority/easy classes) so the expected
   number of episodes per factor per epoch is equal, instead of Task 3/4's plain sequential/random
   batches. Implement via `torch.utils.data.WeightedRandomSampler` or an equivalent explicit
   resampling scheme — your call, document which.
3. **Revised optimization schedule**: lower learning rate (`lr=0.01`, down from `0.05`) and more
   epochs (`300`, up from `100`) — document this as a single deliberate choice, not a
   tune-until-it-works loop; if you do find yourself needing a second adjustment to get a sane
   training curve, report what you tried and why, matching this project's established honesty
   convention (see Task 3/4's factor-discovery reports for the pattern).
4. **Training-dynamics tracking** (new, to directly diagnose whether balancing worked): every 50
   epochs, record (a) training-set accuracy, (b) held-out accuracy, (c) the number of *distinct*
   factors predicted at least once across the held-out set (the diversity metric Task 4 found stuck
   at 16/32) — report this trajectory, not just the final numbers, so a reader can see whether
   balanced sampling actually changes the collapse dynamic during training or just shifts the final
   snapshot.

**Report, plainly, verdict up front**, comparing directly against Task 3/4's baseline:
1. Held-out recovery accuracy (was 45/205 = 21.95%) — report the new number.
2. Number of distinct factors predicted at least once in the held-out set (was 16/32).
3. Prediction concentration on the top-5 most-predicted factors (was 68.78%).
4. Per-factor accuracy breakdown (was 19/32 at exactly 0%) — report how many factors are now at 0%,
   and whether the previously-dead factors specifically improved or whether different factors are
   now dead (a real risk: balancing could fix the aggregate while just moving which factors collapse).
5. The training-dynamics trajectory from step 4 above.

State plainly whether class-balanced training with more episodes fixes the collapse (uniform-ish
per-factor accuracy, most/all factors predicted at least once), only partially fixes it (better
aggregate but a meaningfully-sized dead-factor set remains), or does not fix it (collapse persists
at similar severity) — the last case would be real evidence for reconsidering
`ConditionEncoder`'s architecture rather than its training procedure.

- [ ] Implement the balanced-retraining script and write the report with the comparison above.
- [ ] Commit: `git add src/test/20261003_condition_recovery_balanced_retrain/ docs/reports/2026-10-03_cosir_v2_candidate_a_condition_recovery_balanced_retrain.md && git commit -m "docs(cosir-v2): re-test condition recovery with class-balanced training"`

---

## Self-review

**Placeholder scan:** no TBD/TODO; "your call, document why" points are explicit
implementation-time decisions with a documented default, not silent gaps.
**Scope:** three tasks — set-encoder, mining pipeline, real validation of both together. The
scoring function and swap-loss training (stage d) are deliberately excluded, matching the spec's
own build order.
**Type consistency:** `ConditionEncoder.forward`'s `support_img_codes`/`support_txt_codes`/
`contrast_img_codes`/`contrast_txt_codes` inputs are `Tensor[*, L]` matching
`SharedFactorEncoder.encode_image`/`encode_text`'s `Tensor[B, L]` output exactly (same `L`).
`Episode`'s index lists are positions into the same `(N, L)` `img_codes`/`txt_codes` arrays Task 3
generates from the validated Task 6 recipe — no separate ID space is introduced.
**Review Focus coverage:** anchor/support/contrast overlap (Task 2 tests), near-dead targeted
factor (Task 2 tests + Task 3 Part A report item 1), size-1 support set (Task 1 tests), support/
contrast index collision (Task 2's set-intersection test), chance-level recovery as a real
reportable outcome (Task 3 Part B item 3) — all five have an owning test or report requirement.
