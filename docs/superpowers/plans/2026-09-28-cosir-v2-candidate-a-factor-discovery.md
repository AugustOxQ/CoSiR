# CoSiR v2 Candidate A, Stage (b): shared sparse factor discovery

> **For agentic workers:** TDD, function/class-formal code, no ad hoc scripts. Checkbox steps.

**Goal:** learn a shared sparse image-text factor dictionary on top of frozen CLIP features,
regularized by Block 1's graph/community structure — validated **on its own**, before any
condition-selection mechanism exists (per the spec's build order). This is Candidate A's
"Discovery" stage only — the condition interface, scoring function, and episode-mining training
are separate, later plans.

**Spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md`, "Candidate A
architecture" §1 ("Factor discovery"). **Important context from Block 1's validation** (read
`docs/reports/2026-09-28_cosir_v2_block1_stage1_validation.md` in full, especially the raw-CLIP
baseline and epoch-sensitivity sections): content-only Stage 1's trained embedding turned out
in-sample-emotion-AMI-equivalent to raw CLIP features clustered directly — the teacher graph is
itself built from raw CLIP mutual-kNN, so this is expected, not a bug. **This is why factor
discovery uses raw CLIP features as its primary input, not Stage 1's trained embedding** — Stage
1 contributes graph/community *structure* (which points are neighbors) as an auxiliary
regularizing signal, not a replacement representation. Do not substitute Stage 1's embedding for
raw CLIP features as the encoder input; that would conflate two different things.

## Global constraints

- Branch: `cosir-v2` (`/project/CoSiR-v2`).
- Reuse from Block 1: `build_content_graph`/`GraphConfig` (`src/model/graph.py`) for
  neighbor-consistency structure; `detect_communities`/`community_stats`
  (`src/model/communities.py`) for the community-level sampling hierarchy. Do not rebuild these.
- Reuse `FeatureManager` for cached ArtELingo CLIP features (confirmed working at 308,723-sample
  scale in Block 1's validation).
- No `cuml`/`cugraph`. `seed=42` for every stochastic step.
- Codex is the default implementer (direct `codex e --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check -c mcp_servers={} -C /project/CoSiR-v2 --json -` invocation). Fall back to
  direct implementation only if Codex is genuinely unavailable (per user direction, 2026-09-28);
  resume dispatching to Codex once it's usable again. Never engage `.ccg/tasks/` scaffolding or a
  secondary review layer — Claude is the sole reviewer.
- Real validation runs on the local GPU are pre-authorized (per user direction, 2026-09-28).

---

### Task 1: `SharedFactorEncoder` — sparse factor projection + reconstruction

**Files:** Create `src/model/factors.py`; test `src/test/test_factors.py`.

**Interfaces:**
- `class SharedFactorEncoder(nn.Module)`: `__init__(self, feature_dim: int, num_factors: int,
  dropout: float = 0.1)`. Two independent linear-then-nonnegative-sparsifying heads (one per
  modality — do not share weights between modalities, the whole point is testing whether
  *activations* agree, not forcing identical projections), each mapping `feature_dim -> num_factors`,
  output non-negative (e.g. ReLU or softplus — document your choice) and L1-sparse-encouraged (the
  sparsity itself is enforced by the *loss* in Task 2, not necessarily architecturally — a plain
  linear+ReLU head is a reasonable starting point; don't over-engineer the module itself).
  `encode_image(img_feat: Tensor[B, feature_dim]) -> Tensor[B, num_factors]`,
  `encode_text(txt_feat: Tensor[B, feature_dim]) -> Tensor[B, num_factors]`.
- `reconstruct_image(codes: Tensor[B, num_factors]) -> Tensor[B, feature_dim]`,
  `reconstruct_text(codes: Tensor[B, num_factors]) -> Tensor[B, feature_dim]` — a linear decoder
  per modality (tied or untied to the encoder weights is your call; document which and why —
  untied is simpler and a reasonable default).

- [ ] Write failing tests: output shapes; codes are non-negative; a reconstruction round-trip
  (`encode` then `reconstruct`) has finite values and non-trivial (not all-zero) output on random
  input; gradient flows to every parameter from a dummy reconstruction loss.
- [ ] Implement, run tests, self-review (is anything accidentally shared between the image and
  text heads that shouldn't be — check this explicitly, it's the seed of the split-dictionary
  risk this whole stage exists to test for).
- [ ] Commit: `feat(cosir-v2): SharedFactorEncoder sparse projection + reconstruction heads`

---

### Task 2: factor-discovery losses — reconstruction, paired agreement, graph consistency

**Files:** Create `src/train/factors.py`; test `src/test/test_factor_losses.py`.

**Interfaces:**
- `reconstruction_loss(encoder: SharedFactorEncoder, img_feat, txt_feat) -> Tensor` (scalar) —
  MSE (or your documented choice) between original frozen features and their round-tripped
  reconstruction, summed/averaged over both modalities.
- `paired_agreement_loss(img_codes: Tensor[B, L], txt_codes: Tensor[B, L]) -> Tensor` — encourages
  genuinely paired image/text codes to activate similar factors (e.g. a similarity/alignment loss
  between `img_codes` and `txt_codes` for matched pairs — cosine or L2 on the code vectors,
  document which).
- `graph_neighbor_consistency_loss(codes: Tensor[N, L], graph: scipy.sparse.csr_matrix, sample_idx: np.ndarray) -> Tensor`
  — for a batch of sampled node indices, encourages graph-neighbor pairs (per Block 1's
  `build_content_graph` output) to have more similar codes than non-neighbor pairs (a
  graph-contrastive-style term — reuse the symmetric-InfoNCE *pattern* from
  `src/train/stage1.py` if that fits naturally, or a simpler margin/similarity formulation if
  cleaner; your call, document which and why).
- `sparsity_penalty(codes: Tensor) -> Tensor` — L1 penalty encouraging sparse activation.
- `anti_split_penalty(img_codes: Tensor[B, L], txt_codes: Tensor[B, L]) -> Tensor` — **the
  control against the documented split-dictionary failure mode** (a nominally-shared dictionary
  quietly allocating separate factors to each modality). Concretely: penalize factors whose
  activation is consistently high in one modality and consistently near-zero in the other across
  a batch (e.g. a group-sparsity-style term comparing per-factor mean activation between
  modalities). Read the MGSAE reference this stage is inspired by (cited in
  `docs/reports/2026-09-28_cosir_v2_stage1_genecis_synthesis_brainstorm.md`, "Literature survey"
  table) for the shape of this control if you want a concrete precedent — do not skip this loss
  term or treat it as optional, it's the reason this whole discovery stage exists as a validated
  block rather than an assumption.

- [ ] Write failing tests for each loss term on small synthetic data: reconstruction loss
  decreases when reconstruction improves (sanity); paired-agreement loss is lower for matched
  pairs than randomly-shuffled pairs on a fixture where you control this directly; graph-
  consistency loss is lower when codes respect a known synthetic graph than when codes are
  random; anti-split penalty is measurably higher on a deliberately-constructed split-dictionary
  fixture (e.g. codes where half the factors are always ~0 for text and half always ~0 for image)
  than on a fixture with genuinely shared activation — this is the single most important test in
  this task, do not skip or weaken it.
- [ ] Implement, run tests, self-review.
- [ ] Commit: `feat(cosir-v2): factor-discovery losses (reconstruction, paired agreement, graph consistency, anti-split)`

---

### Task 3: factor-discovery training loop + real ArtELingo validation

**Files:** Create `src/train/train_factors.py`; test `src/test/test_train_factors.py`; create
`src/test/20260928_factor_discovery_validation/run_validation.py` +
`docs/reports/2026-09-28_cosir_v2_candidate_a_factor_discovery_validation.md`.

**Interfaces:**
- `@dataclass FactorTrainingConfig`: `num_factors: int`, `lr: float`, `epochs: int`,
  `batch_size: int`, `lambda_reconstruction: float`, `lambda_paired: float`, `lambda_graph: float`,
  `lambda_sparsity: float`, `lambda_anti_split: float`, `seed: int = 42` — sensible defaults are
  your call to choose and document (there is no validated reference for this exact loss
  combination — say so plainly rather than implying these are "validated" values).
- `train_factors(img_features, txt_features, graph, config) -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]`
  — trains the encoder, returns it plus final `(N, L)` image and text code matrices.

**Real validation, on ArtELingo** (same data-loading pattern as Block 1's Task 5 — reuse it, same
`FeatureManager`/`artelingo_train.json` positional join): train the factor encoder on the real
308,723-sample content graph, then check and report plainly:
1. **Does the anti-split control actually work on real data** — per-factor mean activation
   comparison between image and text codes; report how many factors (if any) are effectively
   modality-private despite the penalty.
2. **Do useful factors span multiple communities** (per the spec's own falsification criterion):
   for each factor, compute its activation distribution across Block 1's Task-4 community labels;
   a factor concentrated in exactly one community is "just renamed a topic," not a real
   cross-cutting aspect — report the fraction of factors that span multiple communities
   meaningfully vs. those that don't, plainly, not just an aggregate average.
3. **Basic sanity**: reconstruction quality, paired-agreement strength (matched vs. shuffled
   pairs), whether training is stable (no NaN, sparsity penalty doesn't collapse all codes to
   zero).

Report the real numbers in `docs/reports/2026-09-28_cosir_v2_candidate_a_factor_discovery_validation.md`,
with a plain verdict up front (does this factor dictionary look usable as a foundation for the
condition-interface stage, or does it show the split-dictionary/topic-renaming failure modes) —
matching this project's established report convention (verdict first, then evidence, then
caveats). If the anti-split control or the cross-community-spanning check fails on real data, say
so directly — do not tune hyperparameters repeatedly until the numbers look good and then present
only the final run; if you do adjust hyperparameters from your first attempt, report what you
tried and why, not just the final result.

- [ ] Implement the training loop and its unit tests (synthetic data, fast).
- [ ] Run the real ArtELingo validation, write the report with real numbers.
- [ ] Commit: `feat(cosir-v2): factor-discovery training loop + real ArtELingo validation`

---

### Task 4: usage-balance regularizer + input whitening (fixing the mass-concentration collapse)

**Added 2026-09-29** after Task 3's real validation found a third, unanticipated failure mode:
2 of 32 factors carried 87.7% of all activation mass (13/32 factors concentrated in a single
community; the anti-split and dead-factor controls both worked as designed, but nothing in the
original loss suite targeted *cross-factor* usage balance). Read
`docs/reports/2026-09-28_cosir_v2_candidate_a_factor_discovery_validation.md` in full for the
exact numbers this task must improve on.

**Files:**
- Create: `src/model/whitening.py`; test `src/test/test_whitening.py`.
- Modify: `src/train/factors.py` (add `usage_balance_penalty`); test addition in
  `src/test/test_factor_losses.py`.
- Modify: `src/train/train_factors.py` (`FactorTrainingConfig` gets `lambda_usage_balance: float`;
  wire the new loss into the training loop).
- Create: `src/test/20260929_factor_balance_fix/run_revalidation.py` +
  `docs/reports/2026-09-29_cosir_v2_candidate_a_factor_balance_fix.md`.

**Scope note**: this task stays on **CLIP features**, matching Task 3's original setup exactly —
do not combine this with the cross-encoder findings from the Block 1 ablation plan
(`docs/superpowers/plans/2026-09-29-cosir-v2-block1-cross-encoder-ablation.md`); that's a
different, later experiment. Isolate one variable (the loss/preprocessing fix) at a time.

**Interfaces:**
- `usage_balance_penalty(img_codes: Tensor[B,L], txt_codes: Tensor[B,L]) -> Tensor` — negative
  entropy of the batch's combined per-factor mean activation, normalized to a distribution:
  `p = mean_activation / mean_activation.sum()`, `penalty = (p * (p + eps).log()).sum()`. This
  quantity is minimized (most negative) when `p` is uniform (maximum entropy) and approaches 0 as
  activation concentrates onto few factors — added to the total loss with a positive
  `lambda_usage_balance`, minimizing it pushes toward uniform per-factor usage. Test this exactly
  like the anti-split penalty was tested: construct a deliberately peaked fixture (activation mass
  concentrated in 2 of 8 factors) and a deliberately uniform fixture, assert the peaked fixture's
  penalty is measurably higher (closer to 0, less negative) than the uniform fixture's.
- `pca_whiten(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, PCA]` in
  `src/model/whitening.py` — fits `sklearn.decomposition.PCA(whiten=True)` on `features`
  (full-rank, `n_components=features.shape[1]` — decorrelate and unit-variance-scale each
  component without discarding dimensions), returns the whitened features and the fitted `PCA`
  object (so a caller can apply the *same* fit to held-out data later, even though this task only
  needs the train-split transform). Test: whitened output has near-zero cross-component
  covariance and near-unit per-component variance on a synthetic correlated-features fixture.

**Real re-validation** (`run_revalidation.py`, reusing Task 3's real-data loading pattern exactly,
CLIP features): re-run factor discovery with (a) `lambda_usage_balance` added at a reasonable
starting weight (your call, document why — comparable in spirit to how `lambda_anti_split` was
scaled) and (b) PCA-whitened CLIP features as the encoder input instead of raw CLIP features.
Report, plainly, up front verdict first:
1. Does the max single-factor (or top-2-factor) activation-mass share drop substantially below
   87.7%? Report the actual new number.
2. Re-check anti-split and dead-factor counts — did fixing mass-concentration break either of the
   two previously-working controls (a real risk: pushing hard for uniform usage could force
   otherwise-dead or modality-private factors to activate anyway, in a way that isn't genuinely
   meaningful)? Report plainly if so.
3. Re-check the cross-community-spanning fraction (was 19/32 = 59.4%) — did it improve, stay flat,
   or get worse?
4. If results still show real problems, say so as directly as Task 3 did — this task's job is to
   test the proposed fix honestly, not to declare victory.

- [ ] TDD for `usage_balance_penalty` and `pca_whiten`, real anti-split-style discriminative
  tests, not shape-checks.
- [ ] Wire both into `train_factors`/`FactorTrainingConfig`.
- [ ] Real re-validation on ArtELingo, write the report with a plain verdict, comparing every
  number directly against Task 3's original figures.
- [ ] Commit (implementation): `git add src/model/whitening.py src/test/test_whitening.py src/train/factors.py src/test/test_factor_losses.py src/train/train_factors.py src/test/test_train_factors.py && git commit -m "feat(cosir-v2): usage-balance regularizer + PCA whitening for factor discovery"`
- [ ] Commit (validation): `git add src/test/20260929_factor_balance_fix/ docs/reports/2026-09-29_cosir_v2_candidate_a_factor_balance_fix.md && git commit -m "docs(cosir-v2): re-validate factor discovery with usage-balance fix"`

---

## Self-review

**Placeholder scan:** no TBD/TODO; the "no validated reference for this loss combination" and
"your call, document why" points are explicit implementation-time decisions, not silent gaps.
**Scope:** three tasks — encoder, losses (with the anti-split control as the load-bearing
piece), training + real validation. Right-sized for one plan; the condition interface and
episode-mining training are deliberately separate, later plans per the spec's build order.
**Type consistency:** `SharedFactorEncoder.encode_image/encode_text` outputs feed directly into
every loss function in Task 2 and into `train_factors`'s return contract in Task 3.
**Honesty check carried from Block 1:** this plan explicitly does not assume factor discovery
will "just work" — Task 3's validation criteria (anti-split check, cross-community-spanning
check) are real falsification tests, not confirmation-seeking, matching the discipline Block 1's
Task 5/5b/5c validation already established.
