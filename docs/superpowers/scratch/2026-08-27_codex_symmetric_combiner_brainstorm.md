# Symmetric conditioning brainstorm

## Scope and conclusion up front

This is a new model variant, not a `combine_side` configuration flip.  The smallest defensible first variant is **two modality applications of one shared `Combiner_new`, supplied with one shared per-sample condition table**, with a symmetric two-output loss and evaluation.  It removes the asymmetric `other_proj` pathway without doubling the combiner or condition-table parameter count.  It is the cleanest test of whether *the one-sided conditioning mechanism* amplifies the observed i2t-only frozen-condition effect.

The more expressive alternative—separate image/text combiners and two condition tables initialized to the same buddy embedding—is scientifically useful, but it is a second experiment.  It simultaneously changes symmetry, parameter count, and the inductive assumption that one latent condition describes both modalities.

Current source references below are line numbers as of 2026-08-27.

## What is asymmetric today

`CoSiRModel` owns exactly one `Combiner_new` at [src/model/cosirmodel.py:87-96](../../src/model/cosirmodel.py#L87-L96), validates and stores the binary `combine_side` at [98-100](../../src/model/cosirmodel.py#L98-L100), and owns an identity-initialized `other_proj` at [102-108](../../src/model/cosirmodel.py#L102-L108).  Its `forward()` combines only `txt_emb` or only `img_emb` ([188-204](../../src/model/cosirmodel.py#L188-L204)); `project_other()` explicitly describes the uncombined side ([176-178](../../src/model/cosirmodel.py#L176-L178)).  The sole predictor is likewise documented as combine-side only ([180-186](../../src/model/cosirmodel.py#L180-L186)).

The training loop repeats that choice: it selects one feature/full-sequence pair, slices one condition table, applies `project_other()` to the opposite side, and sends one `comb_emb` to the criterion ([src/hook/train_cosir.py:1534-1594](../../src/hook/train_cosir.py#L1534-L1594)).  `LabelContrastiveLoss_enhance.forward()` receives a single combined feature tensor ([src/metrics/loss.py:109-123](../../src/metrics/loss.py#L109-L123)); its preservation term explicitly selects the combine-side raw embedding ([198-214](../../src/metrics/loss.py#L198-L214)).

This matters for interpretation.  The current bidirectional cross-entropy of the *one* similarity matrix is symmetric as a ranking loss, but the functions producing that matrix are not: one modality receives a condition-dependent nonlinear update and the other receives a condition-free linear update.  Thus the new result is strong evidence that `combine_side` drives the bridge-node relocation, while the persistent i2t advantage says it is not the sole source of the frozen-vs-trained result.

## Concrete model families

All options below use raw frozen-CLIP outputs `(i, I_full, t, T_full)` and return conditioned outputs `(i', t')`.  They should replace `combine_side` with an explicit `conditioning_mode: symmetric_*`; keeping `combine_side` as a silent no-op would make runs hard to audit.

| Option | Condition state | Combiner / predictor state | Conditioned outputs | Main scientific use |
| --- | --- | --- | --- | --- |
| A. Shared table, shared combiner (recommended first) | One `z_s` per sample | Same `C` applied to both sides; one shared predictor `P`, called on each modality | `i'=C(i,I_full,z_s)`, `t'=C(t,T_full,z_s)` | Isolates bidirectional conditioning while retaining one common latent condition and approximately one combiner's capacity. |
| B. Shared table, separate combiners | One `z_s` | `C_img`, `C_txt`; predictors shared or separate | `i'=C_img(...)`, `t'=C_txt(...)` | Allows modality-specific transformations, but doubles combiner capacity relative to A/current. |
| C. Separate tables, shared combiner | `z_img`, `z_txt`, cloned from identical buddy-init values | One shared `C`; predictors normally separate by target table | `i'=C(i,I_full,z_img)`, `t'=C(t,T_full,z_txt)` | Tests whether each modality needs its own evolving latent state without giving it separate transformation networks. |
| D. Separate tables, separate combiners | `z_img`, `z_txt`, initially equal | `C_img`, `C_txt`, normally `P_img`, `P_txt` | `i'=C_img(...,z_img)`, `t'=C_txt(...,z_txt)` | Most flexible fully symmetric architecture; highest confounding and overfitting risk. |
| E. Tied-table, two-view conditions | Store one base `z`; derive side conditions by small learned, tied/anti-tied adapters | Usually one shared `C` | `z_img=A_img(z)`, `z_txt=A_txt(z)` | A controlled middle ground if A is too restrictive and C is unstable. |
| F. Pair-conditioned rather than per-sample symmetric | One pair/sample condition as A, but train a joint predictor `P(i,t)` too | One or two combiners | Same as A/B, with joint inference available only for paired data | Useful for analysis/training, but not a standalone retrieval-query predictor: an image-only or text-only query lacks its partner. |

### Weight-sharing choices

Option A makes the strongest symmetry claim: the same function is evaluated on both modalities.  The caveat is that `Combiner_new` consumes `emb_full`; text and image sequence lengths/statistics can differ.  Shared weights may therefore be a capacity constraint rather than a neutral symmetry control.  If image full features are not available in the current feature cache, they must be retained; the training loop already has an `img_full` fallback ([src/hook/train_cosir.py:1522-1531](../../src/hook/train_cosir.py#L1522-L1531)), but a zero fallback is not a faithful second combiner input.

Option B/D can maintain **architectural symmetry** (isomorphic networks and equal hyperparameters) without **parameter tying**.  That is likely the eventual performance variant, but should follow—not replace—the tied-weight test.  A practical staged conclusion is: A tests whether conditioning both sides is sufficient; B then tests whether modality-specific functions add value.

For the predictor, use either (1) a shared `P` evaluated on image and text, which matches A's tying but asks the mapping to handle both feature distributions, or (2) `P_img`/`P_txt` of identical architecture.  Separate predictors do not double retrieval transformation capacity, but they do add parameters and make modality-specific prediction quality interpretable.  The clean A baseline should use one shared predictor, then a `shared-table + separate-predictors` ablation can identify predictor underfitting.

### Table choices

With a shared table, one `z_s` gets gradients through both `i'` and `t'`.  This encodes the claim that the buddy-graph state is a shared cross-modal latent property.  It is also economical: at 150k samples and 32 dimensions, the table is about 4.8M float parameters (about 19 MB in fp32), unchanged from today.

With separate tables, clone the completed buddy-init tensor exactly once into `z_img` and `z_txt`; do **not** run graph initialization twice if the intent is identical starting values.  The result adds another ~4.8M parameters plus optimizer state and persistent storage.  Decide explicitly whether the losses contain an agreement term such as `1-cos(z_img,z_txt)`: no agreement tests free divergence; strong agreement turns the variant into a soft version of a shared table.  A sweep of that coefficient is a separate research question and should not be included in the first discriminator experiment.

## Loss design: symmetric in computation, not merely in name

The core contrastive matrix should be

`S = normalize(i') @ normalize(t').T`.

Use the same bidirectional InfoNCE/cross-entropy currently implicit in `LabelContrastiveLoss_enhance`—both `CE(S)` and `CE(S.T)` at [src/metrics/loss.py:123-137](../../src/metrics/loss.py#L123-L137)—but feed it *both conditioned outputs*.  This preserves the total contrastive weighting convention; do not add two full contrastive losses and accidentally double its scale.

Every auxiliary term must be made deliberately two-sided:

* Preservation: average a separate distance-to-input penalty for `i'` vs `normalize(i)` and `t'` vs `normalize(t)`.  The present branch at [198-214](../../src/metrics/loss.py#L198-L214) cannot remain.
* Delta/gate/logit: the combiner must return `(delta_img, gate_img, logit_img)` and the text analog; regularize each then average, and log both plus their difference.  Otherwise one modality can saturate unseen behind an average.
* Laplacian/manifold and mixup: the current laplacian call carries raw `text_features` and the one combined feature ([141-150](../../src/metrics/loss.py#L141-L150)); `imix_loss` is explicitly text-side (`model.combine(text_mixed, ...)`, [src/metrics/loss.py:19-55](../../src/metrics/loss.py#L19-L55)).  Define each side's version and average, or disable these terms in the first symmetric run while holding that setting fixed across its trained/frozen pair.  Leaving them text-only silently reintroduces asymmetry.
* Table regularizers: with a shared table, apply label smoothness/collapse/boundary once, not twice.  With two tables, report and regularize each separately; whether graph smoothness is also applied to their mean is a new hypothesis, not a bookkeeping detail.
* Predictor consistency: shared-table/shared-predictor A should distill both `P(i)` and `P(t)` to the same `z_s`, with equal weights.  Separate tables should use `P_img(i)->z_img`, `P_txt(t)->z_txt`; an optional cross-target term would be an additional coupling hypothesis.

This is more than a signature edit: `LabelContrastiveLoss_enhance` should expose a symmetric pair interface rather than keep ambiguous names like `image_features`, `text_features`, and `combined_features` whose roles swap today in the caller ([src/hook/train_cosir.py:1539-1546](../../src/hook/train_cosir.py#L1539-L1546)).

## Grounded change map

| Area | Current dependency | Symmetric change | Relative effort |
| --- | --- | --- | --- |
| `src/model/cosirmodel.py` | One `combiner`, `other_proj`, one predictor, combine-side branch. | Add explicit image/text combining API; remove `other_proj`/`project_other`; instantiate tied or separate combiners/predictors by variant; return both outputs and both combiner diagnostics. `forward()` must include both full sequences. | Medium, foundational. |
| `src/metrics/loss.py` | One combined tensor; combine-side preservation; text-only mixup/laplacian assumptions. | Change criterion to accept `img_comb`, `txt_comb`, raw inputs, and two diagnostic sets; average side-specific auxiliaries; define shared/separate-table regularizers. | Medium-high; behaviorally important. |
| `src/utils/embedding_manager.py` and `embedding_manager_nocache.py` | One parameter/storage namespace and one `embeddings` access path. | A/B need no manager redesign. C/D need a two-table wrapper or named-table support with two parameters, two save/load/template namespaces, `get[_all]_embeddings(side)`, and cloning from one init tensor. The active training hook imports `TrainableEmbeddingManager` from `src.utils`; inspect which implementation is exported before choosing the migration point. | Low for A/B; high for C/D because persistence/template compatibility is part of validity. |
| `src/hook/train_cosir.py` | One manager initialized in `_init_embedding_manager` ([223-349](../../src/hook/train_cosir.py#L223-L349)); optimizer has special `condition_predictor` and `other_proj` groups ([352-385](../../src/hook/train_cosir.py#L352-L385)); one-sided train step. | Build symmetric optimizer groups; obtain one or two condition batches; combine both sides; call new loss; perform two predictor distillations; make buddy contrastive/refresh and phase diagnostics symmetric or explicitly disable them. Template creation/loading must clone one buddy template for separate tables. | High; many diagnostics currently encode the old orientation. |
| Snapshots/checkpoints in `train_cosir.py` | Condition-viz stores one table, one combiner, one predictor, and `other_proj` ([531-617](../../src/hook/train_cosir.py#L531-L617)); final checkpoint does the same ([1261-1305](../../src/hook/train_cosir.py#L1261-L1305)). | Store named tables, combiner/predictor state dict(s), tying metadata, and separate drift/gate summaries. Remove rather than preserve misleading `other_proj_state_dict`. Update downstream notebook schema deliberately. | Medium. |
| Training evaluator / analysis snapshots | `TrainEvaluator` chooses one side ([src/eval/pipeline.py:183-199](../../src/eval/pipeline.py#L183-L199)); condition analysis/retrieval snapshots branch repeatedly on `combine_side` (for example [train_cosir.py:654-684](../../src/hook/train_cosir.py#L654-L684)). | Score `i' @ t'.T` consistently; evaluate both conditioned outputs. Replace orientation branches and label artifact schema as a new variant. | Medium-high, because paper analyses depend on them. |
| Test oracle and predictor metrics | Oracle projects only the other side and combines only one source ([src/eval/metrics.py:191-249](../../src/eval/metrics.py#L191-L249)); predictor metric repeats the same branch ([487-522](../../src/eval/metrics.py#L487-L522)). | Implement an explicitly defined two-sided oracle and a two-sided predictor score. `combine_side` branches can disappear in the symmetric path. | High, both computationally and scientifically. |
| Dormant non-oracle helpers | Three helpers always combine text against raw image, documented at [metrics.py:308-312](../../src/eval/metrics.py#L308-L312), [359-365](../../src/eval/metrics.py#L359-L365), and [428-434](../../src/eval/metrics.py#L428-L434). | A symmetric common path can replace these with correctly conditioned image/text scoring, eliminating the branch-specific bug rather than adding another branch. This is a real design upside, but it must be separately regression-tested. | Medium. |

## Evaluation semantics must be chosen before implementation

With both sides conditioned, “oracle max over conditions” is no longer uniquely defined.  The current evaluator takes each table condition, transforms only one modality, forms one matrix, and takes the elementwise maximum over candidates ([src/eval/metrics.py:225-280](../../src/eval/metrics.py#L225-L280)).  Its transpose provides the opposite retrieval direction but does not condition the opposite side independently.

Recommended reporting hierarchy:

1. **Coupled-table oracle (primary).** For each table representative `z_k`, condition both full galleries with the *same* `z_k`, form `S_k = i'(z_k) @ t'(z_k)^T`, then elementwise max/mean over `k`.  This is the closest symmetric analogue of the present global-table oracle and costs O(K) matrices.
2. **Two-sided predictor retrieval (deployable metric).** Predict a condition for every image and every text, form `i'(P_img(i)) @ t'(P_txt(t))^T`, and compute normal i2t/t2i recalls.  In A, a single predictor is called twice; in separate-table variants, use the matching predictor/table.  This is the meaningful no-search result.
3. **Independent two-sided oracle (diagnostic upper bound only).** Maximize over image-side and text-side condition candidates independently.  This costs O(K^2) candidate pairs (or needs an approximation) and is far looser, so it must never be compared numerically to the old one-sided oracle as though it were the same metric.

Also retain raw CLIP recall as the unconditioned baseline.  Do not interpret an oracle gain as deployable retrieval until the two-sided predictor metric agrees.

The symmetric path can eliminate all `combine_side` routing in `test_oracle`, `test_pre_diff`, train evaluation, and the three dormant non-oracle helpers.  That reduction is valuable: one `conditioned_similarity(images, texts, image_conditions, text_conditions)` primitive makes the evaluation contract auditable.

## Confounds and failure modes

* **Capacity versus symmetry.** A tied shared combiner/table does not double those parameter counts and actually removes `other_proj`; it is the key matched-capacity symmetry test.  Separate combiners double a large nonlinear component; separate tables add ~4.8M fp32 parameters at the stated scale.  Compare A with the old asymmetric model before making claims from B–D.
* **Optimization scale.** Two conditioning paths create twice as many delta/gate/predictor auxiliary contributions unless averaged.  Equal nominal loss weights can therefore be inequivalent.  Keep the contrastive-loss scale fixed and average paired auxiliaries.
* **Shared-table gradient conflict or domination.** In A, gradients through image and text may cancel, or the modality with larger/generally easier gradients may set `z_s`.  Log per-side gradient norms/cosines on the table, per-side gate/delta distributions, and predictor-to-table error.  A shared table collapsing to a compromise is a result, not proof that symmetry fails.
* **Separate-table overfitting/divergence.** Two tables can memorize modality-specific quirks across 150k samples, especially when conditions remain trainable.  Track `cos(z_img,z_txt)` from their identical initialization, table norms, nearest-neighbor preservation, and held-out predictor recall.  High oracle with weak predictor recall is a warning sign.
* **Sequence-input mismatch.** Tied `Combiner_new` sees `img_full` and `txt_full`; their token counts/statistics differ.  Performance loss could reflect inappropriate weight tying rather than conditioning both sides.  This is why B is a useful follow-up after A.
* **Unequal query cardinalities.** Captions-per-image means condition predictions and condition-table gradient sampling occur more often on text rows unless loss construction normalizes by modality/sample identity.  This can recreate an i2t/t2i bias in a “symmetric” graph.
* **Oracle inflation.** The independent two-sided oracle has a much larger search space.  It cannot adjudicate the hypothesis by itself; use coupled oracle plus predictor recall.
* **Legacy regularizers/diagnostics.** Buddy contrastive targets currently select a non-combine-side feature at [src/hook/train_cosir.py:1370-1387](../../src/hook/train_cosir.py#L1370-L1387), and several phase analyses are guarded by `combine_side == "img"`.  Leaving any enabled asymmetrically invalidates a symmetry claim.
* **Comparison protocol.** Same data split, buddy template, seed, backbone cache, optimizer budget, condition freeze policy, epoch/checkpoint selection, and metric code version are required.  Changing both the condition manager and evaluation definition in one comparison obscures the result.

## Minimal discriminating experiment

### Question and first model

Test the narrow hypothesis: *the one-sided condition-dependent transformation is responsible for most of the roughly 11x amplification of the frozen-vs-trained i2t effect under image-side combining.*

Implement/run only Option A first: one shared table, same buddy initialization, one shared/tied combiner applied to both modalities, one shared predictor applied to both modalities, and all auxiliary terms symmetrized by averaging.  This deliberately does **not** add a second condition table or second combiner.  Use the coupled-table oracle and two-sided predictor evaluation; retain raw recall.

### Smallest clean new run

The smallest new evidence set is **3 matched seeds x 2 condition arms** (trained versus frozen at the identical buddy template) for Option A.  Pair seeds with the already completed asymmetric `combine_side=img` and `combine_side=txt` sweeps wherever the existing run metadata permits.  The existing two arms are comparison baselines; no new reruns are required merely to establish the direction flip already measured.

Per seed and architecture, pre-register:

* `Δ_i2t = R1_frozen - R1_trained` and `Δ_t2i = R1_frozen - R1_trained`, reporting mean, SEM/paired CI, and the signed directional contrast `A = Δ_i2t - Δ_t2i`.
* The amplification comparison `|A_img-asym| / max(|A_symmetric|, epsilon)` and the corresponding txt-asym comparison; use CIs rather than treating the ratio alone as a test statistic.
* Raw, coupled-oracle, and two-sided-predictor R@1/R@5/R@10; condition drift/gate statistics per modality; and the established bridge-node subgroup report with symmetric labels.

Use final checkpoints or the same predeclared checkpoint rule for every arm.  Do not select the best direction-specific epoch.

### Interpretive outcomes

**Supports the architectural-amplification hypothesis:** Option A substantially shrinks `A` toward zero relative to the image-combine asymmetric result, with paired uncertainty inconsistent with the old large i2t-only contrast; both directions now respond comparably to freeze/train while the bridge-side relocation disappears or becomes symmetric.  It is acceptable if absolute retrieval changes—the claim concerns the *freeze effect asymmetry*, not necessarily a performance improvement.

**Refutes/limits it:** Option A retains a large, seed-replicated i2t-only `Δ` comparable to the image-combine baseline, despite both modalities receiving the same condition mechanism and all losses/evaluation being symmetric.  Then the 11x factor cannot mainly be attributed to the one-sided combiner; investigate data/cardinality, retrieval geometry, buddy initialization, or the definition of frozen condition updates.

**Ambiguous:** Symmetric training collapses, predictor recall fails while oracle is high, or both directions degrade so severely that the frozen/trained comparison is noise-dominated.  Diagnose optimization/weight-sharing first; do not infer that graph topology wins.

Only after A is interpretable should the next 3x2 sweep be C (separate cloned tables with one shared combiner) or B (separate combiners with shared table), chosen according to whether A's evidence points to table-gradient conflict or shared-transform mismatch.  D should be last because it combines both confounds.

## Rough cost

This is a **substantial new-model-variant undertaking**, not a cheap config override.  The narrowly scoped Option A avoids a new manager class and avoids doubling persistent table state, but still requires coordinated model, loss, training, snapshot/checkpoint, train-eval, oracle, predictor-eval, and analysis-path changes.  It needs new unit/integration tests for shape/tying, trained-versus-frozen behavior, and symmetric evaluation consistency before expensive runs.

Option C/D add a new two-table persistence/template contract and migration of all condition-drift/viz tools; that is materially larger.  A realistic research sequence is: first implement and validate A as a dedicated model mode, run its 3x2 discriminator, then decide whether the evidence justifies the separate-table/combiners program.

