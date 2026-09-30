# CoSiR v2 Candidate A: affect-signal factor learning (design)

**Status:** design approved section by section in conversation (2026-09-30); awaiting review of this written spec.
**Parent spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md` (amended by §1 below).
**Follows:** the factor-learning 2×2 (`docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md`,
report `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`), which stopped at selection, and
the headroom probe (`docs/reports/auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md`).

## 0. Context: the system and the terms this spec uses

**The task.** CoSiR v2 Candidate A scores an image and a caption *under a condition* given by examples, such as
"the same emotion as these" or "the same art style as these". It retrieves in two directions: **i2t** (an image is the
query; candidate captions are ranked) and **t2i** (a caption is the query; candidate images are ranked).

**Data.** ArtELingo: 308,723 rows, each one painting image with one annotator's English caption. Every caption
carries the annotator's **emotion** label (9 values, one of them "something else"); every painting has one **art
style** (27 values). About 5 captions share each painting's image. Images and captions are represented by **frozen
CLIP ViT-B/32 features** (512-d). ArtELingo's emotion and style labels are used **only to evaluate**, never to train.

**Factors and codes.** Two small encoders (one linear layer plus ReLU per modality, `SharedFactorEncoder`) map CLIP
features to 32 non-negative numbers, the **factors**; an image and a caption each get a 32-number **code**. A row's
**pair code** is `0.5·(image code + caption code)`. The encoders are trained without labels: reconstruction of the
CLIP features, **paired agreement** (InfoNCE: an image and its own caption get matching codes), a content-graph term,
sparsity, usage-balance and decorrelation terms. **R3** (`R3_CONFIG` in `src/train/train_factors.py`) is this recipe;
**R0** is the old recipe that collapsed, kept as the reference level of one gate.

**The naive rule and the score.** A condition is given by 4 **supports** (items that have it) and 4 **contrasts**
(items that do not). The naive rule sets factor weights with no parameters, `w = ReLU(mean support pair code − mean
contrast pair code)`, L1-normalized. A query `q` and a candidate `c` score
`s = β · cos(CLIP_q, CLIP_c) + Σ_l w_l · q_l · c_l`. **β** weighs plain CLIP similarity against the condition-weighted
factor term; it is fixed at **0.3**.

**Label episodes (the evaluation).** For a target label (one emotion or one art style): an anchor with the label, 4
supports with it, 4 contrasts without it, and 13 candidates, 1 positive with the label and 12 negatives from paintings
never given it. **R@1** is the share of episodes in which the positive ranks first (chance 7.7%); reported per
direction and as the mean of the two directions.

**Rows.** Split by painting: train / val / held. Train is split again by painting into **scorer-train** (183,694 rows)
and **selection** (32,413 rows). Val is never used. Held is read only by the final test.

**Condition episodes (training).** Self-generated conditions: a partition of scorer-train rows into groups; an episode
takes an anchor, 4 supports, 4 positives from one group, and 4 contrasts plus 12 random negatives from outside it. The
**naive-rule episode loss** scores the episode with the naive rule at β 0.3 in both directions and rewards ranking the
positives first; its gradient trains the factor encoders (`naive_episode_loss`, `train_factors(..., condition_source=)`).

**What came before.**
- The headroom probe: frozen CLIP holds far more emotion and style than R3's factors express (a label-aligned 36-d
  code reaches a label-oracle R@1 of 49.8% against R3's 20.5%). Emotion is read from captions (linear probe 57.9%),
  hardly from images (35.2%, majority class 28.4%). No label-free partition lines up with emotion (AMI ≤ 0.06).
- The 2×2: **C0** (R3's recipe refit on scorer-train rows with painting-expanded batches) is the matched control.
  **S** (C0 plus condition episodes from 64 k-means clusters of CLIP image features) gained **+3.6 R@1 on art style**,
  in the code itself, but lost emotion (−2.1 at β 0) and was stopped (sparsity gate, emotion guard). Painting-level
  agreement (A) did not free emotion. No label-free route to emotion was found.

**Collapse gates.** Nine pass/fail checks on a factor space (`src/eval/factor_gates.py`, amended thresholds of
2026-09-29): participation ratio ≥ 8; max factor correlation ≤ 0.90; readout of CLIP features no worse than R0's on
the same rows; **sparsity**: active fraction ≤ 0.50; no dead factors; at most 1 modality-private factor; top-2 usage
≤ 0.20; community spanning ≥ 0.75; pair retrieval ≥ half of CLIP's. Here gate 4 (sparsity) is reported only (§5).

**GoEmotions.** A RoBERTa model fine-tuned on GoEmotions (Reddit comments labelled with 28 emotion categories),
`SamLowe/roberta-base-go_emotions`. It was never trained on ArtELingo. PercepT, this project's baseline on the percept
line, uses the same model as its affect input.

**AMI** (adjusted mutual information): how well a partition lines up with a label; 0 unrelated, 1 identical.

## 1. Parent-spec amendment (user decision, 2026-09-30)

The parent spec allowed only self-generated conditions and ruled out any external condition taxonomy. The user
amended this, following PercepT:

- **Allowed:** the GoEmotions RoBERTa (`SamLowe/roberta-base-go_emotions`) as a **training signal**, in the form of
  its 28 sigmoid probabilities per caption.
- **Unchanged:** ArtELingo's emotion and style labels are evaluation-only. No other external model or taxonomy is
  added. At test time the retrieval model still reads only frozen CLIP features; the affect vectors are used only to
  build training conditions.

## 2. Intent

Make the factors carry emotion, with the one mechanism that has worked (condition episodes through the naive-rule
loss), now fed by an affect signal, while keeping art style. Success is an **emotion** gain over C0 on human-label
episodes, with style protected.

## 3. Settled decisions (user-approved, 2026-09-30)

| Decision | Choice | Reason |
|---|---|---|
| Primary criterion | **Emotion gain over C0 + powered style guard** | Emotion is the reason for this experiment; style is protected, not required to improve. |
| Signal form | **28 GoEmotions probabilities per caption row** | Pure emotion content, non-negative, interpretable; the buddy model's validated affect input; per annotation, as emotion is. |
| Approach | **Affect condition episodes (A): cells E and SE** | The direct analogue of S's style gain; reuses the built machinery; SE tests whether style and emotion signals coexist. |
| Gates | **8 structural gates binding; sparsity reported only** | Density is not collapse; condition training makes codes denser; decided before any run. |
| Testing | **Selection on 4,096 episodes per label, then one held test on fresh episodes** | Doubles selection power; held rows are unspent by the 2×2. |

## 4. Affect source

**Extraction.**
- Captions: `annotations[sample_id]["caption"]` via the existing positional join (all 308,723 captions are English;
  median 14 words, 99th percentile 29).
- Rows: **scorer-train rows only** (183,694). Val, held and selection captions are never passed to the model.
- Model: `SamLowe/roberta-base-go_emotions` from the local Hugging Face cache (offline), eval mode, no grad,
  `max_length=64`, truncation and padding, batch 256, fp32. Output: `sigmoid(logits)`, shape (183,694, 28), float32,
  in scorer-train order, cached to the run folder with its SHA-256 recorded.

**Affect partition.** `MiniBatchKMeans(n_clusters=64, random_state=42, n_init=3, batch_size=4096)` on the **raw** 28-d
probability vectors of scorer-train rows (the same settings as the image-cluster source; no normalization). Groups
below 200 rows are not conditions (the source's `min_group_rows`).

**Diagnostics (measured and reported, never used for a choice).** On scorer-train rows:
- AMI of the affect partition with ArtELingo emotion and art style, beside the image partition's (0.035 / 0.318) and
  the caption partition's (0.056 / 0.058).
- A multinomial logistic probe from the 28 affect probabilities to ArtELingo emotion, fit on 80% of scorer-train
  paintings and scored on the other 20% (a painting-grouped split, seed 42), beside a CLIP-caption → emotion probe on
  the same split and the majority class. No selection, val or held caption ever reaches the affect model.

## 5. Cells and training

All cells: `R3_CONFIG` with `painting_batches=True`, pair agreement, `lambda_condition=1.0`, `condition_beta=0.3`,
64 condition episodes per step, 12 random negatives, 2,000 steps, seed 42, trained on scorer-train rows. They differ
only in the condition source.

| Cell | Condition source | Role |
|---|---|---|
| **E** | the affect partition | affect signal alone |
| **SE** | affect partition and CLIP image partition: each condition's partition is drawn with probability ½, then a group uniformly within it | affect and style signals together |
| **C0** | none (existing checkpoint `src/test/20261016_factor_learning_grid/checkpoints/C0_seed42.pt`, SHA-256 recorded, config asserted) | matched control |
| S (reference) | CLIP image partition (existing 2×2 checkpoint) | reported, never a candidate |

**Gates.** `evaluate_factor_gates` with `AMENDED_2026_09_29_THRESHOLDS`, fit rows = scorer-train, eval rows =
selection, readout reference = R0's readout on the same rows (the 2×2's cached value). **Binding:** participation
ratio, redundancy, readout, dead, modality-private, usage concentration, community spanning, pair retrieval.
**Reported only:** sparsity (active fraction).

## 6. Selection (selection rows) and stop points

**Episodes.** `standard_label_episodes(data, groups, selection, label, 4096, seed=42)` per label. Their first 2,048
are stage (d)'s selection episodes (asserted: the SHA-256 of the first 2,048 equals `run_final.SELECTION_SHA256`).

**Scores** of a cell `X` (naive rule at β 0.3; per episode, the mean of i2t and t2i; paired bootstrap over episodes,
5,000 resamples, seed 42; R@1 points):
- **Primary:** `D_emo(X) = R@1(X) − R@1(C0)` on the 4,096 emotion episodes.
- **Guard:** `D_style(X)`, the same on the 4,096 art-style episodes.

**Rule.**
1. **Eligible:** all 8 binding gates pass.
2. **Qualifies:** eligible, `D_emo` lower bound > 0, and `D_style` lower bound > −1.5.
3. **Pick:** the qualifying cell with the highest `D_emo`; cells within 0.5 points tie; a tie goes to E (fewer
   changes), then to the higher `D_emo`.

**Guard power (stated before the run).** At 4,096 episodes the style SE is about 0.42 points (the 2×2 measured about
0.57 at 2,048). A cell with no true style change then passes the −1.5 guard with probability about 0.95, and one with
a true −0.5 about 0.66. The report recomputes these from the measured SE.

**Stop points.**
- C0 fails any binding gate: the setup is broken; stop and report.
- No cell qualifies: stop, report, no held test.

**Replication.** The picked cell and C0 retrained with seeds 43 and 44; `D_emo`, `D_style` and gates reported. The
verdict rests on seed 42.

**Reported, not gating** (every cell, C0, S and original R3): pooled and per-direction R@1; naive R@1 on the β grid
{0, 0.03, 0.1, 0.3, 1} and the balance-matched-β comparison against C0 (as in the 2×2's post-hoc); the label oracle
and its null; per-modality code probes, within-painting caption-residual emotion probe and code AMI (reusing
`src/test/20261016_factor_learning_grid/run_posthoc.py` functions); per-target breakdown; sparsity; the condition loss
and τ over training.

## 7. Held test (pre-registered; run once)

**Episodes.** New held label episodes `standard_label_episodes(data, groups, held, label, n, seed=43)`. `n` per label
is fixed before the run: `SE_sel` = the half-width of the picked cell's `D_emo` CI / 1.96; assumed effect
`0.75 × D_emo,sel`; for `n` in (2048, 4096, 8192), `SE_n = SE_sel × sqrt(4096 / n)` and power `Φ(effect / SE_n − 1.96)`;
the smallest `n` with power ≥ 0.8, else 8192 with the power stated.

**Criterion** (picked cell vs C0, both seed 42, same episodes): **confirmed iff** the `D_emo,held` lower bound > 0 and
the `D_style,held` lower bound > −1.5.

**Also reported:** pooled and per-direction numbers, the label oracle, the replication seeds' held differences,
original R3 naive, CLIP-only.

**Row scope.** Factors only encode held rows; affect vectors are never computed for held rows. The held phase refuses
to run twice; a smoke run of the whole held path on selection rows comes first and its numbers are discarded.

**Disclosure.** Held rows were read in two earlier final tests (the repair plan's and stage (d)'s, seed-42 episodes).
The 2×2 never read them. These episodes are new; the rows and paintings are not.

## 8. Constraints

- **Labels:** ArtELingo emotion and style labels are evaluation-only (diagnostic probes fit on them only to measure,
  never to train or choose). The only external training signal is §1's GoEmotions model.
- **Rows:** training, the affect extraction, the partitions and every fitted statistic use scorer-train rows; the
  selection rows are for choosing; held rows only in §7.
- **No tuning:** every weight, k, step count, margin and threshold above is fixed before the runs.
- **Environment:** seed 42 unless stated; no `cuml`/`cugraph`; conda env `CoSiR`; `/project/CoSiR`, `main`; commit
  locally, never push. At most 3 training processes share the GPU at once (4 ran out of memory in the 2×2).
- **Code:** function/class-formal code in `src/` with unit tests (TDD); defaults unchanged for existing callers. Real
  runs in dated `src/test/yyyymmdd_<name>/` folders with a log and a `.gitignore`. Each modified `src/` file gets a
  `.claude/yyyymmdd_log.md` entry.
- **Reports:** `docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md` and
  `2026-10-19_candidate_a_affect_factor_learning_held.md`; verdict first; every number against a named baseline (C0 for
  criteria; naive on original R3 at β 0.3 as the current system); figures; no em dashes; a row in
  `docs/reports/reports_sum.md` (own lines only) and `scripts/check_reports_sum.py` OK.
- **Execution:** SDD with Claude Code subagents (model per task), per-task reviewers, an Opus whole-branch final
  review, one fix wave; overnight automation allowed, stopping only at §6's stop points or before destructive actions.
  Another session may be committing in this repo: stage only the task's own files.

## 9. Code structure

| Module | Change |
|---|---|
| `src/data/artelingo.py` | Add `join_captions(sample_ids, annotations) -> np.ndarray` (the caption string per row, same positional join). |
| `src/data/affect.py` (new) | `GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"`; `goemotions_probabilities(texts, device, batch_size=256, max_length=64, model_name=GOEMOTIONS_MODEL) -> np.ndarray` of shape (n, 28), float32, sigmoid outputs, offline load. |
| `src/train/condition_sources.py` | Add `MultiPartitionSource(labels_by_view: dict[str, np.ndarray], rows, min_group_rows=200, max_tries=100)`: conditions are groups of any view; `sample_condition` draws a view uniformly, then one of that view's valid groups uniformly. Not swap-capable. |
| `src/test/20261018_affect_factor_learning/run_affect.py` | `--prepare` (extraction, partition, diagnostics, C0/S checkpoint SHAs), `--smoke` (timing), `--run CELL [--seed]`, `--evaluate`, `--replicate`, `--tables`; reuses `run_grid.py` and `run_posthoc.py` helpers via importlib. |
| `src/test/20261019_affect_factor_learning_held/run_held.py` | `--power`, `--smoke` (selection rows), `--run` (once), `--tables`. |

**Key tests.**
- `join_captions`: the positional join returns the right caption per row and rejects bad ids.
- `goemotions_probabilities` (skipped if the model is not in the local cache): shape (n, 28), values in [0, 1],
  deterministic; a clearly joyful sentence scores joy above sadness and a clearly sad one the reverse; batch size does
  not change the output (beyond float tolerance).
- `MultiPartitionSource`: groups come from every view; view keys do not collide; a view with no valid group is
  excluded; view-then-group sampling gives each view about half of the conditions; outside = the same view's other
  groups; works with `mine_condition_episodes(num_hard=0)`.
- The run script asserts row scope (only scorer-train captions reach the affect model), the C0 and S checkpoint
  configs and SHA-256s, and the selection-episode prefix SHAs.

**Build order (the plan's tasks).**
1. `join_captions`, `goemotions_probabilities`, `MultiPartitionSource` (TDD).
2. Run script: prepare (extraction, partition, diagnostics) and the timing smoke.
3. Runs E and SE, selection evaluation, the selection report (**stop points**).
4. Replication (seeds 43/44, picked cell and C0).
5. Power, held smoke, the held test, the held report.
6. Final review and one fix wave.

## 10. Out of scope

- The 768-d RoBERTa embedding; the affect teacher-target approach (B) and affect as an encoder input (C).
- Painting-level agreement (the 2×2's A); capacity changes (more factors, TopK); whitening.
- Tuning any weight, k, step count, temperature or margin.
- Stage (e) (a human-judged set), the free-text condition path, Candidate B.

## 11. Risks and caveats to carry into the reports

- **Taxonomy mismatch.** GoEmotions' 28 Reddit emotions are not ArtEmis's 9; §4's diagnostics measure the alignment.
- **Image side.** Images carry little emotion (probe 35.2% against a 28.4% majority), so i2t emotion gains may be small;
  the primary averages the two directions.
- **Affect clusters may follow content.** Captions' affect probabilities can still track subject matter; the AMI with
  art style is reported.
- **Selection rows have been read many times** (stage (d), the probe, the 2×2). This experiment chooses on them; the
  held test confirms.
- **The control is C0, not R3**; original R3 is shown beside it. C0 and S are the 2×2's seed-42 checkpoints.
- **Code scale acts like β**; the β grid and the balance-matched comparison show it.
- **One seed for the verdict**; seeds 43/44 are reported. Episodes reuse rows, so bootstrap CIs are somewhat optimistic.

## Glossary

- **C0:** the matched control, R3's recipe refit on scorer-train rows with painting-expanded batches.
- **E / SE:** C0 plus condition episodes from the affect partition / from the affect and image partitions.
- **Affect partition:** 64 k-means groups of scorer-train captions by their GoEmotions probability vectors.
- **`D_emo`, `D_style`:** a cell's naive R@1 minus C0's on the emotion / art-style episodes, mean of both directions.
- **Label oracle:** one weight vector per human label, fitted on half of that label's episodes and ranking the other
  half; measures how much label information a code carries.
