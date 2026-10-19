# PercepT core in five settings, and the missing rows of the ablation table: design

> Date: 2026-10-18 23:33 · Status: approved 2026-10-18 23:34; amended 2026-10-19 08:25 (A1, A2 in §4), text brought in line 2026-10-19 09:45; amended 2026-10-19 10:45 (A3 in §4), fix-test count brought in line 2026-10-19 16:26 (build 6) · Kind: experiment
> Builds on: [aspect graph follow-up](../../reports/auto/percept/2026-12-12_aspect_graph_followup.md) and its
> [ablation table](../../user_read/2026-10-18_aspect_graph_followup.md) (rows 1 to 13) · facts brief
> `.scratch/r10-percept-core/facts.md` · grill record `.scratch/r10-percept-core/grill.md` · Constitution:
> [constitution.md](../constitution.md)
> Decision rule: none (descriptive; the baseline's "established" checks are in §6)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** We build PercepT core: PercepT with the four fixes from the code audit, on our own features. It
fills the PercepT rows of the ablation table. These rows set what our model must beat by 2026-10-29. After that, the
table's missing rows of our own network run on the CPU.

**Plan.**
1. Regression check (amended 2026-10-19, A1): only what repeats exactly. The saved September snapshot re-scores to
   its recorded numbers (emotion 0.1225, genre 0.2274); the reused September code is byte-identical; k-means on the
   saved latents gives the saved labels. Our scoring code reproduces the default-embedding row (row 2) exactly. The
   September re-fit left the check: it does not repeat itself, on this machine or between machines.
   PercepT core's k-means start runs on one CPU thread, so its runs repeat exactly (A2).
2. Baseline: PercepT core with three Stage 1 versions and five Stage 2 settings, 3 seeds each, on the local GPU.
   - Stage 1 versions: CLIP + RoBERTa (September encoders); Qwen with the Qwen emotion-instructed caption; Qwen
     with RoBERTa.
   - Stage 2 settings: version 1 with CLIP patch tokens, or with a global CLIP vector; version 2 with the Qwen
     vector, or with CLIP patch tokens (does Stage 1 transfer?); version 3 with the Qwen vector.
3. Ablation, on the CPU after the GPU part: the September encoders clustered directly (row 1), our network on the
   default graph (row 6), on the other side's graph (row 10), and with prompted inputs on it (row 13).

**Compared against.** Our network's image side on the same 5,000 paintings (rows 7 to 12, already measured), and
the Qwen embeddings clustered directly (rows 2 and 3). The PercepT comparator per aspect is the best of its five
settings on that aspect (constitution C1, strongest real comparator).

**Outcomes.** Descriptive: no go or kill. A setting is "established" when the regression check passed, the fixes
pass their tests, and Stage 2 learned its topics (dev AUC at least 0.75). The table then shows which encoder and which
method part carry the score. We expect PercepT core near the Qwen default embedding on the image side. We do not
expect it near our best network on emotion and style. This is a guess, not a bar.

**Amended 2026-10-19 (A3), after run 2.** Algorithm 1 kept 1, 2, 3 or 99 topics by seed, because the largest norm
gap always sat at an end. PercepT core now keeps a fixed K = 40 topics (you chose it after seeing our
recommendation): it starts at 100 topics as before, and after training keeps the 40 topics that hold the most
training pairs (our default; on our data the short norms mark the little-used topics, so a norm cut keeps the wrong
ones). The five arms run again, 3 seeds each: a short build and about 20 GPU minutes. Rows 1, 6, 10 and 13 stay.

**Cost.** About 2.5 to 3 GPU hours and 1.5 CPU hours (estimate; the build measures it). The build runs tonight; the
run follows when the build ends. If the run starts by about 06:00, it ends before 11:00. You hear back with a report
and an updated ablation table.

**Check these.**
- The topic count (A3): 40 topics, the 40 with the most training pairs out of 100. Algorithm 1's count (γ 1.5, 2, 3
  and 5) is still reported beside it, but it no longer sets K. Run 2's γ = 2 results stay in its report.
- Eq. 2 as the code audit reads it, on l2-normalized inputs (the paper does not say).
- The September recipe (60 topics pruned to 40), whose saved snapshot the regression check re-scores (A1), is the
  base; the four fixes and the one-thread k-means start (A2) change it. So September's balance term stays.
- Version 3's RoBERTa vector goes up to 2048 dimensions by a fixed random orthogonal map (you confirmed it in the
  summary; the idea was ours).
- The comparator is the best PercepT setting per aspect, chosen on the same 5,000 paintings that chose our rows.

## 1. Context and goal

Asked: "Q1: 1b I guess? ... Q2: ... Maybe both 2a and 2b so in total 3 settings. Q3: 3a. minor: yes. So this is still
exploration, we get as many trails as possible to enlarge the ablation table and then pick. Also afte the percepT
experiment, do these in the table but still missying experiment." Then "K1b ... K2a yes. All minor: yes", and "yes"
to the closing summary (2026-10-18, about 23:30).

The stage 1 follow-up showed that our network on the right graph keeps most of the instruction's gain without an
instruction at test time. The user asked for an ablation table that changes one part at a time. Its PercepT rows (4,
5) and four of our rows (1, 6, 10, 13) are empty.

Goal: fill the empty rows of the ablation table on the frozen-sample protocol.
Question this step answers: where does PercepT core stand on the image side, and which encoder and which method
parts carry the score?

## 2. Scope

In: the regression check; PercepT core with the four fixes in five settings, 3 seeds; image-side scoring at k = 20
plus PercepT's own topic count; RoBERTa-GoEmotions features per caption; rows 1, 6, 10 and 13; the updated table.
Out: PercepT faithful (the paper's encoders and training details; later); the paper's split and metrics; held data;
stage 2 of our own model (decided after this); a caption side for PercepT (it has no caption-only path; the mean
column stays empty, D6).

## 3. Design

### 3.2 Arms and controls

PercepT core = the September Stage 1 and Stage 2 recipe (the snapshot configuration of 2026-09-24, deterministic)
with four changes (the four fixes) and a one-thread k-means start (A2): (1) Stage 1 rows are image-caption pairs,
not painting means; (2) the paper's Eq. 2 fusion
`h = (2·C·‖E‖ + E·‖C‖)/3`, `C = (I + T)/2`; (3) the topic count (amended, A3): start at 100 topics; after the joint training, assign
every fit pair to its argmax center over all 100 centers and keep the 40 centers with the most fit pairs (ties to the
lower center index); Algorithm 1 (cut at the midpoint of the largest gap in the sorted center norms when that gap
exceeds γ times the mean gap) is computed beside it for γ 1.5, 2, 3 and 5 and reported, not used; (4) Stage 2
targets are the multi-hot union of the topics of a painting's captions.

| Arm | Stage 1 input (I, T, E) | Stage 2 image input | Table row | Role |
|---|---|---|---|---|
| P1 | CLIP B/32 image, CLIP B/32 caption, RoBERTa-GoEmotions (PCA to 512) | CLIP patch tokens, attention pooling | 4a | comparator, closest to PercepT |
| P2 | as P1 (the same Stage 1 run) | CLIP global vector, 2-layer MLP | 4b | comparator; differs from P1 in Stage 2 only |
| P3 | Qwen default image, Qwen default caption, Qwen emotion-instructed caption | Qwen global vector, 2-layer MLP | 5a | comparator; same input as our network |
| P4 | as P3 (the same Stage 1 run) | CLIP patch tokens, attention pooling | 5b | transfer check; differs from P3 in Stage 2 only |
| P5 | Qwen default image, Qwen default caption, RoBERTa-GoEmotions (random orthogonal map to 2048) | Qwen global vector, 2-layer MLP | 5c | differs from P3 in E only; from P2 in I and T only |

Ablation rows of our network (the stage 1 head and scoring, unchanged; 3 seeds; default inputs unless stated):

| Row | Graph | Input | Differs from |
|---|---|---|---|
| 1 | none: k-means on CLIP image; caption side once with CLIP text, once with RoBERTa | — | row 2 by the encoder |
| 6 | default graph: mutual top-20 on the default caption embedding (emotion, genre) or default image embedding (style) | default | row 7 by the instruction in the graph |
| 10 | the other side's aspect graph alone (R0o) for emotion and style (genre exists) | default | row 7 by the graph's side |
| 13 | R0o for all three aspects | prompted | row 10 by the input |

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---|---|---|
| Frozen sample, 13,400 fit paintings (their captions as pairs) | development: training | 1, 2 |
| Frozen sample, 1,489 dev paintings | development: Stage 2 AUC, head early stopping | 1, 2 |
| Frozen sample, 5,000 evaluation paintings | development: scoring only (they also chose our rows) | 1, 2 |
| September split (61,402 train, 9,365 val+test paintings) | read to reproduce the recorded numbers; nothing tuned | 0 |
| ArtELingo test split, other held data | not touched | — |
| Seeds 0, 1, 2 | PercepT seed s sets Stage 1 and its Stage 2; head seeds as in stage 1 | 1, 2 |

### 3.4 Measurement

- **Image side (headline, D6):** the Stage 2 sigmoid output of the 5,000 evaluation paintings, l2-normalized,
  k-means at k = 20 with the stage 1 scoring (3 k-means seeds), AMI against emotion, style and genre. Per arm: the
  mean ± population sd over the 3 PercepT seeds. The column for an aspect is its own label.
- **Beside it (PercepT's own measures):** the topic count K (40 after A3) and how many kept topics hold at least
  one fit pair and at least 1% of them; AMI of the argmax topic of Stage 2 (image) and of Stage 1
  (painting = argmax of the mean topic weights of its pairs) at K; Stage 2 dev macro AUC; K for γ = 1.5, 3 and 5 on
  the same converged centers.
- **Rows 1, 6, 10, 13:** the stage 1 measures (both sides and their mean).
- **No significance tests.** Three seeds give a spread, not a test; differences below about 0.01 are read as level.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check (amended, A1) | the saved September node404 Stage 1 snapshot (`res/percept/percept_stage1_embedding_snapshot.npz`); the September code at `1b70d7f` and `pc_sept.py`; our scoring on the stored default embeddings | (i) the September scripts and every module they import are byte-identical between the record commits (`3b3ce61`, `4e4480b`) and `1b70d7f`; (ii) every function, class and constant `pc_sept.py` copies is text-identical to its source at `1b70d7f`; (iii) the snapshot re-scores to held-out emotion AMI 0.122469 and genre AMI 0.227419 (1e-6); (iv) k-means on the snapshot's pre-DEC latents (K 60, n_init 10, seed 42, one thread as A2) gives its saved initial labels exactly; row 2 = 0.088556, 0.170954, 0.393076 to 1e-6 | stop stage 1, trace the cause, note it for the user; stage 2 still runs |
| 1 | baseline | P1 to P5, seeds 0 to 2 | each arm: "established" checks of §6 | report the arm as not established, with the cause |
| 2 | ablation | rows 1, 6, 10, 13 | a rerun of stage 1's genre R0 seed 0 head matches its target AMI to 1e-6 (else within 0.002, with the cause stated) | stop the head rows, trace the cause |

### 3.7 Frozen vs rerun

Frozen: the frozen sample and its splits; stage 1's graphs, head code and all existing rows; the per-caption Qwen
stores; the CLIP features and patch cache. New: RoBERTa features per caption (once, to the shared cache), the default
graphs of row 6, and everything PercepT core trains.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|---|---|---|---|---|---|
| D1 | Which encoders | CLIP B/32 + RoBERTa-GoEmotions, and Qwen default | September's inputs and our network's inputs (one version only) | user, after seeing the recommendation | 2026-10-18 18:45 |
| D2 | E in the Qwen version | Qwen emotion-instructed caption (P3) | same instruction as our network, in training only (keep RoBERTa; no E) | user | 2026-10-18 23:10 |
| D3 | Stage 2 image input | P1, P2, P3: close to PercepT and a fair CLIP vs Qwen comparison | 2a or 2b alone | user | 2026-10-18 23:10 |
| D4 | CLIP Stage 2 on Qwen topics | add P4: "test whether what we learned in stage 1 is transferable" | three arms only | user | 2026-10-18 23:28 |
| D5 | Qwen with RoBERTa | add P5 | no isolation of E | user (the agent offered it) | 2026-10-18 23:28 |
| D6 | Scoring | image side only; K-native measures beside; mean column empty | both sides (PercepT's caption side sees the image); native K only | user | 2026-10-18 23:10 |
| D7 | Base recipe | the September snapshot configuration; only the four fixes change | the original 67-center or later H2H variants: not the recorded, reproducible numbers | user (minor i), recipe pick agent default | 2026-10-18 23:31 |
| D8 | Data, seeds, RoBERTa cache | frozen sample splits; seeds 0, 1, 2; per-caption RoBERTa features to `/data/SSD2/pre_extract/artelingo_full/` | — | user, after seeing the recommendation (minor ii to iv) | 2026-10-18 23:10 |
| D9 | γ | 2; no qualifying gap keeps all 100; K for 1.5, 3, 5 reported (superseded by A3: K is fixed; the γ counts are reported only) | the paper gives no value | agent default (minor v allowed it) | 2026-10-18 23:33 |
| D10 | Eq. 2 dimensions | P1, P2: PCA of RoBERTa to 512 on fit captions; P5: fixed random orthogonal map 768 to 2048 (seed 0) | PCA of Qwen to 768 (drops content) | user, after seeing the recommendation | 2026-10-18 23:28 |
| D11 | Eq. 2 reading and inputs | the audit's reading; I, T, E l2-normalized first | raw inputs (the paper is silent; September normalized) | agent default | 2026-10-18 23:33 |
| D12 | Stage 2 on a global vector | 2-layer MLP, hidden 1,024; other Stage 2 settings as September | — | user, after seeing the recommendation | 2026-10-18 23:28 |
| D13 | Rows 1, 6, 10, 13 | as §3.2; 3 seeds; after the GPU part | default graph on one fused content graph | user (rows, order); definitions agent default | 2026-10-18 23:28 |
| D14 | Stage types | 0 regression, 1 baseline, 2 ablation | exploration for stage 2: row 6 removes one part, so ablation is the first type that fits | agent default | 2026-10-18 23:33 |
| D15 | The comparator | per aspect, the arm with the highest mean image-side AMI; ties within 0.003 go to the lower arm number | a fixed arm (C1: strongest real comparator) | agent default | 2026-10-18 23:33 |

**Amendments:**

| ID | What changed | Why | Decided by | When |
|---|---|---|---|---|
| A1 | Stage 0 gate (§3.5): the checks that repeat exactly replace the September re-fit (4 decimals) and the Stage 2 macro AUC (0.8296) | the September k-means start sums its centers in a thread order that changes per run, so the re-fit gave three outcomes in eight local runs (load check `src/test/20261019_stage0_load_check/`); script 2 reaches its AUC only after the re-fit. The Stage 2 code is then covered by check (ii) only | user (option a of the r10 run brief; own pick) | 2026-10-19 08:20 |
| A3 | The topic count (§3.2 fix 3): K = 40, the 40 of 100 centers with the most fit pairs (argmax over all 100 after the joint training), replaces Algorithm 1; Algorithm 1's K for γ 1.5, 2, 3, 5 stays as a reported measure. The five arms rerun, 3 seeds; run 2's outputs are kept as `*_run2.*` before the rerun. Rows 1, 6, 10, 13 are not rerun | run 2: with 100 smoothly spread norms the largest gap sat at an end in 9 of 9 Stage 1 runs, so K was 1, 2, 3 or 99 by seed and only P2 was established (two of its seeds with 2 and 3 topics). Rejected: γ alone (the end gap is 4 to 12 times the mean gap, so γ 1.5 to 3 gives the same K); a cut inside the norm range and the September lowest-norm pick (in the three 99-topic runs, norm and pair share correlate positively, Spearman 0.40 to 0.44, so short norms mark little-used topics); a share threshold of 1% (our own rule; it gives about 40 anyway) | K = 40: user, after seeing the recommendation (option b of the r10 decide 9 chat); which 40 (most fit pairs, start at 100): agent default, shown to the user | 2026-10-19 10:45 |
| A2 | PercepT core's Stage 1 runs its k-means start (`pc_sept.initialize_cluster_centers`) under a one-thread limit in `pc_stage1.py`; `pc_sept.py` stays verbatim. Checks: three k-means starts on the snapshot latents under the limit give one center hash; the run repeats Stage 1 `clip_roberta` seed 0 once and compares its saved arrays bitwise (reported, not a gate) | with 8 threads, Stage 1 seeds would carry the same last-bit noise; one thread repeated the whole September fit in two of two runs. Cost: the start takes about 2.5 times longer | user, after seeing the recommendation | 2026-10-19 08:20 |

Places the amendments changed: A1 in the brief (plan item 1, "Check these"), §3.5, §5 and §7; A2 in the brief,
§3.2 and "Check these"; A3 in the brief ("Amended", "Check these"), §3.2 fix 3, §3.4, D9, §6, §8 and §9. §3.2, §5, §7 and "Check these", with §6 and §8 (the 21 fix tests), were brought in
line after the final review of build 5 (2026-10-19), text only. After build 6, §6 and §8 give the fix-test count after
A3 (30: the 21, three top-40 cases and six input checks of the top-40 rule), text only.

## 5. Assumptions and open questions

- Assumption: per-caption Qwen default and emotion-instructed vectors cover all 99,999 frozen captions. Checked: the
  stores exist and the stage 1 code read their painting means; the per-caption join not checked.
- Assumption: CLIP B/32 features and the patch cache (61,402 train paintings) cover the frozen sample. Checked: the
  sample comes from the CLIP-backed train loader; the name join not checked.
- Assumption: the snapshot configuration can be recovered from the September branch. Checked: recovered as
  `pc_sept.py`, text-identical to its source (stage 0 check (ii)); the re-fit does not repeat exactly on this
  machine (load check of 2026-10-19), hence A1.
- Assumption: the RoBERTa-GoEmotions and CLIP B/32 weights are in the shared HF cache. Checked: both listed.
- No open questions.

## 6. Success criteria

**Descriptive mode.** No decision rides on a bar: the user picks from the filled table.
An arm is **established** when: stage 0 passed (the amended gate, A1); unit tests pass for the four fixes (30 tests after A3, 21 before; none skipped: pair count equals the fit
captions; Eq. 2 on a hand-worked example; Algorithm 1 on norms with a known gap; the multi-hot target equals the OR
of the caption topics; after A3 also the top-40 pick on known pair counts, ties included); and its Stage 2 dev macro AUC is at least 0.75 on every seed (September 0.829 with
threshold targets; the paper 0.94).
Reading: an established arm near or above our best network on the image side moves the stage 2 design toward that
arm's parts; P4 vs P3 says whether topics transfer across Stage 2 readers; P5 vs P3 says which emotion source helps.
**Licensed claim (stage 1):** "PercepT core, a fair comparator under the frozen-sample protocol (four fixes, our
features, no tuning on either side), scores X on the image side." After A3 the claim names the count: "with K = 40
topics, the most-used 40 of 100, in place of PercepT's Algorithm 1". Not PercepT as published, not the paper's numbers.
**Licensed claim (stage 2):** each row's effect as a difference of 3-seed means with both sds, on development data:
an attribution as strong as the exploration it explains, a direction, never a result.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | D15, the best arm per aspect |
| C6 sample IDs | pass | joins by extraction sample IDs and painting names, as the stage 1 code |
| C7, C8 rule and disclosure | pass | no bar; §8 discloses selection |
| C12 regression first | exception (A1) | stage 0, amended; see below |
| C13 re-derivation | pass | the comparator's image-side AMI re-derived by separate code |

Exceptions: C12 (A1, the user's decision of 2026-10-19 08:20). Stage 0 reproduces the saved snapshot's numbers by
re-scoring, shows the reused code byte- and text-identical, and repeats the k-means labels. The September re-fit and
the Stage 2 AUC are not reproduced, because the September k-means start does not repeat (load check: 0.1238 / 0.2617
or 0.1287 / 0.2804 against the recorded 0.1225 / 0.2274). Rejected: the 4-decimal re-fit. "Stage 0 passed" in the
report means these checks.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| A port error in the four fixes | unit tests (30 after A3, none skipped); stage 0 checks (i) and (ii) | fix before stage 1 |
| γ = 2 gives K = 100 (no cut) or a very small K | the K table for γ 1.5 to 5 | happened in run 2 (K 1 to 3 or 99); A3 fixes K = 40 |
| K = 40 by pair share is our choice, not the paper's; another K could move the comparator | the kept topics' pair shares; the γ counts beside it | report K with the claim; no K sweep in this spec |
| Selection: stage 1 chose our rows on the 5,000 paintings, and D15 chooses PercepT's arm on them too | stated in the report | carry both picks to fresh data in stage 2 of our model |
| Qwen's content and emotion vectors come from one model, so Eq. 2 adds little in P3 | P3 vs P5, Stage 1 topic AMI | read as a finding |
| Captions per painting differ, so Stage 1 weighs some paintings more | pair counts per painting | note in the report |

## 9. Deliverables and review

- At the end: round folder `src/test/20261018_percept_core/` (module prefix `pc_`); RoBERTa cache; a results sheet;
  the ablation table updated in the user-read report; an auto report with a results brief, a user-read report.
- Reviews: tickets per `matt-chain.md`; the four fixes and the scoring are verdict code (unit tests and ticket
  reviews); the final whole-branch review; the comparator's numbers re-derived independently (C13).
- Stop points: none between approval and the reports (build and run unattended). Time-box: 2026-10-20.
- **A3 rerun:** a short build (the count rule, its unit test, the `*_run2.*` copies; verdict code), then the GPU
  queue only: stage 0, 9 Stage 1 and 15 Stage 2 runs, the repeat check (run 2 took 18 minutes), then scoring, the
  C13 re-derivation and both reports.
- **Budget:** local GPU under the lock, about 2.5 to 3 hours (RoBERTa extraction minutes; 9 Stage 1 runs; 15 Stage 2
  runs; stage 0 about 15 minutes, about 2 after A1; the build's timing gave about 25 GPU minutes for the whole GPU part, not re-measured); CPU about 1.5 hours (21 head runs on three processes, scoring). Estimates, not
  checked: the build's smoke times one run of each kind, and the run chat recomputes the end time. An end after 11:00
  is weighed as `shared-resources.md` says: the GPU part first, the CPU rows may move to the next evening.
- GPU check at 23:31: the driver answers; module and library both 595.99.02; no process and no lock on the GPU.
- Length: the body (about 170 lines) is over the target of about 140, because five arms and 15 grill decisions
  fill §3.2 and §4; each fact stays in one place.
