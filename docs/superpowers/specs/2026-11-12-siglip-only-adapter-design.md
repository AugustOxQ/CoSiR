# r29: the SigLIP-only aspect adapter: design

> Date: 2026-11-12 22:10 · Status: draft · Kind: experiment
> Builds on: the [r29 decide handoff](../handoffs/2026-11-12-r29-decide-handoff.md); r28's numbers sheet
> (`src/test/20261112_siglip_main/paper_cells.md`); the grill in the r29 decide chat (2026-11-12, one key question, a
> scenario check, five minor points and a closing summary; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) (amended 2026-11-10) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: the r14 spec ([2026-11-02-backbones-sweep-design.md](2026-11-02-backbones-sweep-design.md)) §3.4: the WIN,
> LEVEL, BELOW outcomes, their paired 95% bootstrap intervals and the seed ranges for LEVEL; the r23 spec
> ([2026-11-06-text-claim-design.md](2026-11-06-text-claim-design.md)): the text-direction protocol and its interval
> unit (paintings); r28 (`src/test/20261112_siglip_main/`): the read-out menus and picks on validation per cell, the
> ablation arms, the MMArt and photo protocols, the cells sheet · Reviewer: not yet

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** CoSiR on SigLIP 1 still reads texts through Qwen3 at test time, so the paper's PercepT sentence and
its light-model claim are weak. This loop trains the aspect adapter (the image tower and the caption tower) on SigLIP 1
alone. The graphs, the content correction and the alignment targets stay on Qwen3. The caption tower reads the
captions' own SigLIP 1 text embedding. If the new model holds on validation, it becomes the paper's main model, and
SigLIP 2 and CLIP L/14 get the same design.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (regression check): the new code reproduces r14's SigLIP 1 seed 0 run and r28's scores exactly.
2. Stage 1a (sweep, validation, seed 0): the plain caption against the "Represent the caption: " prefix.
3. Stage 1b (incremental, validation, seeds 0 to 4): the pick against the current SigLIP 1 model. A pass rule decides:
   the main model, or an extra row.
4. Stage 1c (claim test, ArtELingo test): the paper cells of Tables 2, 3 and 6. It runs on a pass and on a fail.
5. Stage 2 (only on a pass): MMArt (Table 4), the ablation (Table 5), the photo model (Table 7), and SigLIP 2 and
   CLIP L/14 with their own text encoders (Table 6). The text-time measure (Table 8) runs in every case.

**Compared against.** On validation: the current SigLIP 1 model (its caption tower reads Qwen3). On test: plain
SigLIP 1, with its own text encoder (paper claim 1), and the instructed Qwen3 (paper claim 2, the strongest comparator).

**Outcomes.**
- Pass rule (stage 1b), five-seed means on validation. The image group (AMI and mAP, three aspects) must not fall
  below the current 0.3863 by more than its seed range, 0.0030. The text group (three directions × three aspects, plain
  cosine) must not fall below the current 0.2594 by more than its seed range, 0.0020. Pass: the new main model. Fail:
  an extra row; stage 2 does not run.
- Stage 1b ladder: level 1 is the pass; level 2 adds a text group at or above 0.2594; level 3 puts both groups above
  the current model's best seed.
- Stage 1c and each stage 2 claim test: level 1 is a WIN over the plain embedding in most cells (12 of 15 on
  ArtELingo); level 2 in all cells; level 3 adds a WIN or LEVEL over the instructed Qwen3 in most cells.
- We expect the image group to hold (0.8) and the text group to fall outside its range (a pass at 0.4). On test, level 2
  at 0.6 and level 3 at 0.3: the current model already trails the instructed Qwen3 in the three text emotion cells.

**Cost.** Stage 1 about 1.2 GPU-hours, stage 2 about 5.5 (our estimates), on the local GPU. Build tonight and Friday;
stage 1 on Friday; stage 2 from Friday 19:00, or on DAS6 if a node is reserved. Report by Saturday 11-14; the paper is
due 11-16.

**Check these.**
1. The pass rule uses one mean per group (your pick, after seeing the draft rule). A text emotion loss can hide behind
   style and genre gains.
2. Plain cosine in the pass rule for both models (agent default). The paper cells keep their read-outs picked on
   validation.
3. SigLIP 2 and CLIP L/14 take the caption form that wins on SigLIP 1, with no prefix check of their own.
4. The Affection captions go through SigLIP 1 at its 64-token limit. Longer texts are cut and counted.
5. Stage types (agent default): stage 1c and stage 2 are claim tests, so this rule gets an independent review before
   any code.

**Morning checks.**
| What | Look for |
|---|---|
| `src/test/20261112_siglip_only/records/pass_rule.json` | the two group means beside 0.3863 and 0.2594; pass or fail |
| `src/test/20261112_siglip_only/records/prefix_pick.json` | which caption form won, and by how much |
| `src/test/20261112_siglip_only/paper_cells.md` | every new cell beside the current one, with its outcome |
| the auto report (`docs/reports/auto/percept/`) | its results brief: the level per stage |
| `src/test/20261112_siglip_only/decisions.md` | choices the build and run chats made |

## 1. Context and goal

Asked: "the instruction part and buddy part still from qwen, just the later student part becomes all siglip" (paper
chat, 2026-11-12); "if it works we can possibly try more models (Siglip2 own, CLIP L own, and more), and then this
becomes the new model to replace qwen3+siglip. But only if it work, if not possibly as another abolation or something"
(the grill).

The paper's main model is c* on SigLIP 1 (r14's models, five seeds; r28 filled Tables 2 to 8 from them). Its image
tower reads SigLIP 1, and its caption tower reads the Qwen3 plain caption embedding, so a text needs a Qwen3 pass at
test time (3.6 ms per caption, r28). On test, its text cells are 0.264 / 0.179 / 0.489 (text to image, mAP emotion /
style / genre) against the instructed Qwen3's 0.268 / 0.157 / 0.443 and plain SigLIP 1's 0.211 / 0.152 / 0.419.

Goal: a model with no Qwen3 pass at test time, for images and texts.
Question this step answers: does the aspect adapter keep its cells when its caption tower reads SigLIP 1, and if so,
where does it stand against plain SigLIP 1 and the instructed Qwen3 on test?

## 2. Scope

In: stages 0 to 2 above; the numbers sheet for the paper chat (`paper_cells.md`, as r28's).
Out: graphs or content correction on SigLIP (the wider version in the paper plan's row; the user chose the narrower
idea); CLIP B/16, B/32 and other encoders ("and more": parked in the handoff for after 11-16); other data sets
(SemArt); a settings sweep, including r28's flagged learning-rate row; PercepT reruns; edits to the paper's .tex (the
paper chat).

## 3. Design

### 3.1 Hypothesis and prior

The caption tower learns its aspect spaces from the graphs, not from its input, so a weaker text input may cost little.
Against this, SigLIP 1's text encoder carries less emotion than Qwen3's: plain text-to-text emotion mAP is 0.206 for
SigLIP 1 and 0.253 for Qwen3 (r28, test). The two towers train together (four directions of the graph-positive loss),
so the image cells can move too. Priors: image group holds 0.8; text group within range 0.4 (so a pass 0.4); the prefix
wins 0.4 (SigLIP 1 was not trained to follow instructions). A surprise: the text group above the current model's best
seed.

### 3.2 Arms and controls

| ID | Arm | Differs from the current SigLIP 1 model by | Role |
|---|---|---|---|
| K0 | current SigLIP 1 model: r14's c* on SigLIP 1, seeds 0 to 4, frozen | nothing | comparator (validation); stage 0 target |
| K1 | SigLIP-only, plain caption: caption tower reads the painting mean of the captions' SigLIP 1 text embeddings | the caption tower's input | candidate |
| K2 | SigLIP-only, prefixed caption: as K1, each caption encoded as "Represent the caption: " + caption | input, prefix | candidate (stage 1a) |
| K* | the stage 1a pick (K1 or K2), seeds 0 to 4 | input | the candidate of stages 1b, 1c, 2 |
| P1 | plain SigLIP 1 (own text encoder; r17 image cells, r23 text side row) | no training | comparator, claim 1 |
| Q1 | instructed Qwen3 (r17 image cells, r28 text rows) | the MLLM itself, three passes | comparator, claim 2 |
| E2, E3 | SigLIP 2 and CLIP L/14 with their own text encoders, caption form of K*, seeds 0 to 4 | encoder | candidates (stage 2) |
| A1-A4 | K* with one part removed: instructed graphs, content correction, graph-positive loss, relational term (r28's arms) | one part | ablation (stage 2) |
| F1 | the photo model fine-tuned from K*'s emotion checkpoints, caption mean from SigLIP 1 | the start point and the caption input | candidate (stage 2) |

Everything else is c*'s: settings, graphs (k = 40), losses, early stopping, output width 2048, seeds. Only the caption
input width changes (2048 to 1,152; 768 for CLIP L/14).

### 3.3 Data roles

| Data or seeds | Role | Stage |
|---|---|---|
| ArtELingo validation (6,152 paintings, 30,872 captions) | development: the prefix pick, the pass rule, read-out picks, seed ranges | 0, 1a, 1b |
| ArtELingo test (12,281 paintings, 61,744 captions) | test; read freely, shapes nothing | 1c, A1-A4, E2, E3 |
| MMArt test rows (r15's 16,384 paintings) and its validation picks (r28) | test; development for read-out picks | 2 |
| Affection fit, validation, test; EmoSet validation, test (r14, r28) | as r14 and r28 | 2 |
| Seeds 0 to 4 (seed 0 for stage 1a) | as every round since r10 | all |

### 3.4 Measurement

- **Stage 1a pick:** the text group (nine cells: emotion, style, genre × text to image, image to text, text to text)
  on validation, seed 0, read-out r0 (plain cosine). Higher wins; an exact tie goes to K1.
- **Pass rule (stage 1b):** five-seed means on validation, read-out r0 for text; image AMI and plain-cosine mAP as
  r28's encoder pick (`valpick`: MiniBatchKMeans at 9 / 27 / 10 clusters, n_init 3, states 0 to 2). Image group: the
  mean of the six cells. Text group: the mean of the nine cells. K0's values are r28's records, not recomputed:
  image 0.38629 (per seed 0.38489 to 0.38787, range 0.00298; `records/valpick.json`), text 0.25935 (per seed 0.25862 to
  0.26063, range 0.00201; `records/text.json`, `ours_siglip1/picks/*/*/per_seed/r0`). Stage 0 replays both.
- **Test cells (stage 1c, stage 2):** as r28 (`paper_cells.md`): AMI at the class counts; image mAP and the text
  directions with the read-out picked per cell on validation from r28's menus. Outcomes and intervals as r14 §3.4:
  WIN when the paired 95% bootstrap lower bound of K* minus the comparator is above 0; LEVEL when not WIN and the lower
  bound is above −r, r = K*'s own seed range on validation for that cell; BELOW otherwise. Text intervals use paintings
  as the unit (r23). Per-cell intervals, no multiplicity correction.
- **Cells per claim test:** stage 1c 15 (AMI, image to image, text to image, image to text, text to text × 3 aspects);
  MMArt 6 (AMI, image mAP); photo model 4 (Affection AMI and mAP, EmoSet AMI and mAP); E2 and E3 6 each (Table 6:
  AMI, image mAP). "Most cells" for level 1 and level 3: 12 of 15, 5 of 6, 3 of 4.
- **Ablation:** each removal's change from K* (mean over seeds) in AMI and plain-cosine mAP, with its paired interval,
  as r28's ablation table.
- **Table 8:** ms per caption for the SigLIP 1 text pass and for the new caption tower, as r28's `time_text` (RTX 3090,
  batch 32, 2,000 captions, three repeats).
- **C13:** Codex re-derives, from this rule and the saved outputs, the stage 1a pick, the two group means and the
  pass, and every outcome of stages 1c and 2, within 1e-6, before any verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new trainer with the Qwen3 caption input; the new scorers on K0 | (i) r14 SigLIP 1 seed 0 retrained with the Qwen3 caption input: weights equal r14's seed 0 checkpoints bit for bit (as r10's V4a), and validation cells equal r28's `valpick` seed 0 within 1e-6; (ii) K0's text validation cells equal r28's `text.json` picks within float16 resolution (r28's tolerance); (iii) the SigLIP 1 caption join by sample ID and painting covers every caption (C6); (iv) the prefixed store passes the extractor's norm and finite checks, 0 truncated | stop, trace, fix code bugs only (at most three tries per gate); the rest waits |
| 1a | sweep | K1, K2, seed 0, validation | both trained and scored | that arm out; if both fail, stage 1 is inconclusive |
| 1b | incremental | K* seeds 1 to 4 (seed 0 from 1a), validation | five seeds trained; picks committed before any test number | inconclusive |
| 1c | claim test | K*, ArtELingo test | read-out picks and seed ranges committed first | that cell inconclusive |
| 2 | claim tests, ablation, measure | only after a pass in 1b (Table 8 always) | as 1c; ablation rows trained on five seeds | that piece inconclusive; the others go on |

Stage 2's order: E2 and E3, then MMArt, then the photo model, then the ablation (cheap and central first).

### 3.6 Diagnostics

Measured after the levels are written; they decide nothing. (a) Per-cell validation changes K* minus K0, to show which
aspect and direction moves. (b) The best epoch of each head against K0's. (c) Text to text on validation for K* and for
plain SigLIP 1 per aspect: how much the caption tower adds over its own input.

### 3.7 Frozen vs rerun

Frozen: K0's checkpoints and records (r14, r28); Qwen3's graphs, instructed embeddings and content correction (r10's
store); c*'s settings; r28's read-out menus, the comparator rows (r17, r23, r28) and r28's MMArt and photo picks for
the comparators. Rerun: K1, K2, K*, E2, E3, A1-A4, F1; their read-out picks and seed ranges on validation.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|---|---|---|---|---|---|
| D1 | Direction | a SigLIP-only model as a medium loop with its own decide chat | the PercepT sentence is "a bit problematic" while texts need Qwen3 | user | 2026-11-12 21:10 |
| D2 | Idea | Qwen3 keeps the instructions, graphs, content correction, alignment targets; the adapter reads SigLIP 1 only | "the later student part becomes all siglip" (not the paper plan's wider version) | user | 2026-11-12 21:10 |
| D3 | Place in the paper | decided on validation: a pass makes it the main model and starts stage 2 (all tables, more encoders); a fail gives an extra row or ablation | a row now (option a) or the main model whatever the numbers (option b) | user (own pick, option c with the encoder extension) | 2026-11-12 21:50 |
| D4 | Pass rule form | one mean per group, image and text; a loss in one aspect can be offset | a floor per aspect | user, after seeing the recommendation (scenario check) | 2026-11-12 21:58 |
| D5 | Caption form | run plain and the "Represent the caption: " prefix on seed 0; higher text group wins (an exact tie: plain) | pick one untested | user, after seeing the recommendation (the tie: agent default) | 2026-11-12 21:58 |
| D6 | Settings | c*'s settings unchanged | r28's check: no row clearly ahead (+0.0007 inside 0.0030) | user, after seeing the recommendation | 2026-11-12 21:58 |
| D7 | Device and timing | local GPU; stage 2 after 19:00 or on DAS6 | weekday rule for runs over 1 hour | user, after seeing the recommendation | 2026-11-12 21:58 |
| D8 | Table 8 | measure the SigLIP 1 text pass and the new caption tower | leave the Qwen3 text time | user, after seeing the recommendation | 2026-11-12 21:58 |
| D9 | On a fail | cells go in the numbers sheet; the paper chat places them | a decision here | user, after seeing the recommendation | 2026-11-12 21:58 |
| D10 | Stage types | 1a sweep, 1b incremental, 1c and stage 2 claim tests, A1-A4 ablation | 1c as a confirmation (no rule review) | agent default | 2026-11-12 22:10 |
| D11 | Read-out in the pass rule and the prefix pick | r0 (plain cosine) for text, plain-cosine mAP for images, both models | the picked read-out (a pick inside a pick) | agent default | 2026-11-12 22:10 |
| D12 | Claim ladders | as §6 ("most cells": 12 of 15, 5 of 6, 3 of 4) | a single bar | agent default | 2026-11-12 22:10 |
| D13 | E2, E3 caption form | K*'s form, no own check | a prefix check per encoder (more runs before 11-16) | agent default | 2026-11-12 22:10 |
| D14 | Texts over the encoder limit | cut at the limit, count, flag (constitution row 5) | drop them | agent default | 2026-11-12 22:10 |
| D15 | Stores | new folders only: `artelingo_full/siglip_so400m_capprefix/`, `affection_local/siglip_so400m_txt/` (and E2, E3 prefixed stores if K2 wins) | write into the existing image store folders | agent default | 2026-11-12 22:10 |
| D16 | F1's caption mean | the mean of SigLIP 1 text embeddings of each photo's default explanations, L2 normalised, in K*'s form | Qwen3's mean (not SigLIP-only) | agent default | 2026-11-12 22:10 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the SigLIP 1 caption store joins every ArtELingo caption by sample ID and painting. Checked: the Codex
  facts read compared both index files (308,723 IDs, same pairs); stage 0 (iii) checks it in code.
- Assumption: no ArtELingo caption exceeds 64 tokens with the prefix. Checked: tokenizer count, 0 (facts read). The
  SigLIP 2 and CLIP L/14 stores record 0 truncated.
- Assumption: the Affection explanations fit in 64 tokens. Not checked; D14 covers it.
- Assumption: r28's ablation on K0 ends before stage 2 needs the GPU. Not checked (r28 runs tonight).
- Open questions: none.

## 6. Success criteria (decision rule)

**Decision mode.** This section, with §3.4 and §3.5, is the decision rule (C7). Prior: §3.1.

| ID | What is measured | Compared against | Level 1 | Level 2 | Level 3 | Data |
|---|---|---|---|---|---|---|
| SC1 | stage 1a: text group, seed 0 | K1 vs K2 | pick: the higher | – | – | validation |
| SC2 | stage 1b: image group, text group (five-seed means) | K0 | image ≥ 0.38629 − 0.00298 and text ≥ 0.25935 − 0.00201 (the pass) | L1 and text ≥ 0.25935 | image > 0.38787 and text > 0.26063 | validation |
| SC3 | stage 1c: 15 cells | P1; Q1 | WIN over P1 in ≥ 12 | WIN over P1 in 15 | L2 and WIN or LEVEL over Q1 in ≥ 12 | ArtELingo test |
| SC4 | E2, E3: 6 cells each (each its own verdict; the current model on that encoder shown beside, descriptive) | that encoder's plain embedding; Q1 | WIN in ≥ 5 | WIN in 6 | L2 and WIN or LEVEL over Q1 in ≥ 5 | ArtELingo test |
| SC5 | MMArt: 6 cells | P1; Q1 | WIN in ≥ 5 | WIN in 6 | L2 and WIN or LEVEL over Q1 in ≥ 5 | MMArt test |
| SC6 | photo model F1: 4 cells | P1; Q1 | WIN in ≥ 3 | WIN in 4 | L2 and WIN or LEVEL over Q1 in ≥ 3 | Affection, EmoSet test |
| SC7 | ablation A1-A4 | K* | each removal's change with its interval (attribution) | – | – | ArtELingo test |
| SC8 | Table 8 | Qwen3 text pass, 3.616 ms | measured (descriptive) | – | – | 2,000 captions |

- SC2 level 1: "On validation, the SigLIP-only model is level with the current SigLIP 1 model in images and texts." It
  sets the pick: K* becomes the paper's main model and stage 2 runs. Below level 1, K* is an extra row.
- SC2 level 2: "…and its texts are no worse." Level 3: "On validation, it is above the current model's best seed in
  both groups."
- SC3 to SC6 level 1: "With no Qwen3 pass at test time, training adds over the plain embedding in most cells." Level
  2: "in every cell." Level 3: "…and it is level with or above the instructed Qwen3 in most cells." Each Q1 cell is
  also its own claim: WIN "above", LEVEL "level with", BELOW none.
- Inconclusive: a gate that fails after three tries, or a missing pick record, for the cells it touches.
- Licensed claim: ArtELingo (and MMArt, Affection, EmoSet for SC5, SC6) under c*'s settings and these encoders only;
  development results (SC1, SC2) are never quoted as test results; the sentence never goes beyond the level reached.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | Q1 on test; K0 on validation |
| C6 sample IDs | pass | joins by sample ID and painting; stage 0 (iii) |
| C7 rule before code | pass | §6 reviewed by a fresh Opus `xhigh` worker and committed before any code |
| C8 prior and disclosure | pass | §3.1; disclosure in §8 |
| C12 regression check | pass | stage 0 (i), (ii) |
| C13 re-derivation | pass | §3.4 |
| Decisions never alone | pass | rows 1 to 9 apply as written; "and more" encoders are parked (row 6) |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| The prefix is not applied the same way in training and at test | stage 0 store hash; the text scorer reads the same store | gate failure, fix |
| A wrong caption-to-painting join (C6) | stage 0 (iii); per-painting caption counts equal Qwen3's | stop |
| A width mix-up loads a 2048 caption checkpoint as 1,152 (or the reverse) | shape asserts; checkpoint metadata holds the input width | stop |
| The pass rule hides a text emotion loss (D4) | diagnostic (a); the report shows the per-cell changes | report it; the paper chat decides the wording |
| Stage 2 does not finish before 11-16 | queue estimates at the start of stage 2 | run order of §3.5; unfinished pieces reported as not run |

Selection disclosure: on ArtELingo validation, c*'s settings came from r10's sweeps on Qwen3; SigLIP 1 was picked over
three other encoders by 0.0007 (r28, a close call); this loop adds two caption forms (seed 0) and one pass decision.
Test numbers shape nothing; the 80% and LEVEL bars are fixed here.

## 9. Deliverables and review

The body is over the 150-line target: five stages and one rule in one file.


- At the end: code in `src/test/20261112_siglip_only/` (module prefix `r29_`), outputs in
  `/data/SSD2/cosir_r29/20261112_siglip_only/`, records and `paper_cells.md` in the round folder, an auto report in
  `docs/reports/auto/percept/` with a user-read report beside it.
- Reviews: this rule by a fresh Opus `xhigh` worker before approval; verdict code reviewed per ticket (`matt-chain.md`);
  the final whole-branch review; C13 by Codex.
- Stop points: none between approval and the reports. Time-box: stage 1 done by Friday 11-13 evening; stage 2 by
  Saturday 11-14 noon.
- Budget: stage 0 and 1 about 1.2 GPU-hours (12 training runs of about 6 minutes, one extraction of 6 minutes,
  scoring); stage 2 about 5.5 GPU-hours (E2, E3 10 runs, A1-A4 20 runs of about 7.5 minutes, F1 five runs (time not
  checked), one Affection extraction, scoring); CPU scoring under `res-queue cpu`. Local GPU; jobs over 1 hour not on
  weekdays 10:00 to 19:00.
