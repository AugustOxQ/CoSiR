# r29: the SigLIP-only aspect adapter: design

> Date: 2026-11-12 21:40 (revised 22:05 after the rule review and the re-review) · Status: approved 2026-11-12 22:42 · Kind: experiment
> Builds on: the [r29 decide handoff](../handoffs/2026-11-12-r29-decide-handoff.md); r28's numbers sheet
> (`src/test/20261112_siglip_main/paper_cells.md`); the grill in the r29 decide chat (2026-11-12, one key question, a
> scenario check, five minor points and a closing summary; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) (amended 2026-11-10) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: the r14 spec ([2026-11-02-backbones-sweep-design.md](2026-11-02-backbones-sweep-design.md)) §3.4: the WIN,
> LEVEL, BELOW outcomes and their paired 95% bootstrap intervals; the r23 spec
> ([2026-11-06-text-claim-design.md](2026-11-06-text-claim-design.md)): the text-direction protocol and its interval
> unit (paintings); r28 (`src/test/20261112_siglip_main/`): the read-out menus, the ablation arms, the MMArt and photo
> protocols, the cells sheet; per cell type, the units and sources of §3.4's table
> Reviewer: rule review by a fresh Opus `xhigh` worker, 21:51, pass after fixes
> (`src/test/20261112_siglip_only/r29_rule_review.md`); fixes applied 22:00; re-review by a Sonnet `high` worker,
> 22:04, pass after fixes (`r29_rule_rereview.md`); its fixes applied 22:05

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** CoSiR on SigLIP 1 still reads texts through Qwen3 at test time, so the paper's PercepT sentence and
its light-model claim are weak. This loop trains the aspect adapter (the image tower and the caption tower) on SigLIP 1
alone. The graphs, the content correction and the alignment targets stay on Qwen3. The caption tower reads the
captions' own SigLIP 1 text embedding. If the new model holds on validation, it becomes the paper's main model, and
SigLIP 2 and CLIP L/14 get the same design.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (regression check): with the old caption input, the new code trains exactly as r14's code on this machine
   and reproduces r14's and r28's scores. Each new scoring path gets this check before its first number.
2. Stage 1a (sweep, validation, seed 0): the plain caption against the "Represent the caption: " prefix.
3. Stage 1b (incremental, validation, seeds 0 to 4): the pick against the current SigLIP 1 model. A pass rule decides:
   the main model, or an extra row.
4. Stage 1c (claim test, ArtELingo test): the paper cells of Tables 2, 3 and 6. It runs on a pass and on a fail.
5. Stage 2 (only on a pass): new models trained and their read-outs picked on validation; then MMArt (Table 4), the
   ablation (Table 5), the photo model (Table 7), SigLIP 2 and CLIP L/14 with own text encoders (Table 6). The
   text-time measure (Table 8) runs in every case.

**Compared against.** On validation: the current SigLIP 1 model (its caption tower reads Qwen3). On test: plain
SigLIP 1, with its own text encoder (paper claim 1), and the instructed Qwen3 (paper claim 2, the strongest comparator).

**Outcomes.**
- Pass rule (stage 1b), five-seed means on validation: the image group (AMI and mAP, three aspects) stays within its
  seed range (0.0030) below the current 0.3863; the text group (three directions × three aspects, plain cosine) within
  its range (0.0020) below the current 0.2594. Pass: the new main model. Fail: an extra row; no stage 2.
- Stage 1b ladder: level 1 is the pass; level 2 adds a text group at or above the current model's; level 3 puts both
  groups above the current model's best seed.
- Stage 1c and each stage 2 claim test: level 1 is a WIN over the plain embedding in most cells (12 of 15 on
  ArtELingo); level 2 in all cells; level 3 adds a WIN or LEVEL over the instructed Qwen3 in most cells.
- We expect a pass at 0.4 (the image group holds at 0.8, the text group is the risk). On test, level 2 at 0.6 and level
  3 at 0.3: the current model already trails the instructed Qwen3 in the three text emotion cells.

**Cost.** Stage 0 and 1 about 1.5 GPU-hours, stage 2 about 6, on the local GPU. Build tonight and Friday; stage 1 on
Friday (each job under 1 hour); stage 2 from Friday 19:00, or on DAS6. Report by Saturday 11-14; paper due 11-16.

**Check these.**
1. The pass rule uses one mean per group (your pick, after seeing the draft rule). A text emotion loss can hide behind
   style and genre gains.
2. c*'s settings stay, although a learning rate of 0.0005 led on seed 0 by 0.0007, inside the 0.0030 seed range (your
   pick, after seeing the recommendation).
3. The new model trains on the local GPU; the current model was trained on DAS6. In r14 the machine alone moved single
   cells by up to 0.0024. The pass rule does not correct this; the report shows the change (your yes, after seeing
   the recommendation).
4. Plain cosine in the pass rule; SigLIP 2 and CLIP L/14 take the caption form that wins on SigLIP 1; texts over the
   token limit are cut and counted (agent defaults).
5. Stage types (agent default): 1c and the stage 2 tests are claim tests, so the rule had an independent review and a
   re-review (both pass after fixes, fixes applied).

**Morning checks** (in `src/test/20261112_siglip_only/`, and the auto report).
| What | Look for |
|---|---|
| `records/gates.json` | every stage 0 gate passed; a parked piece is named |
| `records/pass_rule.json`, `records/prefix_pick.json` | the two group means beside 0.3863 and 0.2594, pass or fail; the caption form that won |
| `paper_cells.md` | every new cell beside the current one, with its outcome |
| the auto report (`docs/reports/auto/percept/`) | its results brief: the level per stage |
| `decisions.md` | choices the build and run chats made |

## 1. Context and goal

Asked: "the instruction part and buddy part still from qwen, just the later student part becomes all siglip" (paper
chat, 2026-11-12); "if it works we can possibly try more models (Siglip2 own, CLIP L own, and more), and then this
becomes the new model to replace qwen3+siglip. But only if it work, if not possibly as another abolation or something"
(the grill).

The paper's main model is c* on SigLIP 1 (r14's models, five seeds, trained on DAS6 node402; r28 filled Tables 2 to 8
from them). Its image tower reads SigLIP 1, and its caption tower reads the Qwen3 plain caption embedding, so a text
needs a Qwen3 pass at test time (3.6 ms per caption, r28). On test, its text cells are 0.264 / 0.179 / 0.489 (text to
image, mAP emotion / style / genre) against the instructed Qwen3's 0.268 / 0.157 / 0.443 and plain SigLIP 1's 0.211 /
0.152 / 0.419.

Goal: a model with no Qwen3 pass at test time, for images and texts.
Question this step answers: does the aspect adapter keep its cells when its caption tower reads SigLIP 1, and if so,
where does it stand against plain SigLIP 1 and the instructed Qwen3 on test?

## 2. Scope

In: stages 0 to 2 above; the numbers sheet for the paper chat (`paper_cells.md`, as r28's).
Out: graphs or content correction on SigLIP (the wider version in the paper plan's row; the user chose the narrower
idea); CLIP B/16, B/32 and other encoders ("and more": parked in the handoff for after 11-16); other data sets
(SemArt); a settings sweep, including r28's flagged learning-rate row; a local retrain of K0 (D25); PercepT reruns;
edits to the paper's .tex (the paper chat).

## 3. Design

### 3.1 Hypothesis and prior

The caption tower learns its aspect spaces from the graphs, not from its input, so a weaker text input may cost little.
Against this, SigLIP 1's text encoder carries less emotion than Qwen3's: plain text-to-text emotion mAP is 0.206 for
SigLIP 1 and 0.253 for Qwen3 (r28, test). The two towers train together (four directions of the graph-positive loss),
so the image cells can move too.

Priors: image group holds 0.8; text group within range 0.4; the pass 0.4 (we assume the image group holds whenever the
text group does); the prefix wins 0.4 (SigLIP 1 was not trained to follow instructions). On test: SC3 level 1 0.8,
level 2 0.6, level 3 0.3. SC4 to SC6 hold image cells only, where the current models win over the plain embedding in
every cell (r14, r15): level 1 0.8, level 2 0.65, level 3 0.5 each. A surprise: the text group above the current
model's best seed; on test, a BELOW against plain SigLIP 1 in any cell.

### 3.2 Arms and controls

| ID | Arm | Differs from the current SigLIP 1 model by | Role |
|---|---|---|---|
| K0 | current SigLIP 1 model: r14's c* on SigLIP 1, seeds 0 to 4 (node402), frozen | nothing | comparator (validation); stage 0 target |
| K1 | SigLIP-only, plain caption: caption tower reads the painting mean of the captions' SigLIP 1 text embeddings | the caption tower's input | candidate |
| K2 | SigLIP-only, prefixed caption: as K1, each caption encoded as "Represent the caption: " + caption | input, prefix | candidate (stage 1a) |
| K* | the stage 1a pick (K1 or K2), seeds 0 to 4 | input | the candidate of stages 1b, 1c, 2 |
| P1 | plain SigLIP 1 (own text encoder; caption form by D20) | no training | comparator, claim 1 |
| Q1 | instructed Qwen3 | the MLLM itself, three passes | comparator, claim 2 |
| E2, E3 | SigLIP 2 and CLIP L/14 with their own text encoders, caption form of K*, seeds 0 to 4 | encoder | candidates (stage 2) |
| R1-R4 | K* with one part removed: instructed graphs, content correction, graph-positive loss, relational term (r28's arms) | one part | ablation (stage 2) |
| F1 | the photo model fine-tuned from K*'s emotion checkpoints, caption mean from SigLIP 1 | the start point and the caption input | candidate (stage 2) |

Everything else is c*'s: settings, graphs (k = 40), losses, early stopping, output width 2048, seeds. Only the caption
input width changes (2048 to 1,152 for both SigLIPs; 768 for CLIP L/14). F1 takes r13's final photo settings and r14's
photo protocol instead: seed s starts from K*'s seed-s emotion checkpoint, with training seed 100 + s; only the caption
input changes (D16). The per-cell sources of P1 and Q1 are in §3.4.

### 3.3 Data roles

| Data or seeds | Role | Stage |
|---|---|---|
| ArtELingo validation (6,152 paintings, 30,872 captions) | development: the prefix pick, the pass rule, every read-out pick and seed range of 1c and 2a | 0, 1a, 1b, 1c (picks), 2a |
| ArtELingo test (12,281 paintings, 61,744 captions) | test; read freely, shapes nothing | 1c, 2b |
| MMArt validation (the 5,769 ArtELingo validation paintings MMArt labels, r17) | development: K*'s MMArt read-out picks and ranges | 2a |
| MMArt test rows (r15's 16,384 paintings) | test | 2b |
| Affection fit; Affection and EmoSet validation | F1's training data; development for F1's early stopping, picks and ranges | 2a |
| Affection and EmoSet test | test | 2b |
| Seeds 0 to 4 (seed 0 for 1a); F1 training seeds 100 to 104 | as every round since r10; F1 as r14 | all |

### 3.4 Measurement

- **Stage 1a pick (SC1):** the text group (nine cells: emotion, style, genre × text to image, image to text, text to
  text) on validation, seed 0, read-out r0 (plain cosine). Higher wins; an exact tie goes to K1.
- **Pass rule (SC2):** five-seed means on validation, read-out r0 for text; image AMI and plain-cosine mAP as r28's
  encoder pick (`valpick`: MiniBatchKMeans at 9 / 27 / 10 clusters, n_init 3, states 0 to 2). Image group: the mean of
  the six cells. Text group: the mean of the nine cells. K0's values are r28's records, not recomputed, and are used
  unrounded (D17): image m_I = 0.3862917081 (per seed 0.3848908314 to 0.3878696916, so r_I = 0.0029788602;
  `records/valpick.json`); text m_T = 0.2593512884 (per seed 0.2586239413 to 0.2606346548, so r_T = 0.0020107136;
  `records/text.json`, read-out r0). The seed range is the maximum minus the minimum of K0's five per-seed group
  means. `pass_rule.json` records these values and the bounds. Stage 0 replays both.
- **Test cells (SC3 to SC6):** outcomes as r14 §3.4: WIN when the paired 95% bootstrap lower bound of the candidate
  minus the comparator is above 0; LEVEL when not WIN and the lower bound is above −r; BELOW otherwise. Per-cell
  intervals, no multiplicity correction. Read-out picks, r and the comparators' per-unit values per cell type (D18):

| Cells | Read-out pick (validation only) | LEVEL unit r | P1 / plain values | Q1 values | Interval |
|---|---|---|---|---|---|
| ArtELingo AMI, class counts | none (k-means as r17) | candidate's five-seed range of AMI at the class count, ArtELingo validation (r17) | r17 SC1 | r17 SC3 teacher | r17: 2,000 draws, rows then seeds |
| ArtELingo image mAP | r11's rule (as r14 for c*) | candidate's range of validation mAP at R0 (r14) | r17 SC7 | r11 stage 5 `instructed` per-query files at r11's pick | r11: 10,000 draws, queries and seeds |
| ArtELingo text, three directions | r23's rule (highest five-seed mean; ties r0, center, center_k10) | candidate's range at its picked read-out (r23) | r23 SigLIP 1 side row at r23's pick (form by D20) | r23 teacher rows at r23's pick | r23: 10,000 draws, painting unit, shared seed draw |
| MMArt AMI (9 / 20 / 10), image mAP | r15's rule on MMArt validation (as r17) | candidate's range on MMArt validation (r17) | r17 SC1 / SC7, MMArt plain | AMI: r17 SC3 MMArt teacher; mAP: r15 `stage1/picks_teacher.json` | r17 |
| Affection AMI (k = 9), mAP | r12's rule | F1's range on Affection validation (mAP at R0, as r28 for c*; AMI at k = 9) | AMI: r17 SC4 SigLIP 1 plain; mAP: r14 SC5 SigLIP 1 comparator | AMI: r17 SC4 teacher; mAP: r13 stage 3 `teacher_artwork` | r12 |
| EmoSet AMI (k = 8), mAP | r14 part c step 2 | AMI: F1's range on EmoSet validation; mAP: r14's committed r_E = 0.00695 (D19) | r28 `photos.json` cell c | AMI: r17 SC4 teacher; mAP: r14 SC10 comparator | r14 / r12 |
| E2, E3 (Table 6) | as the ArtELingo AMI and image mAP rows | E2's or E3's own range | r17 SC1 / SC7 (plain side) for SigLIP 2, CLIP L/14 | as the ArtELingo rows | as those rows |

  Where a source holds only point values and a pick, the per-unit values (per-query AP, per-row labels) are rebuilt
  from the stores at that pick; the rebuilt point value must equal the recorded one within 1e-9 (gate (v)). This holds
  for P1 and Q1 in the MMArt, Affection and EmoSet mAP cells, where no stored per-query file exists.

- **Cells per claim test:** stage 1c 15 (AMI, image to image, text to image, image to text, text to text × 3 aspects);
  MMArt 6 (AMI, image mAP); photo model 4 (Affection AMI and mAP, EmoSet AMI and mAP); E2 and E3 6 each (Table 6:
  AMI, image mAP). "Most cells" for level 1 and level 3: 12 of 15, 5 of 6, 3 of 4.
- **Ablation:** each removal's change from K* (mean over seeds) in AMI and plain-cosine mAP, with its paired interval,
  as r28's ablation table.
- **Table 8:** ms per caption for the SigLIP 1 text pass and for the new caption tower, as r28's `time_text` (RTX 3090,
  batch 32, 2,000 captions, three repeats).
- **Crashes (D21):** a run with a non-finite loss or a crash reruns once with the same seed and code (r14 §3.4). A
  second failure makes that arm's stage inconclusive.
- **Order and C13:** Codex re-derives, from this rule and the saved outputs, SC1 before stage 1b trains, and SC2 before
  1c or stage 2 reads any test row. The SC1 and SC2 records, and every read-out pick and seed range, are committed
  before any test number of their piece exists. Codex then re-derives every read-out pick, seed range and outcome of
  1c and 2b, within 1e-6, before any verdict is written. A re-derived value that differs by more than 1e-6 makes that
  pick, range or cell inconclusive until the cause is fixed; if Codex cannot run, a Claude `worker` re-derives
  instead, and the report says so. The night chat never settles a mismatch by choosing a side.
- **P1's caption form (D20):** if K2 wins SC1, P1's form is picked in stage 1a on validation and written to
  `prefix_pick.json` before 1b.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | the new code with the Qwen3 caption input; the new scorers on K0 and frozen records | gates (i) to (viii) below, each before the first candidate number of the stage it serves | fix code bugs only, at most three tries per gate; a gate that still fails parks what it serves (below); the rest goes on |
| 1a | sweep | K1, K2, seed 0, validation; if K2 wins, P1's caption form (D20) | both trained and scored; SC1 re-derived and committed before 1b | that arm out: the other becomes K* with no comparison, flagged in `prefix_pick.json`; both out: stage 1 inconclusive |
| 1b | incremental | K* seeds 1 to 4 (seed 0 from 1a), validation | five seeds trained; SC2 re-derived and committed before any test number | inconclusive |
| 1c | claim test | K*, ArtELingo test | K*'s read-out picks and seed ranges committed first | that cell inconclusive |
| 2a | sweep (development only) | only after a pass in 1b: E2, E3, F1, R1-R4 trained; picks and seed ranges on validation | picks and ranges committed before any test number of that piece | that piece inconclusive; the others go on |
| 2b | claim tests, ablation, measure | the 2a models on test; Table 8 in every case, after 1b | as 1c | that piece inconclusive; the others go on |

Stage 0 gates (tolerances: D23):
- (i) Trainer (D25). (a) On the local GPU, r14's trainer and the new trainer with the Qwen3 caption input each train
  SigLIP 1 seed 0; their weights are equal bit for bit (as r10's V4a). If they differ, r14's trainer runs seed 0 once
  more. If its own two runs also differ, the cause is CUDA, not code (r21's spec): (a) then passes when the new
  trainer's best epochs are within 2 of r14's and its six validation image cells are within K0's per-cell seed ranges
  (as r21's gate 0c), and this is flagged in `gates.json`. (b) At caption width 2048, the new trainer's
  model state at start, loss and gradients on a fixed batch equal r14's exactly (as r28's trainer test); a test shows
  that width 1,152 changes only the caption trunk's first layer and the checkpoint metadata. (c) Given r14's SigLIP 1
  seed 0 weights, the new export code reproduces r14's validation and test exports bit for bit, and r28's `valpick`
  seed 0 cells within 1e-6. A local run is not compared bit for bit with r14's node402 weights. If a DAS6 node is
  reserved, (a) may run there against r14's weights instead.
- (ii) Text scorer: on K0, every per-seed validation mAP at r0, center and center_k10 within 1e-6 of r28's
  `text.json` per-seed values, and the same nine picks. The tower embeddings match at r28's float16 rule.
- (iii) SigLIP 1 caption join: by sample ID and painting, it covers every caption (C6); per-painting caption counts
  equal Qwen3's.
- (iv) Prefixed store: the extractor's norm and finite checks pass, 0 truncated. With an empty prefix, the extractor
  reproduces the first 100 rows of the existing SigLIP 1 text store (cosine ≥ 0.999, as r14 0(i)). The manifest
  records the exact stored prefix after the extractor's lowercasing. The trainer and the text scorer each record the
  caption store's index hash, and the hashes are equal.
- (v) Test scorers: given K0's stored test exports and the comparators' frozen per-unit values, the new outcome code
  reproduces r17's SigLIP 1 SC1, SC3 and SC7 cells (value, difference, interval, outcome) within 1e-9. On the Qwen3 c*
  files, it reproduces r23's nine ArtELingo text cells within 1e-9. Every rebuilt comparator point value equals its
  record within 1e-9: r13 `teacher_artwork` Affection mAP 0.2945474692; r14 SC10 EmoSet teacher mAP 0.4518753521;
  r14 SC5 Affection plain SigLIP 1 mAP 0.2545391742; r28 `photos.json` cell c (mAP 0.3638545779, AMI 0.2672607188);
  r15's MMArt teacher and plain mAP at the `picks_teacher.json` pick (as r17's MMArt cells). A failed rebuild parks
  only the cells that use that comparator.
- (vi) MMArt: given r14's SigLIP 1 weights, the adapted export code reproduces r15's stored MMArt exports; the new
  scorer reproduces r28's MMArt SigLIP 1 c* cells within 1e-9.
- (vii) Photo and ablation: the photo trainer with the Qwen3 caption input equals r14's photo trainer on a fixed batch
  (state, loss, gradients). Given r14's photo weights, the scorer reproduces r28's `photos.json` cell a and r20's
  Affection photo cells within 1e-9. The ablation trainer at width 2048 equals r28's on a fixed batch, and the
  ablation scorer reproduces r28's `ablation_ref_replay.json`.
- (viii) New stores (the Affection SigLIP 1 text store for F1; the prefixed SigLIP 2 and CLIP L/14 stores if K2 wins;
  the SigLIP 2 and CLIP L/14 text stores): one row per key, finite values, unit norm within 1e-3, keys join by ID and
  painting (C6), truncation count recorded.

What a failed gate parks: (i) and (iii) all training; (ii) SC1 and SC2; (iv) K2 only, so K1 becomes K* (flagged in
`prefix_pick.json`); (v) the outcomes of SC3 to SC6; (vi) SC5; (vii) SC6 and SC7; (viii) the piece whose store fails.

Stage 2's order (D24): E2 and E3, then MMArt, then the photo model, then the ablation (cheap and central first).

### 3.6 Diagnostics

Measured after the levels are written; they decide nothing. (a) Per-cell validation changes K* minus K0, to show which
aspect and direction moves. (b) The best epoch of each head against K0's. (c) Text to text on validation for K* and for
plain SigLIP 1 per aspect: how much the caption tower adds over its own input. (d) Per test cell, K* minus K0 with its
paired interval: what the switch costs. (e) The local minus node402 change in the six validation image cells and the
nine text cells, from gate (i)(a)'s local seed 0 run, shown beside the pass rule.

### 3.7 Frozen vs rerun

Frozen: K0's checkpoints and records (r14, r28); Qwen3's graphs, instructed embeddings and content correction (r10's
store); c*'s settings; r28's read-out menus; the comparator values of §3.4's table and their picks. Rerun: K1, K2, K*,
E2, E3, R1-R4, F1; their read-out picks and seed ranges on validation; P1's text rows on the prefixed store if D20
picks it.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|---|---|---|---|---|---|
| D1 | Direction | a SigLIP-only model as a medium loop with its own decide chat | the PercepT sentence is "a bit problematic" while texts need Qwen3 | user | 2026-11-12 21:10 |
| D2 | Idea | Qwen3 keeps the instructions, graphs, content correction, alignment targets; the adapter reads SigLIP 1 only | "the later student part becomes all siglip" (not the paper plan's wider version) | user | 2026-11-12 21:10 |
| D3 | Place in the paper | decided on validation: a pass makes it the main model and starts stage 2 (all tables, more encoders); a fail gives an extra row or ablation | own pick, option c, with the encoder extension (rejected: a row now, option a; the main model whatever the numbers, option b) | user | 2026-11-12 21:30 |
| D4 | Pass rule form | one mean per group, image and text; a loss in one aspect can be offset | a floor per aspect | user, after seeing the recommendation (scenario check) | 2026-11-12 21:34 |
| D5 | Caption form | run plain and the "Represent the caption: " prefix on seed 0; higher text group wins (an exact tie: plain) | pick one untested | user, after seeing the recommendation (the tie: agent default) | 2026-11-12 21:34 |
| D6 | Settings | c*'s settings unchanged | r28's check: no row clearly ahead (+0.0007 inside 0.0030) | user, after seeing the recommendation | 2026-11-12 21:34 |
| D7 | Device and timing | local GPU; stage 2 after 19:00 or on DAS6 | weekday rule for runs over 1 hour | user, after seeing the recommendation | 2026-11-12 21:34 |
| D8 | Table 8 | measure the SigLIP 1 text pass and the new caption tower | leave the Qwen3 text time | user, after seeing the recommendation | 2026-11-12 21:34 |
| D9 | On a fail | cells go in the numbers sheet; the paper chat places them | a decision here | user, after seeing the recommendation | 2026-11-12 21:34 |
| D10 | Stage types | 1a sweep, 1b incremental, 1c claim test, 2a sweep (development only), 2b claim tests, R1-R4 ablation | 1c as a confirmation (no rule review); stage 2 as one claim-test stage (it trains and picks on validation) | agent default | 2026-11-12 21:38 (2a, 2b: 22:00) |
| D11 | Read-out in the pass rule and the prefix pick | r0 (plain cosine) for text, plain-cosine mAP for images, both models | the picked read-out (a pick inside a pick) | agent default | 2026-11-12 21:38 |
| D12 | Claim ladders | as §6 ("most cells": 12 of 15, 5 of 6, 3 of 4) | a single bar | agent default | 2026-11-12 21:38 |
| D13 | E2, E3 caption form | K*'s form, no own check | a prefix check per encoder (more runs before 11-16) | agent default | 2026-11-12 21:38 |
| D14 | Texts over the encoder limit | cut at the limit, count, flag (constitution row 5) | drop them | agent default | 2026-11-12 21:38 |
| D15 | Stores | new folders only: `artelingo_full/siglip_so400m_capprefix/`, `affection_local/siglip_so400m_txt/` (and E2, E3 prefixed stores if K2 wins) | write into the existing image store folders | agent default | 2026-11-12 21:38 |
| D16 | F1's caption mean | the mean of SigLIP 1 text embeddings of each photo's default explanations, L2 normalised, in K*'s form | Qwen3's mean (not SigLIP-only) | agent default | 2026-11-12 21:38 |
| D17 | Pass tolerance | r = maximum minus minimum of K0's five per-seed group means, unrounded; pass at ≥ m − r; level 2 at ≥ m; level 3 at > the best seed | the numbers the user saw at D4; rejected: the mean of the per-cell ranges (0.0078 image, 0.0054 text, 2.6 times wider) | agent default | 2026-11-12 22:00 |
| D18 | LEVEL unit, read-out pick and comparator values per cell type | §3.4's table: each cell type keeps the protocol that made its paper cell (r11, r12, r14, r15, r17, r23) | one unit for every cell (it would change the paper cells' protocols) | agent default | 2026-11-12 22:00 |
| D19 | EmoSet mAP LEVEL unit | r14's committed r_E = 0.00695 (the Qwen3 photo model's EmoSet range), as r28 used for every arm | the same unit as Table 7's other EmoSet cells; rejected: F1's own r_E | agent default | 2026-11-12 22:00 |
| D20 | P1's caption form if K2 wins | the form with the higher P1 text group on validation (r0; an exact tie: plain); read-outs by r23's rule on that form | C1, the strongest comparator; rejected: P1 always plain | agent default | 2026-11-12 22:00 |
| D21 | Crash rule | one rerun, same seed and code; a second failure: that arm's stage inconclusive | r14 §3.4; rejected: inconclusive at the first crash | agent default | 2026-11-12 22:00 |
| D22 | Inconclusive cells in a ladder | count as neither WIN nor LEVEL; "level L, inconclusive above" when they could raise the level | r14 needed all cells, but this ladder counts cells | agent default | 2026-11-12 22:00 |
| D23 | Gate tolerances | weights and exports bit for bit; validation cells 1e-6; replayed test cells 1e-9; tower embeddings r28's float16 rule (≥ 99.5% equal, the rest one float16 step); store replay cosine ≥ 0.999; unit norm within 1e-3; gate (i)(a) when CUDA itself is not bit-exact: best epochs within 2, validation cells within K0's per-cell seed ranges | as the earlier rounds' gates (r10, r14, r21, r28); rejected: a bit-for-bit rule only, which a CUDA difference could fail and so park all training | agent default | 2026-11-12 22:00 |
| D24 | Stage 2 order | E2, E3; MMArt; photo; ablation | cheap and central first; rejected: the paper's table order | agent default | 2026-11-12 22:00 |
| D25 | Machine | K* trains on the local GPU against r14's node402 K0; gate (i) checks the trainer locally and replays r14's weights; the local minus node402 change is diagnostic (e) | a bit-for-bit retrain of node402 weights cannot pass locally (r14: up to 0.0024 per cell); rejected: retrain K0 locally (changes D4's numbers and the paper's model) | user, after seeing the recommendation | 2026-11-12 22:42 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the SigLIP 1 caption store joins every ArtELingo caption by sample ID and painting. Checked: the Codex
  facts read compared both index files (308,723 IDs, same pairs); gate (iii) checks it in code.
- Assumption: no ArtELingo caption exceeds 64 tokens with the prefix. Checked: tokenizer count, 0 (facts read). The
  SigLIP 2 and CLIP L/14 stores record their truncation count (gate (viii)).
- Assumption: the Affection explanations fit in 64 tokens. Not checked; D14 covers it.
- Assumption: every comparator in §3.4's table can give per-unit values. Checked by the re-review: the record keys
  exist and hold the stated point values and picks; the MMArt, Affection and EmoSet mAP comparators have no stored
  per-query file, so they are rebuilt from the stores, and gate (v) checks each rebuild against its recorded value.
- Assumption: the local GPU and node402 give numbers within one seed range. Checked only on r14's seed 1 (AMI −0.0017
  to +0.0024); diagnostic (e) measures it.
- Assumption: r28's ablation on K0 ends before stage 2 needs the GPU. Not checked (r28 runs tonight).
- Open questions: none.

## 6. Success criteria (decision rule)

**Decision mode.** This section, with §3.4 and §3.5, is the decision rule (C7). Prior: §3.1.

| ID | What is measured | Compared against | Level 1 | Level 2 | Level 3 | Data |
|---|---|---|---|---|---|---|
| SC1 | stage 1a: text group, seed 0 | K1 vs K2 | pick: the higher | – | – | validation |
| SC2 | stage 1b: image group, text group (five-seed means) | K0 | image ≥ 0.3833128479 and text ≥ 0.2573405749 (the pass; m − r, §3.4) | L1 and text ≥ 0.2593512884 | image > 0.3878696916 and text > 0.2606346548 | validation |
| SC3 | stage 1c: 15 cells | P1; Q1 | WIN over P1 in ≥ 12 | WIN over P1 in 15 | L2 and WIN or LEVEL over Q1 in ≥ 12 | ArtELingo test |
| SC4 | E2, E3: 6 cells each (each its own verdict; the current model on that encoder shown beside, descriptive) | that encoder's plain embedding; Q1 | WIN in ≥ 5 | WIN in 6 | L2 and WIN or LEVEL over Q1 in ≥ 5 | ArtELingo test |
| SC5 | MMArt: 6 cells | P1; Q1 | WIN in ≥ 5 | WIN in 6 | L2 and WIN or LEVEL over Q1 in ≥ 5 | MMArt test |
| SC6 | photo model F1: 4 cells | P1; Q1 | WIN in ≥ 3 | WIN in 4 | L2 and WIN or LEVEL over Q1 in ≥ 3 | Affection, EmoSet test |
| SC7 | ablation R1-R4 | K* | each removal's change with its interval (attribution) | – | – | ArtELingo test |
| SC8 | Table 8 | Qwen3 text pass, 3.616 ms | measured (descriptive) | – | – | 2,000 captions |

- SC1 lets us say: "On validation seed 0, the <plain / prefixed> caption gives the higher text group (x against y)."
- SC2 level 1: "On validation, the SigLIP-only model's image and text group means are within one seed range of the
  current SigLIP 1 model's." It sets the pick: K* becomes the paper's main model and stage 2 runs. Below level 1, K* is
  an extra row. Level 2: "…and its text group mean is at or above the current model's." Level 3: "On validation, both
  group means are above the current model's best seed."
- SC3 to SC6 level 1: "With no Qwen3 pass at test time, training adds over the plain embedding in N of 15 cells" (of 6,
  of 4). Level 2: "in every cell." Level 3: "…and it is level with or above the instructed Qwen3 in N of 15 cells."
  Each Q1 cell is also its own claim: WIN "above", LEVEL "level with", BELOW none.
- SC7 lets us say: "On ArtELingo test, removing <part> changes AMI by x and mAP by y (interval), as strong as the
  result it explains."
- SC8 lets us say: "On one RTX 3090, the SigLIP 1 text pass takes x ms per caption against 3.616 ms for the Qwen3 pass
  (r28 protocol)."
- Inconclusive: a gate that fails after three tries, a second crash (D21), or a missing pick record, for the cells it
  touches. An inconclusive cell counts as neither WIN nor LEVEL; a level holds when its count holds on the decided
  cells; if every inconclusive cell as a WIN would raise the level, the report gives "level L, inconclusive above"
  (D22). If SC1 or SC2 is parked, stage 1 is inconclusive: 1c and stage 2 do not run; Table 8 runs.
- Pick: K*'s place (main model or extra row) is set by SC2 alone. If it changes after any test number exists, the
  report says the choice was made after reading test data (constitution row 2).
- Licensed claim: ArtELingo (and MMArt, Affection, EmoSet for SC5, SC6) under c*'s settings and these encoders only;
  development results (SC1, SC2) are never quoted as test results; the sentence never goes beyond the level reached.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | Q1 on test; K0 on validation; P1 in its better caption form (D20) |
| C6 sample IDs | pass | joins by sample ID and painting; gates (iii), (viii) |
| C7 rule before code | pass | §6 reviewed by a fresh Opus `xhigh` worker (pass after fixes), fixes applied and re-reviewed, committed before any code |
| C8 prior and disclosure | pass | §3.1; disclosure in §8 |
| C12 regression check | pass | stage 0 (i) to (viii) |
| C13 re-derivation | pass | §3.4, "Order and C13" |
| Decisions never alone | pass | rows 1 to 9 apply as written; "and more" encoders are parked (row 6) |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| The prefix is not applied the same way in training and at test | gate (iv): equal store index hashes in the trainer and the text scorer | gate failure, fix |
| A wrong caption-to-painting join (C6) | gate (iii); per-painting caption counts equal Qwen3's | stop |
| A width mix-up loads a 2048 caption checkpoint as 1,152 (or the reverse) | shape asserts; checkpoint metadata holds the input width; gate (i)(b) | stop |
| The pass rule hides a text emotion loss (D4) | diagnostic (a); the report shows the per-cell changes | report it; the paper chat decides the wording |
| The machine change moves K* against K0 (D25) | diagnostic (e) beside the pass rule | report it; the pass rule stands |
| Stage 2 does not finish before 11-16 | queue estimates at the start of stage 2 | run order of §3.5; unfinished pieces reported as not run |

Selection disclosure: on ArtELingo validation, c*'s settings came from r10's sweeps on Qwen3; SigLIP 1 was picked over
two other encoders (SigLIP 2, CLIP L/14) by 0.0007 (r28, a close call). r28 also tried four settings rows on SigLIP 1
seed 0 validation (k11 = 10, 20, 80; learning rate 0.0005), none adopted. This loop adds two caption forms (seed 0) and
one pass decision. Test numbers shape nothing; the 80% and LEVEL bars are fixed here.

## 9. Deliverables and review

The body is over the 150-line target: five stages, eight gates and one rule in one file.

- At the end: code in `src/test/20261112_siglip_only/` (module prefix `r29_`), outputs in
  `/data/SSD2/cosir_r29/20261112_siglip_only/`, records and `paper_cells.md` in the round folder, an auto report in
  `docs/reports/auto/percept/` with a user-read report beside it.
- Reviews: this rule by a fresh Opus `xhigh` worker before approval, and a re-review of the must-fix changes; verdict
  code reviewed per ticket (`matt-chain.md`); the final whole-branch review; C13 by Codex.
- Stop points: none between approval and the reports. Time-box: stage 1 done by Friday 11-13 evening; stage 2 by
  Saturday 11-14 12:00. Stage 2 pieces not finished by then are reported as not run (inconclusive cells, D22).
- Budget: stage 0 and 1 about 1.5 GPU-hours (stage 0: two training runs and the replays; stage 1: six training runs of
  about 6 minutes; one extraction of 6 minutes; scoring). Stage 2 about 6 GPU-hours (E2, E3 10 runs; R1-R4 20 runs of
  about 7.5 minutes; F1 five runs of about 3 minutes (r14: 187 s per seed on node402); one Affection extraction; if K2
  wins, two prefixed extractions (SigLIP 2, CLIP L/14, with a truncation count at CLIP's 77-token limit); scoring).
  CPU scoring under `res-queue cpu`. Local GPU; each stage 1 job (one training run or one scoring pass) stays under
  1 hour on weekdays 10:00 to 19:00.
