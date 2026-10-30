# Frozen c* on Affection photos, with a CRL bar on ArtELingo and Affection: design

> Date: 2026-10-30 21:25 · Status: approved 2026-10-30 21:17 (from the brief, r12 decide; no container restart) · Kind: experiment
> Builds on: [report 12-28](../../reports/auto/percept/2026-12-28_recipe_sweep.md) (c*, held clustering) and
> [report 12-29](../../reports/auto/percept/2026-12-29_retrieval.md) (held retrieval); the grill in the r12 decide chat
> (2026-10-30, ending 20:20; its answers are §4) · Constitution: [constitution.md](../constitution.md)
> Decision rule: inline in §6, with §3.4 and §3.5 · Inherits: [r11 spec](2026-10-29-retrieval-design.md) (retrieval
> measurement, read-out menu, selection rule) and r10's held clustering statistic (round 20261027, `DECISION_RULE.md` §2.4)
> Reviewer: a fresh Opus `xhigh` worker, 2026-10-30 21:10 (pass after fixes: three must-fix and fifteen should-fix
> items, all applied; nits applied except the brief's length)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** c* passed its claim tests on ArtELingo paintings. This loop checks whether the result holds on new
data: Affection, 85,007 real-world photos with emotion labels only. c* is used as it is: its heads were trained on
paintings and are not retrained. Beside it, c* gets a bar against our re-implementation of CRL, on ArtELingo and on
Affection.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): the new code reproduces earlier numbers (the store with FI added, the instructed and CLIP images,
   c*, PercepT core, CRL, r10's and r11's held numbers). Then c*'s seed ranges are measured on Affection validation
   and committed.
2. Stage 1 (sweep, validation only): each model picks its read-out; the teacher picks its wording (artwork or photo)
   per task; CRL picks its version (CLIP L/14 or Qwen backbone; painting or photo wording) per dataset. All picks are
   committed before any held number.
3. Stage 2 (confirmation, Affection test): emotion clustering (AMI) and emotion retrieval (mAP with the picked
   read-outs), c* against the plain and instructed Qwen images. Plain-cosine rows, graded emotion and the switch column
   are reported beside the verdict.
4. Stage 3 (claim test, held): c* against the picked CRL, retrieval with the picked read-outs, on ArtELingo (three
   aspects) and on Affection (emotion).
5. Impressions: a figure of example retrievals only, since it has no class labels.

**Compared against.** The verdict: the plain Qwen image and the Qwen image instructed for emotion (c*'s teacher). The
CRL bar: the stronger CRL version on validation. Without a bar: PercepT core (beside PercepT's published Affection
AMI 0.18 and its own simpler variants, 0.21 and 0.25, all under another protocol), CLIP L/14, and the CRL versions and
teacher wording not picked.

**Outcomes.** Per comparison: WIN (interval above zero), LEVEL (lower bound above minus one seed range) or BELOW.
Holds: c* WINs over the plain image in clustering and retrieval, and is LEVEL or WIN against the teacher in both.
Qualified: the same, with the teacher LEVEL or WIN in one task only. Fails: anything else. A yes/no line says whether
the ArtELingo clustering WIN over the teacher repeats; it decides nothing. Each CRL bar is its own claim: a failed one
gives no claim and keeps its row. We expect WIN over the plain image. Against the teacher, LEVEL or BELOW in each task,
so any of holds, qualified or fails: c*'s heads never saw a photo. We expect the CRL bar to pass on ArtELingo;
Affection is open.

**Cost.** About 1.4 h of heavy GPU on the local GPU this weekend: FI about 28 min; the two instructed emotion
wordings and the CLIP L/14 images, validation and test only, about 55 min. CPU about 2 h at 8 threads. Build tonight,
run on Saturday; the report is expected on Sunday 2026-11-01, at the latest 2026-11-02 23:00.

**Check these.**
1. Impressions as a figure only: your pick after seeing our recommendation.
2. The new instructed and CLIP L/14 views cover only Affection validation and test (12,752 images); if r13 trains on
   the train part, r13 extracts it then (agent default).
3. The retrieval verdict uses the read-outs picked on validation, as r11 decided for ArtELingo; plain cosine is
   reported beside it (agent default).
4. The ArtELingo CRL bar is set after r11's descriptive CRL numbers were seen (c* led by about 10 to 30 seed ranges);
   only the Qwen version of CRL is new. The report says so beside any ArtELingo CRL claim.
5. Agent defaults: the photo wordings (the teacher's emotion instruction with "artwork" replaced by "photo"; CRL's "a
   photo with the emotion of {value}"); the types of stages 1 and 3 (sweep, claim test).

## 1. Context and goal

Asked: "1. Let's go with b" (r12 as planned: frozen c* on more datasets, clustering and retrieval, as a
confirmation). Then: "We go for at least 4b and with 4a include but not directly decide the verdict" (user,
2026-10-30).

c* (r10, arm k11=40) beats the instructed Qwen image in held ArtELingo clustering on emotion (AMI 0.1066 against
0.0921, +0.0145 [+0.0113, +0.0177]). In retrieval with picked read-outs (r11's stage 5) it is level on emotion (mAP
+0.0007 [−0.0002, +0.0016]) and above the plain image (+0.0208). The meeting of 2026-10-28 asked for more datasets
before 11-10. PolyArt waits for the user's link.

Goal: test frozen c* on held Affection photos for emotion clustering and retrieval, and test it against CRL.
Question this step answers: does c*'s ArtELingo pattern (above the plain image, at or above its teacher) hold on
photos its heads never saw?

## 2. Scope

In: Affection's validation and test images (all five sources, FI added); emotion only; ArtELingo held and validation
for the CRL bar; Impressions for one figure; the auto report and a user-read report.
Out: PolyArt (the user's link); style and genre on photos (no labels); retraining and Affection's train part (r13);
Impressions scores (option 1b, the supplementary on request); PercepT fitted on Affection; SigLIP 2 on Affection;
text queries.

## 3. Design

### 3.1 Hypothesis and prior

Claim checked: report 12-29's stage 5 sentence and report 12-28's clustering WIN, for emotion, on a second domain.
Mechanism: c*'s emotion head maps the default image toward the teacher's emotion view. If that mapping is not specific
to paintings, it carries to photos. Prior, before any number: WIN over the plain image in both tasks (high
confidence); against the teacher, LEVEL or BELOW in each task (no number; c*'s heads never saw a photo, while the
teacher is a general model). The photo wording may beat the artwork wording on photos. CRL: c* WIN on ArtELingo (point
gaps 9.7, 19.0 and 29.4 seed ranges in r11); on Affection, open, since emotion words carry over between domains.
Surprise: c* below the plain image.

### 3.2 Arms and controls

| Arm | What it is | Role |
|---|---|---|
| c* | r10's arm k11=40, weights frozen, seeds 0 to 4, CPU inference on the default Qwen image; the emotion head (the style and genre heads only for the switch column) | candidate |
| plain Qwen | Qwen3-VL-Embedding-2B default image | bar (verdict) |
| teacher, artwork wording | the image under r10's `emotion_focus` instruction, unchanged | bar if picked (per task) |
| teacher, photo wording | "Represent the emotion that this photo, or this description of a photo, conveys. Focus only on the emotion." | bar if picked (per task) |
| CRL, four versions on Affection | r11's D12 with r11's 50 emotion values; image and text encoder CLIP L/14, or Qwen3-VL-Embedding-2B (image side: the plain Qwen image; text side: the default instruction); value text "a painting with the emotion of {value}" or "a photo with the emotion of {value}"; vector: cosine with the unit value-text vectors, then L2-normalized, as r11's CRL | CRL bar if picked |
| CRL on ArtELingo | r11's CLIP L/14 version, plus a Qwen version (painting wording, three aspects, built as above) | CRL bar if picked |
| PercepT core | r11's arm (the full-scale P3 mapper, fitted on ArtELingo, seeds 0 to 4), applied frozen; clustering in r10's two readings (r20 and own) | descriptive; the own reading beside PercepT's published Affection 0.18 and the same paper's 0.21 and 0.25 (h and h_concat without DEC; literature note 12-05), all under another protocol |
| CLIP L/14 | image embedding | descriptive |

No arm is trained on Affection. Every arm picks its read-out from r11's menu on the same validation images.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| Affection test, 8,501 images (6,367 in the store, 2,134 FI) | held: scored only | 2, 3 |
| Affection validation, 4,251 images (3,129 + 1,122 FI) | development: seed ranges, read-out fits, all picks | 0b, 1 |
| Affection train, 72,255 images | not touched (FI default views are extracted to complete the store; nothing reads them) | none |
| ArtELingo held 12,281 / validation 6,152 paintings | held: the CRL bar and the replays / development: Qwen-CRL's read-out and the version pick | 0a, 3 / 1 |
| Impressions, one image per photo ID | figure only | 4 |

An image's key is the string `<dataset>/<image_name>` from the annotation file's two fields. An image's split is the
split of its rows other than "ignored" (no image has two). For validation and test, the evaluation order is the sorted
list of keys (Python `sorted`). It fixes the bootstrap rows I_b, the order of queries and gallery (ties keep this
order) and the cross-fit halves (parity of the position). Every join is by key, never by store position (C6).

### 3.4 Measurement

- **Terms.** R0 is plain cosine. The picked read-out is each arm's pick from this spec's stage 1, fit on all
  validation images and applied as r11's stage 5 does.
- **Labels.** From the preprocessed annotation file (version 0), per image key: the plurality emotion over its rows
  whose split is not "ignored". A tie goes to the emotion first in alphabetical order among the nine names (r10's
  `emotion_labels`: 0 amusement … 8 something else). Graded relevance: the emotion distribution over the same rows.
  Gate (stage 0a; C13 re-derives it): validation class counts 798/54/622/1,387/189/620/203/270/108 with 907 ties;
  test 1,651/81/1,460/2,626/351/1,185/448/501/198 with 1,917 ties (22.6%).
- **Clustering.** r10's stage 4 protocol on Affection test: MiniBatchKMeans (as r10: n_init 3, batch 4,096, one
  thread), 20 clusters, three random states, mean AMI; 5 seeds for c* and PercepT core. Statistic: r10's paired
  bootstrap with partitions fit once (2,000 draws, images then seeds, `default_rng(0)` per statistic).
- **Retrieval.** The r11 spec §3.4 and its read-out rules (§6), for emotion only: queries and gallery are the split's
  images, AP per query, mAP; nDCG with the overlap gain; 10,000 bootstrap draws over queries and seeds.
- **Seed ranges.** r_clu: max minus min over seeds 0 to 4 of c*'s emotion AMI (mean over three states, k = 20) on all
  4,251 validation images. r_ret,mAP and r_ret,nDCG: the same for c*'s emotion mAP and nDCG at R0, all 4,251 validation
  images as queries and gallery. r_clu serves SC1, SC3 and the clustering wording pick; r_ret,mAP serves SC2, SC4, SC8
  and the Affection read-out, wording and CRL picks; r_ret,nDCG the nDCG rows; r11's r_a the ArtELingo picks and
  SC5 to SC7. All unrounded.
- **Outcomes.** WIN when the lower bound > 0; LEVEL when not WIN and the lower bound > −r; BELOW otherwise.
- **C13.** Codex re-derives the labels, the seed ranges, every pick, the SC values, intervals and outcomes from this
  rule and the data, within 1e-6, before the verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0a | regression check | new code paths | (i) to (vii) below, and the label counts of §3.4 | stop, trace, note it |
| 0b | measurement | Affection validation | r_clu, r_ret,mAP, r_ret,nDCG written and committed before any Affection test number | stop, note it |
| 1 | sweep | validation | every pick of §6 (read-outs, teacher wordings, CRL versions) written to one file and committed before any Affection test number and any ArtELingo held number other than stage 0a's replays | report |
| 2 | confirmation | Affection test, picks frozen | SC1 to SC4 | report |
| 3 | claim test | held, picks frozen | SC5 to SC8 | report |
| 4 | descriptive | Impressions | the figure | report |

Stage 0a:
- (i) Store: the sha256 of the old store's arrays and index is recorded and committed before FI is added. FI's 21,808
  images and 134,418 texts are appended after the existing rows; the first 63,199 image rows, the first 392,331 text
  rows and their index entries hash the same as before; FI rows pass the extraction gate (finite, unit norm within
  1e-3).
- (ii) The generalized instructed extractor, batch 32, reproduces the first 200 rows (store order) of the stored
  ArtELingo `emotion_focus` store with per-row cosine ≥ 0.999 (the project's parity gate). Run with the default
  instruction on the first 200 Affection validation keys including 100 FI images, it reproduces the default store's
  rows by key with cosine ≥ 0.999. Maximum absolute differences are reported.
- (iii) The new scoring code gives r11's stage 5 per-arm mAPs, c*'s per-seed mAPs and SC10 to SC15's d and intervals
  (`stage5/verdict.json`) within 1e-9; r11's stage 4 scores within 1e-9 and its seven picks; r10's stage 4 emotion
  block within 1e-9 (A_cand 0.10658729378204021, A_T 0.09207439052277983, d 0.014512903259260385, interval
  [0.011288090988306501, 0.017671180133450672]).
- (iv) c*'s inference code reproduces the saved ArtELingo validation outputs of all 15 heads within 1e-3 (r11's
  0b(iv)).
- (v) PercepT core's new loader gives r11's full-scale held P3 arrays for seed 0 (hash) and the own-reading emotion
  AMI 0.08742922158584242.
- (vi) The CLIP L/14 extraction reproduces 200 stored ArtELingo CLIP L/14 rows with cosine ≥ 0.999.
- (vii) The CRL code, given CLIP inputs, reproduces r11's CLIP-CRL validation vectors (hash); the Qwen text path
  reproduces 200 stored Affection explanation rows with cosine ≥ 0.999.

### 3.6 Diagnostics

After the verdicts, deciding nothing: R0 (plain cosine) rows; graded emotion nDCG; the switch column (c*'s emotion
head against its other heads, emotion labels); the verdict with tied images dropped; per-source mAP (FI is itself an
emotion dataset); per-class AP; c*'s held per-seed spread beside r; all rows not picked; chance mAP.

### 3.7 Frozen vs rerun

Frozen: c*'s and PercepT core's weights; r11's CRL value lists; the seed ranges and every pick before held scoring.
Rerun: c* inference on Affection (CPU); PercepT core inference on its training CUDA device (light, under the lock), as
in r11; the new feature views; scoring.

## 4. Decisions

Grill rows: the r12 decide chat, 2026-10-30, three rounds ending 20:20.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Impressions | example retrievals only, no score | it has no class labels; text-based relevance (1b) or new labels (1c) invite questions | user, after seeing the recommendation | grill |
| D2 | Affection split | official: validation for picks and seed ranges, test held, train untouched | (all held: no validation, and r13 could never train on it) | user | grill |
| D3 | Teacher wording | artwork and photo; the better one on validation, per task | (artwork only; photo only) | user | grill |
| D4 | The verdict | holds, qualified or fails as in §6; a yes/no line on the clustering WIN beside it | (ArtELingo's WIN bar decides; the plain-image bars only) | user (wording of "qualified" ours, confirmed) | grill |
| D5 | CRL versions | CLIP L/14 and Qwen backbones; on Affection two wordings each | (CLIP only; Qwen only) | user | grill |
| D6 | Which CRL faces the bar | the better one on validation, per dataset | (always the Qwen version) | user | grill |
| D7 | Where the CRL bar sits | ArtELingo (three aspects) and Affection (emotion), with the picked read-outs; a failed bar gives no claim, keeps its row | (ArtELingo only) | user | grill |
| D8 | PercepT on Affection | our PercepT core, frozen, descriptive, beside the published numbers | (cite only; a bar) | user (unsure) | grill |
| D9 | Minor protocol | k = 20 and r10's statistic; plurality labels, ties-dropped check; per-task wording pick; emotion-only instructed views; local GPU this weekend | the grill's minor batch | agent default (accepted by the user) | grill |
| D10 | Retrieval reading of the verdict | the picked read-outs; R0 reported | as r11's D15 (R0 alone rejected) | agent default | 20:45 |
| D11 | Views extracted | FI default views for all splits; instructed and CLIP L/14 views for validation and test only | the train part is not read (all splits: about 4 h more) | agent default | 20:45 |
| D12 | Keys, label ties, pick scores and ties | §3.3, §3.4, §6 | alphabetical codes as r10; the same units as r11 (seed ranges); ties go to the published or learned form | agent default | 21:25 |
| D13 | Impressions figure | one image per UUID: the file without a headline overlay, else the first overlay file in file-name order; queries: the first 6 UUIDs in sorted order; gallery: the other 821; ranking by R0 cosine of c*'s seed-0 emotion head and of the plain image; top 5 each | a fixed rule, no picking by hand | agent default | 21:25 |
| D14 | Stage types | §3.5 | stage 2 checks an existing claim; stage 3 feeds a new paper claim; stage 1 picks settings | user, after seeing the recommendation (stage 2, decision A); agent default (stages 1, 3, 4) | 20:45 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: the FI images from the T4SA-2.0 copy are the ones Affection's annotators saw. Checked: all 21,808 names
  match and open; pixels not checked.
- Assumption: PercepT core's full-scale mapper runs on Affection's default Qwen images with only a new loader path.
  Not checked; stage 0a(v) checks the loader on ArtELingo.
- Assumption: no Affection image is an ArtELingo or WikiArt painting. Not checked; the five sources are photo
  collections (8 sampled Emotional Machines images are photos).
- Assumption: the instructed extraction runs at about 10.2 images/s (the ArtELingo `emotion_focus` store's rate).
  Not checked on Affection.
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | c*'s emotion AMI | plain Qwen | WIN | Affection test |
| SC2 | c*'s emotion mAP, picked read-out | plain Qwen, its picked read-out | WIN | Affection test |
| SC3 | c*'s emotion AMI | the teacher, clustering wording | LEVEL or WIN | Affection test |
| SC4 | c*'s emotion mAP, picked read-out | the teacher, retrieval wording, its picked read-out | LEVEL or WIN | Affection test |
| SC5 to SC7 | c*'s mAP, picked read-out, emotion / style / genre | the picked CRL on ArtELingo | WIN | ArtELingo held |
| SC8 | c*'s emotion mAP, picked read-out | the picked CRL on Affection | WIN | Affection test |

- **Verdict (stage 2):** holds when SC1 to SC4 pass; qualified when SC1 and SC2 pass and exactly one of SC3 and SC4
  does; fails otherwise. Beside it: "the ArtELingo clustering WIN over the teacher repeats: yes" when SC3 is WIN, else
  "no"; the comparator is the clustering wording picked on validation (ArtELingo's WIN was against the artwork
  wording). A failed stage 0 leaves no verdict.
- **CRL claims (stage 3):** each of SC5 to SC8 is its own claim; a failed one is reported as failed, gives no claim,
  and keeps its row. SC5 to SC7 were set after r11's descriptive held CRL numbers were seen; the report states this
  beside any ArtELingo CRL claim.
- **Selection rule (stage 1).**
  - Affection read-outs: r11's menu, cross-fit halves and tie order, per arm; the score is (mAP − mAP(R0)) / r_ret,mAP
    for emotion, seeded arms averaged over seeds.
  - ArtELingo: c* and the CLIP L/14 CRL keep r11's stage 4 picks (center + k10; none + k10), fit as in r11's stage 5.
    Only the Qwen CRL picks, by r11's stage 4 rule (three-aspect mean gain, guard, r11's r_a).
  - Teacher wording, per task: clustering by validation emotion AMI (r10's protocol); retrieval by the cross-fit
    validation mAP at each wording's own read-out pick. The photo wording only when it beats the artwork wording by
    more than 0.001 r (r_clu, r_ret,mAP).
  - CRL version, per dataset: the highest mean over the dataset's aspects of cross-fit validation mAP / r at its own
    read-out pick (ArtELingo: three aspects, r11's r_a; Affection: emotion, r_ret,mAP). Within 0.001 r of the best:
    the first of CLIP L/14 painting, CLIP L/14 photo, Qwen painting, Qwen photo.
- **Licensed claim (holds):** "On 8,501 held Affection photos, c*, whose heads were trained only on ArtELingo
  paintings over frozen Qwen3-VL-Embedding-2B image features and not retrained, clusters and retrieves photos by
  emotion better than the plain Qwen image, and level with or better than the Qwen image instructed for emotion in the
  wording picked on validation (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds)." Qualified:
  "… better than the plain Qwen image in clustering and retrieval, and level with or better than the instructed image
  in <task>; in <other task> it is BELOW." Not licensed: style or genre on photos, Impressions, PercepT's published
  numbers, claims against CLIP L/14.
- **Licensed claim (each CRL bar that passes):** "c* retrieves by <aspect> better than our re-implementation of CRL,
  with the stronger of its CLIP L/14 and Qwen versions picked on validation, on held <dataset>."

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the teacher in its better wording; CRL in its better version; PercepT core beside PercepT's published numbers, including its stronger simple variants |
| C6 sample IDs | pass | keys `<dataset>/<image_name>`, joins by key; stage 0a(i) checks the store's old rows |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker, committed before any code |
| C8 prior and disclosure | pass | prior in §3.1; disclosure in §8 |
| C12 regression check | pass | stage 0a |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass | all picks and seed ranges on validation; Affection test and ArtELingo held are scored only; query expansion reads test embeddings at test time (no labels), with k fixed on validation, as in r11; SC5 to SC7 were set after r11's descriptive held CRL numbers (disclosed in §6 and §8) |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| c*'s heads are tied to paintings, so c* falls below the teacher on photos | SC3, SC4 | report "qualified" or "fails"; the retrain question goes to r13 |
| 22.6% of test labels are ties broken by alphabetical order | the ties-dropped diagnostic | reported; no re-scoring after results |
| FI images are an emotion benchmark collected from Flickr, so one source may carry the result | per-source mAP | reported |
| The validation seed range is very small or large, so LEVEL is too strict or loose | c*'s held per-seed spread beside r | reported; r not changed after results |
| Selection disclosure: on Affection validation, 2 wordings, up to 25 read-outs per arm and 4 CRL versions are tried; on ArtELingo, c* was picked among about 60 configurations, and the CRL bar comes after r11's descriptive CRL numbers | validation numbers are inflated | only held numbers are results; every pick is frozen before held scoring |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261030_affection_confirmation/` (module prefix `ac_`) with code, tests, log
  and per-query outputs; the new feature views in `/data/SSD2/pre_extract/affection_local/`; the auto report
  `docs/reports/auto/percept/2026-12-31_affection_confirmation.md` with its results brief and index row; a user-read
  report.
- Reviews: this rule by a fresh Opus `xhigh` worker before the commit (done); tickets per `matt-chain.md`, with the
  label, pick, outcome and bootstrap code as verdict code (an Opus `high` ticket review); the C13 re-derivation by
  Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-11-02 23:00.
- Budget: heavy GPU about 1.4 h, local, under the lock, on Saturday 2026-10-31 (a weekend day: no working-hours
  limit). FI 21,808 images and 134,418 texts, about 28 min at the store's measured rates; instructed views 2 × 12,752
  images at about 10.2 images/s, about 42 min; CLIP L/14 12,752 images (rate not checked, about 10 min). CPU about
  2 h at 8 threads (MiniBatchKMeans on 8,501 images for 5 seeds × 3 states × arms; bootstraps). No DAS6.
