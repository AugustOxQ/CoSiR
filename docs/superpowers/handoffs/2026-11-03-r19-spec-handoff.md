# Handoff: r19 spec: write the spec from the finished grill, have its rule reviewed, get the user's approval, cut to the build

## Brief
- Done: the r19 grill, two rounds and a closing summary, confirmed by the user (19:20).
- State: c* is tested unchanged on DINOv2, CLIP B/32 and B/16 (ArtELingo, MMArt, photos), on SemArt's new artists (all seven encoders) and in a cost probe.
- Next: write the spec from §2 and §4 below, have its rule reviewed (fresh Opus `xhigh`), paste the brief for approval.
- Blocked: nothing. DAS6 node402 is free (3 GPUs, ends about 11-08 14:00).

> Written 2026-11-03 19:30 by tab "r19 decide" (cut at the context meter, 200k, after the grill's closing summary).
> Loop step reached: step 5 done (A and B made), step 6 next. Direction log entry: [2026-11-03 · MMArt test
> read](../direction_log.md), bullets "The user's reading of the experiment overview" and "Decided (A)".

## 1. Read in this order

1. This handoff, §2 (the grill's answers) and §4 (the draft design). Write the spec from them; no question rounds
   (`spec-templates.md`, "No second interview").
2. The facts: [`src/test/20261103_past_picks/pp_facts_brief.md`](../../../src/test/20261103_past_picks/pp_facts_brief.md)
   (Codex's brief, sections A to I, plus Claude's checked additions at the end).
3. The two specs r19 inherits, for the bar form, gates and wording: r14
   [`2026-11-02-backbones-sweep-design.md`](../specs/2026-11-02-backbones-sweep-design.md) (§3.2 to §3.5, §6 to §9:
   part a only) and r15 [`2026-11-02-mmart-test-design.md`](../specs/2026-11-02-mmart-test-design.md) (§3.2 to §3.5,
   §4, §6). Cite them ("inherits r14 part a; changes: …") instead of restating.
4. `~/.claude/templates/spec_core.md` and `spec_experiment.md`; `~/.claude/rules/spec-templates.md`.
5. The SemArt check: [`src/test/20261103_semart_check/20261103_semart_check_log.md`](../../../src/test/20261103_semart_check/20261103_semart_check_log.md).

Skills: none (the spec is written by the agent, `spec-templates.md`).

## 2. Decided by the user, not to be reopened

Grill: r19 decide chat, 2026-11-03, round 1 about 18:55, round 2 about 19:10, closing summary confirmed about 19:20.
Answers in the user's words where given.

| Row | Decision | Why (the user's line) | Decided by |
|---|---|---|---|
| Items | 16 (current c*, unchanged, on DINOv2, CLIP B/32, B/16), 6 (SemArt genre test on new artists), 19 (cost table) | decision A, 18:30 | user |
| No re-tuning | "we update that with the current model" | 18:30 | user |
| Paper section in the report | each loop's report states shortly how much it changes the paper: related work, method, experiments (draft `docs/paper/cvpr2027_aspect_spaces/drafts/v1/`) | 18:33 | user |
| Q1 bar (1a) | r14's bar unchanged, per encoder; all results in the paper's encoder table whatever they are; a fail or qualified result gets a limit sentence naming the encoder and cells | "1a is the current setting and (possibly) the least hard one" | user (before our recommendation; ours agreed: an extra margin for the new encoders only would be unfair the other way) |
| Q2 SemArt test set (2a) | the new-artist set at the 0.95 cut as it is: 11,162 paintings by 2,717 artists (unknown masters out) | asked "will this cause some big issues?"; answered no (§3 below) | user (ours agreed after the close-pair check; we withdrew the stricter 0.90 cut) |
| Q2 SemArt encoders (2f) | all seven: Qwen, SigLIP 2, SigLIP 1, CLIP L/14, DINOv2, CLIP B/32, CLIP B/16 | "they are not expensive, though we might expect some encoders to fail already" | user (no recommendation shown) |
| Q3 test sets for the three new encoders (3c) | ArtELingo held, MMArt test and Affection photos (r14's photo cells): every test the four picked encoders had | "the direct answer to the question, plus it's not expensive" | user (ours agreed) |
| m1 cost table | images per second for each encoder's default pass and each Qwen instructed pass (fixed batch, warm-up, with and without image decoding); peak GPU memory; parameters; passes per aspect; c* counted as one encoder pass plus three trunk-and-head networks of 4,197,376 parameters each (paper §3.6); whole-set figure: time to embed MMArt's 16,384 for three aspects; one probe of about 30 min on the 3090, on 2,000 MMArt paintings, short and light (day hours allowed) | "all minors yes" | user, after seeing the recommendation |
| m2 SemArt read-out | seed ranges (LEVEL unit) and retrieval read-out from ArtELingo validation, so SemArt shapes nothing | as above | user, after seeing the recommendation |
| m3 SemArt labels | all 10 types kept ("other" 253 and "study" 24 included); genre clustering and genre retrieval | as above | user, after seeing the recommendation |
| m4 runs | seeds 0 to 4, c*'s settings unchanged; c* and photo training on DAS6 (one GPU per encoder, beside r18); extractions on the local 3090 | as above | user, after seeing the recommendation |
| m5 DINOv2 matched arm | reruns on the full training set, seeds 0 to 4 (r10 has it only at the sample scale); B/32 and B/16 reuse r10's full M3 and M6 | as above | user, after seeing the recommendation |
| m6 read-out | claim read-out stays AMI at 20 clusters and mAP; r17's new metrics (NMI, ARI, ACC, cluster counts) as context with r17's module after it merges | as above | user, after seeing the recommendation |
| m7 SemArt context row | the SemArt cells again without the 217 paintings at cosine ≥ 0.93 to ArtELingo; no bar change; the report says whether any verdict changes | "yes" | user, after seeing the recommendation |
| Scenario checks | (1) DINOv2 level with PercepT in style clustering is "qualified", in the table, with a limit sentence; (2) a reviewer's copy (the Brueghel Proverbs pair) is answered by m7; (3) B/32 holding on paintings but failing on photos is reported as such; (4) each test set keeps its own verdict: a SemArt hold never makes up for an ArtELingo fail | "yes" to all | user, after seeing the recommendation |
| Plan page | none; the summary stayed in chat | "plain" | user |

No key decision was taken only after seeing our recommendation, so none needs a flag for that reason.

## 3. Where things stand

- c*'s record (direction log "Now"): holds on Qwen, SigLIP 2, SigLIP 1, CLIP L/14 on ArtELingo held (r14) and MMArt
  (r15: 16,384 paintings, every frozen-image and PercepT cell WIN; teacher 18 WIN, 4 LEVEL, 2 BELOW).
- The three new encoders were tested only with the older model R6-5 (r10), on another bar (emotion AMI ≥ 0.0100 above
  PercepT core's strongest entry): B/32 +0.0011 [−0.0043, +0.0025], B/16 +0.0070 [+0.0016, +0.0085], DINOv2 −0.0041
  [−0.0094, −0.0030] at the sample scale. On development, R6-5 beat its frozen image on all three (lift table in the
  facts brief). Prior for the spec's §3.1: WIN over the frozen image likely; PercepT cells less sure, DINOv2 least.
- The answer to "2a, big issues?": the closest new-artist pairs are mostly same-subject paintings by other artists,
  plus a family copy; true copies of ArtELingo paintings (Leonardo's Last Supper) belong to ArtELingo artists and are
  out by construction. Remaining risks: artist-name misses and family or workshop copies; both favor c* and matched
  PercepT (trained on ArtELingo images) over the frozen image and teacher; size not checked, at most a few hundred.
  The spec's §8 carries this, and m7 is the guard.
- Stores (facts brief §E and Claude's additions): ArtELingo has DINOv2, B/32, B/16 on all 61,402 paintings. Needed:
  MMArt B/32, B/16, DINOv2 (16,782 each); Affection B/32, B/16, DINOv2 (train, validation, test; 85,007 each); SemArt
  SigLIP 1, CLIP L/14, B/16, DINOv2 (21,382 each); SemArt Qwen genre teacher view (`genre_word`). All on the 3090,
  about 1¾ to 2 h together (estimate from r10, r14 and MMArt prep rates), heavy: evening, night or weekend only.
- Costs (measured): c* about 516 s per seed on an A6000 (r14 units), photo fine-tune 180 to 260 s, matched arm 35 to
  46 s per seed at full scale. So node402 needs about 3 GPU-h, about 1 h of wall time on three GPUs.

## 4. The draft design (agent defaults for the spec; refine, do not re-ask)

- **Kind and types.** Claim tests for items 16 and 6 (they feed paper claims; decision rule inline in §6 with §3.4 and
  §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit, as r14 and r15, C7). Item 19: a claim test
  with a simple bar (below). Stage types: 0 regression, 1 sweep on development only, 2 and 3 claim tests, 4
  descriptive.
- **Arms.** X-c* and X photo model for X in {DINOv2, CLIP B/32, CLIP B/16}: r14 part a's arms with the first layer
  width 1024 or 512. X frozen image. X matched arm: B/32 M3 and B/16 M6 (r10 full, held outputs and mappers; check
  the mappers are saved), DINOv2 a new full M5 with 10-25's Stage 2 settings. The teacher: Qwen instructed per aspect
  (ArtELingo, Affection artwork wording, MMArt as r15; SemArt: `genre_word`, new extraction). On SemArt every arm is
  frozen; the matched arms are applied frozen with r15's reading (argmax of the 40 topic probabilities; k-means 20 on
  the L2-normalized probabilities; the larger reading, r14 D16).
- **Criteria.** SC1 to SC3: the three new encoders on ArtELingo held (r14 SC1 to SC3). SC4, SC5: their photo models on
  Affection test (r14 SC4, SC5). SC6: their teacher cells on ArtELingo and Affection (r14 SC6: LEVEL or WIN, each cell
  its own "level with or better than" claim). SC7 to SC9: the three on MMArt test (r15 SC1 to SC3). SC10: their MMArt
  teacher cells; suggest every cell licenses only "level with or better than" (r15's D7 "better than" cells came from
  an earlier ArtELingo record that these encoders lack). SC11 to SC13: all seven encoders on SemArt: genre AMI against
  the frozen image, genre mAP (picked read-out) against the frozen image, genre AMI against the matched arm; per
  encoder holds when all three WIN, qualified when SC11 and SC12 WIN and SC13 does not, fails otherwise. SC14: SemArt
  teacher genre cells (AMI, mAP), each its own claim. SC15 (cost): per encoder, c*'s time to embed an image for all
  three aspects against the teacher's three instructed passes, same GPU and batch; holds when c*'s slowest of three
  repeats is faster than the teacher's fastest of three.
- **Verdicts** per encoder and per test set, each its own (scenario check 4): ArtELingo (r14 part a form), photos (r14),
  MMArt (r15), SemArt (above). A failed verdict stands and goes in the paper with a limit sentence.
- **Read-outs and ranges** (stage 1, committed before any test number). ArtELingo: r14's rule with each X-c*'s
  validation ranges. Affection: r12's rule with each X photo model's ranges. MMArt: r15 D4, the 5,769 ArtELingo
  validation paintings with MMArt labels. SemArt (m2): the arm's ArtELingo validation genre range and its ArtELingo
  read-out pick, with maps fitted on ArtELingo validation features and applied to SemArt; note under "Check these"
  that SemArt labels do not exist on ArtELingo paintings, so this uses ArtELingo's genre labels (r15 used the test
  set's own labels on those paintings).
- **SemArt set.** Recomputed from `sc_check.py`'s rule and `sc_nearcopy.parquet`: gate on 11,162 paintings and 2,717
  artists. Keys: `image_file`; evaluation order by sha256 of the key (r15 D9); exact md5 duplicates inside the set:
  keep the first in that order (r15 D10's form); internal pairs ≥ 0.95 reported. m7's row: without the 217 at ≥ 0.93.
- **Stage 0 gates** (C12), suggested: (i) extraction parity, cosine ≥ 0.999 by key: the first 100 ArtELingo rows of
  the B/32, B/16 and DINOv2 stores; the first 100 SemArt keys of the stored CLIP B/32 store; the first 100 Affection
  validation keys of the stored CLIP L/14 store; MMArt as the prep queue's parity (200 shared paintings); store checks
  (one row per key, finite, unit norm within 1e-3, keys join, C6). (ii) r14's trainer, imported unchanged, reproduces
  r14's CLIP L/14 c* seed 0 on node402 (best epoch and best development loss within 1e-4), and r14's CLIP L/14 photo
  model seed 0 likewise. (iii) Export code reproduces r14's CLIP L/14 held and Affection exports within 1e-3. (iv)
  Scoring replays within 1e-9: r14's CLIP L/14 SC1 to SC6 values and intervals; r15's CLIP L/14 SC1 to SC4. (v)
  Matched arms: M3 and M6 stored outputs replay their `reported_stage4` readings within 1e-9 and their mappers on CPU
  within 1e-5 (r15 (iv)); the new DINOv2 M5 code reproduces r10's sample-scale M5 seed 0 held reading (tolerance
  stated in the spec). (vi) The SemArt set count (above). (vii) The read-out rule reproduces r11's picks (r15 (vi)).
  (viii) The cost probe: its Qwen default rate on the 2,000 paintings within 20% of the MMArt prep log's 86.3 ms per
  image (end to end), else stop and trace.
- **Licensed sentences**: r14 part a and r15 with "<X> is one of three encoders that failed or were unclear in an
  earlier held emotion test with an older model (R6-5)". SemArt: "On 11,162 paintings by 2,717 artists with no
  painting in ArtELingo's train set (SemArt, Web Gallery of Art; images at cosine ≥ 0.95 to an ArtELingo painting
  removed), frozen c* on <X> clusters and retrieves paintings by type (SemArt's 10 classes, close to genre) better
  than the plain <X> image, and clusters them better than PercepT on the <X> image." Not licensed: emotion or style
  on new artists; artist matching is by name (misses possible).
- **Budget and timing.** node402 about 3 GPU-h (15 c*, 15 photo, 5 matched, stage 0 runs), about 1 h wall on three
  GPUs if r18 leaves them free. Local 3090: extractions about 1¾ to 2 h, heavy (after 19:00 on weekdays, ending by
  11:00), the probe about 30 min, light. CPU about 3 to 4 h at 8 threads (ArtELingo 390 to 520 s and retrieval about
  120 s per encoder, Affection about 90 s, MMArt about 15 min per encoder, SemArt seven encoders, C13 by Codex). Disk:
  about 6 GB of run outputs on SSD2 (15 X-c* at about 0.38 GB, 15 photo models at about 0.18 GB), stores under 1 GB.
  Report target: 11-06 (the plan said about 11-10).
- **Folders.** Round folder `src/test/20261103_past_picks/` (prefix `pp_`; the facts brief is already there); outputs
  `/data/SSD2/cosir_r19/20261103_past_picks/outputs/`; new stores in new folders under
  `/data/SSD2/pre_extract/<dataset>/<encoder>/`.
- **Report** (§9): the auto report with its results brief, the user-read report, and a short section on what the
  results change in the paper: related work (SemArt moves from related work, `paper_v1.md` line 128, to a test set),
  method (§3.7 names three encoders and "We changed no setting for any encoder"), experiments (the encoder table, the
  cost table beside §3.2 and §3.6's one-pass sentences).

## 5. Code and pitfalls

- Earlier round folders are frozen: import r14 (`bs_`), r15 (`mt_`), r10, r11 modules, never edit them; copy and
  extend in `pp_`. r17's scoring module is imported read-only after r17 merges into main tonight; r19's build starts
  after that merge.
- The build is unattended (night mode); before opening it, ask the user about a container restart and run the GPU check
  (`loop-chats.md`, "How to cut", step 3). The r17 and r18 chats share the local GPU and DAS6: GPU lock locally; on
  DAS6 the first job to ask gets a GPU.
- r19's main chat is open already ("r19 main", session 8812272b, opened 18:33); no `--main` call is needed.
- Codex has room again (five-hour 85%, weekly 98% at 18:46): the C13 re-derivation and build tickets can go to Codex
  per `agent-routing.md`.
- zsh: `git commit -F - <<'EOF'`. Several chats commit in this working tree: commit only your own paths.

## 6. State at handoff

- Running: nothing of r19. Beside it: the r17 and r18 chats.
- Uncommitted: nothing of r19 after this cut's commit.
- On disk over 1 GB: nothing new.
