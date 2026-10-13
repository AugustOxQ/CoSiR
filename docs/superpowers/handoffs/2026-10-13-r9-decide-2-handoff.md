# Handoff: r9 decide 2: grill AFF's scope plan (step 5), then write the spec (step 6)

## Brief
- Done: decisions C, A and B. AFF stays frozen; the grouping line is closed. Scope: Qwen3-VL-Embedding-2B, then SigLIP 2; CUB and SemArt; Affection and ArtELingo-28.
- State: the facts brief is ready. ArtELingo-28 is on disk in full (1,658 paintings, 28 languages). Qwen features cover every ArtELingo row.
- Next: the grill, key questions answer-first; then the spec.
- Blocked: Affection's data waits for the user's access request.

> Written 2026-10-13 13:00 by `r9 decide` (cut at the context meter, 200k). Loop step reached: after decision B and
> step 3 (literature check and facts brief), before step 5. Direction log entry:
> [2026-10-13 · Round 8](../direction_log.md) (decisions C, A and B, who decided each).

## 1. Read in this order

1. `docs/superpowers/direction_log.md`: "Now" and the 2026-10-13 entry.
2. The facts brief `.scratch/r9-scope/facts_brief.md` (local only, 454 lines): AFF's parts table (what depends on the
   backbone, the labels, the captions), timings, data readiness, the held budget, the paper bar's out-of-date points
   (§6), open items (§9). Each fact is tagged (file, computed, estimate, inference). Spot-checked by the controller:
   the ArtELingo-28 split CSV has 173,745 rows, 28 languages, 1,658 paintings, 27 styles, 2 splits.
3. `docs/reports/literature/2026-12-01_affect_caption_datasets.md`: Affection, ArtELingo-28, and the addendum
   (WikiArt Emotions, EMOTIC, MELD: none fits).
4. `~/.claude/references/grill.md` (protocol) before the grill; `~/.claude/rules/spec-templates.md` and
   `~/.claude/templates/spec_experiment.md` before the spec.
5. The CVPR plan spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`: only the sections
   the facts brief cites (K2, K4, §10).

## 2. Decided by the user, not to be reopened

- **C** (2026-10-13): the user's reading of round 8, recorded before ours (direction log).
- **A** (2026-10-13): AFF stays frozen; the grouping line closes. Backbones: Qwen3-VL-Embedding-2B first, then SigLIP
  2, each a claim test with its own held read (user, own pick). Datasets: CUB and SemArt as scope runs, plus a second
  dataset with affective captions (user, own pick). The paper runs in a `claude-paper` chat with round-by-round ARS
  review (user, own pick). DINOv2 parked (user, after seeing the recommendation); FCMI with a longer cap parked.
- **B** (2026-10-13): Affection with a COCO-category join as the second aspect, after a feasibility check, and
  ArtELingo-28 as a language check (user, own pick).
- Round 8's storage plan (user, approved and done; disk ledger).

## 3. Where things stand

From the facts brief, unless marked:
- **ArtELingo-28 is on disk in full**, at `/data/PDD/artelingo/ArtELingo/` (`ArtELingo-28/` and
  `semeval-artelingo28/`, a CSV with a split column and `images.zip`). Of its 1,658 paintings, 1,160 are in our data
  (248 held, 691 scorer-train, 99 selection, 122 val) and 498 are new. No document rules on whether they count as held
  rows.
- **Qwen3-VL-Embedding-2B:** features for every ArtELingo row, held included (`/data/SSD2/pre_extract/artelingo_full/
  qwen3vl_emb_2b`, 2048-d). No loader builds the `load_artelingo()` object from them yet.
- **SigLIP 2:** no full ArtELingo features; its extraction code lives only in round folders. Estimate: 21 to 58 min of
  GPU for all of ArtELingo.
- **AFF is not generic over the backbone.** Every stage loads CLIP features by default, checks SHA-256 pins of
  CLIP-made files, and builds the train/val/held split from a hash of CLIP image vectors. Nothing is hard-coded to
  512. The factor model A3 (621 s of GPU on CLIP) feeds B.
- **AFF's compute is small:** round 3 took about 15 min of CPU. Estimate: 1 to 1.5 h of compute per new backbone,
  without code porting and the claim-test process.
- **CUB and SemArt:** no SemArt loader, no episode wiring for either, no v2 result. Features: CUB for CLIP, Qwen and
  SigLIP 2; SemArt for CLIP and Qwen. The plan's aspects: CUB primary colour, bill shape, wing colour; SemArt type,
  school, timeframe.
- **Held budget:** the ledger counts per dataset (ArtELingo: H5 used, reserve unused; CUB and SemArt: 0 of 2), while
  constitution C5's amendment 2 (2026-10-09) says one read per final method and backbone. Settle which applies.
- **AFF without affective captions:** a closed gate scores like B; on held style × genre AFF is −0.43 [−0.65, −0.21]
  against B′(A0). The GoEmotions model (`SamLowe/roberta-base-go_emotions`) is English only and runs offline on the
  training captions.
- **Affection:** access by a request form, which the user was asked to send on 2026-10-13 (not confirmed).

## 4. Open points for the grill

Key (answer-first, at most three per round):
- **Order and dates:** what runs first, the dates by which a result must exist to enter the paper (abstract
  2026-11-10, paper 2026-11-16), and overnight runs ending by 11:00.
- **The time cap:** the user leans to drop or loosen a cap like round 8's 60 minutes, since the cost of new backbones
  and datasets is unknown. Set it from the brief's timings.
- **"Frozen" on a new backbone:** carry CLIP's choices (A3, thresholds, cells), or re-pick them on the new backbone's
  seed 42 by a fixed recipe? No document decides. Also how the split is made off CLIP.
- **Held reads:** per backbone (C5) or per dataset (ledger); ArtELingo-28's 248 held paintings and 498 new ones; whether
  a language check is a held read.
- **ArtELingo-28's languages:** translation to English (PercepT used GemmaX2-28-9B) or a multilingual emotion
  classifier; which backbones (CLIP reads English only).
- **CUB and SemArt:** the aspect pairs, the type (claim test or confirmation) and what result counts.
- **Affection:** the feasibility check's go/no-go (the COCO share, the leakage of object words), and the date after
  which no access means it is dropped.
- **Type and loop size per stage** (agent default to propose), and when the paper chat opens (its handoff is written
  by this decide line).

Minor (one batch, with recommendations, to the spec's "Check these"): a rule drop is logged as "dropped under the rule"
(not "failed"); a loader for the Qwen features; SigLIP 2 extraction in `scripts/extract_features.py`; CUB and SemArt
in the cluster `DATA_MAP`; whether to delete the remote branch `origin/r8-grouping` (round 7's was deleted); the
plan page question (grill rule 7).

## 5. Code and pitfalls

- A new round folder needs a new module prefix (`r9_`) and the next sequence date, 2026-12-02 (the literature note
  took 12-01).
- AFF's frozen code: `src/test/20261121_round3_affect_gate/`; its held read: `src/test/20261125_artelingo_held_test/`.
- The SHA-256 pins on CLIP-made files fail on other backbones by design; the spec says which pins are re-made.
- ArtELingo-28 sits on the shared `/data/PDD`: read only.
- The GitHub repo is public: never mention it in paper material.

## 6. State at handoff

- Running: nothing.
- Uncommitted: only the untracked round 8 results trees (`results/`, `results_m4_09999/`), as before.
- On disk over 1 GB: nothing new; round 8's items are decided in the disk ledger.
