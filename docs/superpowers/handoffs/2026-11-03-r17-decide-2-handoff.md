# Handoff: r17 decide 2: write the spec for new scores from saved outputs, get the rule reviewed, then the user's brief approval

## Brief
- Done: the r17 grill. Cluster counts per aspect; four metrics with intervals at the class count; a WIN bar against CRL on the same encoder.
- State: every input is saved; CPU only, about 10 CPU hours (our estimate). Two small CPU exports are needed: the CRL vectors and PercepT's validation probabilities.
- Next: write the spec from the grill (no new questions), get the rule reviewed by Opus, then the user approves the brief.
- Blocked: nothing.

> Written 2026-11-03 19:25 by tab "r17 decide" (cut at the 200k context meter, after the grill's closing summary was
> confirmed). Loop step reached: step 5 done, step 6 (spec) next. Direction log entry:
> [2026-11-03 · MMArt test read](../direction_log.md), bullet "r17 grill (B)".

## 1. Read in this order

1. This handoff, sections 2 to 5: the grill's answers and the agent defaults proposed for the spec.
2. [`src/test/20261103_new_scores/ns_saved_outputs.md`](../../../src/test/20261103_new_scores/ns_saved_outputs.md):
   what is saved per cell, class counts, the scoring code to import, seed-range files, missing outputs and their cost.
3. The r15 spec as the model for form and protocol:
   [`specs/2026-11-02-mmart-test-design.md`](../specs/2026-11-02-mmart-test-design.md) (§3.4 measurement, §3.5
   gates, §6 rule, D7 and D12). Then the r14 spec's SC table (`specs/2026-11-02-backbones-sweep-design.md`, line 340 on).
4. `~/.claude/rules/spec-templates.md`, `~/.claude/templates/spec_core.md`, `~/.claude/templates/spec_experiment.md`.
5. For the decisions before the grill: [`2026-11-03-r17-decide-handoff.md`](2026-11-03-r17-decide-handoff.md), §2 and §5.

Skills: none (the spec is written straight from the grill, with no question rounds of its own).

## 2. Decided by the user, not to be reopened

All of the first decide handoff's §2 still holds (items 1, 2, 7, 8, 9, 11, CRL clustering on paintings, PercepT
retrieval; untold retrieval's form; the report's paper-writing section). The grill, 2026-11-03 19:00 to 19:20:

- **Q1, cluster counts (1a, user, before seeing ours; ours the same):** emotion 5, 8, 9, 16, 20 (all sets; EmoSet's
  class count is 8); genre 5, 10, 16, 20; style 10, 20, 27, 40 on both painting sets (class count 27 on ArtELingo
  held, 20 on MMArt test). Why: "more reasonable than the random number I suggest".
- **Q2, intervals (2a, user, before seeing ours; ours the same):** at the class count, AMI, NMI, ARI and ACC with
  paired intervals and WIN, LEVEL or BELOW in every claim cell (claim test). At the other counts, AMI and ARI as point
  values only (mean over 5 seeds and the min-max over seeds), for a figure. The existing AMI-at-20 claims keep their
  intervals. Intervals at every count (2b) are a later option if time allows (user): not in this spec; note it in §2 Out.
- **Q3, CRL bar (3a, user, after seeing the recommendation; first pick 3b; flag in Check these):** a WIN bar in all 12
  cells of c* on CLIP L/14 against CRL on CLIP L/14: ArtELingo held and MMArt test, three aspects, clustering and
  retrieval. A failed cell gives no claim and keeps its row. The report says that the ArtELingo retrieval point
  values were seen before the bar (CRL 0.2554/0.2816/0.5033, r11; c* on CLIP L/14, r14).
- **Scenario checks (confirmed by the user):** a lower ARI point value at 5 emotion clusters is written as "lower
  point value", no verdict word, and the class-count claim stands; a LEVEL CRL clustering cell fails its WIN bar, no
  claim, row kept.
- **Minor batch m1 to m8 (user, after seeing the recommendation):**
  - m1: ACC and NMI at the class count only; ACC by one-to-one (Hungarian) matching; NMI with scikit-learn's default
    (arithmetic) normalization; AMI and ARI as scikit-learn.
  - m2: the LEVEL margin of each new metric is its seed range over c*'s 5 seeds on the same set's validation split,
    at the class count, per encoder. (See §4 for a correction to how this was put to the user.)
  - m3: untold retrieval: each head unit-normalized, the three joined (6,144 wide), cosine; read-out picked on
    validation by r11's rule; comparator the plain image with its existing pick; WIN bar per aspect; ArtELingo held
    and MMArt test, four encoders; told c* beside it, descriptive.
  - m4: PercepT retrieval: its 40 topic probabilities (L2-normalized) as the embedding; validation probabilities made
    on CPU with the saved mappers; read-out picked on validation; WIN bar per aspect. Cells: ArtELingo held SigLIP 2,
    SigLIP 1, CLIP L/14; MMArt test, all four encoders.
  - m5: CRL on MMArt with r11's value texts (50 per aspect, painting wording); read-out picked on MMArt validation;
    Qwen c* against CRL on CLIP L/14 beside it (r12's form), descriptive.
  - m6: emotion per class: AP per class, c* minus teacher, four encoders, ArtELingo held and MMArt test; per-class
    paired intervals, no verdict; class sizes beside; run after the verdicts.
  - m7: inherited protocol: k-means settings, 3 states, 2,000 (clustering) and 10,000 (retrieval) joint bootstrap
    draws over items and seeds, no multiplicity correction; a regression check first reproduces r14's and r15's AMI
    and mAP exactly.
  - m8: folder `src/test/20261103_new_scores/`, prefix `ns_`; the scorer takes embeddings and labels and returns a
    metric table; its interface is frozen at merge, for r18 and r19 to import read-only.
- **Glossary (user, yes):** "told, untold retrieval" added to `GLOSSARY.md` (this chat).
- **HTML plan page:** no (user: "plain").

## 3. Where things stand

- Class counts (labelled classes present; label code, 2026-11-03): ArtELingo held emotion 9, style 27 (smallest
  class 15 paintings), genre 10 (9,949 labelled); MMArt test emotion 9 (16,218 labelled), style 20, genre 10 (13,420
  labelled); Affection test emotion 9; EmoSet test emotion 8 (17,348 after the cut).
- Claim cells for the class-count metrics: 72 painting cells (2 sets × 4 encoders × 3 aspects × plain image, matched
  PercepT, teacher) and 10 photo cells (Affection test: 4 photo models × plain, teacher; EmoSet test: Qwen photo
  model × plain, teacher).
- Teacher bars of the AMI-at-20 cells these extend (from the specs): ArtELingo held LEVEL or WIN (r10, r14 SC6);
  MMArt test WIN in r15 D7's cells, LEVEL or WIN elsewhere; Affection test LEVEL or WIN (r13 SC3, r14 SC6); EmoSet
  test WIN (r14 SC9). Plain image and matched PercepT: WIN everywhere.
- CRL on the same encoder, already seen: ArtELingo held emotion mAP, c* on CLIP L/14 0.2671 against CRL 0.2554
  (+0.0117, about 7 seed ranges; CLIP L/14's ArtELingo validation range 0.0017, r14 records). Not seen: CRL
  clustering on paintings; CRL on MMArt in both tasks.
- Cost (our estimate from the r14 and r15 logs, not measured): about 10 CPU hours with the re-derivation, about 4 hours
  of wall time in three processes at 8 threads each. r15's scorer: clustering 637 to 825 s, retrieval 250 to 488 s
  per encoder; read-out picks 540 to 957 s per encoder.
- Codex: weekly limit at 5% until 2026-11-04 08:27, so the rule review, the build and the re-derivation start on Claude.

## 4. Open points for the next step: agent defaults proposed for the spec

None needs the user; decide them as agent defaults and list the most consequential under "Check these".

- **Correction to m2 (flag in Check these):** the user was told that Qwen's ArtELingo AMI range would be "recomputed
  and checked against the glossary values". That check cannot work: the glossary's 0.0046/0.0082/0.0128 are D2's
  ranges on the old 5,000 development paintings, not c*'s. So the new class-count ranges for every encoder, Qwen
  included, come from c*'s 5 seeds on that set's validation split (Qwen: the r10 validation exports, Table 4 of the
  inventory); the AMI-at-20 claims keep their recorded ranges.
- **Stage types:** stage 0 regression checks; stage 1 sweep on development only (the new seed ranges, the read-out
  picks for untold c*, PercepT and CRL on MMArt), committed before any new test number; stage 2 claim test; stage 3
  descriptive (the count sweep, the emotion per-class look, Qwen c* against CRL on MMArt, told beside untold), after
  the verdicts. The loop stays medium.
- **Bars of the class-count cells:** each takes the bar of the AMI-at-20 cell it extends (§3 above); the CRL cells
  take the WIN bar (3a) at 20 clusters and at the class count. Each metric cell is its own claim; the licensed
  sentence names the metrics that pass.
- **Matched PercepT at the class count:** only the k-means reading at K on its probabilities (its own reading has up
  to 40 topics, so it does not fit a fixed count).
- **PercepT retrieval cells:** also include Qwen on ArtELingo held (its r11 values were seen: picked 0.2483/0.1523/
  0.5049 against c*'s 0.2691/0.3235/0.6151), so 24 cells; untold retrieval 24 cells.
- **Priors (C8), first guesses:** class-count cells against the plain image WIN 0.85 (AMI, NMI), 0.75 (ARI), 0.65
  (ACC); against matched PercepT 0.75; teacher cells as their AMI-at-20 outcomes, 0.7; CRL retrieval WIN 0.9, CRL
  clustering WIN 0.85; PercepT retrieval WIN 0.9; untold against the plain image WIN 0.8 in style and genre, 0.5 in
  emotion (the joined vector carries the style and genre heads too).
- **Stage 0 gates:** the new clustering scorer at 20 clusters reproduces r14's SigLIP 2 ArtELingo and r15's MMArt
  AMI values and intervals (same draws) within 1e-9; the retrieval scorer reproduces r15's SC2 and r11's stage 5
  values and intervals; the rebuilt CRL reproduces r11's CRL held mAP; the CPU mappers reproduce their stored held
  probabilities within 1e-5 (r15 gate (iv) a).
- **Re-derivation (C13):** separate code from the rule and the saved data, before any verdict is written; Codex if
  `codex-usage --room` says room, else a Claude worker.
- **Rule review:** a fresh Opus worker (r15 used `xhigh`) before the approving commit; apply its fixes, then paste the
  brief.
- **Report:** auto report under `docs/reports/auto/percept/` with the next sequence date taken at report time (r18
  and r19 also write reports), its results brief, its index row, a user-read report, and the paper-writing section
  (related work, method, experiments of `docs/paper/cvpr2027_aspect_spaces/drafts/v1/`).

## 5. Code and pitfalls

- First handoff §5 holds (frozen round folders, imported, never edited; the stable interface; zsh commit form; commit
  only your own paths, since the paper chat and the r18 and r19 chats commit here too).
- `ac_cluster.partitions` asserts k = 20; `bs_emoset_score.partitions_k(x, k)` is the only k-means with a k argument.
  `rt_stats.paired` needs float64 inputs. `mt_ret.score_encoder` reads only committed stage-1 records.
- EmoSet exports are uncut: use `bs_emoset.keys(split, cut=True)`. MMArt test is the 16,384 subset in r15's
  `records/cutsets.json`. Retrieval and bootstrap order follow r15 D9 (sha256 of the key).
- c*'s three heads of one seed are three separate training runs; check key equality before joining them.
- No c* exports on Affection or EmoSet exist for SigLIP 2, SigLIP 1 or CLIP L/14; the photo cells use the photo
  models, so none are needed.
- At brief approval, open r17's main chat: `loop-next ~/.claude/templates/main_chat_start.md --label "r17 main"
  --main --new-workspace "CoSiR r17" --model sonnet --effort high --notify`. Before opening the build chat
  (`--unattended`), ask the user about a container restart (`loop-chats.md`, "How to cut").

## 6. State at handoff

- Running: nothing of r17. Beside it: the r18 and r19 decide chats.
- Uncommitted: nothing (this handoff, the inventory, the glossary row and the direction-log bullet are committed).
- On disk over 1 GB: nothing new.
