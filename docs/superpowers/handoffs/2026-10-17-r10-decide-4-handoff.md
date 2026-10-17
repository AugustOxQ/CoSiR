# Handoff: r10 decide 4: run the image-only transfer pilot overnight, write the morning report, then go on with the grill

## Brief
- Done: premise check of the user's prompted-aspect idea (Qwen3-VL-Embedding-2B, 10 instructions) and its auto report.
- State: the emotion instruction lifts caption emotion AMI from 0.122 to 0.278 (GoEmotions 0.209). The style instruction lifts image style AMI from 0.222 to 0.395 (CSD 0.317). Image emotion moves only from 0.069 to 0.093.
- Next: night mode: the image-only transfer pilot (the student premise), then the emotion-word masking check if time allows; end by 11:00; then a user-read report and the grill.
- Blocked: the grill waits for the user (morning).

> Written 2026-10-17 05:35 by `r10 decide 3` (cut at 200k context, after the premise check). Loop step reached:
> step 5 (grill) in progress, round 1 partly answered; decision B open; decision A still a draft. Direction log entry:
> [2026-10-15](../direction_log.md) (no new entry yet; it is written at the next wrap-up).

## 1. Read in this order

1. **The check's auto report**,
   [2026-12-08_prompted_aspect_views_check](../../reports/auto/percept/2026-12-08_prompted_aspect_views_check.md). It
   has the facts, the results brief, and our reading folded away. The round log
   `src/test/20261017_prompted_aspect_views/20261017_prompted_aspect_views_log.md` has the commands and conventions.
2. **The step 3 note**, [2026-12-06_new_line_candidates](../../reports/auto/percept/2026-12-06_new_line_candidates.md):
   it has the candidates (a), (b) and (c), and the parked ideas.
3. **The literature check**, [2026-12-07_prompted_aspect_views](../../reports/literature/2026-12-07_prompted_aspect_views.md):
   prior art, the main attack ("just prompting Qwen") and the comparators a reviewer will want.
4. Before the grill: `~/.claude/references/grill.md`. Before the user-read report:
   `~/.claude/references/user-read-report.md`. Before a Codex job: `~/.claude/references/codex-jobs.md`.

Skills: `grilling` and `domain-modeling` for the grill (morning); none for the night work.

## 2. Decided by the user, not to be reopened

- The bases of the new line (2026-10-17): GeneCIS (conditional similarity as the core idea), PercepT (the main model
  framework), the conditional buddy (our own idea). CVPR is a soft target: something works by about 2026-10-29, else
  keep trying to the abstract deadline (2026-11-10), else no submission.
- **Decision B:** the user is most interested in (a) and (b): "Discuss me more with them, start with (a)." (a) was
  discussed; (b) is still to be discussed. The recommendation (user saw it after their answer): (a) then (b), the same
  direction.
- **Label-free:** "a mostly-hard setting … if possible we want to keep the whole training label free. But this is not
  definite." Tentative: no labelled data; aspect names in prompts are allowed.
- **The PercepT baseline:** run the core reimplementation on our protocol now (painting-grouped held-out split, matched
  k, per-label AMI), with a no-DEC row. The faithful reimplementation comes later. The user's view: "at the end we need
  to do the faithful version", possibly with a non-held-out run compared directly with the paper's numbers. Names in
  `GLOSSARY.md` (commit 5d6d7d8).
- **The user's central idea:** prompt Qwen3-VL-Embedding with one instruction per aspect (aspect names only), to get
  K embedding sets, and run the buddy model on top of them. On wording, the user asked whether "focus only on emotion"
  works better than negation, so the check tested the focus, one-word and negation forms.
- "If this MLLM part succeeds then perhaps a paper." The student (a small image-only network, one head per aspect,
  distilled from the per-aspect buddy graphs) answers the attack "just prompting Qwen". It was explained to the user,
  but its requirements are not decided yet (Q4 below).

## 3. Where things stand

The facts are in the auto report. Its results brief has the key numbers, each beside its comparator. In short:
- **Captions:** the instruction makes caption geometry follow emotion, above a supervised emotion classifier.
- **Images:** the instruction makes image geometry follow style, above a dedicated style encoder.
- **Image emotion stays weak** for every view, the probes included (40.9% vs 35.2% majority).
- **Aspect partitions differ:** two aspect prompts agree at AMI 0.27 to 0.47, and seed reruns at 0.61 to 0.69.
- **Wording:** the focus and one-word forms are best; negation is weakest for emotion.
- **Not checked:** whether the caption lift is more than reading the emotion word.

## 4. The night work (agent default; present both as pilots, with no design choice beyond the grill)

The run mode is night (set by the user at 02:38). Do not wait for the user, delete nothing (use
`.scratch/pending_deletions.md`), and end GPU and CPU use by 11:00.

**Pilot 1: image-only transfer (the student premise).** Question: can a small head on image embeddings learn the
caption-side emotion geometry, and beat prompted Qwen on the image alone?
- **Paintings:** the 19,889 paintings of `sample.npz` (round folder). Evaluate on the fixed 5,000-painting subset,
  which has every prompted image view. Train on the other 14,889, and hold out a 10% painting-disjoint dev slice for
  early stopping. Early stopping reads the training loss only, never labels.
- **Teacher target:** for each painting, the l2-normalised mean of its `emotion_focus` caption embeddings (the store
  covers all 99,999 captions).
- **Control target:** the same mean, from `default` captions. It is the matched control: it drops only the emotion
  instruction.
- **Student input:** the default Qwen image embedding. For all paintings it is in the existing store
  `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/img.npy`, which used the same default instruction. Join by
  painting name through the store's `index.json` `images` list (35 MB: never print it). As a sanity check, the store's
  vectors for the 5,000 subset have cosine near 1 with `default/img.npy`. Optional second input: `emotion_focus` image
  embeddings for the 14,889 training paintings (about 22 min GPU).
- **Student:** a 2-layer MLP with an l2-normalised output. Loss: InfoNCE between student(image) and its own teacher
  target, with in-batch negatives. Three training seeds.
- **Scores** on the 5,000, with the conventions of `pav_score.py` (reuse its functions):
  - painting emotion, genre and style AMI at k = 20;
  - the emotion probe;
  - top-20 neighbour overlap with the teacher space, also for the default and `emotion_focus` image spaces.
- **Bar, stated before the run:** student image emotion AMI of at least 0.12 (prompted Qwen image: 0.093; teacher:
  0.217). It must also beat the control student by more than the seed sd. Report what the student loses on genre and
  style.
- **Build:** Codex if `codex-usage --room` says room, else a Sonnet `high` worker. The controller reviews it and runs it
  under `flock -o -w <s> -E 75 /tmp/gpu0.lock`. New folder `src/test/20261017_image_transfer_pilot/`, with module
  prefix `itp_` (`apc_` and `pav_` are taken). Expected: about 45 min to build and under 30 min to run.

**Check 2, if time allows: emotion words removed from captions.**
- Mask the nine ArtELingo emotion names and their common forms (amused, excited, angry, afraid, scared, sad, and so
  on) in the 99,999 captions.
- Re-embed them for `default` and `emotion_focus` (about 13 min GPU), and score caption emotion AMI.
- Report the share of captions that contain an emotion word.

**Then:** add the pilot results to the auto report (a new section, and an updated results brief), or write a second
auto report. Write one user-read report that covers both checks, in `docs/user_read/`. Commit and push.

## 5. Open points for the grill (morning, with the user)

1. **Present the results first** (facts first, then the user's reading, then ours).
2. **Key: the discussion of (b) and decision B.**
3. **Key, Q4: what the student must show.** Options: image-only transfer; better topics than the teacher; a gain from
   the shared trunk. The pilot informs it.
4. **Key, Q5: which aspects.** Options: content+affect+style; content+style; content+affect.
5. **Key: what counts as "something works" by 2026-10-29.**
6. **Minor batch:** the second domain (COCO-4c mood or Affection), the backbone (2B or 8B), the retrieval sets
   (GeneCIS focus tasks).

After decision B, the candidates not chosen and the parked ideas go to the direction log's "Parked ideas", one line
each:
- culture as the condition (ArtELingo-28, +0.052 Jensen-Shannon on 746 paintings);
- within-painting caption variation (probe 57.3% → 46.6%).

Then the spec (step 6).

## 6. Code and pitfalls

- **Joins are positional** (constitution C6): use the sample IDs from extraction. The pav store's `ids.npz` holds
  them.
- **Gitignored:** `*.json` and `*.log`. `sample.npz` (67 MB) stays out of git.
- **Earlier problem:** the auto-mode classifier once denied two harmless commands right after a Codex launch, then let
  the next one through. Do not repeat those calls.
- **The scorer's probe** prints lbfgs ConvergenceWarnings at 300 iterations. This matches the earlier checks, so keep
  it.
- **Held data:** the train side only. No step reads held data.

## 7. State at handoff

- Running: nothing (the queue ended at 05:20; the GPU lock is free).
- Uncommitted: `sample.npz` and Codex's leftovers in the round folder (listed in `.scratch/pending_deletions.md`).
- On disk over 1 GB: `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/` (4.6 GB). Its ledger row
  says "deferred: recommend keep"; the next storage summary asks the user.
