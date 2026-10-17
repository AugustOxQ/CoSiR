# Handoff: r10 decide 5: show the user the night results (decision C), then go on with the grill

## Brief
- Done: both night pilots, their auto report and one user-read report covering the three checks.
- State: masking 132 affect words keeps most of the emotion instruction's caption lift (AMI 0.187 vs 0.069 default). The image-only student fails its bar (0.074, control 0.074, bar 0.12).
- Next: show the results brief without our reading, get the user's reading, then the grill (candidate (b), decision B, Q4, Q5).
- Blocked: the grill waits for the user (morning).

> Written 2026-10-17 06:38 by `r10 decide 4` (cut at 200k context, after the night work). Loop step reached: step 5
> (grill) in progress; decision C for the premise check and the night pilots open; decision B open; decision A still
> a draft. Direction log entry: [2026-10-15](../direction_log.md) (no new entry yet; it is written at the next wrap-up).

## 1. Read in this order

1. **The user-read report**, [2026-10-17_prompted_aspects_and_student_pilot](../../user_read/2026-10-17_prompted_aspects_and_student_pilot.md).
   It covers the premise check and both night pilots for the user. Its advice is folded away.
2. **The night pilots' auto report**,
   [2026-12-09_image_transfer_and_word_masking](../../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md):
   results brief (our reading folded away), tables, analysis.
3. **The premise check's auto report**,
   [2026-12-08_prompted_aspect_views_check](../../reports/auto/percept/2026-12-08_prompted_aspect_views_check.md).
4. **The earlier handoff**, [2026-10-17-r10-decide-4-handoff](2026-10-17-r10-decide-4-handoff.md), sections 2 and
   5: the user's decisions so far and the grill's open points. They still hold; not copied here.
5. **The step 3 note** (candidates (a), (b), (c)) and the literature check, as linked in that handoff.
6. Before the grill: `~/.claude/references/grill.md`.

Skills: `grilling` and `domain-modeling` for the grill.

## 2. Decided by the user, not to be reopened

All of section 2 of the [decide 4 handoff](2026-10-17-r10-decide-4-handoff.md): the three bases of the line (GeneCIS,
PercepT, the conditional buddy), the soft CVPR target, interest in (a) and (b) (start with (a)), tentative label-free,
the PercepT core baseline now and the faithful version later, and the central idea (one instruction per aspect to
Qwen3-VL-Embedding, buddy graphs on top). Nothing new was decided tonight; the user was asleep.

## 3. Where things stand

Premise check (from the 12-08 report): the emotion instruction lifts caption emotion AMI from 0.122 to 0.278
(GoEmotions 0.209); the one-word style instruction lifts image style AMI from 0.222 to 0.395 (CSD 0.317); image
emotion moves only from 0.069 to 0.093.

Night pilots (agent default, from the 12-09 report; numbers read from `ewm_results.json` and `itp_results.json`):
- **Word masking:** caption emotion AMI k = 20, default vs emotion_focus: unmasked 0.122 vs 0.278; label words masked
  0.095 vs 0.231; 132 affect words masked 0.069 vs 0.187. On the 59,413 captions with no broad-list word (unmasked):
  0.070 vs 0.197. Label words are in 13.2% of captions, broad words in 40.6%. Not checked: affect words outside the
  list ("dark", "death", "smile").
- **Image-only student** (2-layer MLP, InfoNCE toward the mean caption embedding, 13,400 training paintings, 3 seeds,
  scored on the 5,000 subset): bar (AMI at least 0.12 and above the control by more than the seed sd) fails for both
  inputs. Default input 0.074 ± 0.000 vs control 0.074 ± 0.002; emotion_focus input 0.074 vs control 0.075. The
  student loses genre (0.449 → 0.424) and style (0.222 → 0.165). Probe: student 45.3% vs control 44.4%, default image
  39.3%.
- **Neighbour transfer** (controller exploration after the fail; mean teacher of the 20 nearest training images): 0.091
  (default input; control targets 0.070) and 0.102 (emotion_focus input; control 0.083). Teacher 0.217.

Our reading (folded in the reports; show it only after the user's reading): emotion lives on the caption side and
style on the image side; the student should show its worth on style or retrieval. Suggested next pilot: a style
student (about 1 h).

## 4. Open points for the next step

1. **Key: decision C.** Show the user-read report's results (or the 12-09 results brief) without our reading; ask for
   the user's reading in a sentence or two; then open ours and say where they differ. Record the reading for the
   wrap-up ("user" or "user, after seeing ours").
2. **Key: the discussion of (b) and decision B** (handoff 4, section 5, item 2).
3. **Key, Q4: what the student must show.** The pilot changes the options: image-only emotion transfer is now weak.
   Options: a style student; better topics than the teacher; a gain from the shared trunk; a gain on retrieval.
4. **Key, Q5: which aspects** (content+affect+style; content+style; content+affect). The pilots suggest affect from
   captions and style from images.
5. **Key: what counts as "something works" by 2026-10-29.**
6. **Minor batch:** the second domain (COCO-4c mood or Affection), the backbone (2B or 8B; the 8B could test image
   emotion, about 3 h), the retrieval sets (GeneCIS focus tasks).
7. **Storage summary** at the wrap-up: the aspect-prompt store is now 6.3 GB (ledger row deferred, recommend keep);
   `.scratch/pending_deletions.md` has five rows from r10 decide 3 and 4 (Codex leftovers, smoke folders, 118 MB of
   student outputs).

After decision B, the candidates not chosen and the parked ideas go to the direction log's "Parked ideas" (list in
the decide 4 handoff, section 5). Then the spec (step 6).

## 5. Code and pitfalls

- Round folders: `src/test/20261017_image_transfer_pilot/` (prefix `itp_`) and
  `src/test/20261017_emotion_word_masking/` (prefix `ewm_`). Taken prefixes: `apc_`, `pav_`, `itp_`, `ewm_`.
- New stores under `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/`: `masked_label/`,
  `masked_broad/` (99,999 masked captions per key) and `emotion_focus_trainimg/` (the 14,889 training paintings' images
  with the emotion_focus instruction; with the 5,000 subset this gives that view for all 19,889 paintings).
- The default image view for all paintings is the store `qwen3vl_emb_2b/img.npy`, joined by painting name
  (`itp_data.default_images`).
- Gitignored: `*.json`, `*.npy`, `*.log`, `outputs/`. The results files are on disk only.
- Held data: none read. Everything is the train side.

## 6. State at handoff

- Running: nothing (both queues ended; GPU lock free since 06:31).
- Uncommitted: Codex leftovers in the three round folders (listed in `.scratch/pending_deletions.md`), `sample.npz`.
- On disk over 1 GB: `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/` (6.3 GB, ledger row
  deferred).
