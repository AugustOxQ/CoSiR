# Handoff: r10 decide 6: write the pilot's short report, do the wrap-up, then draft the spec as an exploration sweep with a number sheet

## Brief
- Done: the grill (decision B is B2: named aspects now, learned aspects later), the purification pilot and a CLIP content-graph check.
- State: the purified buddy-pair image student gets image emotion AMI 0.094 vs control 0.070 (bar 0.12 fails); the shared edges stay the most emotion-consistent with a CLIP content graph (lift 2.61 vs 2.42).
- Next: the pilot's short auto and user-read report, the wrap-up, then the spec draft for the user's approval.
- Blocked: the storage summary and the spec brief wait for the user (back after a workout).

> Written 2026-10-18 16:14 by `r10 decide 5` (cut at 200k context). Loop step reached: step 5 (grill) closed, its
> closing summary confirmed by the user; decision C given for the night pilots and for the purification pilot;
> decision B made; decision A still recorded as a draft. Direction log entry: [2026-10-15](../direction_log.md) (no
> new entry yet; the wrap-up writes it). Run mode: night (set 16:14, "user at workout; automate"). The container was
> restarted at 15:53 (container clock); the GPU check passes (driver 595.99.02).

## 1. Read in this order

1. **This handoff, sections 2 and 3**: every decision of the grill, with who decided.
2. **The pilot's tables**, `src/test/20261018_purified_buddy_pilot/pbp_tables.md`, its log
   `20261018_purified_buddy_pilot_log.md` (method, bar) and `clip_content_stats.md` (the CLIP check).
3. **The night pilots' reports**, for the report format and the earlier numbers:
   [2026-12-09_image_transfer_and_word_masking](../../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md)
   and the user-read report
   [2026-10-17_prompted_aspects_and_student_pilot](../../user_read/2026-10-17_prompted_aspects_and_student_pilot.md).
4. **The step 3 note**, [2026-12-06_new_line_candidates](../../reports/auto/percept/2026-12-06_new_line_candidates.md)
   (candidate (a) is the base of B1), and the literature check
   [2026-12-07_prompted_aspect_views](../../reports/literature/2026-12-07_prompted_aspect_views.md) (prior art:
   DIOR, COCO-Facet, ESMC, guideline-driven clustering; the attack "just prompting Qwen").
5. Before the report: `~/.claude/references/user-read-report.md`, `~/.claude/templates/auto_report_template.md`,
   `results_brief.md`. Before the spec: `~/.claude/rules/spec-templates.md` and its templates.

Skills: none for the report and the wrap-up; `domain-modeling` only if the spec needs a new glossary term.

## 2. Decided by the user, not to be reopened (2026-10-18, this chat)

Earlier decisions (bases, soft CVPR target, label-free tentative, PercepT core baseline now and faithful later, the
central prompted-aspect idea) stand: [decide 4 handoff](2026-10-17-r10-decide-4-handoff.md), section 2.

- **Decision C, night pilots** (user, before ours): "there are aspects that we should not expect an image-only
  student model can learn". Ours agreed and went further (no more image-only emotion runs; a style student next).
- **Decision C, purification pilot** (user, before ours): "something but not conclusive; the teacher-student net
  failed on cross-modal emotion transfer; the Stage 1 finding matters: purification lowers content-likeness but does
  not raise emotion purity; maybe the views of one model share a lot". Ours added a data explanation (content and
  emotion co-occur in art); the CLIP check supports it (section 3).
- **Decision B: B2** (user, after seeing the recommendation; flag in the spec's brief). B1 now: named aspects. Then
  (b), learned aspects, if 2b below is met and time allows.
- **Architecture (Q1):** one shared trunk with one head per aspect (option 1c, user); clustering each prompted view
  apart (1b) is the cheap reference it must beat (user agreed).
- **Mapping (Q2):** cross-modal mapping (user: 2b), in the wider form (after seeing ours): each aspect space is built
  from the side where the aspect is visible (emotion from captions, style from images, genre from both); both the
  image and the text mapper predict every aspect space; any backbone (CLIP) for Stage 2. Training uses image-text
  pairs.
- **Aspects:** emotion, style and genre get heads (the three with labels); content is a helper graph only (user).
- **Graph combination:** first form "remove all shared edges" for emotion and style, genre kept whole (user, 2a).
  Open design piece (user's idea): how to combine the aspect graphs with the content graph: pruning, fusion (edge
  weights from all graphs), or pruning that guides fusion. Not to be picked blind: see "the user's guidance" below.
- **Loss forms:** a (independent heads, own buddy positives), then b (plus cross-aspect negatives), then c (one
  embedding with a mask per aspect, CSN form); each head also has an image-caption positive term; chosen on the
  train-side dev split; a stays on a tie (user; tie rule after seeing ours).
- **Retrieval:** aspect-named retrieval (GeneCIS form: query plus "same emotion / style / genre", ArtELingo labels as
  ground truth) carries the claim; one ranking from the heads (PVSE form) is the check (user, after seeing ours);
  gallery images only (user, after seeing ours). The comparator to beat: Qwen prompted with the same instruction.
- **Weak image emotion:** acceptable if the design improves it; cross-modal retrieval is one of the tasks (user).
- **"Something works":** by 2026-10-29, per-aspect held-out AMI beats the PercepT core baseline (2a), with 1b
  reported beside it (after seeing ours); then by 2026-11-10, aspect-named retrieval beats prompted Qwen (2b). (b)
  waits until 2b is met (user confirmed the scenario).
- **Minor batch, all accepted** (user, after seeing ours): instructions emotion_focus, style_word, genre_word; content
  helper = default caption graph plus default image graph, a dedicated content instruction as a variant (about 40
  min GPU); backbone 2B, one 8B image-emotion check (about 3 h GPU); retrieval sets as above, GeneCIS optional;
  ArtELingo first, Affection after the first result (not extracted; no genre labels).
- **Parked ideas** (to the direction log, one line each; user yes): B3, an aspect list proposed by an LLM from the
  captions (no way yet to map it back to the three named aspects); (c), PercepT under a fair protocol as the whole
  paper (the baseline stays); culture as the condition (ArtELingo-28, +0.052 Jensen-Shannon on 746 paintings);
  within-painting caption variation (probe 57.3% to 46.6%).
- **No html plan page** (user).
- **The user's guidance for the spec** (2026-10-18): "we are still mostly doing exploration jobs … try a broad range
  of options (how to prune vs how to fuse vs how to gain from prune to help fuse) and get a basic number sheet to help
  us decide later, not by just listing several options and pick one with no clue." Saved as project memory
  `feedback_exploration-number-sheet`.

## 3. Where things stand

**Purification pilot** (agent default design; `pbp_tables.md`; 13,400 training paintings for the graphs and
students, scored on the fixed 5,000 train-side paintings; image-only students on the default Qwen image; buddy-pair
InfoNCE; 3 seeds; image emotion / genre / style AMI at k = 20):
- control S_D (default caption graph) 0.070 ± 0.002 / 0.419 / 0.155; S_E (emotion graph) 0.090 ± 0.003 / 0.425 /
  0.167; S_P (purified) 0.094 ± 0.002 / 0.414 / 0.171; S_PN (purified plus content-only negatives) 0.096 ± 0.001 /
  0.419 / 0.169.
- References: default image 0.070 / 0.449 / 0.222; emotion_focus image 0.093; night student (own-caption target)
  0.074; neighbour average 0.102; teacher 0.217.
- Bar (S_P at least 0.12 and above S_D by more than the seed sd): fails. Effects: buddy pairs vs own caption about
  +0.02; purification +0.004 (about 2 sd); hard negatives +0.001 (inside the sd).
- All 12 runs stopped at epoch 1 to 3 (dev loss rises after). Probe about 46.5% for all four, the control included.

**Graphs** (fit paintings; lift = same-label rate over the random rate): emotion graph 57,727 pairs, emotion / genre
/ style lift 2.42 / 3.95 / 2.28; purified 45,505 pairs, 2.38 / 3.75 / 2.04; shared with the Qwen content graph 12,222
pairs, 2.55 / 4.66 / 3.18. 21% of emotion edges are content edges; 705 paintings lose all edges.

**CLIP check** (`clip_content_stats.md`, Sonnet worker, numbers checked against the file by the controller): with a
CLIP B/32 content graph, the shared edges (5,131 pairs) have emotion lift 2.61, above the emotion graph's 2.42; the
rest of the emotion graph 2.40. Overlap 9% of emotion edges (Qwen: 21%). The CLIP and Qwen content graphs share about a
third of the CLIP edges. No uncertainty computed. Reading: content and emotion co-occur in the data; the shared model
does not explain the pattern.

## 4. Open points for the next step

1. **The pilot's short auto report and user-read report** (purification pilot plus the CLIP check), as the night
   pilots got: an auto report in `docs/reports/auto/percept/` (CoSiR sequence date, next after 2026-12-09) with its
   results brief, our reading folded; a half-page user-read report in `docs/user_read/`. Decision C is already given:
   record the user's reading in the brief as given, not "left for the user". Index row in `reports_sum.md`, then
   `python scripts/check_reports_sum.py`.
2. **The wrap-up** (research-loop.md): the direction log entry for r10 (decisions C, A and B, the grill answers of
   section 2, the parked ideas), its "Now" paragraph; `wrapup-check`; `/project/claude-config/sync.sh`.
3. **Key: decision A is still a draft.** Ask the user to confirm the direction and its type and loop size, with the
   user's own line "still mostly doing exploration jobs" in view. It is the user's decision; ask answer-first.
4. **Key: the storage summary** waits for the user: the aspect-prompt store
   `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/` (6.3 GB, ledger row deferred, recommend
   keep) and the seven rows of `.scratch/pending_deletions.md` (two new from this chat: Codex leftovers with the
   about 100 MB smoke folder, and the 235 MB student outputs).
5. **The spec (step 6)**, drafted while the user is away, approved by the user. It follows the user's guidance:
   - an exploration sweep with one shared protocol (the painting-grouped split, the dev split for every choice, the
     same metrics) and a number sheet with one row per option;
   - the open design piece first: prune vs fuse vs prune-guided fusion of the aspect graphs and the content graph,
     scored on graph lift and on a downstream student or head;
   - then the multi-head student (loss forms a, b, c), the two-sided mappers, the 1b and PercepT core rows, and
     aspect-named retrieval against prompted Qwen;
   - an end-time estimate for any overnight run (end by 11:00) and the GPU budget (the 8B check, the content
     instruction).
   The spec's brief flags decision B as "user, after seeing the recommendation".
6. **Minor: ask the user** whether the number-sheet guidance should become a global rule (habit-scope.md); it is in
   project memory for now.

## 5. Code and pitfalls

- Round folder `src/test/20261018_purified_buddy_pilot/`, prefix `pbp_`. Taken prefixes: `apc_`, `pav_`, `itp_`,
  `ewm_`, `pbp_`. Its graphs (`graphs/fit.npz`, `dev.npz`, 7.8 MB, not in git) rebuild in about 35 s with
  `pbp_graphs.py`. `pbp_clip_content.py` builds the CLIP graphs (11 s).
- The pilot ran on CPU (the GPU was down before the restart). Students train in seconds per epoch on CPU.
- The auto-mode check refused harmless reads of Codex's json log twice right after a Codex job ("Create Unsafe
  Agents"). Read the files Codex writes (its log in the round folder) instead of its json output.
- Gitignored: `*.json`, `*.npy`, `*.log`, `outputs/`. Results JSON is on disk only; tables `.md` are committed.
- Held data: none read. Everything is the train side.

## 6. State at handoff

- Running: nothing (the queue ended 15:49; the CLIP check ended; no subagent, no GPU lock).
- Uncommitted: Codex leftovers in the round folders (listed in `.scratch/pending_deletions.md`), `sample.npz`, the
  untracked archive result folders from before.
- On disk over 1 GB: `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/` (6.3 GB, ledger row
  deferred).
