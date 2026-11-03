# Handoff: r16 report 2: list the open items for the user, then the stage note, then the direction

## Brief
- Done: r16 report answered the user's questions on retrieval (protocol, read-out, mAP, CRL, CLAY, TPIPS); the user marked untold retrieval doable and a TPIPS comparator doable given a run time.
- State: c* is ahead of the teacher in clustering almost everywhere; in retrieval, emotion is level (3 of 8 cells below), style and genre mostly ahead.
- Next: list the open items for the user; then the stage note (ask its form and readers first); then decision A.
- Blocked: nothing (the five merged GitHub branches still wait for the user's delete).

> Written 2026-11-03 16:47 by tab "r16 report" (cut at the context meter, 200k). Loop step reached: r16 after
> decision C; decision A open, paused by the user. Direction log entry:
> [2026-11-03 · MMArt test read](../direction_log.md).

## 1. Read in this order

1. The direction log's r16 entry ([direction log](../direction_log.md)): the two options this chat added (untold
   retrieval; TPIPS) and the open direction. "Parked ideas" now holds TPIPS as a teacher.
2. This handoff's section 4: the open items to list for the user first, as the old chat promised.
3. For the stage note: the last stage report [`2026-10-28_recipe_sweep.md`](../../reports/stage/2026-10-28_recipe_sweep.md)
   (with its `.html` outside-view page and `_devlog.md` post), then `~/.claude/rules/report-kinds.md`,
   `~/.claude/references/outside-view-report.md` and `~/.claude/templates/stage_report_template.md`.
4. Only if the user asks again: r11 retrieval [report 12-29](../../reports/auto/percept/2026-12-29_retrieval.md) and
   [spec](../specs/2026-10-29-retrieval-design.md) (§3.3 read-out rule, §3.4 measurement, §4 D1 to D12); r12
   [report 12-31](../../reports/auto/percept/2026-12-31_affection_confirmation.md) (CRL versions and bars).

Skills: none.

## 2. Decided by the user, not to be reopened

- From the earlier r16 handoff: no further experiments for now; the order (questions, stage note, direction, r16);
  the r15 storage; the deadlines (abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23).
- Untold retrieval (form c) is doable; it runs only after the report (2026-11-03, direction log r16 entry).
- TPIPS (arXiv 2607.18237, NeurIPS 2026, the other authors' feedback): cite it in the paper; a comparator run is
  doable given a run-time estimate; as a teacher, parked (no time) (2026-11-03, direction log).

## 3. Where things stand

Answers given in the r16 report chat (sources: reports 12-29, 12-31, 2027-01-01, 2027-01-02, 2027-01-03; r11 spec;
c*'s weights; literature notes 2026-10-21, 2026-12-04, 2027-01-03):

- **Retrieval protocol (r11 to r15):** query-by-example label retrieval; the query is a painting plus an aspect name,
  which picks c*'s head for query and gallery; gallery = the other labelled paintings; cosine after the read-out
  (c*, teacher and plain all picked centering plus average query expansion, k = 10, on validation; CRL picked no
  centering, k = 10); AP over the full ranking, mAP; per-query AP averaged over 5 seeds. No text queries and no
  COCO-style retrieval (r11 D1). Chance mAP on ArtELingo held: emotion 0.2007, style 0.0792, genre 0.1466.
- **Read-out effect** (ArtELingo held, Qwen, emotion mAP): c* 0.2642 to 0.2691; teacher 0.2665 to 0.2684; the
  teacher cell moved from BELOW (−0.0023) to LEVEL (+0.0007).
- **No retrieval training:** c* was picked in r10 by mean AMI gain on development data; the photo model (r13) was
  picked with a score that includes Affection validation mAP.
- **c* has a caption side** (trunk and three heads on the Qwen caption mean; `cap.trunk`, `cap.output` in
  `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/*.pt`), trained by the buddy terms
  (image of A against caption of neighbour B); the own-caption term was removed in r10 (R6-5). Never scored. So
  r11 D1's reason, "c* has no text side", is not exact (correction not yet recorded; item d below).
- **Teacher cells by task** (4 encoders): clustering WIN on emotion and style on ArtELingo and MMArt; genre LEVEL
  on ArtELingo, WIN 3 and LEVEL 1 on MMArt. Retrieval: emotion LEVEL or BELOW (BELOW: CLIP L/14 on ArtELingo; Qwen
  and CLIP L/14 on MMArt); style WIN 3, LEVEL 1 (Qwen); genre WIN 4. Non-Qwen encoders compare across encoders
  (the teacher is always the Qwen image).
- **CRL:** our re-implementation on CLIP L/14 (agent default, r11 D12: a local Qwen3-VL-8B writes 50 values, a
  painting template, L/14 instead of the public B/32); training-free. The Qwen version lost r12's validation pick
  (ArtELingo 140.00 against 148.71; higher on emotion alone, 0.2691 against 0.2600). c* minus CRL: ArtELingo held
  +0.0137, +0.0418, +0.1118 (WIN); Affection frozen c* +0.0016 (LEVEL, no claim); photo model +0.0433 (WIN). The bar
  was set after r11's descriptive numbers. No CRL in r14 or r15. Matched encoder, point values only: c* on CLIP L/14
  0.2671 against CRL 0.2554 (ArtELingo held emotion).
- **CLAY:** not run (r11 D5, the user chose CRL only); training-free, text condition; code availability not checked.
- **TPIPS facts** (a Sonnet web-fetch agent; Tables 3 to 7 and the appendix seen only through a summary): two
  images plus free-text aspect, image to image; Qwen3-VL-Embedding-8B with LoRA; 24,342 synthetic FLUX triplets,
  about 1M human votes; physical aspects (lighting, material, pose; emotion, style and genre not found); released:
  three 8B models (late fusion `sywang/TPIPS-Embed-Qwen3VL-8B`, early and mid fusion), no 2B; Adobe Research License,
  noncommercial research use; `pip install tpips` (torch>=2.7, transformers>=5.0, peft>=0.19); one 8B pass per image
  per aspect (`model.embed(images, factor=...)`).
- **TPIPS run-time estimate** (a guess, not measured): 34,817 images (ArtELingo held 12,281, validation 6,152, MMArt
  16,384); three aspects 104,451 passes, with "overall" for untold 139,268. At 3 to 4 times Qwen 2B's 86 to 95 ms
  per image on the 3090: about 7.5 to 11 h for three aspects, 10 to 15 h with "overall"; ArtELingo only 4 to 6 h; CPU
  scoring about 1 h. A long heavy GPU run: DAS6 first if a node is reserved (`cluster status` failed to open, not
  checked), else a weekday night ending by 11:00. The CoSiR env has torch 2.11.0+cu130 and transformers 5.6.2; peft
  not checked. A 10-minute timing probe would replace the guess.

## 4. Open points for the next step

First, list these for the user (the old chat promised it), with the recommendations:

- a. Key: the stage note says the frozen encoder keeps its own retrieval unchanged (recommended: yes).
- b. Key: text queries by aspect through c*'s caption side on the decision A list (recommended: yes; a small
  exploration first: emotion, hits by the caption's own emotion label, ArtELingo validation, against plain Qwen text
  and the teacher text; per-modality centering in the read-out).
- c. Key: the TPIPS comparator: a timing probe first, then on the decision A list (the user asked for the estimate,
  section 3).
- d. Minor: the D1 correction as one line in the r16 direction log entry, no edit to the r11 spec (recommended).
- e. Key: the matched-encoder CRL comparison in the stage note as point values; a short CPU bootstrap later under a
  bar set before looking (recommended).
- f. Key: CRL on MMArt (CPU, features exist) and CLAY (check its code first) on the decision A list (recommended).
- g. Key: paper framing of fairness: a "training images from the domain" column; CRL, CLAY and TPIPS as context
  rows; the teacher, matched PercepT and frozen MMArt and EmoSet carry the fairness (recommended).

Then:

- Key: the stage note. "Stage note" is the user's word; ask its form and readers first (likely the next stage report
  since 2026-10-28, r11 to r15, or a shorter note for the collaborator and the supervisor).
- Key, after the stage note: decision A (options in the direction log r16 entry).
- Minor: the user deletes five merged origin branches (`.scratch/pending_deletions.md`).
- Minor: `routing-check` prints three items deferred to a tooling chat after 2026-11-09.

## 5. Code and pitfalls

- zsh here: `echo ====` fails; use `git commit -F - <<'EOF'`.
- `*.json` and `*.log` are gitignored: records need `git add -f`. The paper chat also commits here: commit only your
  own paths.
- Do not `pip install tpips` into the CoSiR env without a check: it may move transformers or peft; a separate env
  is safer.
- Auto report names keep CoSiR's sequence dates; stage reports carry the real date.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of this chat.
- On disk over 1 GB: nothing new (`/data/SSD2/cosir_r15/20261102_mmart_test/outputs/exports/` 9.0 GB, kept).
