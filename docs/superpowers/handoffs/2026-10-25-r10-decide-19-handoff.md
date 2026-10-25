# Handoff: write the spec and decision rule of decide 17's two-part loop, check them, then open the build chat

## Brief
- Done: grill round 2 (recipe rule, backbones, comparator, minor batch), the closing summary, the user's approval.
- State: the user approved the brief through the closing summary (option a); night mode; no container restart.
- Next: write the spec and the decision rule, run the rule review and the cold check, commit, open "r10 build 16".
- Blocked: nothing. The user is asleep: ask nothing.

> Written 2026-10-25 03:04 by "r10 decide 18" (cut at the context meter, about 206k). Loop step reached: step 5 done,
> step 6 (the spec) next. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md) (its Idea (B)
> line is filled).

## 1. Read in this order

1. This handoff. §2 is the grill record; the spec writes from it with no question rounds (`spec-templates.md`).
2. `~/.claude/rules/spec-templates.md`, `~/.claude/templates/spec_core.md`, `spec_experiment.md`,
   `decision_rule_template.md`; the constitution (`docs/superpowers/constitution.md`: C1, C6, C7, C8, C12, C13).
3. The claim test's rule, `src/test/20261020_claim_test/DECISION_RULE.md`: part 2's rule reuses its statistics (§3),
   validity checks (V1 to V4, R-COL) and outcome names, and changes only the candidates and the entry set.
4. The ablation spec `docs/superpowers/specs/2026-10-21-d2-ablation-design.md` (part 1 inherits its protocol) and its
   report's results ([12-22](../../reports/auto/percept/2026-12-22_d2_ablation.md), seed ranges and row labels).
5. Facts: `.scratch/r10-decide-17/grill_facts.md`, `.scratch/r10-decide-16/f2_facts.md` §3 to §5,
   `.scratch/r10-run-12/a28/a28_overlap_facts.md`. The raw grill record: `.scratch/r10-decide-18/grill_round2.md`.

Skills: none (the spec is written by the agent, `spec-templates.md`).

## 2. Decided by the user, not to be reopened

Decision C (2026-10-21) and decision A (2026-10-25) are in the direction log. Round 1 (decide 17): the fresh sample
is all 29,232 train-side paintings outside the 19,889-painting sample, training stays on the 13,400 fit and 1,489
stopping paintings, the old 5,000 are reported beside it (user); a fixed rule picks part 2's recipe (user, after the
recommendation); ArtELingo-28 is (a) a test set on the 746 read-set paintings and (c) a descriptive culture split
(user). Round 2 (this chat, 2026-10-25 02:35 to 03:02):

| Question | Answer | Who decided |
|---|---|---|
| The recipe rule's form | Guarded replacement: a recipe replaces D2 (teacher 1/5) when it is at least "small" (interval above 0, gain of at least one seed range) on one aspect or more, and no aspect is more than one seed range below D2; if several pass, the largest mean gain in seed ranges wins; none passes: D2. Why: "more comprehensive ... keep the best one" | user |
| Its what-if (genre guard blocks a larger style gain; none passes: D2 at 1/5) | "fine" | user |
| Backbones | SigLIP 2, then CLIP B/32; DINOv2 only if the rule picks image endpoints only (the user first chose all three "in order", then changed to this) | user, after seeing the recommendation (flag) |
| The student's caption side | Qwen's default captions for every student: only the image input changes | user, after seeing the recommendation (user was unsure) (flag) |
| The bars | All three reported: (a) PercepT core as now, (b) a matched PercepT core arm (Qwen Stage 1 topics, the backbone's image in Stage 2), (c) the backbone's frozen image | user |
| The claim's bar | The stronger of a and b per aspect: the matched arm becomes one more PercepT core entry, the bar is the strongest entry, as in the claim test | user, after seeing the recommendation (flag; user's first answer was b) |
| Which paintings | Held 12,281, trained on the 13,400 first; then the full 42,969 split | user |
| When the full stage runs | Unless stage 1 is a NO-GO (d ≤ 0), as the claim test | user, after seeing the recommendation (flag) |
| Minor batch | All accepted as recommended (list below) | user, after seeing the recommendation |
| html-plan page | No ("plain") | user |
| Closing summary | Confirmed ("yes") | user |
| Brief approval | Option a: the closing-summary yes counts as approval; a cold Sonnet `high` worker checks the spec and rule against this record; the brief says "approved through the closing summary" | user |
| Container restart before the build | No: "the GPU will be fine in the next 12 hours at least" (GPU check passed 02:58: 595.99.02 on driver, module and library; GPU idle) | user |

The closing summary the user confirmed: part 1 scores six recipes on one seed and the 29,232 paintings with the
guarded rule, beside the ceiling probe and the four-view sheet; part 2 runs SigLIP 2 and CLIP B/32 students (DINOv2 if
image endpoints only wins) with Qwen captions, graphs and teacher, and the new recipe on Qwen if it changes; the bar
is the strongest PercepT core arm per aspect including the matched arm, the frozen image reported; the claim test's
form (emotion, d ≥ 0.01 and lower bound > 0, 5 seeds, held, stage 1 then full unless NO-GO); ArtELingo-28 scoring of
every student on the 746 with its own images, the culture split descriptive; local 3090, about 2 GPU h and 4 to 6 CPU
h (a guess), planned to end before 11:00.

**Minor batch, accepted** (user, after seeing the recommendation; they go to the spec's "Check these" as needed):
1. Six recipes: D2, image endpoints only (ablation R7), no own-caption terms (R6), each at teacher 1/5 and 1/3.
   Teacher weight 0.25 gives the caption-free variants 1/5; weight 1.0 gives D2 1/3 (loss = buddy + own-caption +
   w·teacher).
2. One seed (seed 0) for the pick (global `held-data.md`, "Seeds"); the rule's seed ranges are fixed at 12-22's
   values (emotion 0.0046, style 0.0082, genre 0.0128); part 2 tests the pick on 5 seeds.
3. The rule reads early-stopped exports only; paired bootstrap over the 29,232 paintings, 2,000 draws; ties (equal
   mean gain to two decimals, in ranges) go to fewer changes from D2: D2 at 1/3, then no own-caption terms, then
   image endpoints only, 1/5 before 1/3.
4. A fixed-epoch export at epoch 4 beside early stopping, reported only.
5. The ceiling probe (literature note 12-20, idea 1) for all three aspects: ridge and a 4-layer MLP predict the
   instructed image from the default one; early stopping on the regression loss of the 1,489 (no labels); fit on the
   13,400; scored on the 29,232 and the old 5,000.
6. The four-view statistics sheet on the 29,232 and the old 5,000; genre labels on the 29,232 counted and one
   export's scoring time measured before the run, then the end time estimated.
- a. Order: SigLIP 2, then CLIP B/32 (a new `artelingo_full/clip_b32` store, about 10 GPU min); DINOv2 only if image
  endpoints only wins; its encoder code is built anyway.
- b. Students: the picked recipe; only the image network's first layer takes the backbone's width; Qwen default
  captions on the caption side; teacher, graphs and 2048-wide outputs unchanged; nothing tuned per backbone.
- c. If the rule picks a new recipe, it also runs on Qwen through both stages; D2's claim result stays beside it.
- d. Bars: the claim test's PercepT entries plus one matched arm per backbone (the claim test's Qwen Stage 1 topics,
  saved at both scales; Stage 2 on the backbone's image); the claim's bar is the strongest entry per aspect; the
  frozen backbone image is reported.
- e. Pass: the claim test's form on emotion; style and genre reported against all three bars; 5 seeds per student
  and arm; stage 2 unless S1-NO.
- f. Local 3090; each job short; plan to end before 11:00; no heavy run over 1 hour on weekdays 10:00 to 19:00.
- i. ArtELingo-28's own image files (its stores) for all 746 paintings; its Qwen image differs from ours on the 248
  shared ones (mean cosine 0.934, cause not checked), stated in the report.
- ii. Emotion: "other" merged into "something else", plurality over nine classes, ties to the smallest code (729
  unique, 17 ties).
- iii. Genre: its nine (no "nude painting"); style: all 27, which match ours.
- iv. The instructed bar on ArtELingo-28: three instructed Qwen images of the 746 (about 3.5 GPU min).
- v. Every part 2 student, and Qwen, scored on the 746; DINOv2 would extract its 746 images.

## 3. Where things stand (facts for the spec)

- Claim test bars on held emotion (B1-own, core's own assignment, the strongest entry): stage 1 0.0855 (D2 0.1001,
  d +0.0146 [+0.0093, +0.0157]); stage 2 0.0874 (D2 0.1008, d +0.0133 [+0.0081, +0.0146]). The claim test's held
  outputs of B1 to B5 are saved under `src/test/20261020_claim_test/outputs/` and can be reused (frozen).
- Saved Stage 1 artifacts, seeds 0 to 4, both scales: `qwen_qwen__s*` (P3's topics, for the matched arm) and
  `clip_roberta__s*` (P1/P2's topics), under `outputs/percept/core/stage1/` and `outputs/stage2/full/percept/core/stage1/`.
- PercepT core's best arm per aspect on development (12-15): P3 for emotion and genre (0.077, 0.436), P2 (CLIP +
  RoBERTa topics, CLIP global image) for style (0.229). P2 was not in the claim test; agent default to settle in the
  spec: add P2 as an entry at both stages for the style report (cheap; its topics are saved).
- Matched-arm estimate for CLIP (not measured): about 0.067 / 0.223 / 0.405, below a on all three; CLIP's frozen
  image 0.065 / 0.300 / 0.412 (12-15). SigLIP 2: no bar measured.
- Development references (12-22, old 5,000): D2 0.0962 / 0.3697 / 0.4540; image endpoints only +0.0057 / +0.0248 /
  −0.0086; no own-caption terms +0.0055 / +0.0180 / +0.0063; instructed Qwen image 0.0929 / 0.3952 / 0.4956.
- Stores: SigLIP 2 (1152) covers all 61,402 paintings and captions and ArtELingo-28; CLIP B/32 (512) has the legacy
  caption-indexed store and ArtELingo-28; DINOv2-large (1024): weights cached, encoder not wired; Qwen default 11.5
  images/s, SigLIP 2 74.7, CLIP 106 (grill facts B4).
- Genre labels on the 29,232: not checked (the old 5,000 had 4,053).
- Disk: system disk 248 GB free (03:00); the claim test's outputs took 7.3 GB, the ablation's 4.6 GB.

**Agent defaults for the spec** (this chat's proposals; the spec chat may refine them, flag the consequential ones):
- Types: part 1's pick is a confirmation with a selection rule (decision A); the probe and the sheet are
  exploration; part 2's code is engineering; part 2's held tests are claim tests (stricter than decision A's
  "incremental", allowed by `spec_experiment.md`; flag it). Claim test needs the rule reviewed by a fresh Opus
  reviewer (C7; the claim test used Opus `xhigh`) and committed before any code.
- Run order: SigLIP 2, CLIP B/32, then Qwen with the new recipe (only if the recipe changed), then DINOv2 (only if
  image endpoints only wins), so DINOv2 is dropped first if time runs short.
- Regression checks (C12): part 1 reproduces 12-22's R0, R6 and R7 seed-0 exports on the old 5,000 bit for bit;
  part 2's width-opened code with Qwen input reproduces the claim test's K1 seed-0 stage 1 held output and B1 seed 0
  bit for bit.
- Each backbone's verdict is its own sentence; no multiplicity correction; every backbone is reported, pass or not.
- Prior guesses (C8): SigLIP 2 stage 1 d about +0.003 to +0.012; CLIP about −0.005 to +0.008.
- DINOv2 encoder: the pooled CLS output, L2-normalized; a new canonical store for 61,402 paintings and the 746.
- ArtELingo-28: the full-split students when they exist, else stage 1's; the culture split per language, descriptive.
- Timing: Sunday 10-25 is a weekend day. The run chat estimates the end at start; if part 2 would end after 11:00 it
  chooses by itself (for example part 1 first, part 2 in the evening) and records why (`shared-resources.md`).
- New round folder and module prefix (for example `src/test/20261025_recipe_backbones/`, prefix `rb_`); the spec's
  next report sequence name is 12-24.

## 4. Open points for the next step

None for the user. The spec chat:
1. Writes the spec, `docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md` (status "approved 2026-10-25
   03:02 through the closing summary"), and the decision rule in the new round folder, committed before any code.
2. Runs, in parallel: the rule review (a fresh Opus worker, C7) and the cold check (a Sonnet `high` worker: do the
   spec and the rule say what §2 records; the user's approval depends on it). Applies the fixes, commits, pushes.
3. Does not ask about a restart (answered: no). In night mode, runs the GPU check only.
4. Opens the build: `loop-next <build handoff> --label "r10 build 16" --unattended`; commits `chats.tsv` and
   `briefs.md`, pushes, and stops.

## 5. Code and pitfalls

- Change no code in `src/test/20261021_d2_ablation/` or `src/test/20261020_claim_test/`: their `*_row.json` and
  verdict files bind code hashes. A new round folder and module prefix.
- Every D2 and PercepT core path asserts 2048-wide inputs (f2 §5); part 2 opens only the image network's first layer
  (and PercepT Stage 2's input), while the teacher and outputs stay 2048.
- Joins are positional (C6); ArtELingo-28's store uses `row_id`, ours `sample_id`.
- `held-data.md`: no selection step reads held paintings; part 1 reads development paintings only.

## 6. State at handoff

- Running: nothing of this chat. Run mode: night (set 02:58 by this chat).
- Uncommitted: nothing after this commit (`.scratch/` is gitignored).
- On disk over 1 GB: nothing new.
