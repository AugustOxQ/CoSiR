# Handoff: write the spec and decision rule of the next loop (four more encoders, backbone analysis, genre variants) from the closed grill; the user approves the brief; then open the build chat

## Brief
- Done: report 12-24 read (decision C), wrap-up (33 GB deleted), decision A and the grill; closing summary confirmed.
- State: next loop: four more encoders under 12-24's rule; a backbone analysis on development paintings; genre variants toward the instructed value 0.4989.
- Next: write the spec and the amended rule, get the rule reviewed, then the user approves the brief.
- Blocked: nothing. Day mode.

> Written 2026-10-25 17:40 by "r10 decide 20" (cut at the context meter, after the grill). Loop step reached: step 5
> done (decisions A and B made); step 6 (spec) next. Direction log entry:
> [2026-10-25 · Recipe and backbone round read](../direction_log.md).

## 1. Read in this order

1. This handoff. §2 is the grill record; the spec carries it into its §4 without asking again.
2. The direction log entry of 2026-10-25 (decisions C, A and B, with who decided).
3. The facts brief, `.scratch/r10-decide-20/facts_decide20.md` (Codex, gitignored). We checked it: the frozen AMI
   values and rule line 233 were re-read from the files. It lists every rule line and code site that names the old
   encoder list, the extraction sizes and times, the per-encoder AMI per split, and R6-5's loss terms.
4. The previous spec, [2026-10-25-recipe-and-backbones-design.md](../specs/2026-10-25-recipe-and-backbones-design.md),
   and its rule, `src/test/20261025_recipe_backbones/DECISION_RULE.md`. The new spec and rule inherit them
   ("inherits ...; changes: ...") and record only what is new.
5. `~/.claude/rules/spec-templates.md`, `~/.claude/templates/spec_core.md`, `spec_experiment.md`,
   `decision_rule_template.md`; the [constitution](../constitution.md) (C1, C6, C7, C8, C12, C13).
6. Literature note `docs/reports/literature/2026-12-20_ways_to_improve.md`, ranking rows 2 and 3 (stage iii).

Skills: none.

## 2. Decided by the user, not to be reopened (decide 20, 2026-10-25)

**Decision C** (user, before ours): "I think for SigLIP2 and Qwen thats a good result, CLIP B/32 suprisingly beging
bad." Ours and the differences are in the direction log entry.

**Decision A** (user, after seeing the recommendation, "as above"): the user's three proposals as one medium loop.

| Item | Decision | Who |
|---|---|---|
| Stage i: more encoders | claim test on held paintings under 12-24's rule, only the encoder list changed: OpenAI CLIP ViT-B/16 (`openai/clip-vit-base-patch16`), CLIP ViT-L/14 at 224 px (`openai/clip-vit-large-patch14`), SigLIP 1 So400m/14 at 384 px (`google/siglip-so400m-patch14-384`), DINOv2-large | user, after seeing the recommendation (models) |
| Stage i: the claim | one verdict sentence per encoder (option 3a); no pooled "across encoders" sentence | user |
| Paper reporting of the verdicts | open: "not really, we might exclude models that cannot pass, too soon to determine the paper writing now". Our note, recorded for the paper: a paper that leaves out failed encoders must say they were tested | user |
| Stage ii: backbone analysis | development first (option 1b): the sample-scale students (13,400 paintings, which stage i trains anyway) and PercepT arms of all seven encoders scored on the 29,232 development paintings: frozen AMI, own PercepT arm, student, lift, encoder statistics; then the held table, which chooses nothing | user |
| Stage ii: what-if | a method idea from the analysis is tested on development paintings first, and on held paintings only as a new claim test in a later loop, never in this one | user (confirmed our check) |
| Stage iii: target | the instructed Qwen genre value on the 29,232, 0.4989: "2c would be the target, but we might not be able to reach it directly from this round, so keep weights on these methods that close to the targeted number" | user |
| Stage iii: guard | a variant never wins if another aspect loses more than one seed range (answer "no" to: a variant at genre 0.4950 that loses two emotion seed ranges wins?) | user, after seeing the recommendation |
| Stage iii: "keep weights" | (i) save the trained weights of every variant within one genre seed range of the target (genre ≥ 0.4861), and (ii) two more weights for the method that comes closest (about 3 extra minutes) | user, after seeing the recommendation ("as recommended") |
| Minor points m1 to m6 (below) | accepted | user, after seeing the recommendation |
| Plan page | none ("plain") | user |
| The abstract | not now | user |
| Aspect-named retrieval (11-10 gate) | the next loop: "the retrieval is important" | user |
| Loop size | medium; step 3 skipped (the options are clear; stage i reuses a reviewed rule) | agent default (not contested) |

**Closing summary, confirmed by the user (about 17:30):**
1. Stage i (claim test, held): CLIP B/16, CLIP L/14 at 224 px, SigLIP 1 So400m/14 at 384 px and DINOv2-large under
   12-24's rule with only the encoder list changed; DINOv2 runs under the new recipe; each also gets the ArtELingo-28
   stage. One verdict per encoder; how the paper reports them is decided later.
2. Stage ii (backbone analysis, development first): the sample-scale students and PercepT arms of all seven encoders
   scored on the 29,232 development paintings: frozen AMI, PercepT arm, student, lift, encoder statistics. The held
   table is read after that and chooses nothing. A method idea from it goes to development first, and to held only as
   a new claim test in a later loop.
3. Stage iii (incremental, development, Qwen, new recipe, seed 0): five genre variants, plus two more weights for the
   method that comes closest. Target: the instructed genre value 0.4989. GO: genre reaches it; promising: within one
   seed range of it (at least 0.4861); a variant never wins if another aspect loses more than one seed range. The best
   guarded variant runs on three fresh seeds if its genre gain is at least one seed range. Weights are kept for every
   variant within one seed range of the target.
4. Not in this loop: retrieval (next loop), the abstract (not now), a held test of a genre winner (a later loop).
   Minor points m1 to m6 accepted; no plan page.

**Minor points m1 to m6** (accepted as recommended):
- m1. Stores: images and texts for CLIP B/16, CLIP L/14 and SigLIP 1; images only for DINOv2. The student reads only
  images; texts are cheap and the next loop's retrieval may use them.
- m2. DINOv2 runs under the new recipe (the old rule ran it only under image endpoints only; its missing text side
  affects neither the student nor the PercepT arm, facts brief §2). Each new encoder also gets the ArtELingo-28 stage.
- m3. Stage iii variants on Qwen, the new recipe (R6-5), seed 0, five in all: the genre teacher weight ×2 and ×4; a
  relational term toward the teacher for genre at weights 0.5 and 2 (the existing KL term); standardised regression
  through a two-layer adaptor that is discarded, one setting. Idea 2's probe interpolation is left out: it needs the
  Qwen image at test time, which breaks other encoders.
- m4. Stage ii statistics per encoder on its default image: norms, anisotropy, 20-neighbour label purity, silhouette,
  hubness. The lift-to-statistic correlation over seven encoders is descriptive only (seven points).
- m5. Glossary: add five terms that reports already use, with plain meanings: recipe, teacher share, matched arm (an
  encoder's own PercepT arm), frozen input, seed range. `GLOSSARY.md` now defines only PercepT core.
- m6. Timing: the build chat runs tonight; the run starts when the build is done if it can end by 11:00, else Monday
  2026-10-26 at 19:00 (long heavy GPU runs avoid weekdays 10:00 to 19:00).

## 3. Where things stand

All from [report 12-24](../../reports/auto/percept/2026-12-24_recipe_backbones.md) and its outputs (facts brief §6, §9).

- Held emotion AMI, full scale (12,281 paintings, 5 seeds): students SigLIP 2 0.1024, CLIP B/32 0.0885, Qwen 0.1029;
  their own PercepT arms (own reading) 0.0793, 0.0716, 0.0874 (B1); frozen images 0.0569, 0.0642, 0.0708. The bar
  was B1-own, 0.0874; gains +0.0150, +0.0011, +0.0154.
- Development, the 29,232 paintings, seed 0 (emotion / style / genre): R6-5 0.1071 / 0.3987 / 0.4674; instructed Qwen
  image 0.0969 / 0.4023 / 0.4989; default Qwen image 0.0690 / 0.2305 / 0.4623; ceiling probe, ridge 0.0961 / 0.3933 /
  0.5040; D2 0.1010 / 0.3748 / 0.4604. Seed ranges 0.0046 / 0.0082 / 0.0128.
- Costs: one R6-5 run on the sample about 85 s (three aspects); part 2 per student about 3 min (sample) and 8.5 min
  (full) training, CPU scoring about 7 min per stage, report queues about 1 h per stage; extraction of the existing
  stores: CLIP B/32 images 310 s, SigLIP 2 822 s, Qwen 5,350 s on 61,402 images (facts brief §3, §4).
- Codex: five-hour window 90% left at 16:37 (reset 21:26), weekly 93%.

## 4. Open points for the next step

a. **Write the spec** in `docs/superpowers/specs/` and the **rule** in a new round folder (a new module prefix; the
   report's sequence name is 12-25). Stage types: i claim test; ii, development first, as an exploration or a
   confirmation (agent default, listed in "Check these"; the grill settled its data and what may follow from it); iii
   incremental with the guarded rule toward the target above, written in the rule before code (C7).
b. **Rule details** left as agent defaults: IDs for the new encoders and their matched arms and frozen images; the
   amended DINOv2 trigger; a prior per new encoder and for stage iii (C8); the stage iii selection rule (metric, tie
   band, tie-break); the fresh seeds of the stage iii winner; the regression checks (C12) and re-derivations (C13).
c. **Reviews before approval:** the rule by a fresh Opus `xhigh` worker (C7), then a cold Sonnet `high` check of spec
   and rule against §2 of this handoff (as in decide 19).
d. **Approval:** paste the brief in chat with the file link; the user approves from it.
e. **Then cut to the build chat:** run the GPU check and ask the user about a container restart (`loop-chats.md`,
   "How to cut"), then `loop-next <handoff> --unattended --label "r10 build 21"`.
f. **For the paper, not this loop:** how to report encoders that fail (§2).

## 5. Code and pitfalls

- 12-24's code: `src/test/20261025_recipe_backbones/` (prefix `rb_`, code commit 08cab72). Change no round code in
  place; a new round needs a new module prefix (round folders go on `sys.path`).
- `scripts/extract_features.py` knows only `clip_b32`, `qwen3vl_emb_2b`, `siglip2_so400m`, `dinov2_large`, and forces
  texts for every encoder except DINOv2 (facts brief §2). No `dinov2_large` store exists yet for ArtELingo; the HF
  cache (`/data/SSD2/HF_home`) has `facebook/dinov2-large` but not CLIP B/16, CLIP L/14 or SigLIP 1 (downloads needed).
- The DINOv2 guard requires an R7 pick (`rb_student.py:99–104`, `rb_run.py:189–203`).
- The four-view sheet's loader is Qwen-specific and asserts four named views (`rb_sheet.py:135,196–228`); per-encoder
  statistics need a default-only path. Its kernels take any width.
- The relational KL term exists (`src/test/20261019_model_sheet/ms_train.py:102–116`); `src/test/20261021_d2_ablation/ab_rows.py:43–45`
  requires direct distillation for switched recipes.
- Stage ii reads `src/test/20261025_recipe_backbones/outputs/part2/sample/weights`; stage iii compares against
  `outputs/part1/train/R6-5` and `R0-5`. Both are kept until this loop's wrap-up (disk ledger).
- Codex: a resumed session re-reads the whole session; a fresh session is far cheaper (12-24 run).

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: the 12-24 outputs, 24 GB on the system disk (kept, ledger rows of 2026-10-25); 219 GB free.
