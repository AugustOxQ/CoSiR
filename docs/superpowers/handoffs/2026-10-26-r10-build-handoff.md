# Handoff: build the relational genre round from its approved spec (tickets, build, ticket reviews, final whole-branch review, merge into main), then open the run chat

## Brief
- Done: spec and decision rule of the relational genre loop, reviewed (Opus `xhigh`, pass after fixes) and approved by the user at 19:57.
- State: held genre, 9,949 paintings: the recipe 0.4534 against the instructed Qwen genre image 0.5062. GO: level within 0.0128; partial: half the gap closed.
- Next: tickets from the spec and rule, the build per `matt-chain.md`, the final review, merge into main, the run handoff.
- Blocked: nothing. No container restart (user, 20:01; the GPU check passes).

> Written 2026-10-26 19:59 by "r10 decide 25". Loop step reached: step 6 done (spec approved); step 7 (build) next.
> Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md) (its "Now" names the spec).

## 1. Read in this order

1. This handoff.
2. The spec, [2026-10-26-relational-genre-design.md](../specs/2026-10-26-relational-genre-design.md), from §1 on
   (skip the brief), and its rule in full, `src/test/20261027_relational_genre/DECISION_RULE.md` @ 2e9529a (it
   governs where they differ). The rule inherits 12-25's rule (stage iii: §2.3, §3.3, §4.3), 12-24's rule (§3.1; part 2,
   §3.2, §4.2) and the claim test's rule (§3); read the sections it cites.
3. The base code (§5 below names the files and what each gives).
4. `~/.claude/references/matt-chain.md` and `~/.claude/rules/agent-routing.md` (Codex first when
   `codex-usage --room` allows; ticket reviews and unit tests for verdict code only).

Skills: none beyond those `matt-chain.md` names.

## 2. Decided by the user, not to be reopened

- The spec's §4 rows D1 to D20 and the minors M1 to M10, decided in the r10 decide 23 and 24 chats (2026-10-26, the
  grill's answers 18:58 to 19:20), and the approval of the brief at 19:57 (r10 decide 25).
- Run mode is "day" at the time of writing; the user or the main chat may switch it to night.

## 3. Where things stand

- The rule is committed at 2e9529a before any code (C7); the spec at 074664a ("approved"). The round folder
  `src/test/20261027_relational_genre/` holds only the rule. Module prefix `rg_`; outputs go on `/data/SSD2`
  (as 12-25: `/data/SSD2/cosir_r10/20261027_relational_genre/outputs`, linked from the round folder).
- Frozen inputs on disk (checked 2026-10-26 19:59), all under `src/test/20261025_encoders_genre/outputs/` (EG) or
  `src/test/20261025_recipe_backbones/outputs/` (RB):
  - EG `stage_iii/train/GR-8/` (seeds 0 to 3) and `stage_iii/train/R6-5/` (seeds 1 to 3); `stage_iii/score_first.json`,
    `score_all.json`, `score_fresh.json`, `reported.json`.
  - RB `part1/train/R6-5/` (seed 0 exports on P); `part2/full/` (K4's full-scale weights and held outputs);
    `part2/reported_stage4.json` (K4's held margins "T", "a_B1_to_B6", "b_matched").
- Disk: system disk 258 GB free; `/data/SSD2` 1.9 TB free. Codex at 19:58: five-hour window 100% left (reset 00:58),
  weekly 96%.

## 4. Open points for the build (agent work; nothing for the user)

a. Tickets from the spec and rule with `to-tickets` (hard rule 6). The coverage line names every rule ID: VA1 to VA4,
   EQ, the arms GR-16, GR-32, the climb (cap 256), GN-8, GS-0.02, GS-0.01, GQ-8, the guard, failed runs, A-NEW and
   A-SAME with v*_A; VB1 to VB4, GR-2 and GR-4 at seeds 1 to 3, C-HOLDS (and C-QUAL, C-FAILS), B-PICK and B-NONE with
   v_held; VC1 to VC4, G at full scale (seeds 0 to 4) with its held inference, d_T, d_R, X, G_half, PC, G-GO, G-MM,
   G-PART and G-NO; the weights rule (§4.3 "Weights"); the loss-scale run and the diagnostics of spec §3.6.
b. Verdict code (ticket reviews and unit tests, spec §9): the three switches and the anchors loss with EQ, the climb,
   the seed-0 winner, the stage B statistics and pick, the held scorer with d_T and d_R, the re-derivations.
c. Plumbing (controller diff read and one end-to-end smoke): the queues per stage, the loss-scale run's per-step
   logging, the full-scale training and held inference of G, the verdict and pick files.
d. Short GPU work during the build (smokes, VA1, EQ, VB1, VC1) is allowed under the lock; the whole run is light
   (spec §9), so the weekday and 11:00 rules are not touched.
e. The run chat's handoff: ticket list, queue order (spec §3.5: VA1 and EQ, then A with GR-16 and GR-32 first and the
   climb, then the other arms; B; VC1 and C), the commands, the end estimate (reports by 2026-10-27 22:00). Then the
   GPU check and `loop-next <handoff> --label "r10 run" --unattended`.

## 5. Code and pitfalls

- The base to extend (change no `eg_`, `rb_` or `ms_` module in place; copy into `rg_` modules):
  - `src/test/20261025_encoders_genre/eg_train3.py`: `switched_head_loss`, the genre switches m, lam, w; today
    m > 0 is asserted, so GN-8 (m = 0) needs the new trainer.
  - `src/test/20261019_model_sheet/ms_train.py`: `distill_loss`, with 0.05 hard-coded on both rows; the
    teacher-side temperature changes only the teacher's row (the student's stays at 0.05).
  - `eg_variants.py` (ladders, fresh seeds, T_g, the seed ranges), `eg_score3.py` and `eg_vg2.py` (stage iii's
    scorer and re-derivation).
  - `src/test/20261025_recipe_backbones/rb_part2_report.py` (the held margins: its margin "T" code is the model for
    d_T, "b_matched" for d_R) and `rb_rows.py` (R6-5).
- Three new switches on the genre head only: the direct term off (m = 0), the teacher-side temperature, the anchors
  loss with two anchor sets A_s and A_t (rule §2.1) and its equivalence test EQ against `distill_loss` 'relational'
  on three real fit batches. A batch is 1,024 pairs = 2,048 rows (`ids = cat(a, b)`), same-painting rows masked.
- The regression gates: VA1 (switches off at seeds 0 and 1 bit for bit against 12-24's `part1/train/R6-5` and 12-25's
  `stage_iii/train/R6-5/s1`; the GR-8 switches bit for bit against `stage_iii/train/GR-8/s0`; the scorer against
  `stage_iii/score_all.json`), VB1 (hashes from `score_fresh.json` `identity.img_sha256`; its `effects` within
  1e-6), VC1 (the full-scale K4 weights and held outputs bit for bit; the held scorer against
  `reported_stage4.json`'s K4 genre margins "T", "a_B1_to_B6", "b_matched"). The climb's re-derivation runs before
  each next rung (VA2).
- The re-derivations VA2, VB2, VC2 by separate code, never given our numbers: Codex when `codex-usage --room` says
  room (`~/.claude/references/codex-jobs.md`), else a Sonnet `high` worker.
- The held variant is written to the stage B verdict file before any full-scale run of a candidate; the held read
  gets a row in `docs/superpowers/held_ledger.md` (the run chat).
- Round folders go on `sys.path`: the prefix `rg_` (taken: `eg_`, `rb_`, `ab_`, `ms_`, `r4_` to `r9_`).
- Unattended: delete nothing; list items in `.scratch/pending_deletions.md` (empty now). GPU under the lock.
- Codex: a resumed session re-reads the whole session; a fresh session is far cheaper (12-24 run).

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing after this commit (`chats.tsv` and `briefs.md` are committed right after `loop-next`).
- On disk over 1 GB: unchanged (the rounds' outputs on `/data/SSD2`, kept; the shared feature stores).
