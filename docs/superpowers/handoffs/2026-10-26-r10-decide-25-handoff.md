# Handoff: r10 decide 25: paste the spec's brief for the user's approval, then record it, write the build handoff and cut to the build chat

## Brief
- Done: the grill of the genre loop (three key questions and the minors answered), the spec (draft) and the decision rule, reviewed by a fresh Opus `xhigh` worker (pass after fixes, all applied) and committed (rule 2e9529a, spec bcc4811); glossary entries "relational term" and "teacher anchors".
- State: held genre, 9,949 labelled paintings: the recipe 0.4534 against the instructed Qwen genre image 0.5062; the bar: level within 0.0128 (GO) or at least half the gap closed over the recipe (partial).
- Next: paste the brief for the user's approval; on the yes, mark the spec approved, rewrite the direction log's Now, write the build handoff, ask about the container restart, open the build chat with `--unattended`.
- Blocked: nothing (the approval needs the user; the decide 24 chat cut at its context threshold).

> Written 2026-10-26 19:51 by "r10 decide 24". Loop step reached: step 6 (spec and rule written, reviewed and
> committed), before the user's approval of the brief. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md) (its "Now" holds decisions A and B).

## 1. Read in this order

1. This handoff.
2. The spec's brief, `docs/superpowers/specs/2026-10-26-relational-genre-design.md` (the "Brief" block only). Paste
   it into chat with the file link; the user approves from the brief (`~/.claude/rules/spec-templates.md`, item 8).
   Read §4 only if the user asks what was decided.
3. `~/.claude/references/loop-chats.md`, "How to cut" (the container question, `loop-next --unattended`) and
   `~/.claude/templates/handoff.md` (the build handoff).
4. Only if the user asks a question about the bar or the statistics: the rule,
   `src/test/20261027_relational_genre/DECISION_RULE.md`, §3.2 and §4.

Skills: none.

## 2. Decided by the user, not to be reopened

- Everything in the spec's §4 (D1 to D20), decided in the r10 decide 23 and 24 chats on 2026-10-26 (the grill's
  answers 18:58 to 19:20): the held bar as option (c) with X = 0.0128 (user, after seeing the recommendation);
  weights 2, 4, 8 and a new seed-0 winner compete on fresh seeds; the ladder climbs while it rises, cap 256; the
  anchors arm is a full candidate behind the equivalence test; the minors M1 to M10 as recommended; no html-plan
  page. Not to be re-asked.
- The two glossary rows (user's yes, 19:20), already in `GLOSSARY.md`.
- The stage report, its page and its post stand (decide 23); aspect-named retrieval for the 11-10 gate waits
  (decide 20); the write-up of the encoder claims comes after this loop.

## 3. Where things stand

- Committed and pushed: the rule (2e9529a, header revised after the review; its hash is in the spec's header), the
  spec (bcc4811, "Status: draft"), the glossary. The review: four must-fix (the equivalence test's form, the file
  that holds weight 8's seed-0 values, a stage B regression gate, one reading of "rises"), eleven should-fix and ten
  nits, all applied; the spec's §9 and the rule's header record it.
- The loop (spec brief): stage A, a sweep on seed 0 (weights 16 and 32, higher rungs while the newest rung is
  guarded and above every other guarded rung, cap 256; the relational term alone at weight 8; teacher temperature
  0.02 and 0.01 at weight 8; teacher anchors at weight 8); stage B, fresh seeds 1 to 3 (weights 2, 4, 8 and a new
  winner; the confirmed arm with the largest three-seed mean goes to held); stage C, the held genre claim test (GO:
  level with the instructed Qwen genre image within 0.0128; partial: at least half the held gap over the recipe,
  interval above zero; margin met and NO-GO as the other outcomes).
- Cost (spec §9): about 1 GPU hour and 3 CPU hours, light work; build tonight, the run on 2026-10-27, reports by
  its evening.

## 4. Open points for the next step

- **The approval (key, the user).** Paste the brief. On "yes": edit the spec header to "Status: approved
  2026-10-26 HH:MM (from the brief, r10 decide 25)", commit. On a change request: amend the spec, and if the rule
  changes, amend the rule before any code and update every place the old text appears in one commit (hard rule 5),
  then paste the brief again.
- **After the approval, in this order:**
  a. The direction log's "Now" paragraph: replace its sentence "The grill and the spec run in the r10 decide 24
     chat." with one or two sentences on the decided loop (the spec's path, the rule's hash 2e9529a, the bar, the
     build tonight and the run on 10-27); commit.
  b. The build handoff, `docs/superpowers/handoffs/2026-10-26-r10-build-handoff.md`, from the template: the read
     list (the spec from §1, the rule in full, `~/.claude/references/matt-chain.md`, the base code named in §5
     below), the tickets' coverage of every rule ID (hard rule 6), the round folder
     `src/test/20261027_relational_genre/` with the prefix `rg_`, the pitfalls of §5 below; §6 state. Commit and push.
  c. The GPU check and the container question (`~/.claude/rules/container.md`, "The GPU and a container restart";
     `loop-chats.md`, "How to cut", item 3): give the check's result, name what a restart would end, ask. On a yes:
     push, do not open the build chat, give the user the `loop-next` command to run after the restart, and stop.
  d. `loop-next docs/superpowers/handoffs/2026-10-26-r10-build-handoff.md --label "r10 build" --unattended`;
     commit `chats.tsv` and `briefs.md`; push; say which tab continues; stop.
- The weekday rule and the 11:00 rule are not touched (light work, spec §9); the run chat re-estimates at its start.

## 5. Code and pitfalls (for the build handoff)

- The base to extend: `src/test/20261025_encoders_genre/eg_train3.py` (`switched_head_loss`, the genre switches m,
  lam, w; today m > 0 is asserted), `src/test/20261019_model_sheet/ms_train.py` (`distill_loss`: 0.05 hard-coded on
  both rows), `eg_variants.py` (ladders, fresh seeds, T_g, the seed ranges), `eg_score3.py` and `eg_vg2.py` (stage
  iii's scorer and re-derivation), `src/test/20261025_recipe_backbones/rb_part2_report.py` (the held margins; the
  margin "T" code is the model for d_T, "b_matched" for d_R), `rb_rows.py` (R6-5).
- Three new switches on the genre head only: the direct term off (m = 0), the teacher-side temperature (the
  student's row stays at 0.05), the anchors loss with two anchor sets A_s and A_t (rule §2.1) and its equivalence
  test EQ against `distill_loss` 'relational' on three real fit batches. A batch is 1,024 pairs = 2,048 rows
  (`ids = cat(a, b)`), same-painting rows masked.
- The regression gates: VA1 (switches off at seeds 0 and 1 bit for bit against 12-24's `part1/train/R6-5` and
  12-25's `stage_iii/train/R6-5/s1`; the GR-8 switches bit for bit against `stage_iii/train/GR-8/s0`; the scorer
  against `stage_iii/score_all.json`), VB1 (hashes from `score_fresh.json` `identity.img_sha256`; its `effects`
  within 1e-6), VC1 (the full-scale K4 weights and held outputs bit for bit; the held scorer against
  `reported_stage4.json`'s K4 genre margins "T", "a_B1_to_B6", "b_matched"). The climb's re-derivation runs before
  each next rung (VA2).
- The re-derivations VA2, VB2, VC2 by separate code, never given our numbers: Codex when `codex-usage --room` says
  room (`~/.claude/references/codex-jobs.md`), else a Sonnet `high` worker.
- The held variant is written to the stage B verdict file before any full-scale run of a candidate; the held read
  gets a row in `docs/superpowers/held_ledger.md` (the run chat).
- Round folders go on `sys.path`: the prefix `rg_` (taken: `eg_`, `rb_`, `ab_`, `ms_`, `r4_` to `r9_`). Weights per
  rule §4.3 "Weights"; delete nothing, `.scratch/pending_deletions.md` (empty now). GPU under the lock. The build
  chat opens with `--unattended`.

## 6. State at handoff

- Running: nothing (the review worker finished; no background job, GPU lock or cluster job of this chat).
- Uncommitted: nothing in the project (`chats.tsv` and `briefs.md` are committed right after `loop-next`).
- On disk over 1 GB: unchanged (the rounds' outputs on `/data/SSD2`, kept; the shared feature stores).
