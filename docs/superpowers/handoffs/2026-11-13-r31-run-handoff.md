# Handoff: r31 run: r29's stage 2 under A1 (part a from 19:00, the picks and SC9 committed and re-derived, part b), the ladders, then the auto and user-read reports

## Brief
- Done: the r31 build: tickets 01 to 03 merged into main (f4ea768f); final review fixes merged; scoped re-review pass.
- State: SC9 replay gates pass (SigLIP 2 and CLIP L/14 reproduce r28's valpick.json within 1.2e-16); no stage 2 output yet.
- Next: GPU check, then `r29_run_stage2.sh a --amendment A1` at 19:00 or later; the picks and encoder_pick.json committed and re-derived by Codex; then part b.
- Blocked: nothing.

> Written 2026-11-13 17:04 by "r31 build 2". Loop step reached: step 7 done; this chat does steps 8 and 9. Direction log
> entry: [2026-11-13 · r29 read](../direction_log.md). Run mode: night (`run-mode`).

## 1. Read in this order

1. This handoff, all of it.
2. The r29 spec, `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`, from §1 (skip the brief): §4 rows
   D3, A1, A2, D21 to D24; §3.4 "Order and C13"; §3.5 (stages; row 2a); §6 (SC4 to SC7, SC9 and its bullet); §9 budget.
3. The first build handoff, [2026-11-13-r31-build-handoff.md](2026-11-13-r31-build-handoff.md), §2 and §6 (what this
   chat does after the run).
4. The runner headers in `src/test/20261112_siglip_only/`: `r29_run_stage2.sh` (pieces, order D24, estimates) and
   `r29_run_common.sh` (flags). Both take `--dry-run`.
5. The decision lines, `src/test/20261112_siglip_only/decisions.md` (rows from 2026-11-13 16:15 are this build's).
6. `~/.claude/references/run-planning.md`, `loop-chats.md` (unattended chats), `codex-jobs.md` (the C13
   re-derivations). At step 9: `~/.claude/templates/auto_report_template.md`, `results_brief.md`,
   `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`.

Skills: `cluster-run` only if DAS6 is used (not planned); `diagnosing-bugs` for a crash.

## 2. Decided by the user, not to be reopened

- A1 (2026-11-13 14:40, user, after seeing the recommendation): stage 2 runs although SC2 failed; `records/pass_rule.json`
  stays "fail" and is never edited. Place (15:25, the user with the co-authors): K* is an ablation or supplementary
  row, not the main model ("the fewer numbers we change now the better").
- A2 (14:40, user, after seeing the recommendation): the encoder pick among the SigLIP-only models on validation
  (SC9); its rule is r28's (agent default, spec §6).
- All of stage 2 runs (E2, E3, MMArt, photo model F1, ablation R1 to R4; order D24), on the local GPU from 19:00 (D7).
- The spec's §4, D1 to D25 (2026-11-12; "who decided" per row there).

## 3. Where things stand

- Code: main at f4ea768f (integration branch `r31-stage2` merged). Round folder `src/test/20261112_siglip_only/`;
  outputs `/data/SSD2/cosir_r29/20261112_siglip_only/` (5.8 GB: K*'s models and exports, which stage 2 reads).
- r29 stages 0 and 1 are done and reported (auto report 2027-01-18). K* = `siglip1_prefix`. SC2 fail (text group
  0.2549 against the bound 0.2573); SC3 level 1.
- SC9 replay gates (committed, `records/gates/sc9_replay_*.json`), both pass at try 1 on r14's c* validation exports:
  SigLIP 2 0.3855510963 (largest difference 1.1e-16), CLIP L/14 0.3756306625 (5.6e-17), tolerance 1e-6. The runner
  skips them.
- Stage 2 dry runs (controller and reviewers, 2026-11-13): part a with `--amendment A1` lists the replays (skipped),
  then per encoder store gate, training, `export-val`, `image-groups` (CPU, background), `picks`; MMArt; the photo
  model; the ablation; `encoder-pick` last. Without the flag a real run refuses. Every Python step carries
  `R29_AMENDMENT=A1`.

The tickets (for information; spec IDs and the coverage line in `.scratch/r31-stage2/issues/00-index.md`):

| # | Title | Blocked by | Spec IDs | Verdict code |
|---|---|---|---|---|
| 01 | The A1 flag: stage 2 on the fail, only with a committed A1 record | none | A1, D3 (as amended) | no |
| 02 | SC9: the encoder pick among the SigLIP-only models on validation | none | A2, SC9 | yes |
| 03 | Stage 2 wiring: part a runs SC9; part b waits for its record; the sheet's SC9 section | 01, 02 | A1, A2, SC9, §3.5 row 2a, C13 order | no |

Reviews: ticket 02 (Opus `high`) pass; final whole-branch review (Opus `xhigh`) found two must-fixes (the sheet took
A1 from the environment; the SC9 writers had no stage 2 gate and no A1 line) and four should-fixes; one fix wave
(Codex, 8 fixes, 424 tests pass); scoped re-review (Sonnet `high`) pass (each new test fails without its fix; 424 tests pass on main). Summary:
`.scratch/r31-stage2/notes/final_review.md`.

## 4. What this chat does

1. **At 19:00 or later** (Friday; the runner refuses weekdays 10:00 to 19:00, D7): the GPU check
   (`container-ops.md`), `res-queue status`, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`.
   Estimate the end (part a plus part b about 6 to 9 hours, spec §9; aim to end GPU and CPU use by 11:00 Saturday,
   `run-planning.md`).
2. **Part a**, from the main checkout: `cd /project/CoSiR/src/test/20261112_siglip_only && bash r29_run_stage2.sh a
   --amendment A1` in a background Bash. Do not wrap the runner in `res-queue` (it queues each step itself) and do not
   set `R29_AMENDMENT` yourself (the runner refuses a caller's value and sets it from the flag).
3. **After part a:** commit the records it wrote with `git add -f` (`image_picks_*`, `mmart_picks_*`, the photo picks,
   `image_groups_*`, `encoder_pick.json`, the gate records). Codex re-derives every pick and range and
   `encoder_pick.json` from the rule and the data, never our numbers (C13; `codex-jobs.md`); compare at the spec's
   tolerance (D23); decision lines for any difference.
4. **Part b:** `bash r29_run_stage2.sh b --amendment A1`. Then Codex re-derives every test cell (C13), then the ladders
   (SC4 per encoder, SC5, SC6), the ablation sheet (SC7) and `r29_ladder.py sheet`, each with `R29_AMENDMENT=A1` in the
   environment (the ladder writers need it; the sheet reads A1 from the committed record).
5. **Step 9:** the auto report and the user-read report (results brief first; A1 flagged "after results"; K* written
   as an ablation or supplementary row; SC9 in the spec's wording: "Among the SigLIP-only models, <encoder> did best on
   the validation split (x against y and z)", with the margin, seeds ahead and close call). The spec's time box is
   Saturday 2026-11-14 12:00; pieces not done by then are reported as not run (D22).
6. Then open the next loop's decide chat (`loop-chats.md`, "Unattended chats", last item). r30 (a small check from
   saved outputs) runs after r31's runs end (direction log "Now"): say in the brief when the GPU is free.

## 5. Code and pitfalls

- Records are gitignored (`*.json`): commit with `git add -f`. Never edit `records/pass_rule.json` or
  `records/amendment_A1.json`.
- SC9 steps are fast: each replay took 1.5 minutes (estimates now 10 minutes). `image-groups` and `encoder-pick` write
  once (`--force REASON` keeps the old record in a side file); `encoder-pick` refuses unless both replay gates pass,
  both group records exist with the default root, and K*'s groups equal `pass_rule.json`'s.
- If E2 or E3 fails in part a, there is no `encoder_pick.json`, and part b skips both encoders' test steps (decision
  line, "maybe" on which SC4 cells exist). SC9 is then reported as not run; the other pieces go on.
- The C13 refusals (picks and `encoder_pick.json` committed before test steps) are in the runner only: run part b
  only through the runner, never `r29_encoders.py cells` or `export --kind held` by hand.
- `r29_image_cells.py` and `r29_photo.py store|gate-viii` have no stage 2 gate when called by hand (their inputs come
  only from gated training): call them only through the runner.
- A crash: D21 (one same-seed rerun, then a crash record; the other pieces go on). Debug code bugs only, with
  `diagnosing-bugs`; the spec's debug budget holds.
- Codex's untracked notes in `src/test/20261112_siglip_only/rederive/` stay out of git.

## 6. State at handoff

- Running: nothing (queue idle). GPU check: passed at 17:04 (driver 595.99.02; library and kernel match; container started 2026-11-10 17:08).
- Uncommitted: nothing of the build.
- On disk over 1 GB: nothing new (r29 outputs 5.8 GB, disk ledger).
- Pending deletions (listed in `.scratch/pending_deletions.md`): worktrees `/project/CoSiR-r31`, `-r31-t01`, `-r31-t02`,
  `-r31-t03`, `-r31-fix` and their branches (local and remote `r31-stage2`, `r31-t02`; local `r31-t01`, `r31-t03`,
  `r31-fix`); the tracker `.scratch/r31-stage2/` goes at the wrap-up. The user still runs one approved deletion: the
  two remote r29 branches.
