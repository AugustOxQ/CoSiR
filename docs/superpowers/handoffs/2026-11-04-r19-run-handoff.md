# Handoff: r19 run chat: follow the run sheet from parity and node402's jobs to the auto report

## Brief
- Done: tickets 01 to 14, final review, fix wave and re-review; build merged into main (dd7e7c0b), 499 tests pass, smoke passes stages 0 to 4.
- State: review found two blockers, both fixed; every SC bar equals §6; r14's and r15's 117 stored outcomes reproduce.
- Next: run chat starts parity and node402's jobs from worktree r19-int, then follows the run sheet.
- Blocked: nothing. The local GPU is held by r18 until about 04:08; node402 does not wait for it.

> Written 2026-11-04 02:23 by tab "r19 build 5" (cut at the 200k context meter, after the merge). Loop step reached: step 7 done,
> step 8 next. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The user's reading of
> the experiment overview" and "Decided (A)".

## 1. Read in this order

1. `src/test/20261103_past_picks/pp_run_sheet.md`: the run's order and commands. It is binding.
2. `/project/CoSiR/.scratch/r19-past-picks/notes/controller_notes.md`: from "From build 3" on (probe, node, gate files).
3. `/project/CoSiR/.scratch/r19-past-picks/notes/final_review_findings.md`: what the review found, what held, what it did
   not reach.
4. The spec [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md): §3.5, §6, §8, §9 (deliverables).
5. `~/.claude/references/codex-jobs.md` before C13; `~/.claude/rules/report-kinds.md` before the report.

Skills: cluster-run.

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D25) and A1. The run's end time: CPU scoring may run to about 13:00 on 11-04; GPU work ends
  by 10:00 on a weekday. See the r19 build handoff §2.

## 3. Where things stand

**Run from `/project/CoSiR/.claude/worktrees/r19-int`** (branch `r19-past-picks`, at the merge dd7e7c0b). Its HEAD subject
carries `cluster run`, which `cluster sync` needs, and `res/r19_pp` already links to OUT. Main moves with the r17 and r18
chats, so do not sync from main. `PP.REPO` is this worktree, so stage 1's record commit lands on `r19-past-picks`. Merge
it into main with the report.

**Checks on the merged code:** full round suite 499 passed, 0 failed (24 min). `pp_run_smoke.py` stages 0 to 4 all passed
(tree `/tmp/claude-0/-project-CoSiR/bade26c5-7a42-4fc8-81cd-8d7cd60d4e18/scratchpad/smoke_merged`). The re-review (Sonnet
`high`) passed, three sabotages caught.

**The final review.** Three Opus `xhigh` passes: the first stopped at the context meter, two narrow ones finished the rest.
- Blockers, both fixed: stage 3's argv crashed outside smoke (`pp_probe cells --enc`), which would have voided SC15 for all
  seven. Gate (v) read `OUT/matched_sample`, which no step created.
- `OUT/matched_sample` was copied at 01:13 from build 3's scratchpad: 9 files, 6.8 MB, weights sha256 equal to the stored
  run's. The controller accepted this local rerun (spec says node402): record it in the run log. Do not rerun it.
- Fixed failure paths, so one lost input voids only its own cells (§3.5): X's photo failure, a failed matched arm, a missing
  SemArt store, a missing SemArt `genre_word` store, one encoder's missing heads in the probe.
- Also fixed: gate (iii) now has one file per part; the stale check covers `cells_cost`; gate (vii)'s r14 result splits by
  path.
- Held by runs: 137 stand-in cells through the real writers into `decide`; each gate flipped alone against §3.5; r14 57 of
  57 and r15 60 of 60 stored outcomes reproduced; `collect_gates` flipped per gate with 0 misses; all `tasks()` commands
  parse outside smoke; every SC bar equals §6.

**Choices of this chat that shape the run (agent default; the report shows them):**
- **`pp_void_scope.py`** is new. A scorer drops SC3, SC9, SC13 when `runs.X.matched` or `gates.v.X` failed, and SC14 when
  the `genre_word` gate failed, reading `pp_outcome.Voids`. Any other missing input still raises.
- **`semart_set.json`** gains `missing_stores`; this replaces the assert the controller notes asked for, since `i_store`
  and `i_semart_genre_word` fail closed.
- **`pp_run.py 3 --enc X`** no longer passes `--enc` to the probe outside smoke; the probe builds every encoder with passes.
- **`r17_metrics`** (ticket 14, D13): NMI, ARI, ACC, cluster counts at K = 20 and the class count; held sets at K = 20 only;
  r17's count sweep left out (about 1,400 more fits). Stage 4's diagnostics estimate is now 1 h. Context only.
- The earlier chats' choices stand: `PP_CPU_JOBS` default 3, `PP.file_lock`, the order `3 → 0 → 1 → commit → 2 → C13 → 4`.

**Not proven before the run:**
- a. No check has run on real SemArt rows (stores not extracted). Stage 0's gates cover these.
- b. `bs_cstar.export_held`, `bs_photo.export_arm`, `mt_export.export`, `parity_cases` and `gate_v_dinov2` were not run live.
  Stage 0 covers them, and each fails loudly.
- c. Eight `pp_diag` kinds and all held rows have not run on real-shaped data. A failed kind can rerun with `--only`; it
  decides nothing.
- d. SC15's shared decode inside the timing code was not re-derived.

**Two notes from the re-review, left as they are:**
- `pp_run.py:179` reads `gate_viii.json` without mapping it to `cells_cost`. A `cells` crash before that file is written
  would leave an old one read as current. No old file exists today; if you rerun the probe, check the step's rc too.
- `pp_diag.py:511,567` call the SemArt scorer with default flags, so with `genre_word` failed those two kinds are skipped.

**Resources at 2026-11-04 02:23:** the local GPU is held by r18 (`tpips_local0` to about 04:08, then `turtle_ref`). CPU load about 40, with
r17's and r18's jobs. Parity waits for the GPU in the background and must not hold up node402. GPU check: driver 595.99.02 in
module, library and `nvidia-smi`. Night mode. Codex: five-hour window 57%, resets 04:35; weekly 83%. C13 goes to Codex.

**Timing:** about 10 minutes behind the build 5 estimate. MMArt and SemArt extraction need about 2 h and must end by 10:00,
so they should start by about 07:00, as before. The probe needs load under about 16.

**The ticket list (for the record):**

| # | Ticket | Blocked by | Spec IDs | Verdict code |
|---|---|---|---|---|
| 01 | Store reader `pp_stores` and the registration of the new three into r14's and r15's tables (`pp_register`) | none | §3.2, C6, D2, D11 | yes |
| 02 | Extraction `pp_extract` (new MMArt, Affection, SemArt stores), store checks, gate (i), the GPU queue script | 01 | §3.5 (i), §9 stores, D11 | no |
| 03 | The SemArt set `pp_semart`: rule recompute, gate (vi), md5 step, order, labels, D14 subset, inner pairs | none | §3.3 SemArt set, (vi), D5, D10, D14, D19 | yes |
| 04 | Node jobs `pp_node`: X-c* and X photo training for the new three, gate (ii), data map, launch plan | 01 | §3.2, §3.4 Training, (ii), §3.5 node order, D2, D11, D24 | no |
| 05 | Matched arms `pp_matched`: DINOv2 M5 at full scale, gate (v), mapper probabilities on SemArt and MMArt | 01, 03 | §3.2 matched arm, (v), D12, D18, D23 | yes |
| 06 | c* exports `pp_export`: new three (ArtELingo, Affection, MMArt), all seven on SemArt (genre), gate (iii) | 01, 03 | §3.2, §3.7, (iii) | yes |
| 07 | ArtELingo and Affection scores and stage 1 for the new three: SC1 to SC6; gate (iv) r14 part, (vii) r14 part | 01 | SC1 to SC6, §3.4, (iv), (vii), D21 | yes |
| 08 | MMArt scores and stage 1 for the new three: SC7 to SC10; gate (iv) r15 part, (vii) r15 part | 01 | SC7 to SC10, §3.4, (iv), (vii), D21 | yes |
| 09 | SemArt scorer: SC11 to SC14 (seven encoders), SemArt units and picks record, gate (iv) SemArt part | 01, 03 | SC11 to SC14, §3.3, §3.4 SemArt, (iv), D9, D17, D18, D20 | yes |
| 10 | The cost probe `pp_probe`: SC15 inputs, gate (viii), the cost table's measured rows | 02 | SC15, §3.4 Cost, (viii), D8, D22 | yes |
| 11 | Outcome code `pp_outcome`: cell outcomes, five verdict kinds, D17 claims, the voids table, licensed sentences | none | §6, §3.5 voids table, D4, D15, D17, D24 | yes |
| 12 | Stage runner `pp_run`, records, status, CPU and GPU queue scripts, node plan; end-to-end smoke | 01 to 11 | §3.5 stages 0 to 4, §3.7, §6 selection rules, D16, D25, A1 | no |
| 13 | Diagnostics `pp_diag` (§3.6, without r17's metrics) | 07 to 10 | §3.6, D14 | no |
| 14 | r17's metrics as context rows (NMI, ARI, ACC, cluster counts) | r17's merge into main; 13 | §3.6, D13 | no |

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- The earlier handoffs' §5 hold: a `cd` in a Bash call moves the chat; zsh takes a bare `==` as a command; `cluster watch` and
  `pull` need the worktree's cwd in a subshell, with `--node node402` on every call.
- **Read each gate's pass key, never the exit code.** A reduced replay exits 0 with `passed` false.
- **Gate (iii) files are now `gate_iii_r10.json` and `gate_iii_r14.json`.** There is no `gate_iii.json`.
- **`py_compile` writes `__pycache__`** even with `PYTHONDONTWRITEBYTECODE=1`. Use `python -c` to check syntax.
- **Tests:** the full round suite takes about 25 minutes at load 50; run it through `res-queue cpu`.

## 6. State at handoff

- Running: nothing of r19 (no subagent, no Codex job, no queued job, no cluster job).
- Uncommitted: nothing of r19's in git (`.scratch/r19-past-picks/` is outside git by design).
- On disk: `OUT/matched_sample` 6.8 MB (keep). The worktrees and branches of this chat are on `.scratch/pending_deletions.md`;
  `r19-int` stays until the report.
