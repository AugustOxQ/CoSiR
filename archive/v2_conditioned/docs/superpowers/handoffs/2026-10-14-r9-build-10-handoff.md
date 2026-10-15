# Handoff: r9 build 10: the final review's fix wave, its re-review, the stage 0a refresh, then the merge to main

## Brief
- Done: 13, 23 and 22 merged; stage 0b re-passed; CLIP dry run passed; final review in 5 areas, all confirmed with fixes.
- State: Qwen stage 2 passed (pins written). SigLIP stopped: its caption k-means has a 1-row group the heads cannot fit.
- Next: diagnose the SigLIP stop and ask the user (key); meanwhile the fix wave, its re-review, stage 0a, merge.
- Blocked: SigLIP's reads (H7, H10) wait for the user's decision on the 1-row caption group.

> Written 2026-10-14 02:48 by `r9 build 9` (cut at the context meter, 200k). Loop step reached: step 7, the build
> (final review done, fix wave next). Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build9.md`**: every merge, number and review finding of build 9, with file:line.
2. The five final-review reports (each in its snapshot worktree, `src/test/20261202_r9_scope/final_review/`):
   `/project/CoSiR-wt/r9-fr-a1/…/fr_a1.md` (statistics, apply step, phase 2), `r9-fr-a/…/fr_a2.md` (held runners, C5,
   A28 episodes, descriptive; partial), `r9-fr-b/…/fr_b1.md` (A28 data path), `r9-fr-b2/…/fr_b2.md` (stages, pins,
   recipe, phase 1), `r9-fr-c/…/fr_c.md` (22). Their scripts sit beside them; copy the folders into r9-scope's
   `final_review/` and commit them (as round 6 did).
3. `F/t22_handback.md` in r9-scope: the controller commands for stage 0a, the phase-1 agreements and the pre-reads.
4. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 2026-10-13 wording changes (rule SHA f9b07b34), the build 5 to 8 agent defaults, and
  the r9_bundle `picked.get("smoke", False)` fix (user, 2026-10-13 22:4x).

## 3. Where things stand

- **`r9-scope`** @ d4f05c5 (pushed): every ticket merged (13 fix 2: 300e04b; 23 + fix: 4458634; 22: d4f05c5).
- **Stage 0b** re-passed on 4458634 bytes (case 1, 131 items exact): `results/stage0b.json` SHA 14e58cf4…; the old
  record kept as `results/stage0b_ee7a76d.json`. **CLIP dry run** passed (`results/t13_clip_dryrun/`).
- **Stage 2 chains** (main checkout, scratchpad script `stage2_chains.sh` of build 9): **Qwen passed** (exit 0, 6190 s; record `results/stage2_qwen3vl_emb_2b.json` "ready"; pins
  `F/pins_qwen3vl_emb_2b.json` SHA 9a0fae5e…). **SigLIP stopped** in the groups step (exit 1, 578 s; log
  `results/stage2_run_siglip2_so400m.log`): `r9_heads.fit_all` :239, "caption/img: classes are not 0..63". Its
  caption k-means 64 on scorer-train has a group of 1 row (next 683; Qwen's smallest 1,510; image groups min 773), so
  the heads' row draw cannot hold all 64 classes. Partial outputs stay in `results/recipe/siglip2_so400m/`.
- **Final review** (agent default: five Opus xhigh areas, since one reviewer per area ran out of context after reading):
  A1, B1, B2, C confirmed with fixes; A2 partial (no verdict yet: its guard sweep and rule 6 left, see §4). No HIGH, no
  C5 leak, every load-bearing number re-derived equal. **No fix touches the 17 closure modules**, except B2's L-1 (wait).
- **N5 still binding:** after the stage 2 chains, no closure module may change (r9_pins, r9_stats, r9_score, r9_bundle,
  r9_common, run_r9_stage0a, …; `run_r9_stage2.recipe_module_shas()` lists the 17). `run_r9_stage2.py` itself must
  not change either (pins_evidence compares its SHA). Tests may change.
- **Codex** at 01:55: five-hour window 15% left (resets 04:38), weekly 87%.
- **Deadline:** rule §10, a read not started by 2026-10-18 does not start.

## 4. Open points for the next step

- **Key, the user's (SigLIP stage 2 stop).** The spec's stage table covers only "switched off" for stage 2; a recipe
  stop is not covered. First diagnose, read-only: which scorer-train row forms the 1-row caption group, its caption text,
  its SigLIP text embedding (norm, duplicates, distance to the nearest centroid), and whether the same k-means call
  gives it on a rerun. Then put the options to the user with a recommendation, e.g. (a) a recipe change that keeps
  CLIP and Qwen bit-equal (any closure edit forces the stage 0b re-compare and Qwen's stage 2 rerun, N5); (b) report
  "the fixed recipe does not run on SigLIP 2" as a result, no H7 or H10 read; (c) a data fix if the row is degenerate.
  Day mode: ask with the 30-minute default; night mode: note it and go on with the fix wave. Deadline §10: 2026-10-18.

## 4b. The fix wave (one wave, then a scoped Sonnet high re-review)

Before any read (blocking):
1. **C13 comparer** (`rederive/`): B2 M-1 (phase 1 cannot read a real `pins_<B>.json`: rd9_inputs.py:93, the logical
   keys, the `seed42.arrays` schema; first check it on d4f05c5, since 22 changed rd9_compare's phase 1); A1 M1 (compare
   SE, x, x95 and the seed-42 σ inputs); tests for B2 M-2 (phase-1 comparisons of σ*, cells, B0, λ, τ, P-points) and
   C MEDIUM-3 (minimum array sets, CLIP B/B0/B1 means); B1 LOW-2 (pin `near_duplicates_a28.json` in rd9_inputs, read it
   with read_pinned; also in run_r9_held.py:733); then rerun B2's `fr_b2_phase1_probe.py`.
2. **Descriptive** (`run_r9_descriptive.py`): A2 F1 (bind_read compares the code name with the ledger name: every real
   verdict is refused; build the test verdict through `AR.apply`); B2 M-4 (it imports r6_common and r6_descriptive,
   which fails test_r9_pins.py's independence test); A2 F2 tests (bind_read refuses another read's and a null verdict).
Tests only: B1 MEDIUM-1 and LOW-3; C MEDIUM-1 (`fr_c_gate_test.py` into test_launch_stop) and MEDIUM-2; B2 M-3
(test_r9_stage2.py only); A1's untested `assert_rule` (:229, :234, :238, :630).
LOWs, cheap ones in the same wave: C LOW-1 to LOW-4, A1 L1 to L3 and L5, B1 LOW-5 and the rd9_compare.py:128 nit, B2
L-4. Defer: B2 L-1 (in the closure).
Also: **A2b**, a fresh reviewer (Opus xhigh, lean) for A2's survivors in `r9-fr-a/…/fr_a2_mut_results_all.json`
(205 run: 113 caught, 90 survived, 2 timed out; survivors: run_r9_descriptive 67, r9_descriptive 13, r9_episodes 7, r9_bundle 3; the sweep used A2's FAST test set, so rerun them with FR_FULL=1 first), the CLIP H8 smoke output of build 9 (`<build 9 scratchpad>/fr_a2/smoke_h8_ctl/smoke/`: equal to
r9_stats on its arrays?), and hard rule 6 for A2's scope; its findings join the wave. A1's last r9_stats mutants:
done in build 9 (21 of 21 killed; r9_stats survivors only :97 and :200, each behind another check).
For the report (run chat, not code): B1 LOW-1 (0/0/72 near-duplicates against all 49,121 non-held paintings; two read
paintings are the same artwork as a train-side one), B1 LOW-4 (list the 6 style-differs paintings), A1 L4 (the user
decides the "keeps the same sentence" wording under D9).

After the wave and its re-review: stage 0a in the main checkout on final bytes (6 to 10 min, CPU); merge r9-scope into
main, push; the handoff to the run chat lists the tickets and the pre-read order (phase-1 derivations and agreements,
pre-reads, ledger rows; `t22_handback.md`).

## 5. Code and pitfalls

- The main checkout is shared: merge with `--no-ff`, `git status` first; log conflicts: union helper
  `/tmp/claude-0/-project-CoSiR/2f29e134-1907-458c-b32a-999b8c5a13f7/scratchpad/union_merge.py <file> <file> …` (zsh:
  list the paths literally; a `$L` list is not split), then `git add -f`.
- Codex launch: the command starts with `codex e --dangerously-bypass-approvals-and-sandbox` (see build 9's notes).
- **Keep reviewers lean**: in build 9 three of five Opus xhigh reviewers hit 200k after reading; tell each to grep,
  read only what it needs, write the report early and hand back near 150k.
- A scratch pre-read run writes `F/smoke_record.json` and `rederive/rd9_out/smoke/` even with `--results` scratch (C
  LOW-3): run it in a worktree, never the main checkout, until that is fixed.
- The pre-read needs stage 2's sigma file and its own r9_stats recompute exactly equal; a dtype or order difference
  stops it on real data (C's note).

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no background run). GPU idle.
- Uncommitted: nothing of ours (notes in the gitignored `.scratch/`).
- New worktrees this chat: r9-t22, r9-fr-a, r9-fr-a1, r9-fr-b, r9-fr-b2, r9-fr-c; scratchpad worktrees `t13_fix2_review`,
  `t23_fix_review`; scratchpad folders `fr_a1/`, `fr_a2/` (three full round-folder copies `wk1..3`), `fr_b/`, `fr_b2/`,
  `fr_c/`, `codex/`. All go on `.scratch/pending_deletions.md` at the build's end, with build 7's and 8's lists. Keep
  `r9-replay` while stage 0b's record points at it.
- Not ours: an old zsh (pid 1078632, cwd r9-t27, a ticket 27 probe command) still alive; holds no GPU or lock.
