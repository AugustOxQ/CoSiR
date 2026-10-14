# Handoff: r9 build 11: re-review fx-rr, stage 0a on final bytes, merge to main, then hand off to the run chat

## Brief
- Done: final-review fix wave (5 tickets) merged and pushed; both re-reviews PASS WITH NOTES; their findings fixed.
- State: no HIGH; N5 holds (no closure edit). SigLIP stop: a dead k-means centre, not bad data; a fix is pre-built.
- Next: re-review fx-rr, stage 0a (CPU), merge r9-scope to main, handoff to the run chat.
- Blocked: SigLIP's reads (H7, H10) wait for the user's pick: refit rule (recommended) or report the stop.

> Written 2026-10-14 03:56 by `r9 build 10` (cut at the context meter, 200k). Loop step reached: step 7, the build
> (fix wave done, scoped re-review of its last round next). Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build10.md`**: every merge, review finding and number of build 10.
2. `F/fx_rr_handback.md` (F = `src/test/20261202_r9_scope`) and the re-reviews `F/final_review/rr_fixwave_1.md`,
   `rr_fixwave_2.md`; the final-review reports beside them (`fr_a1`, `fr_a2` + `fr_a2b`, `fr_b1`, `fr_b2`, `fr_c`).
3. `.scratch/r9-scope/notes/siglip_stop_diagnosis.md` and `F/fx_siglip_a2_handback.md` on branch `r9-siglip-a2`.
4. `F/t22_handback.md`: the controller commands for stage 0a, the phase-1 agreements and the pre-reads.
5. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/fx_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 2026-10-13 wording changes (rule SHA f9b07b34), the build 5 to 8 agent defaults, and
  the r9_bundle `picked.get("smoke", False)` fix (user, 2026-10-13 22:4x).

## 3. Where things stand

- **`r9-scope`** @ 9f631bd (pushed). Fix wave merges: fx-held 039fc93, fx-c13 c48d75c, fx-tests 1e6e55f, fx-desc
  a32970c, controller ab1367e, fx-rr 9f631bd. Final-review folders of all six areas and both re-reviews are in
  `F/final_review/`.
- **Final review:** A1, A2 (via A2b), B1, B2, C all CONFIRMED WITH FIXES; no HIGH, no C5 leak. Every MEDIUM fixed
  (list and tests in reviews_build10.md). Re-reviews RR1 and RR2: PASS WITH NOTES, no HIGH; their M1s and the LOWs
  N1, N2, N3, LOW-2, LOW-5 went to fx-rr; accepted as they are: RR2 LOW-1 (a stale heldfix1 smoke record shadows
  newer ones, as fix1/crash1 already did), LOW-3, LOW-4, RR1 N4. Deferred: B2 L-1 (closure), B2 L-4 (no backbone is
  switched off). B2's open check of `F/pins_qwen3vl_emb_2b.json`: holds (RR2).
- **Full suite** at ab1367e (main checkout, 16 min): 1938 passed, 16 failed, all test-side and sent to fx-rr (14
  `test_r9_a28_stats.py` fixtures without `read`/`backbone`; a subprocess without HOME; a test assuming Qwen's
  partitions are absent). fx-rr (562c0c8) fixed all 16 and the re-reviews' findings: 767 + 264 tests passed in its
  worktree, 23 of 23 mutants; items 8 and 9 also pass read-only against the main checkout. The full suite on the
  final bytes is not yet run.
- **Stage 0b** (14e58cf4…) and **Qwen stage 2** (pins 9a0fae5e…) stand: no closure module changed (N5).
- **SigLIP stage 2 stop**, diagnosed (read-only): the 1-row caption group is a dead MiniBatchKMeans centre on an
  ordinary row (norm 1.0, no duplicates); bit-reproducible at seed 42; seed 43 gives a smallest group of 939. CLIP and
  Qwen have no group under 244 rows. Branch **`r9-siglip-a2`** (pushed, not merged; fff621a, 79324aa): refit with the
  next seed when a group has under 100 rows; CLIP and Qwen refit byte-equal; SigLIP lands on seed 43.
- **Codex** at 03:1x: five-hour window 15% (reset 04:38), weekly 87%; build 10 used none.

## 4. Open points for the next step

- **Key, the user's: SigLIP.** (a2) merge `r9-siglip-a2` (recommended: keeps three backbones, leaves CLIP and Qwen
  byte-equal; one recorded SigLIP deviation; cost: stage 0b replay step 07 about 12 min GPU and the compare 5 min,
  Qwen stage 2 rerun about 1 h 45, SigLIP stage 2 about 1 h 45; the old Qwen and SigLIP partial outputs moved aside
  first, the driver refuses to overwrite). (b) report "the fixed recipe does not run on SigLIP 2" as a result, no H7 or
  H10. Data fix: not applicable. Ask in day mode with (a2) as the 30-minute default (it keeps both options open; (b)
  can still be chosen after); in night mode note it, do the rest, and leave it for the run chat. Decide it **before
  any held read launches**: (a2) changes a closure module, so every pre-read made before it is stale. Deadline §10:
  a read not started by 2026-10-18 does not start.
- **For the run chat's report** (not code): B1 LOW-1 (0/0/72 near-duplicates against all 49,121 non-held paintings;
  two read paintings are the same artwork as a train-side one), B1 LOW-4 (list the 6 style-differs paintings), A1 L4
  (the user decides the "keeps the same sentence" wording under D9).

## 4b. This chat's steps

1. Scoped re-review of fx-rr (Sonnet high, worker, lean; brief pattern `notes/rr_fixwave.md`): its N1 (selection mode
   skips the margins) and N3 (apply step checks `sensitivity_sha256`) are verdict code.
2. The full round-folder suite on the final bytes (16 min, background; no xdist).
3. Stage 0a in the main checkout on final bytes (6 to 10 min CPU; command in `t22_handback.md`).
4. Merge r9-scope into main (`--no-ff`, `git status` first), push. List worktrees and branches on
   `.scratch/pending_deletions.md` (below).
5. Handoff to the run chat: the ticket list (`.scratch/r9-scope/issues/00-index.md`), the SigLIP decision first, then
   the pre-read order (phase-1 derivations and agreements with `--reported-sensitivity`, pre-reads, ledger rows;
   `t22_handback.md`), and the report items above. Open it with `loop-next … --label "r9 run" --unattended`.

## 5. Code and pitfalls

- The main checkout is shared: merge with `--no-ff`, `git status` first. The tracked log `.claude/20261014_log.md`
  conflicts on every ticket merge: union helper
  `/tmp/claude-0/-project-CoSiR/2f29e134-1907-458c-b32a-999b8c5a13f7/scratchpad/union_merge.py <file>`, then `git add -f`.
- No pytest-xdist in the CoSiR env: `-n` fails; run the suite serially.
- Tests that read the main checkout's `F/results/` go stale as stages write real outputs (two found in build 10).
- Keep reviewers lean (grep, write the report early, hand back near 150k).
- `F/final_review/fr_c_gate_test.py` is collected by pytest (`*_test.py`) and passes; `fr_a2b_probe_tests.py` is not
  collected and now fails its F2/F3a cases by design (its fixture writes the old code name): a review record.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no background run). GPU idle.
- Uncommitted: nothing of ours (notes in the gitignored `.scratch/`).
- New worktrees this chat (all for `.scratch/pending_deletions.md` at the build's end, with build 7 to 9's lists):
  r9-fx-c13, r9-fx-held, r9-fx-tests, r9-fx-desc, r9-fx-rr, r9-siglip-a2 (keep until the SigLIP decision), snapshots
  r9-fr-a2b, r9-rr1, r9-rr2. Untracked leftovers in r9-fx-c13's F: `fx_c13_mutations_0_22/22_50/0_43/part_42_43.json`.
  This chat's scratchpad (`/tmp/claude-0/-project-CoSiR/06992efa-c388-41ec-bd52-4b5c99c57e84/scratchpad/`): siglip_diag,
  fr_a2b (wt, wt2), wk, wk2, wk3, m, headcopy, logs. Keep `r9-replay` while stage 0b's record points at it.
- Not ours: an old zsh (pid 1078632, cwd r9-t27) may still be alive; holds no GPU or lock.
