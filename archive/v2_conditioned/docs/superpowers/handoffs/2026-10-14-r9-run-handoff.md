# Handoff: r9 run: settle SigLIP (a2 or b), then the agreements, pre-reads, held reads H6 to H10 and the reports

## Brief
- Done: r9 build closed: all re-reviews pass, suite 2246 passed, stage 0a passed, r9-scope merged to main.
- State: CLIP and Qwen stage 2 ready; SigLIP stage 2 stopped on a 1-row k-means group (a dead centre, not bad data).
- Next: SigLIP pick (a2, refit rule, default; or b, report the stop), then agreements, pre-reads, held reads.
- Blocked: no pre-read or held read before the SigLIP pick; a read not started by 2026-10-18 does not start.

> Written 2026-10-14 04:18 by `r9 build 11`. Loop step reached: step 7 done (build merged); the run chat does steps 8 and 9.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

F = `src/test/20261202_r9_scope`.

## 1. Read in this order

1. Sections 3 and 4 below: the state and the one open decision.
2. `.scratch/r9-scope/notes/reviews_build11.md`, then `reviews_build10.md`: the last two build chats, every merge and
   number.
3. The SigLIP stop: `.scratch/r9-scope/notes/siglip_stop_diagnosis.md`, and `F/fx_siglip_a2_handback.md` on branch
   `r9-siglip-a2` (`git show r9-siglip-a2:F/fx_siglip_a2_handback.md`; lines 43-60 list what a merge makes stale and
   the rerun steps).
4. `F/t22_handback.md`: the pre-read and its commands (the scratch `--results` roots there are the ticket's; production
   runs in the main checkout with the default results root). `F/rederive/rd9_commands.md`: the phase-1 and per-read
   agreement commands, with `--reported-sensitivity`.
5. The spec `docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md` (§3 design, §6 success criteria, §9 time-box)
   and the rule `F/DECISION_RULE.md` (SHA f9b07b34); the ledgers `docs/superpowers/held_ledger.md` and
   `episode_seed_ledger.md`.
6. The ticket list: `.scratch/r9-scope/issues/00-index.md` (shared interfaces at its end).

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 2026-10-13 wording changes (rule SHA f9b07b34), the build 5 to 8 agent defaults, and
  the r9_bundle `picked.get("smoke", False)` fix (user, 2026-10-13 22:4x).

## 3. Where things stand

- **Code:** `main` @ 71843c1 (merge of `r9-scope` @ 5e7541d, `--no-ff`), both pushed. The main checkout is on
  `main`.
- **Build:** final review of six areas, all CONFIRMED WITH FIXES, no HIGH, no C5 leak. Fix wave (5 tickets), then fx-rr
  and fx-rr3. Re-reviews: RR1, RR2, RR3 PASS WITH NOTES, RR4 PASS (all in `F/final_review/`). Accepted LOWs are listed
  in the notes (RR2 LOW-1, LOW-3, LOW-4; RR1 N4; RR3 LOW-3, LOW-4; RR4 LOW-A, LOW-B). Deferred: B2 L-1, B2 L-4.
- **Tests:** full round-folder suite on beca6ce, 2246 passed, 1 skipped, 0 failed (15 min). fx-rr3 changed two files
  after it: their tests, 377 passed (apply, preread, rederive compare) and 228 passed (apply, descriptive).
- **Stage 0a** on 78ea537 (final bytes): PASSED, 131 items, 0 failed, 0 differences, 284 s; its 45 module SHAs equal the current files. Its record `F/results/stage0a_regression.json` is bound to
  `r9_common.r9_module_shas()`: any edit of an `r9_*.py` or `run_r9_*.py` in F (or `scripts/run_r9_*.sh`) makes it
  stale; rerun it then (CPU, about 5 min; command in `t22_handback.md`).
- **Stage 0b** (14e58cf4…) and **Qwen stage 2** (pins 9a0fae5e…) stand: no closure module changed since (N5). CLIP's
  pins: `F/pins_clip.json`.
- **SigLIP stage 2** stopped at the groups step: one caption group of 1 row, a dead MiniBatchKMeans centre on an
  ordinary row (norm 1.0, no duplicates); bit-reproducible at seed 42; seed 43 gives a smallest group of 939. CLIP
  and Qwen have no group under 244 rows. Branch `r9-siglip-a2` (fff621a, 79324aa; pushed, not merged; worktree
  `/project/CoSiR-wt/r9-siglip-a2`): refit with the next seed when a group has under 100 rows; CLIP and Qwen refit
  byte-equal; SigLIP lands on seed 43.
- **Independent derivations (C13):** the main checkout has no `F/rederive/rd9_out/` yet (the pre-read and the apply
  step read it). Worktree `r9-fx-c13` holds CLIP and Qwen phase-1 derivations (`rederive/rd9_out/`, made 03:01-03:13
  on fx-c13 bytes; the rd9 prepare and phase-1 code is unchanged since, only `rd9_compare.py` changed). Worktree
  `r9-t25` holds CLIP phase 1 and the H8 selection derivation. Regenerate in the main checkout or copy, your choice;
  the bundles are 1.9 GB (CLIP) and 5.7 GB (Qwen) on the system disk (208 GB free at 04:13).
- **Probes** (CUB, SemArt, ticket 27): done (2083 s), outputs in `F/results/probes/{semart,cub}/`, not read; stage
  3's decide chat reads them (about 2026-10-17).
- **Codex** at 04:0x: five-hour window 15% (reset 04:38), weekly 87%; build 11 used none.

## 4. Open points for the next step

- **Key, the user's: SigLIP.** (a2) merge `r9-siglip-a2` (recommended: keeps three backbones, CLIP and Qwen stay
  byte-equal; one recorded SigLIP deviation). (b) report "the fixed recipe does not run on SigLIP 2" as a result, no
  H7 or H10. Data fix: not applicable.
  - **When to ask:** run `run-mode` first. Day mode: ask with (a2) as the 30-minute default (`loop-chats.md`,
    "Unattended chats"). Night mode: take (a2) as the agent default, since (b) stays open after it (the user can still
    drop H7 and H10 from the claim), and record "agent default, night mode, no answer possible" in the run log, the
    handoff and the report's brief.
  - **Decide before any pre-read.** (a2) edits `r9_recipe_groups.py`, a closure module in `r9_module_shas()`, so it
    makes stage 0a, stage 0b, Qwen's stage 2 and every pre-read made before it stale.
  - **If (a2):** merge `r9-siglip-a2` (based on c48d75c; `.claude/20261014_log.md` conflicts: union helper
    `.scratch/r9-scope/notes/union_merge.py <file>`, then `git add -f`); run its tests and stage 0a again; stage 0b
    replay step 07 (GPU, about 12 min, a fresh replay folder) and its compare (about 5 min); move aside (never delete)
    Qwen's `F/results/recipe/qwen3vl_emb_2b/` and `results/stage2_*qwen*` files and SigLIP's 7 partial files in
    `F/results/recipe/siglip2_so400m/` (the driver refuses to overwrite); then the stage 2 chains
    (`.scratch/r9-scope/notes/stage2_chains_build9.sh`: Qwen then SigLIP, about 1 h 45 each). Stale after the merge:
    `stage0b.json`, Qwen's recipe, stage 2 records and pins, `t13_clip_dryrun`, `t22_mutation_results.json`. Not
    stale: `pins_clip.json`, `pins_a28.json`.
- **Then, per backbone and read** (`t22_handback.md`, `rd9_commands.md`): the phase-1 derivations and agreements; the
  pre-reads in the main checkout (default results root), committing each successful smoke-record copy; the ledger
  rows; the held reads; the apply step; the descriptive pass (rule §8); the auto report and the user-read report.
- **Time:** the local machine is free by 11:00 (`shared-resources.md`). With (a2) the GPU part alone is about 4 h;
  the pre-reads take 10 to 40 min CPU each (CLIP H8, Qwen H6 and H9, SigLIP H7 and H10). Estimate at the start and
  weigh the options if it runs past 11:00. Spec §9: all reads by 2026-10-18.
- **For the report (not code):** B1 LOW-1 (0/0/72 near-duplicates against all 49,121 non-held paintings; two read
  paintings are the same artwork as a train-side one); B1 LOW-4 (list the 6 style-differs paintings); A1 L4 (the user
  decides the "keeps the same sentence" wording under D9). The committed H8 selection agreement (129 items) predates
  the seven quantity labels; a rerun is expected to agree with 136 items and margins not compared (fx-rr hand-back).

## 5. Code and pitfalls

- **The main checkout is shared** (the paper chat and side chats work there): `git status` before every merge; merge
  with `--no-ff`. The stage 2 driver takes the GPU lock itself.
- **Agreements are strict now:** the apply step refuses an agreement without `sensitivity_sha256` (it must equal the
  SHA of the sensitivity file the step reads) and one that carries `margins_not_compared` (written only by the
  selection replay). Agreement and smoke files are exclusive writes: a redo uses `_fix1`.
- **The full suite:** no pytest-xdist (`-n` fails), so it runs serially, about 15 min. To collect the rederive tests
  too: from F, `pytest -q -p no:cacheprovider . rederive/ --override-ini="python_files=test_*.py *_test.py
  rd9_test_*.py"`. `final_review/fr_c_gate_test.py` is collected and passes; `fr_a2b_probe_tests.py` is not collected
  and fails its F2/F3a cases by design (a review record).
- Tests that read the main checkout's `F/results/` go stale as stages write real outputs (two found in build 10).
- `fx_rr_mutants.py` counts a collection error as "caught" (RR3 LOW-3): run it only in a real git checkout.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no background run). GPU idle.
- Uncommitted: nothing of ours (notes in the gitignored `.scratch/`).
- On disk over 1 GB: worktrees `r9-fx-c13` (7.3 GB, the phase-1 derivations), `r9-t05` (2.4 GB), `r9-t25` (1.9 GB);
  all listed on `.scratch/pending_deletions.md` with every other r9 worktree, ticket branch and build scratchpad.
  Keep `r9-replay` while stage 0b's record points at it, and `r9-siglip-a2` until the SigLIP decision.
- Not ours: an old zsh (pid 1078632, cwd r9-t27) may still be alive; holds no GPU or lock.
