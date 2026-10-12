# Handoff: r7 build 3: Opus reviews of tickets 08, 14 and 15, merge, then 11, 12, 13, 16, stage 0, chain smoke, final review, then cut to r7 run

## Brief
- Done: tickets 02 to 07, 09 and 10 merged (09, 10, 07 after Opus reviews); 08, 14 and 15 built; token statistics extracted (cache check 0.9999997).
- State: integration branch `r7-design-l`; no stage 1, 2 or 3 number computed.
- Next: Opus reviews of 08, 14, 15; merge; then 11 and 12, later 13 and 16.
- Blocked: nothing. Night mode.

> Written 2026-10-12 04:07 by the `r7 build 2` chat (cut past 200k context). Loop step reached: step 7, build in progress.
> Direction log: this round's entry is written at its wrap-up.

## 1. Read in this order

1. `.scratch/r7-design-l/issues/00-index.md`: tickets, shared interfaces, verdict JSON keys. It was amended in this
   chat (F2/F3 "not tested" form, finalize compares core keys only, notes for tickets 08, 11, 12, 13, A1 feature
   names, heads root, tests with `--rootdir=.`). Each ticket's `Status:` line is current.
2. `.scratch/r7-design-l/notes/build_log.md`: every launch, review, fix round and merge, with Codex token use.
3. `.scratch/r7-design-l/notes/implementer_brief.md` and `notes/reviewer_brief.md` (new: the brief every
   verdict-code reviewer gets; scripts in the reviewer's own `rev<NN>/` scratch folder).
4. The previous handoff's §1 item 4 (`2026-10-12-r7-build-2-handoff.md`): process references and the rule.

## 2. Decided by the user, not to be reopened

- Everything of `2026-10-12-r7-build-handoff.md` §2: spec D1 to D27 and the rule, approved 2026-10-12 02:34; build
  and run unattended (night mode).

## 3. Where things stand

| Ticket | State |
|---|---|
| 01 to 07, 09, 10 | merged into `r7-design-l` (last merge 03040de); 07, 09, 10 after Opus `high` reviews (each MERGE AFTER FIXES, no wrong number; fixes made, read by the controller) |
| 08 scorers (`r7_fusion`) | built b0ae941 on `r7t/08-scorers` (Codex), 15 tests; frozen AFF, told A1, R1[A1] match exactly; needs its Opus review, then merge |
| 14 array re-derivation (`rederive/rd7_*`) | built b66106d on `r7t/14-rederive-arrays` (Codex), 10 tests; needs its Opus review (include an independence check: it must not read `r7_*.py` other than `r7_common`) |
| 15 statistics re-derivation (`rederive/rd7_stats.py`) | built dcebaaa on `r7t/15-rederive-stats` (Claude worker, sonnet high), 92 tests, 33 self-mutations; needs its Opus review. It guessed key names the index leaves open (scorer names as array names `R1_V1`, `XAFF_V1`; `bar_comparator` strings; F1 `passes` dict, `holm_order` list; `FV` "GO"/"NO-GO"; chain entries only pick, alpha, F1, F2, F3): the reviewer compares them with what `r7_gate.gate_stats` and `r7_chain` return (the reviewer may read both; the implementer could not). Its `--overlay` reads ticket 14's phase files, untested against 14's real output |
| 11, 12, 13, 16 | not started. 11 needs 06 and 08; 12 needs 03, 04, 06, 08, 09; 13 needs 10 and 12; 16 needs 12 and 13 |

- Token statistics: `/data/SSD2/pre_extract/artelingo/clip_b32_tokstats/` written once (61,402 × 4,608 float32,
  SHA 639a56f4…); cache check minimum cosine 0.9999997 with TF32 kernels on, as the cache was made (fp32 kernels miss
  0.9999 on one image; recorded in `meta.json`; accepted as matching L5's intent). Dropped token share 0.0 in all three
  blocks (descriptive, for the report). The generic family (V5, V6) can go ahead.
- ClustPy 0.0.3 is installed in the CoSiR env with `--no-deps` (rule L7); torch unchanged (2.11.0+cu130).
- Codex at the cut: 5h window 41% left (reset 08:26), weekly 91% left. Every Codex launch in this chat passed the
  permission check.

## 4. Open points for the build (agent defaults; nothing for the user)

- **Next steps.** Opus `high` reviews of 08, 14, 15 (brief `notes/reviewer_brief.md`; for 14 and 15 also check
  independence and that their outputs use the pipeline's key names). Fix rounds by Codex resume for Codex tickets
  (`thread_id` in the first line of `run/codex_NN.log`), by a fresh worker for 15. Then 11 (stage 0 runner) and 12
  (runner: prep, ceiling, gate) on Codex; 13 after 12; 16 after 12 and 13. At most three implementers at once.
- **Controller runs:** stage 0 after 11 (G1 to G4); the chain smoke after 16; unit tests of verdict code; the final
  whole-branch review (Opus `xhigh`), one fix wave, a Sonnet re-review; merge into main and push. The full grouping
  build (03, 04: under an hour), the heads (05) and readers (06, `--jobs 3`) belong to the runner's prep stage, which
  the run chat starts.
- **Notes already in the index for the next tickets:** 12 must build the base heads (affect, image, caption) before
  readers, pass exactly the 4,096 style × genre episodes to `ceiling_stats`, and put `BX_R0__r1`, `BX_N__r1`,
  `BX_Z{k}__r1` and every `{name}_cf__gain` in `arrays.npz`; 13 passes `ceiling_zone` to `friday_verdict`; 11 (G3)
  maps `csd`↔`R0` feature names in column order.

## 5. Code and pitfalls

- **Tests in a worktree** run from the round folder with `pytest --rootdir=.`: otherwise pytest puts the worktree
  root first on `sys.path` (the root has `setup.py`) and `r7_common.load` correctly refuses the worktree's `src`.
- `R.load("r6_stats")` needs `R.load("r6_common")` first.
- Codex launches start with `codex e --dangerously-bypass-approvals-and-sandbox` from `.scratch/r7-design-l/run/`
  (`cd` in a separate call), so the allow rule matches; fix rounds `codex e ... resume <thread_id> - < fix_NN.txt`.
- Unattended: nothing is deleted; the hook refuses `rm` even of a probe file inside a worktree. Use the session
  scratchpad for probes.
- Pitfalls from the previous handoff §5 still hold.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing in git (ticket branches committed; `.scratch/` is outside git). The disk ledger row for the
  token cache is appended in `/project/claude-config/storage/disk_ledger.md`, not yet committed (the wrap-up's
  `sync.sh` commits it).
- Worktrees: `/project/CoSiR-wt/r7-{int,01,02,03,04,05,06,07,08,09,10,14,15}`; trial outputs in r7-02 (9.1 MB),
  r7-03 (6.0 MB), r7-04 and r7-07 (`results/smoke/ticket07_tmp`, 100 KB); all listed for the build's end.
- On disk over 1 GB: the token cache (1.1 GB, SSD2, ledgered keep); `res/cluster_jobs/` 3.1 GB (round 6, keep).
