# Handoff: finish the relational genre build (gate checks in the real outputs, merge into main, run handoff, open the run chat)

## Brief
- Done: fix wave for stages A, B and C merged; scoped re-review PASS (83 mutations); follow-up fix merged; smoke chain: A and B clean.
- State: the stage C smoke stops at the held scorer: a smoke trains seed 0 only, the scorer needs five seeds (smoke plumbing, not the real path).
- Next: fix the stage C smoke and rerun queue C, the gate checks (GPU, about 35 min), merge into main, the run handoff.
- Blocked: nothing. No container restart for this build (user, 20:01); ask about one before the run chat.

> Written 2026-10-26 23:30 by "r10 build 5" (cut at 200k context; no subagent, Codex job or GPU lock of this chat running).
> Loop step reached: step 7 (build), fix wave and re-review done. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. [2026-10-26-r10-build-4-handoff.md](2026-10-26-r10-build-4-handoff.md) §4 e and f and §5, and through it
   [2026-10-26-r10-build-3-handoff.md](2026-10-26-r10-build-3-handoff.md) §4 c and §5 (the gate checks, code-hash gates,
   pitfalls). They still hold; this file adds only the state.
3. `/project/CoSiR/.scratch/r10-relational-genre/notes/build_progress.md`, the entries from 22:30 on (this chat's), and
   `notes/rereview.md` (the re-review: what is fixed, the 7 surviving minor mutations, the open points below).
4. `src/test/20261027_relational_genre/rg_queue.md` (the run's manual: order, gates, recovery, times) and the ticket
   index `.scratch/r10-relational-genre/issues/00-index.md` (ticket table, output layout) for the run handoff.
5. Before the cut: `~/.claude/references/loop-chats.md` ("How to cut", "Unattended chats").

Skills: none.

## 2. Decided by the user, not to be reopened

- As the build handoffs: the spec's D1 to D20, M1 to M10, the brief approved at 19:57; no container restart for this
  build (user, 20:01).

## 3. Where things stand

Integration branch `r10-relational-genre` at fdef661 (worktree `/project/CoSiR-r10-rg`, pushed). Not yet merged into main.

- **Fix wave** (all merged, pushed):
  - 27fbc0b, stage C (Codex; build 4).
  - 6ce2669, scorer, trainer, queue A (Codex, 13% of the five-hour window): `read_va1` at the start of `climb` and `a`
    (scorer half pass and current code hash; training half pass unless smoke); queue A stops on a failed `va1_scorer`
    outside a smoke; `rg_code_sha256` (rg_train, rg_loss, rg_variants) in `va1.json` and every `run.json`, required
    equal in `climb`, `a`, `b`; `eq.json` bound to rg_loss and distill_loss by hash (a stale one is refused: move it
    aside); VA4 checks `replaced/`; non-finite measured terms recorded as null with counts; dirty flags as rg_held's.
  - f4a78e4 (merge 89cd46f), rg_v2, report, queue B (Sonnet `high` worker): rg_v2's A-NEW form and `rerun_of` as the
    scorer writes them, stand-ins built by the scorer's own functions; `report_ab` a soft step; VB1 gate reads the vb1
    step's exit code; "rerun of" column; run notes.
  - b99f538 (controller): the report reads the new non-finite term format (it would have failed on every run).
  - 005020a (merge fdef661) (Sonnet `high` worker, after the re-review): VA4 accepts a `replaced/<arm>/s0*` folder without
    `run.json` (an interrupted run, never scored) and prints `VA4 note: interrupted run moved aside, never scored`;
    the report tolerates an empty `term_means`; rg_queue.md notes for both and the stale `eq.json` recovery.
- **Re-review** (Sonnet `high`, `notes/rereview.md`): PASS. 83 mutations; all fix-wave guards, M13, S5 and the stage C
  leftovers killed; 7 minor survivors (test gaps, listed there).
- **Smoke chain** on b99f538, 22:47 to 23:27 (`RG_SMOKE=1`, outputs
  `/tmp/claude-0/-project-CoSiR/b4527a1e-8cfa-4e8e-a2ab-28076068d8c6/scratchpad/rg`, kept for point a; status files,
  logs and the smoke `reported_ab.md` copied to `notes/smoke_build5/`):
  - Stage A clean in 12 min: VA1 noted as designed (2 epochs), va1_scorer and EQ pass, climb, `va2_a` agree, A-SAME
    (v*_A GR-8), loss scale ok.
  - Stage B clean in 22 min: VB1 holds, VB2 agrees (457 s), B-PICK v_held GR-8, `report_ab` 502 s with the new
    loss-scale table, the "rerun of" column and the GR-2/GR-4 line.
  - Stage C: `vc1_train`, `vc1_scorer`, `train_g` (seed 0), `held_g` ran; `score_c` stopped: `rg_held.py:403` "missing
    held output of G: .../stage_c/full/held/GR-8/genre__s1.npz". Cause: `rg_full.py:285` allows only seed 0 in a smoke
    (and `rg_queue_c.sh:24` passes seed 0), while `rg_held.run_score` reads `SEEDS` (0 to 4) in every mode. VC2 and
    the stage C report were not reached.
  - Ticket 02's smoke check holds on GQ-8 and GN-8: two epoch rows, 5,000 pairs and 10 calls each; GN-8 has no direct
    term.
- Full suite on fdef661: 427 passed in 150 s.
- **Real outputs** (`/data/SSD2/cosir_r10/20261027_relational_genre/outputs`): empty apart from old partition caches in
  `stage_a/partitions/`. The stale `regression/va1_scorer.json` (code 4a536ba) was moved to
  `/data/SSD2/cosir_r10/20261027_relational_genre/moved_aside/2026-10-26_va1_scorer_code4a536ba.json` (22:31).
- Codex at 23:28: five-hour window 17% left (reset 01:12), weekly 83%.
- GPU check at 23:28 passes (driver, module and library 595.99.02; container up since 2026-10-25 18:44).

## 4. Open points for the build (agent work; nothing for the user)

a. **Stage C smoke** (plumbing; the controller decides, Sonnet `medium` worker or Codex when it has room): let a smoke
   train and infer the five seeds at 2 epochs (`rg_full.py:285` and `infer_held`'s seed checks; `rg_queue_c.sh:24`),
   so `rg_held score`, `rg_vc2` and `rg_held report` run in a smoke; or, if that is not possible, make the smoke stop
   with a note after `held_g`. Then move the smoke's `stage_c/` aside within its own folder and rerun queue C alone on
   the kept smoke outputs (`RG_SMOKE=1 RG_OUTPUTS=<that folder> bash rg_queue_c.sh`; it needs queue B's status
   file, which is there). `rg_full` is under the VC1 code gate, so do this before point b.
b. **Gate checks in the real outputs**, from the round folder of `/project/CoSiR-r10-rg`, GPU work under the lock, one
   at a time (build 3 handoff §4 c): `rg_train regress va1` (GPU, about 8 min), `rg_train eq --device cpu`,
   `rg_score check va1`, `rg_score check vb1`, `rg_full regress vc1` (GPU, full-scale R6-5 seed 0, about 4 min),
   `rg_held check vc1` (about 6 min). Each must pass; read each record's `pass` and items from the file itself. The
   ticket smokes of build 3 §4 c are covered by the smoke chain: 02 (term means, above) and 04 (`b --smoke`) in this
   chat's chain; 06 (`held --smoke`) ran in `held_g`.
   Any code edit after these checks means rerunning them (code-hash gates).
c. Then build 4 handoff §4 e and f: merge the integration branch into main, push; worktrees and ticket branches on
   `.scratch/pending_deletions.md` (list below); the run handoff (ticket list from the index; queue order and times
   from `rg_queue.md`: A 82 min, worst 106; B 65 to 70, plus about 10 min for `rg_v2 --phase b`; C 52; total about
   210 min, worst 240); the GPU check; ask the user about a container restart (day mode: default "no restart" after
   30 minutes); `loop-next <handoff> --label "r10 run" --unattended`.
d. **For the run handoff** (agent defaults to state there, not the user's):
   - Not fixed: C9 (rg_full's guard does not hash the frozen ms_train and ct modules); B6 (GR-2 and GR-4 get no
     old-5,000 or Leiden reading; the rule's §4.2 governs over spec §3.6; the report says so in one line).
   - Re-review points left: no unit test of rg_report (plumbing; the smoke covers it); a stale `eq.json` is found only
     at `rg_score a` (cannot happen if no code changes after the gate checks; recovery in rg_queue.md, 4a).
   - The rule's line 48 says a batch is 1,024 pairs; the frozen trainer uses 512 pairs (1,024 rows), which VA1 needs.
     The code is right; the report must not repeat the rule's figure (tell the user).
   - The run chat copies every `VA4 note: interrupted run` line of the `score_a` log into its report and handoff.
   - Do not commit in the main checkout while a queue trains: an interrupted VA3 rerun compares the origin's commit
     with HEAD (rg_train.py, `run_variant`) and refuses after any new commit. Doc edits are fine (dirty flags read only
     the round's `.py` files).

## 5. Code and pitfalls

- As build 3's and build 4's handoffs §5. Added by the fix wave: `climb`, `a` and `b` refuse runs whose
  `rg_code_sha256` differs from `va1.json`'s, so an edit of rg_train, rg_loss or rg_variants after VA1 means VA1 and the
  arms trained since are redone; a stale `va1_scorer.json`, `vb1.json` or `eq.json` is refused, not kept: move it to
  `<outputs>/aside/` (never delete) and rerun.
- The full suite must run in a subprocess-safe way now (it does: 426 pass in one session, about 2.5 min).

## 6. State at handoff

- Running: nothing of this chat (no subagent, Codex job, GPU lock or background job).
- Uncommitted: nothing of this build (`.scratch/` is outside git).
- On disk over 1 GB: `/tmp/claude-0/final_review_B/` (4.8 GB, on `.scratch/pending_deletions.md`). This chat's smoke
  outputs (8.3 GB, system disk) are kept for point a and are on `.scratch/pending_deletions.md`; the hook lets only
  this session delete them, so the storage summary decides.
- Worktrees and branches to list at the merge: `/project/CoSiR-r10-fixA` (r10-rg-fixA), `/project/CoSiR-r10-fixC`
  (r10-rg-fixC), `/project/CoSiR-r10-t01`, `-t02`, `-t09`, `-t10` (their r10-rg-tNN branches),
  `.claude/worktrees/agent-*` with branches r10-rg-t03 to t08, t11, `worktree-agent-a9d5b858956a4a87a` (fix B),
  `worktree-agent-a39bef15491807aa9` (follow-up), and build 4's `smoke_wt` (already listed).
