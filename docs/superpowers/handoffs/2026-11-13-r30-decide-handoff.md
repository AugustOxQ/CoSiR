# Handoff: read r21's results with the user (step 1, decision C), wrap up r21, then decide the next direction

## Brief
- Done: r21 (a DEC loss on c*'s heads) is reported: auto report 2027-01-17 and user-read report 2026-11-13, both checked.
- State: all three DEC forms raise style AMI (+0.006 to +0.008 on 0.411); the λ = 0 control shows no clear change.
- Next: show the user the results brief without our reading, get their reading, then wrap up and decide (step 2).
- Blocked: the user's answer on the direct contrast "DEC form minus control" (option a, asked 2026-11-13 02:10).

> Written 2026-11-13 04:15 by "r21 run 4". Loop step reached: r21 step 9 done (both reports committed); the next loop
> starts at step 1 (decision C on r21). Direction log entry: none yet; this chat writes r21's entry at its wrap-up.

## 1. Read in this order

1. The results brief at the top of the auto report,
   `docs/reports/auto/percept/2027-01-17_dec_heads.md`: show it to the user **without** the folded "Our reading and
   recommendation" block (`research-loop.md`, "Decision C and the results brief").
2. The user-read report, `docs/user_read/2026-11-13_dec_heads.md`: the user may read it first; its advice block is
   folded the same way.
3. The round log, `src/test/20261111_r21_dec_heads/20261111_r21_dec_heads_log.md`, sections "Run chat 3" and "Run chat
   4" only, when a question needs the run's detail.
4. For the wrap-up: `~/.claude/templates/direction_log_entry.md`, `~/.claude/references/storage-summary.md`.

Skills: none for step 1. Step 5 (the grill), if a new idea comes: `grilling` and `domain-modeling` (`grill.md`).

## 2. Decided by the user, not to be reopened

- r21's design: three DEC forms, settings picked on validation seed 0, five seeds on test against saved c*, no claim
  ladder (spec `docs/superpowers/specs/2026-11-11-r21-dec-heads-design.md`, approved 2026-11-11 06:31).
- Internal scores SI, DBI, CHI on test only (user, 2026-11-12 19:44); the λ = 0 control and the figures (user,
  2026-11-12 20:29 and 20:45); "run now" for the control (user, 2026-11-13 01:59). All in the round log.

## 3. Where things stand

- Test, 12,281 paintings, seeds 0 to 4, change from saved c* (the brief's table has the words): style AMI raises in all
  three forms (+0.0077, +0.0062, +0.0078 on 0.4109); style and genre mAP raise in all three; genre AMI raises only in
  the teacher form; emotion no clear change. The λ = 0 control: no clear change in AMI on all aspects (style +0.0015
  [-0.0016, +0.0043]); emotion mAP lowers (-0.0012).
- On style AMI the plain form's interval clears the control's; the start and teacher forms overlap it at the edges.
- Tightness on test changes little and not in one direction; the control is looser than c* on all aspects. Long DEC
  (50 DEC epochs, validation, one seed) raises CHI but lowers AMI on all three aspects.
- Checks: both re-derivations PASS (stage 2 and the λ = 0 row); Claude's claim review recomputed the brief's words and
  numbers (report section "Claim review"); the user-read report had a cold read and a fidelity check, one fix wave.

## 4. Open points for the next step

1. **Key, decision C:** the user's reading of r21, given before ours (a sentence or two). Then open our reading and say
   where the two differ. Record it in the direction log entry.
2. **Key, decision A:** the next step among the brief's options: a. the direct contrast "DEC form minus λ = 0" (paired
   by seed, the same bootstrap, an interval; `r21_table.py` compares only against c*, so it needs a comparator option;
   a small code change and about 1 h of one CPU slot); b. close the DEC line, DEC as an ablation row; c. a DEC setting
   that keeps AMI (new GPU runs, stage 1 again). Ask the user first; our recommendation (b, with a only if the paper
   cites the size of the DEC effect) comes after their answer. Option a is the question open since 02:10.
3. Minor: r25, r26 and r28 also have "the user's reading waits" in the direction log's "Now"; mention it in one line,
   and read them only if the user wants to.

## 5. Code and pitfalls

- `scores/stage2_lam0.json` has `complete` False by design; the paper-facing table stays `scores/stage2_table.*`
  (4 arms). The λ = 0 row is a diagnostic.
- Names: the glossary renamed "teacher" to the instructed embedding (2026-11-12); the auto report keeps "teacher pull"
  (the spec's and code's name) and describes it at first use; the user-read report says "instructed-image pull".
- The r29 run chat uses the local GPU now (through `res-queue`); r21 needs no GPU unless option c is chosen.

## 6. State at handoff

- Running: nothing of r21.
- Uncommitted: nothing of r21. (`docs/superpowers/handoffs/2026-11-13-r29-run-handoff.md` is modified by the r29
  chats; not ours.) The Codex job ledger in `/project/claude-config/codex/jobs.md` has three new r21 rows, uncommitted
  in that repo with other chats' rows: the wrap-up's `sync.sh` and commit cover it.
- On disk over 1 GB, for the wrap-up's storage summary:
  - `/data/SSD2/cosir_r21/20261111_dec_heads/` 12 GB (`full/` 12 GB: training outputs and exports of all arms;
    `scores/`, `figures/`, `logs/` about 20 MB). The reports are done; keep the scores and figures; `full/` is the
    user's call (the paper's ablation rows cite these scores; a re-score or option a needs the test exports).
  - r21's items on `.scratch/pending_deletions.md` (lines of 2026-11-11 08:26 and 2026-11-12 21:19): ten merged
    worktrees (`/project/CoSiR-r21*`, about 1.3 GB), their local branches, build scratchpads (2 GB and more), a partial
    gate record, Codex's `/tmp/r21_*` notes.
- `wrapup-check` printed nothing at 04:14.
