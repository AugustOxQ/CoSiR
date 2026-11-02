# Handoff: finish the r14 build: pull the node smoke, GPU checks after the queue, EmoSet exports and stage 2 record, final whole-branch review, merge into main, then open the run chat

## Brief
- Done: the c* trainer bug for non-2048 encoders is fixed (Codex, Opus review approved); tickets 10a, 10b, 11, 12a, 12b merged. Every ticket is merged.
- State: integration branch at dda5225, pushed and synced to node402; the node smoke (tag 20261102-053122-dda5225) runs on GPU slot 0.
- Next: watch and pull the node smoke; GPU checks 0(iv), 0(v) after the extraction queue (about 08:00); then the final review.
- Blocked: nothing.

> Written 2026-11-02 06:33 by "r14 build 4" (cut at the context meter, 200k). Loop step reached: step 7, building.
> Direction log entry: [2026-11-02](../direction_log.md).

## 1. Read in this order

1. This handoff, then the first build handoff's sections 2 to 5: [2026-11-02-r14-build-handoff.md](2026-11-02-r14-build-handoff.md).
2. The spec from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md).
3. `.scratch/r14-backbones-sweep/` (gitignored, main checkout only): `issues/00-index.md` (ticket 12 is now 12a and
   12b); `notes/controller_notes.md` (all review notes, the GPU check commands, the stage 0 results; new sections for
   12a and 03b at the end); `notes/build4/t03b_verification.log` (the fix's smoke commands and the GPU replay commands);
   `notes/reviewer_brief.md`.
4. `~/.claude/references/matt-chain.md` ("The final whole-branch review") and, before a Codex job,
   `~/.claude/references/codex-jobs.md`.

Skills: `cluster-run` (watch and pull the node smoke).

## 3. Where things stand

- **Integration branch** `r14-backbones-sweep` (main checkout), dda5225, pushed and synced to node402. Merged in this
  chat: 10a (39fefd7), 10b (6fd7989), 11 (5a4bd6f), 12a (20559ed), 03b (effeafc), 12b (dda5225).
- **The blocker, fixed (ticket 03b):** `rg_loss.py:88` asserted `teacher.shape == img.shape`, true only for Qwen. The
  loss compares the 2048-wide teacher with the 2048-wide output `z`, never with the input. `bs_cstar_loss.py` copies
  the function with only that assert changed (`(len(img), rb_student.OUT)`); `training_context` patches it in only for
  non-Qwen encoders, so Qwen keeps r10's function (0(iv) still applies). Teachers, graphs and captions unchanged
  (spec §3.2), as in the 10-25 rounds. Codex (gpt-6.1-sol high); Opus `high` review APPROVE with notes, four
  sabotages caught; 73 tests in `test_bs_cstar.py` pass. CPU smokes: SigLIP 2 and CLIP L/14 train (about 100 s each)
  and held and validation exports pass. Nothing changes what X-c* trains on beyond the image input, so there is
  nothing for the run chat's spec questions.
- **Ticket 12 split** by this chat: 12a `bs_run.py` (stage0 to stage5, `status`, `standin`, `diag`), queue scripts
  `bs_run_cpu.sh <stage0-cpu|stage1|...>` and `bs_run_gpu.sh` (stage 0 GPU gates under `flock -o -w`), smoke
  `bs_run_smoke.py` (75 s, SMOKE OK); 12b `bs_diag.py` (§3.6, 11 items, `run_all`), smoke 22 s. Both plumbing,
  controller diff read. Their choices are in `controller_notes.md` (12a) and below (12b).
- **Node smoke:** tag `20261102-053122-dda5225` (launched 06:31, slot 0, SLURM job 63128): four units (X-c* SigLIP 2
  smoke, a Qwen knob row with scoring, a Qwen photo row with scoring, X photo SigLIP 2 from the smoke X-c*), then
  `estimate`. Data sync passed (all four checks yes). Next: `cluster watch <tag>` (background), then before the pull
  `mkdir -p res && ln -sfn /data/SSD2/cosir_r14/20261102_backbones_sweep/smoke_pull res/r14_bs_smoke` (the plan in
  `bs_node_plan.sh`, step 7), `cluster pull --tag <tag>`. This smoke is also the GPU proof of the fix, so a local
  SigLIP 2 GPU smoke is optional.
- **Extraction queue** (detached, no chat): at 06:32 done to `emoset_qwen_default`; left `emotion_focus`,
  `emotion_focus_photo` (about 38 min each), CLIP L/14, SigLIP 2, SigLIP 1, then `check`; expected end about 08:00.
  Status file `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage0/extract_queue_status.txt`. Watcher:
  `bash .scratch/r14-backbones-sweep/notes/build4/watch_queue.sh` (background; exits on each new status line).
- **After the queue** (local GPU, under the lock, one at a time): 0(iv) `bs_cstar.py regress-iv` (expected best epochs
  emotion 4, style 1, genre 18; pass=True), 0(v) `bs_photo.py regress-v` (best epoch 1, best dev loss
  6.508296407189546 within 1e-4); or `bash bs_run_gpu.sh`, which runs them with the parity cases and the two 0(ix)
  checks. Then ticket 04's EmoSet exports (`bs_photo.py export --arm photo_qwen|cstar_qwen --dataset emoset`) and
  ticket 08's `bs_emoset_score.py stage2 --write-record` (or `bs_run.py stage2`, which commits the record).
- **Frontier:** the final whole-branch review (Opus `xhigh`, a `worker`; brief per `matt-chain.md`: spec, tickets 01
  to 12b, hard rule 6 of `spec-templates.md`, the constitution), one fix wave, a Sonnet `high` re-review, merge into
  main, push, then the handoff to the run chat with the ticket list (index table) and open it.
- **node402:** reserved until 2026-11-04 18:06; 3 A6000.
- **Codex:** 5h 65% left at 06:20 (resets 07:50); weekly 36% (resets 11-04 08:27). Ledger row for the fix added
  (`/project/claude-config/codex/jobs.md`, not yet committed in claude-config).

## 5. Code and pitfalls

- 12a: a failed 0(vii)-sha or 0(vii)-matched is "pending"; resolve with `bs_run.py stage0 --resolve ID=ok|failed` or
  `--close-pending`. The D18 CPU re-export of an r10 ablation export has no code (manual if ever needed; 0(vii) passed
  in build 2). `status.json` is written only when every gate, all three `OUT/node/part_a_<enc>.json` and the stage 2
  status exist. `--no-commit` or `BS_RUN_NO_COMMIT=1` switches git off; `BS_RECORDS` redirects the records.
- 12b: real-data runs of the ArtELingo and Affection held sections are untested on X data (the ArtELingo section ran
  on r10's c* via the Qwen stand-in, 173 s); the uncut EmoSet section needs EmoSet exports that hold the removed
  photos, else it reports "not run". It gives no R0 for the teacher arms (only picked read-outs are stored).
- The fix: `rs_knobs.py:196` keeps the same strict assert, unreachable (knob rows are Qwen only).
- `*.json` is gitignored: committed records need `git add -f` (12a's commit step does it).
- Cluster CLI: `CLUSTER=/root/.claude/skills/cluster-run/cluster`; literal paths, no `$VAR` or `~` in arguments; run
  `watch` bare in the background. A launch needs the commit on the node (`cluster sync`); the commit carrying the
  run's change has "cluster run" in its subject (effeafc and dda5225 do).
- The unattended hook refuses `cluster clean`: node leftovers go on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: the node smoke `20261102-053122-dda5225` (cluster job; the next chat watches and pulls it); the extraction
  queue (detached). No subagents, no background job of this chat (its queue watcher was stopped).
- Uncommitted: nothing in the CoSiR repo except this cut's handoff files (committed with it); `jobs.md` in
  claude-config (synced at the wrap-up).
- On disk over 1 GB: as in the build 4 handoff (EmoSet images 2.3 GB, `EmoSet-118K.zip` 11.3 GB, the new stores; all
  keep). New: the node smoke's outputs on node402 under `/local/wding/res/r14_bs_smoke` (small). Merged worktrees and
  scratch of this chat are on `.scratch/pending_deletions.md`.
- GPU check at 06:32: driver 595.99.02 answers; the kernel module and the library match; the GPU holds only the
  extraction queue (11.4 GB). Run mode: night.
