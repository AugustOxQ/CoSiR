# Handoff: run r14: launch part a on node402 first, then stage 0 on the local machine after the extraction queue, then the stages to the report

## Brief
- Done: the r14 build. The final review (Opus xhigh) found 1 blocker (no export step) and 4 should-fixes; one fix wave fixed them; the re-review approved; main is at 4519762, pushed. The node smoke passed.
- State: node402 idle (3 A6000, until 11-04 18:06). The local extraction queue ends at about 08:00. Stage 0 has not run in production yet.
- Next: sync and launch part a on node402 (plan steps 3 to 5); after the queue, stage 0 locally.
- Blocked: nothing.

> Written 2026-11-02 07:30 by "r14 build 5" (cut at the context meter, 200k; the build is done). Loop step reached:
> step 8, run. Direction log entry: [2026-11-02](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The spec from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md)
   (§3.4 to §3.7 and §6 are the rule; §9 the timing; the report is due 2026-11-04 23:00).
3. The run order: `bash src/test/20261102_backbones_sweep/bs_node_plan.sh` (prints the node commands) and the
   docstring of `src/test/20261102_backbones_sweep/bs_run.py` (the local stages, `exports`, `decide`).
4. `.scratch/r14-backbones-sweep/` (gitignored, main checkout): `issues/00-index.md` (the ticket list and the shared
   interfaces), `notes/controller_notes.md` (all build notes; the last section is the final review),
   `notes/build5/final_review_part1.md` and `final_review_partC.md` (findings, including the notes left for the report).

Skills: `cluster-run`.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D10, approved 2026-11-02 04:13) and the direction log, 2026-11-02. Agent default of this
chat: the exports run on the local CPU after the part a pull, not on the node (every store is local; gate 0(ix)'s c*
part passed on CPU, max abs diff 6.1e-05).

## 3. Where things stand

- **Code:** main = `r14-backbones-sweep` = 4519762 (pushed), round folder `src/test/20261102_backbones_sweep/`.
  Round suite: 309 passed (70 s). Runner smoke `bs_run_smoke.py <scratch dir>`: SMOKE OK.
- **Final review** (three Opus xhigh parts; part 1 cut at 200k context). Re-derived and matched: 0(iv)'s expected
  values (best epochs 4 / 1 / 18) from r10's records and 0(v)'s (best epoch 1, dev loss 6.508296407189546) from r13's;
  the EmoSet near-copy cut, field for field (val 135 of 5,905 removed, test 368 of 17,716); 0(xi) `--recompute` PASS
  (62 items, diff 0.0); one 0(xii) K cell, diff 0.0; the 10-25 and 10-26 teacher partitions are byte-equal. 9 of 12
  sabotages caught by this round's tests, the other 3 by older rounds' tests or a run-time check (2 tests added since).
- **Fixed in the fix wave:** an `exports` step (F1); an X photo failure no longer stops the X-c* seeds (F2); stage 2
  now runs after part a, so the X photo models' EmoSet rows are scored (F3); stage 3 writes cells only, and a new
  `decide` refuses until `RECORDS/c13.json` (C13 pass) is committed with matching hashes (N1); the strongest PercepT
  entry is bs_diag item 12 (N2); small fixes F4, N3 to N7, an estimate label. Then the re-review's two notes: exports
  always run (the CLIs check existing files against the current weights), and the C13 hash map also covers
  `status.json` and `nearcopy_emoset.json`.
- **The strongest PercepT entry** (descriptive, held AMI, 5-seed mean; 20-cluster / own topics): emotion B1 (Qwen)
  0.0799 / 0.0874; style B6 (CLIP B/32) 0.2280 / M7 (CLIP L/14 matched) 0.2306; genre M2 (SigLIP 2 matched)
  0.4558 / 0.4243. Matched arms are counted, as in the 10-25 spec's list (line 117).
- **Node smoke** `20261102-053122-dda5225`: four units ok (X-c* SigLIP 2, a Qwen knob row, a Qwen photo row, X photo
  SigLIP 2), pulled (54 files, `res/r14_bs_smoke`). It is the GPU proof of the c* width fix. Its estimate: all node
  work done by 09:09 with 3 GPUs, but from smoke units (2 epochs, unscaled); with the 3090's full training times it is
  about 11:33 (both counted from 06:39). The A6000 speed is not known. Both end before 11-04 18:06.
- **Extraction queue** (detached, not this chat's): done to `emoset_emotion_focus_photo` at 07:24; left EmoSet
  CLIP L/14, SigLIP 2, SigLIP 1, then `check`; expected end about 08:00. Status file
  `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage0/extract_queue_status.txt`.
- **Stage 0 in production:** not run (`OUT/stage0` has no `gate.json`). The light checks of the build (0(iii), 0(ix),
  0(xii) F) ran into OUT or scratch; the runner's `stage0` records the gates.

## 4. The run order

Node (bs_node_plan.sh), from now:
1. `cluster sync --node node402`. HEAD's subject has no "cluster run": the node code changed in fix A (2284e2e,
   `bs_node.py`, merged at 650f346) and 4519762 is the reviewed state. Pass `--allow-any-branch` on that sync with a
   one-line note in the run log, or put "cluster run" in the subject of the run chat's first commit that changes code.
   No empty commit for the marker (SKILL.md, step c).
2. Plan step 3: 0(viii) and prep (one GPU each); wait for prep's done line before step 5 (the k80 graph race).
3. Plan step 4: part a, three launches (siglip2, siglip1, clip_l14). Plan step 5: the sweeps, three workers.
4. After each job: `cluster watch <tag> --node node402`, then `cluster pull --tag <tag> --node node402` (first create
   `res/r14_bs -> /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs`, plan step 7), then `bs_node.py estimate`.

Local (`bash src/test/20261102_backbones_sweep/bs_run_cpu.sh <part>`; GPU: `bs_run_gpu.sh`):
1. After the queue's `check` line: `bash bs_run_gpu.sh` (gates i, iv, v, ix under the lock; about 15 to 20 min), then
   `stage0-cpu` (gate ii needs the sealed stores; the 0(xii) K replay takes over 4 min: run it in the background).
2. `exports-qwen` (about 1 min).
3. After each encoder's part a pull: `exports-part-a --enc <enc>`.
4. `stage1`, then `stage2` (it refuses before part a and the exports), then `stage3` (cells only; it prints the hash
   map and writes `OUT/stage3/c13_sha256.json`).
5. C13: Codex re-derives, with separate code, from the spec's rule and the saved outputs (§3.4, line 220): the
   near-copy sets, the seed ranges, the picks, SC1 to SC10 with intervals and outcomes, within 1e-6. On a pass, the
   run chat writes and commits `RECORDS/c13.json` (format in `bs_run.py` `run_decide`'s docstring; `git add -f`).
6. `decide`, `stage4`, `stage5`, then the report (auto report with its results brief, and the user-read report).

## 5. Code and pitfalls

- **Cluster CLI:** `CLUSTER=/root/.claude/skills/cluster-run/cluster`; always `--node node402`: without it `status`
  and `watch` look at node401 and fail. Literal paths, no `$VAR` or `~` in arguments. The unattended hook refuses
  `cluster clean`: node leftovers go on `.scratch/pending_deletions.md`.
- **For the report** (not fixed, by design): F5, a failed stage0.paintings gate also makes the photo SC6 cells
  inconclusive (conservative, as ticket 09 says); N9, the photo sheet's d is r13's cross-fit difference, its interval
  the pooled bootstrap; N10, bs_diag has no R0 rows for the teachers; N8, the D18 re-export and matched-rerun
  branches have no code (moot while 0(vii) passes).
- 12a: a failed 0(vii)-sha or 0(vii)-matched is "pending" (`stage0 --resolve` or `--close-pending`).
  `status.json` is written only when every gate, all three `OUT/node/part_a_<enc>.json` and the stage 2 status exist.
- `*.json` is gitignored: committed records need `git add -f` (the runner does it; `c13.json` by hand).
- The CoSiR env has no pytest-xdist (no `-n`). `bs_run_smoke.py` needs a scratch dir argument.
- `git fetch .` fails here (dubious ownership): move a ref with `git update-ref` after `merge-base --is-ancestor`.
- `rs_knobs.py:196` keeps the strict width assert (knob rows are Qwen only).

## 6. State at handoff

- Running: the extraction queue (detached; ends about 08:00). No subagents, no cluster jobs, no background job of
  this chat. The main checkout is on `r14-backbones-sweep` (= main): do not switch branches while the queue runs (it
  reads its scripts from this checkout); after it ends, `git checkout main` changes no files.
- Uncommitted: nothing in the CoSiR repo except old untracked files.
- On disk over 1 GB: as in the build 4 handoff (EmoSet images 2.3 GB, `EmoSet-118K.zip` 11.3 GB, the new stores; all
  keep). New this chat: the node smoke pull (264 MB local, 263 MB on node402) and the build's worktrees and scratch,
  all on `.scratch/pending_deletions.md`.
- GPU check at 07:25: driver 595.99.02 answers; the kernel module and the library match; the GPU holds only the
  extraction queue (2.5 GB). Run mode: night.
- Codex: 5h 65% left at 07:02 (resets 07:50); weekly 36% (resets 11-04 08:27). No Codex job in this chat (the fix
  wave went to Claude: no room).
