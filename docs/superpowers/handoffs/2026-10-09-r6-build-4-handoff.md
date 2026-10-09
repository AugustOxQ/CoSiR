# Handoff: build round 6 from ticket 07 onward (implement-spec), unattended, then cut to the run chat

> Written 2026-10-09 15:16 by the `r6 build 3` chat, cut at a clean point because its context passed 200k tokens.
> Loop step reached: step 7. Eleven of fifteen tickets are merged on the integration branch, and DAS6 preparation is done.
> Direction log entry: [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r6-held-test/issues/00-index.md` (local build tracker, outside git). Each
   ticket's `Status:` line says done or ready. Tickets 06, 07, 08, 11 and 13 end with "Controller notes" that bind
   their implementers. 07, 08 and 13 have a new note on `score_seed(..., include_pm)`.
2. In `.scratch/r6-held-test/notes/`:
   - `implementer_brief.md` and `reviewer_brief.md`: both now carry the delete-nothing line; update the brief's
     session line to your chat.
   - `contracts.md`: read every "Amendment (controller ...)", the newest at 14:50 (`include_pm`, R1's counterpart).
   - `deferred_nits.md`: held for the final fix wave; T06 and T11 rows added.
3. Still binding:
   - the build-3 handoff [2026-10-09-r6-build-3-handoff.md](2026-10-09-r6-build-3-handoff.md) §2, §4 (after the
     tickets) and §5;
   - the build-2 handoff [2026-10-09-r6-build-2-handoff.md](2026-10-09-r6-build-2-handoff.md) §5 (the final review;
     what the run handoff must carry);
   - the first build handoff [2026-10-09-r6-build-handoff.md](2026-10-09-r6-build-handoff.md) §2 to §5.
4. `~/.claude/references/matt-chain.md` ("implement-spec, with our changes") and `~/.claude/references/loop-chats.md`
   ("Unattended chats").
5. The run log `src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md` on the integration branch,
   rows from 13:33 on.

## 2. Decided by the user, not to be reopened

- As the first build handoff §2: the spec's §4 and the rule, approved 2026-10-09 04:53. Build and run are unattended.
  The run chat closes by opening `r6 read` with `--notify`.
- **Delete nothing in this unattended build** (user, relayed by the CoSiR main chat at 14:00). This covers
  implement-spec's step 9, `git worktree remove`, `git branch -d/-D`, `cluster clean`, and `rm` outside the session
  scratchpad. Each item instead gets one row in `/project/CoSiR/.scratch/pending_deletions.md` (`| added | chat | path
  or branch | size | what it is | why it can go |`) and a line in the handoff's storage list. A verified `cluster pull
  --tag` still removes the job's code copy on the node (storage.md's automatic step). Every subagent brief carries the
  line.

## 3. Where things stand

- Integration branch `r6-held-test`, worktree `/project/CoSiR-r6`, pushed.
- Merged and reviewed: 01, 02, 03, 04, 05, 09, 10 (build 2), and in this chat:
  - 12 (controller-checked);
  - a `das6_sync_r6.py --ckpt` action that ships the two FT checkpoints;
  - 11, reviewed. Its fix round made failed-parse rows score as cosine at every λ (λ = inf included) and stopped late
    DTS from being called built;
  - 06, reviewed. Its fix round keeps R1's counterpart behind `include_pm=True`. Merged at a5be181; the head is
    d7399ab.
- Re-derived by reviewers in this chat:
  - B, B0, B1 seed-42 targets from round 4's `dev_seed42.json`, through the runner's own path;
  - round 3's `seed42_arrays.npz` bit for bit by `score_seed`;
  - rule §7's parsing, CRL, tuning subset, tie order, stop (> 9,406, int64) and budget, character for character.
- **Comparator clock (rule §7.7): started 2026-10-09 12:29** (6beb360). The 24 hours end **2026-10-10 12:29**. DTS
  must be built by then (DTS-N sanity passed and the chosen setting run on all 12,288 seed-42 episodes).
- **DAS6 preparation is done** (log rows 13:38 to 14:10):
  - node401, node402 and node408 each have an RTX A6000 on slot 0, three GPUs free, the env present, and the
    selftest passed;
  - check-only smokes of the verbaliser, listing, reranker and FT feature jobs all succeeded on node401;
  - seed-42 images (5,245, 1.72 GB) are staged on all three nodes;
  - the FT checkpoints are on node401 only (`--ckpt` copies them elsewhere);
  - held images ship after the run chat builds held episodes.
- Measured rates on DAS6 (for the run plan):

  | Job | Rate | Run-sized load |
  |---|---|---|
  | Verbaliser | 0.80 s per call, steady state | seed 42: 24,576 tuning + 18,432 chosen-wording calls, about 9.5 GPU-hours (about 1.1 h on nine GPUs) |
  | Reranker | 4.4 s per episode | held: 36,864 episodes, about 45 GPU-hours (about 5 h on nine GPUs) |
  | Model load | 17 s warm, 60 s cold | |

## 4. Next steps

- Wave plan, at most three implementers at once:
  1. **07** (blocked by 04 and 06; Opus).
  2. **08** (07) and **13** (07, 09) together.
  3. **14** (11, 12, 13).
  4. **15** (08, 09, 14) last.
- Every verdict-code ticket gets an Opus reviewer before its merge. Give the reviewer the load-bearing numbers to
  re-derive and where the stored values are (rule §6.4's regression list for 07).
- After all tickets:
  1. The final whole-branch review on Opus, folding in `deferred_nits.md`, then one fix wave and a scoped re-review.
  2. Merge into main and push.
  3. **No removal of worktrees or branches**: list them in `pending_deletions.md`. Nine rows are already there, plus
     tickets 11's and 06's; add the later tickets'.
  4. Fix CoSiR's `CLAUDE.md` test command: add `MKL_NUM_THREADS=8`.
  5. Write the run handoff (contents as build-2 handoff §5, plus §5 below), then `loop-next <handoff> --label "r6
     run"` (no `--notify`).

## 5. Code and pitfalls (new in this chat; also for the run handoff)

- **Every describe-then-score model call goes to DAS6, none to the local GPU.**
  - DAS6 runs torch 2.14.0 and transformers 5.16.1; local runs 2.11.0 and 5.6.2. One of four local verbaliser answers
    differed from DAS6's.
  - Within DAS6 the answers repeat exactly (4 of 4).
  - Listing answers batched vs one by one agreed 3 of 4 on DAS6, so the run chat passes earlier listing folders to the
    listing job with `--cache`, and no (phrase, K) pair is generated twice. Otherwise `merge_listings` refuses.
- Feed the chosen-setting stage only one verbaliser job's outputs per key. A key answered twice with different text
  refuses by design.
- A DAS6 launch needs HEAD's commit subject to contain "cluster run" before `cluster sync`. Make the log commit carry
  it.
- `cluster watch` and `cluster logs` need `--node` for jobs off the default node.
- `das6_sync_r6.py` runs with `/usr/bin/python3`. The job folder name in `/local/wding/r6_jobs/` is the local folder's
  name.
- The run log's rows stamped 13:50 to 14:12 by build 2 ran ahead of the clock (corrected in the 13:33 row).
  Take times from `TZ=Europe/Amsterdam date`.
- Subagent prompts end with the permission line ("do the work another way or report it; do not wait") and the
  delete-nothing line.

## 6. State at handoff

- Running: nothing. No subagent, no GPU job, no DAS6 job (every smoke and selftest tag pulled).
- Uncommitted: nothing; `.scratch/` is local by design.
- Storage (nothing deleted; for the next storage summary):
  - `/project/CoSiR/.scratch/pending_deletions.md`: eleven merged ticket worktrees (about 68 MB each) and their branches.
  - `/project/CoSiR-r6/src/test/20261125_artelingo_held_test/results/` (30 MB): the smoke GPU inputs and outputs
    under `smoke/gpu/`; `prep_gpu/` (seed-42 episodes and the pre-stage job folder, which the run chat may reuse);
    `gpu_images/` (symlinks).
  - `/project/CoSiR-r6/res/cluster_jobs/20261009-11*` (1.4 MB): five pulled smoke and selftest tags.
  - DAS6 node401, under `/local/wding/r6_jobs/`: `s9001_verbalise`, `smoke_listing`, `s9001_rerank`, `s9001_ft`,
    `s42_verbalise_prestage`, `images/` (about 2.5 GB, which the run needs) and `ckpt/` (50 MB).
  - node402 and node408: `s42_verbalise_prestage` and `images/` (1.72 GB each, which the run needs).
