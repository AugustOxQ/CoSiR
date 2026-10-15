# Handoff: build round 6 from its 15 tickets (implement-spec), unattended, then cut to the run chat

> Written 2026-10-09 05:24 by the `r6 build` chat, cut at its first clean point because its context passed 200k
> tokens (the rule, the spec and two code maps). Loop step reached: step 7, tickets written, no code yet. Direction log
> entry: [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r6-held-test/issues/00-index.md` (local build tracker, outside git): the
   15 tickets, blocking edges, verdict-code flags, models, the coverage line (checked: every SC and D ID of the spec
   and every rule section maps to a ticket or a named non-ticket step) and the wave plan.
2. `.scratch/r6-held-test/notes/implementer_brief.md` and `contracts.md`: what every implementer gets, and the APIs
   and file formats shared between tickets. Implementers also read the code maps `cpu_path.md` and `gpu_path.md`
   there; the controller opens them only to settle a question.
3. The first build handoff [2026-10-09-r6-build-handoff.md](2026-10-09-r6-build-handoff.md) §2 to §5: still binding
   (decisions, where the build ends, pitfalls, the amendment: the run chat closes by opening `r6 read`).
4. `~/.claude/references/matt-chain.md` ("implement-spec, with our changes") and `~/.claude/references/loop-chats.md`
   ("Unattended chats"). The rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md` when a review needs it.
5. The run log `src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md` on the integration branch.

## 2. Decided by the user, not to be reopened

- As the first build handoff §2: the spec's §4 and the rule, approved 2026-10-09 04:53; build and run unattended.

## 3. Where things stand

- Integration branch `r6-held-test` (from main at ba6d290), checked out in the worktree `/project/CoSiR-r6`; keep
  `/project/CoSiR` on main, where a parallel `r7 decide` chat may commit (it stays off DAS6 and off the round folder).
  The run log is committed there (3d9ba46) and the branch is pushed.
- No ticket started. Nothing launched locally or on DAS6.

## 4. Open points for the next step

None for the user. Agent defaults set in the contracts (record them in the log as they land): GPU jobs get
pre-permuted candidates and no candidate order, labels or aspect names (rule §8.3 read strictly); "first line" of an
answer is taken after stripping surrounding whitespace; normalisation strips Unicode P* characters; the listing job
batches with a fixed batch size; value strings are embedded after normalisation, without a template; the head
predictions on selection rows stay one call on exactly the selection rows.

## 5. Code and pitfalls

- **Run the tickets** per implement-spec: at most three implementers at once, each `isolation: "worktree"`, in the
  background, with the ticket path and the brief path as the prompt's pointers. Each implementer resets its worktree
  onto `r6-held-test` first. Verdict-code tickets get a reviewer subagent (not haiku) before the merge; ticket 12 is
  checked by the controller (diff and tests). Merge clean branches yourself in `/project/CoSiR-r6`; push the branch.
- **Data in worktrees:** gitignored inputs exist only in `/project/CoSiR`; `r6_common.MAIN` resolves it (contracts §1).
- **The comparator's 24-hour clock** (rule §7.7) starts at the first commit of DTS code (ticket 10). Log that commit's
  Amsterdam time and pass it to the run chat; the wave plan starts ticket 10 only after ticket 02 is merged.
- **DAS6 preparation (controller, after tickets 10 and 12 are merged):** read the `cluster-run` skill; a commit whose
  subject says "cluster run"; `cluster sync` per node; `cluster launch --node <n> -- cluster-selftest gpu` on node401,
  node402 and node408 (GPU model and memory into the log); sync the 8B snapshot (likely already on the shared
  `/var/scratch/wding/cache/hub`), the seed-42 and smoke job inputs and their images with `scripts/das6_sync_r6.py`
  (`/local/wding` is per node); then a 2-episode `--check-only` smoke of the verbaliser, listing, reranker and FT
  feature jobs on selection rows. Watch the `cluster pull` fallback (once pulled 29.95 GB). Never hand-rolled ssh or
  rsync; never `/tmp` on a node. A local GPU smoke instead is allowed only under `flock -o -w` on `/tmp/gpu0.lock`.
- **No held row in the build.** The run chat creates held episodes; the build tests use selection rows and synthetic
  data.
- After all tickets: the final whole-branch review on Opus (`final-review.md`: re-derive every load-bearing number,
  hard rule 6 with its four gap types), one fix wave, a scoped re-review; merge `r6-held-test` into main locally, push
  main, remove the ticket worktrees and branches. Then the run handoff and `loop-next <handoff> --label "r6 run"` (no
  `--notify`).
- **The run handoff must carry:** the ticket list (title, blocked by, spec IDs, verdict code; from the index); the DTS
  clock start; the order of rule §6 with each runner's command; the H5 row format (contracts §7) to commit before
  launch; the agreement record format the re-derivation agent writes (contracts §8); the GPU job commands and the
  held image sync; that it closes by opening `r6 read` with `--notify`; that an `r7 decide` chat may run in parallel
  off DAS6; the storage list for the next summary (nothing is deleted in unattended chats).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing; `.scratch/` is local by design.
- On disk over 1 GB: nothing. The worktree `/project/CoSiR-r6` is about 67 MB.
