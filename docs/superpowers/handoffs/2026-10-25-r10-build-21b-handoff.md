# Handoff: build the encoders-and-genre round from its 16 tickets (implementers, ticket reviews, merges, gates, final whole-branch review, merge into main), then open "r10 run 21"

## Brief
- Done: 16 tickets and their index written from the spec and rule; foundation (paths, encoder table, stage iii variant constants) committed on branch `r10-encoders-genre`; a code map of the reused 12-24 code; the three new checkpoints cached and checked loadable.
- State: no ticket started; Codex 77% of its five-hour window left (reset 21:26).
- Next: dispatch the first wave (01 Codex, 05 and 12 Sonnet `high`), then work the frontier.
- Blocked: nothing.

> Written 2026-10-25 19:12 by "r10 build 21" (cut at 200k tokens of context, after the tickets). Loop step reached:
> step 7 (build), tickets done, no implementation yet. Direction log entry:
> [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the build handoff before it, [2026-10-25-r10-build-21-handoff.md](2026-10-25-r10-build-21-handoff.md)
   §2 to §5 (decided items, inputs on disk, open points a to e, pitfalls); they still hold.
2. The ticket index `.scratch/r10-build-21/issues/00-index.md` (conventions, shared interfaces, output layout,
   ticket table with implementer per ticket, coverage line), then each ticket when you dispatch it.
3. `~/.claude/references/matt-chain.md`, `~/.claude/rules/agent-routing.md`, `~/.claude/references/codex-jobs.md`.
4. Only when a brief or a review needs it: the spec, the rule, `.scratch/r10-build-21/notes/code_map.md` (52 KB; cite
   its sections in briefs, do not paste it).

Skills: `tdd` (named in verdict-code implementer briefs), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

The spec's §4 rows marked "user" or "user, after seeing the recommendation" (D1 to D17) and the brief's approval at
18:35. The container restart before the build (D16) was done at 18:44 (container start time); the GPU check passed
at 18:46 (driver 595.99.02, GPU idle). This cut is inside the same build phase, so no new restart question was asked;
the cut to the run chat asks again (`loop-chats.md`, "How to cut").

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at efc45a4, worktree `/project/CoSiR-r10-int`, pushed. It holds
  `EG/eg_paths.py`, `eg_encoders.py` (`extend_rb_maps()` tested: adds K6 to K8, M6 to M8 to every rb_ map, K5 entries
  unchanged), `eg_variants.py`, `conftest.py`, and the `.gitignore` line for `EG/outputs`.
- **Outputs** folder `/data/SSD2/cosir_r10/20261025_encoders_genre/outputs` exists (empty), linked from the
  integration worktree's `EG/outputs`; the main checkout gets its link after the merge (`ln -sfn`).
- **Checkpoints** in the HF cache, loaded on CPU 2026-10-25 19:00: CLIP B/16 (projection 512), CLIP L/14 (768),
  SigLIP 1 (SiglipModel, 1152, SiglipProcessor, 64-token padding works; transformers 5.6.2). CLIP B/16's main branch
  has only `pytorch_model.bin`; transformers also fetched the safetensors revision, as for B/32. Spec §5's first
  assumption is now checked.
- **X0's old extractor:** `eg_paths.X0_OLD_COMMIT` = 4572189 (main when the branch was cut; the extractor last
  changed at 647867f, so the code is the same).
- **Codex:** five-hour window 77% left at 19:09 (reset 21:26), weekly 91%. An implementer job needs 80% room
  (estimate 40%), so before 21:26 ticket 01 may have to go to Claude unless `codex-usage --room implementer` says
  room; after the reset, Codex can take 01 (if not started), then a re-derivation ticket (09 or 14, kind
  `rederivation`, 10%).

## 4. Open points for the build (agent work; nothing for the user)

a. **Waves** (at most three implementers at once; verdict-code tickets get an Opus `high` reviewer before merge):
   first 01, 05, 12 (heads of the longest chains: 05 to 06 to 10; 12 to 13 to 14); then 08, 11, 03, 04, 07, 02 as
   slots free; then 06, 13, 09, 14, 10, 15; 16 last. The implementer column in the index is the plan; Codex takes
   01, 09 or 14 when `--room` allows.
b. **Brief template per ticket:** the ticket path, the index path, "work in your worktree based on
   `r10-encoders-genre`; merge its tip before you hand back", the code map sections the ticket names, and for
   verdict code `tdd`. Implementers run with `isolation: "worktree"`; the controller merges clean branches itself.
c. **GPU work during the build** (short, under the lock, one at a time): the smokes, X0 (after 01), X1 (after 03, 04,
   05: K2 and M2 seed 0 on the sample, about 5 min), X2 (full scale, about 12 min), VG1 and the fresh-seed parity
   (after 11, 12: about 6 min), VL1 (after 07, 08). Their JSON files then already exist for the run chat; record
   each result in the run handoff. Store extraction (about 1.5 h) and every stage are the run chat's.
d. **Final whole-branch review** (Opus `xhigh` worker, `matt-chain.md`), one fix wave, a Sonnet `high` re-review;
   then merge into main, push, link `EG/outputs` in the main checkout, list worktrees and ticket branches on
   `.scratch/pending_deletions.md`.
e. **The run chat's handoff** as the build handoff §4 e says: ticket list (title, blocked by, spec IDs, verdict code),
   the queue order (stage iii first), the commands, the gates already passed in the build, the end estimate against
   11:00 Monday 2026-10-26 (spec D15: else start Monday 19:00), deferral order K5, K8, K7, K6; then the GPU check,
   the restart question (`loop-chats.md`, "How to cut", step 3) and `loop-next <handoff> --label "r10 run 21"
   --unattended`.

## 5. Code and pitfalls

- The code map's §2 table names every rb_ body that hard-codes the old lists (`rb_score.arms_for`, `output_path`,
  `rb_run`, `rb_a28.default_models`, the CLIs); those are copied, never edited. `extend_rb_maps()` covers the maps.
- rb_ held inference calls 12-24's own gate (`part2/regression/stage2.json`); tickets 03 and 04 copy the gate with
  this round's semantics.
- `rb_matched._same` compares values, not bits; X1 and X2 compare raw bytes (ticket 04).
- Adding the GA adaptor's parameters to AdamW could change the base trajectory through foreach batching: VG1 runs with
  no adaptor at all (ticket 11).
- `rb_paths.OUTPUTS` reads `RB_OUTPUTS` at import; never set it in this round, and pass output folders explicitly.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing (tickets and notes are in `.scratch/`, outside git).
- On disk over 1 GB: the new checkpoints in the HF cache on `/data/SSD2` (CLIP B/16 1.2 GB, CLIP L/14 1.6 GB, SigLIP 1
  3.3 GB; shared cache, kept).
