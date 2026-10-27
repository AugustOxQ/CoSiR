# Handoff: finish the recipe sweep build fast (the 11 fix wave and ticket 15 at once, node checks, final review), then cut to the run chat

## Brief
- Done: 09, 10, 12 and 13 merged (09, 10, 12 after Codex fix waves and re-reviews); gates V1a, V0a and V4a pass; 11's and 12's self-checks pass.
- State: 13 of 15 tickets merged. Ticket 11 is reviewed, with 2 must-fix items: the F = G key, and partitions never checked equal.
- Next: 11's fix wave and ticket 15 at once; cluster sync, N1, N2; then the final review.
- Blocked: nothing. The user wants stage 1 as soon as possible, with no step skipped. Codex has no room until 18:02.

> Written 2026-10-27 14:36 by "r10 build 10" (cut at the context meter's 200k note). Loop step reached: step 7, the
> build, 13 of 15 tickets merged (01 to 10, 12, 13, 14). Direction log entry: [2026-10-27](../direction_log.md).

## 1. Read in this order

1. `.scratch/r10-recipe-sweep/notes/build8_reviews.md` (outside git), from "# Build 10 additions" on: every hand-back,
   review and decision of this chat, with paths.
2. `notes/review_11.md` (the 11 review, partial: no mutation table) and `notes/rereview_12.md`.
3. The briefs this chat used, as models: `notes/brief_12fix.txt` (a fix wave under the independence rule C13),
   `notes/brief_09rereview.txt` (a re-review), `notes/brief_15.txt` (ticket 15, ready: fill WT, branch and tip).
4. The ticket index `.scratch/r10-recipe-sweep/issues/00-index.md`; `notes/implementer_brief.md` (commit lines: say
   build 11 and the new session).
5. The spec (`docs/superpowers/specs/2026-10-27-recipe-sweep-design.md`) from §1, the rule
   `src/test/20261027_recipe_sweep/DECISION_RULE.md`, `~/.claude/references/matt-chain.md`; the build 10 handoff
   §3 to §5 (`2026-10-27-r10-build-10-handoff.md`; the gate list and pitfalls still hold).

Skills: `cluster-run` (cluster sync, N1 and N2 on node402).

## 2. Decided by the user, not to be reopened

- Decisions A, B, C, the grill and the brief: as in the build 7 handoff §2 (approved 2026-10-27 03:48).
- 13:00: the Codex fix is done; the normal routing is back (Codex first, `agent-routing.md`).
- 14:2x, asked when stage 1 would start on the node (estimate about 20:00, range 19:00 to 21:00; an early start
  before the final review was offered and not recommended): "start as quick as possible, but not at the cost of
  skipping processes". So the normal order holds (the final review before stage 1), and the remaining work runs in
  parallel.
- Night mode (`run-mode`: night at 12:57 and at the cut). GPU check at the cut passed (driver 595.99.02 on all three); no restart asked (night mode: check only).

## 3. Where things stand

| Ticket | State | Where |
|---|---|---|
| 01 to 08, 14 | merged (builds 7 to 9) | `r10-recipe-sweep` |
| 13 analysis | merged (diff read, smoke 28 s; G's real table: diagonal 0.104 / 0.406 / 0.500, T 0.092 / 0.403 / 0.506) | cae1c7a |
| 10 ablation | merged after the Codex fix wave (all 13 items; 26 of 26 mutants caught at re-review; effects equal the old smoke) | 26f556d |
| 09 stage 4 | merged after the Codex fix wave (re-review: merge, 25 of 26 mutants; dry run equal: emotion +0.0118 T-WIN, style +0.0032 T-LEVEL, genre −0.0060 T-LEVEL, margin +0.0164 PE-GO). One test gap left: MS3e | 844cad8 |
| 12 dev re-derivation | merged after the Codex fix wave (re-review: merge, 27 of 29 mutants; e2e PICK 165 items and ablation 99 items, 0 differ; F = G function equals rs_full's on 100 of 100 cases); full self-check 45 items, 0 differ (57 to 45 is expected: the draws mode swaps point items for interval items). Left: run the points-only self-check once (`rs_vp.py self-check`, no `--draws`; expect 57 items, 0 differ) | f91fcfa |
| 11 held re-derivation | finished (82 tests; full self-check 75 items, 0 disagree, difference 0); reviewed (Opus, partial): 2 must-fix, 2 should-fix, 5 nits, no mutation check yet | `worktree-agent-a9b6b08a3a20079fc`, f538a16 |
| 15 queues | not started; brief ready (`notes/brief_15.txt`, run notes a to h) | |

Gates (real outputs, `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/gates/`):
- V1a training: passes again at 13:04 (bd34337; s0, s1 bit for bit; about 2 minutes, not 15). The stale record is in
  `.../stale/20261027_v1a_before_build10/`.
- V0a: passes at 13:13 (cae1c7a; cpu 381 s, gpu 8 s).
- V3a: passes (12:10, build 9). V4a: the scorer passes at 14:34 (max difference 0, 165 s) and the training passes at 14:35 (844cad8).
- `rs_full.py` (changed by 09) is hashed only by V4a: rs_stage3, rs_sweep, rs_graphs and rs_stage0 do not import it.

## 4. Open points for the build (agent defaults, no user question)

Order, as fast as the rules allow (at most three implementers at once):
a. At once, in parallel: ticket 15 (Codex after 18:02 if `codex-usage --room implementer` says room, else a Sonnet
   `medium` worker: plumbing; brief `notes/brief_15.txt`, with the controller's worktree); the 11 fix wave (Sonnet
   `high`, or Codex after 18:02: both must-fix, both should-fix, cheap nits, C13 kept; give it 09's merged
   no-outcome shape as facts: top-level `v_failure`, outcome None per aspect and on the margin, numbers omitted); a
   Sonnet `medium` worker adds the MS3e test (09's `refuse_second_candidate` call in `pick_validity`,
   rs_stage4.py:295), checked red on the mutant.
b. 12: merged. Run its points-only self-check once (CPU, background).
c. After 11's fixes: a Sonnet `high` re-review that also runs the 30 to 40 mutants the review could not (clone
   `/tmp/claude-0/r10-review-11/mut/` at f538a16, original `rs_vh_orig.py` there; `-k 'not k4_sample'`, about 6
   mutants per call). Then the full self-check again (`rs_vh.py selfcheck`, about 8 minutes), then merge.
d. Gates after the last merge that touches their hashed files: V4a passes after 09; the full-scale
   removal smokes in scratch (`G-no_direct`, `G-no_content_correction`, `G-no_instructed_graphs`; commands in
   `/tmp/claude-0/r10-fix-07/` notes and `rs_full.py --help`; GPU, minutes).
e. `cluster sync` (a commit with "cluster run" in its subject) and N1, N2 on node402 (build 9 handoff §4 d) as soon
   as 09 is merged: they do not wait for 11, 12 or 15 (the node runs stage 1 and 2 code, merged and gated). Check
   the node402 reservation first (not checked in this chat).
f. The final whole-branch review (Opus `xhigh`) as soon as every ticket is merged; one fix wave; a Sonnet `high`
   re-review; merge into main, push; then cut to the run chat (ask about a container restart: night mode, check only).
g. Codex: the five-hour window is at 8% until 18:02. Three fix waves cost about 92% together (ledger rows added):
   a review fix wave costs like an implementer job, not the 10% "bug" estimate. Note it at the wrap-up's routing check.

## 5. Code and pitfalls

- Agent worktrees start at `main`: each new implementer resets onto `r10-recipe-sweep`.
- `.claude/20261027_log.md` is tracked; merges conflict there: keep both entries.
- F = G: the scorers write G's run key as 'GS-0.02' (`rs_paths.G_ARM`), never 'G'. The one rule for F = G without a
  stage 3 file is `rs_full.sweep_ended_without_candidates`; the re-derivations (11, 12) write their own from the rule.
- A `v4c.json` written before 25e84d3 lacks `output_files` and stage 4 refuses it (the real c* held run comes later).
- Reviewers mutate a `git clone --shared` copy under `/tmp/claude-0/`; git there needs a `safe.directory` entry in a
  temporary HOME; restore by copying, never `git checkout`.
- Example files for end-to-end checks: 09 `/tmp/claude-0/r10-fix-09/outputs/stage4/`; 10
  `/tmp/claude-0/r10-fix-10/outputs/ablation/`; a stage 3 PICK `/tmp/claude-0/r10-review-12/pick_outputs/`. Work in
  symlink folders; never write into them.
- Gates bind code SHA-256s; rerun a gate after the last merge that touches its hashed files.

## 6. State at handoff

- Running: nothing (every subagent and Codex job handed back; no GPU lock, cluster job or background run of this chat).
- Uncommitted: nothing after the cut commit (`.scratch/` notes, briefs and ticket status lines are outside git).
- On disk over 1 GB: `/tmp/claude-0/r10-rs-10/` (4.1 GB), `/tmp/claude-0/r10-fix-10/` (1.7 GB),
  `/tmp/claude-0/r10-review-12/` (1.3 GB), `/data/SSD2/cosir_r10/20261027_recipe_sweep/stale/` (1.0 GB), node402
  `/local/wding/r10_rs/` (27.4 GiB). All on `.scratch/pending_deletions.md`, with the smaller scratch of this chat.
