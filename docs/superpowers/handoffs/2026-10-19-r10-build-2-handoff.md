# Handoff: finish the PercepT core build (tickets 02, 04, 05, 08), smoke, final review, merge, then open the run chat

## Brief
- Done: tickets 01 (readers), 03 (the four fixes), 06 (scoring) and 07 (rows 6, 10, 13 and the gate), built by Codex,
  reviewed by Opus, merged into `r10-percept-core`. Cut at 200k context.
- State: the new scorer reproduces row 2 (PASS). No real runs yet.
- Next: tickets 02, 04 and 06b on Codex (task files ready), then 05, then 08; smoke; final review; merge.
- Blocked: nothing.

> Written 2026-10-19 00:19 by "r10 build 1". Loop step reached: step 7 (build), half done. Direction log entry:
> [2026-10-18, aspect graph follow-up](../direction_log.md).

## 1. Read in this order

1. This handoff, then the ticket index `.scratch/r10-percept-core/issues/00-index.md` (shared interfaces, agent
   defaults, coverage) and tickets 02, 04, 05, 08 in the same folder.
2. [The spec](../specs/2026-10-18-percept-core-design.md), from §1. The build's only source of what to build.
3. `~/.claude/references/matt-chain.md` (our changes to implement-spec; the final whole-branch review) and
   `~/.claude/references/loop-chats.md` ("Unattended chats", "How to cut").
4. The code map `.scratch/r10-percept-core/notes/code_map.md` (cited as CM-<item>), only where a ticket cites it.

Skills: implement-spec (read `~/.claude/skills/implement-spec/SKILL.md`; slash-only).

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15) and its approval, 2026-10-18 23:34. No container restart (23:34).
- Run mode night since 2026-10-18 23:40 (the user: "switch to overnight-run mode. I am going to sleep."). So no
  questions: agent defaults, recorded in the handoff.
- The user frames this as exploration: "as many trials as possible to enlarge the ablation table and then pick".

## 3. Where things stand

| # | Ticket | State |
|---|---|---|
| 01 | Frozen-sample readers (`pc_data`) | merged; Opus review: merge (every join re-derived, CLIP re-encoding confirms); controller fix: RoBERTa reader returns an array |
| 02 | RoBERTa extraction (`pc_roberta`) | ready; `codex/task_02.txt` |
| 03 | The four fixes (`pc_fixes`) | merged; Opus review: merge (18 of 19 mutants caught); controller fix: fixed sign rule for the PCA and QR maps |
| 04 | Stage 1 runner (`pc_stage1`) | ready (01, 03 merged); `codex/task_04.txt` |
| 05 | Stage 2 runner (`pc_stage2`) | blocked by 04; `codex/task_05.txt` |
| 06 | Scoring and sheet (`pc_score`) | merged; Opus review: merge (row 2 and row 1 on CLIP re-derived exactly); its findings are ticket 06b |
| 06b | Review fixes for `pc_score` | ready; `codex/task_06b.txt` (worktree name `06b scorefix`) |
| 07 | Rows 6, 10, 13 and gate (`pc_rows`) | merged; Opus review: merge; controller fix: sd only with 3 seeds, gate tolerance tests |
| 08 | Stage 0 and queues | blocked by 02, 04, 05, 06, 07; `codex/task_08.txt` |

- The integration branch `r10-percept-core` is checked out in `/project/CoSiR` (the untracked data lives there) and
  pushed. `main` is unchanged since `b1447f9`.
- `.scratch/r10-percept-core/mkwt.sh NN slug` makes a ticket worktree from the integration tip, with links to the
  untracked data (frozen sample, stage 1 graphs and outputs, pilot graphs; excluded in `.git/info/exclude`).
- Codex task files = `codex/common.txt` + the ticket path + review notes. Launch from a background Bash:
  `codex e --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c mcp_servers={} -m gpt-6.1-sol -c
  model_reasoning_effort=high -C <worktree> --json - < <task> > <out log> 2>&1`. The last agent message in the log is
  the report. Codex 5h at 59% (00:15), resets 02:03; weekly 88%.
- GPU check at 00:20 passed (driver, module and library 595.99.02; no process, no lock). Disk: system 260 GB free, SSD2 1.9 TB.
- Verdict-code reviews: an Opus `high` worker with `.scratch/r10-percept-core/review_brief.md` plus a focus list.
- Agent defaults at ticketing: index items 1 to 6. One item for the report: the head rows are 24 runs plus the gate
  rerun (row 6: 9, row 10: 6, row 13: 9); the spec's budget says 21.

## 4. Open points for the next step

None for the user. Agent work left:

- Tickets 02, 04 and 06b now (three Codex jobs), 05 after 04, 08 last. Reviews for 04, 05 and 06b (verdict code); 02
  and 08 by the controller's diff read and the smoke.
- The index now fixes the exact sidecar keys `pc_score` asserts ("Exact sidecar keys"); tickets 04, 05, 08 must match.
- The smoke (ticket 08 `--smoke`, GPU under the lock) times each run kind; with it, estimate the run's end time.
  Optional and useful: the stage 2 gate rerun (about 9 minutes, CPU) and stage 0 (about 15 minutes, GPU) as real
  checks inside the smoke; their results then go to the run chat.
- The final whole-branch review (Opus `xhigh`), one fix wave, a scoped re-review (Sonnet `high`); merge into main;
  push; worktrees and ticket branches on `.scratch/pending_deletions.md`.
- The run chat's handoff lists the tickets (title, blocked by, spec IDs, verdict code), the agent defaults and the
  controller fixes. The comparator's re-derivation (C13) is the run chat's verify step.

## 5. Code and pitfalls

- The September modules collide with main's `src` package (CM-3): run them in a fresh process with the September tree
  first on `sys.path` (parity tests in 04 and 05, stage 0 in 08). Read the branch with `git show 1b70d7f:<path>`.
- The September snapshot keeps the highest 40 center norms; only stage 0 runs that. PercepT core uses Algorithm 1.
- The extended-seed Stage 2 script refits Stage 1 itself; seed 42's recorded AUC is 0.8296.
- CLIP image features are not normalized. The patch cache's token 0 is not the CLIP image embedding (cosine about
  0.6); P2 uses the embedding (index default 1).
- `outputs/` is gitignored by the repo's rule; `sept_stage0/` needs its own line (ticket 08).
- `ags_head.run` writes to `root(smoke) / outdir`; `pc_rows` passes an absolute outdir.
- Implementer briefs set `CUDA_VISIBLE_DEVICES=''`; GPU checks are the controller's, under
  `flock -n -o -E 75 /tmp/gpu0.lock` (queued: `flock -o -w`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (`.scratch/` is outside git). All round tests pass on the integration tip: 61 in 40 s.
- Worktrees and branches of tickets 01, 03, 06, 07: merged, listed on `.scratch/pending_deletions.md`.
- On disk over 1 GB: unchanged (stage 1 `outputs/` 1.1 GB and `outputs_fu/` 1.2 GB, deferred).
