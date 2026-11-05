# Handoff: r19 read: read r19's results with the user (decision C), then wrap up r19

## Brief
- Done: the user-read report passed both checks; r19 is merged into main and pushed. No number was wrong.
- State: 23 of 23 verdicts hold. Against the teacher: 0 "better than", 15 "level with or better than", 41 no claim.
- Next: show the results brief without our reading, record the user's reading, wrap up r19.
- Blocked: this chat waits for the user's reading. r20's decide chat still holds r17, so r19 has its own tab.

> Written 2026-11-04 22:30 by tab "r19 run 5" (a planned cut, and past the 200k context meter). Loop step reached: r19 step 9 done, step 1
> next, with decision C. Direction log entry: none yet for r19; the wrap-up writes it.

## 1. Read in this order

1. The results brief at the top of [the auto report](../../reports/auto/percept/2027-01-05_past_picks.md). Show it to the user **without**
   its folded block, and ask for their reading in a sentence or two. Then open ours and say where the two differ.
2. [The user-read report](../../user_read/2026-11-04_past_picks.md), for the user. Its advice is folded too.
3. `~/.claude/rules/research-loop.md`, "Decision C and the results brief" and "Wrap-up and handoff".
4. `~/.claude/rules/storage.md`, "After every task".

Skills: none.

## 2. Decided by the user, not to be reopened

- All of the r19 spec's §4 (D1 to D25) and A1: [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md).
- The six-loop plan, r17 to r22 (decision A, 2026-11-03 18:30, direction log).
- r17's results enter the 11-04 weekly only after the user's reading in the r20 chat, and stay out of the 11-04 stage report. The next
  stage report starts from r17.

**Agent default, not the user's:** this tab. Nothing in the plan says where r19 is read. r20's decide chat holds r17, and r21 and r22
start from 11-06. So r19 got a small tab in its own workspace, with no new loop. If the user prefers to read r19 in r20's decide
chat, this chat closes after a short note there.

## 3. Where things stand

- **Merge:** main holds r19 (merge 82df7f3a, index row after it), pushed. `reports_sum.md` is OK and `docs-check` prints nothing.
- **Result:** 23 of 23 verdicts hold, GO. The three answers: the first two are against the plain image and PercepT; cost is against the
  teacher's three passes.
- **Teacher, 56 cells:** 0 "better than", 15 "level with or better than", 41 no claim. DINOv2 is BELOW in all 16 cells.
- **Cost not measured:** a teacher that encodes the image once. Our arithmetic gives about 203 ms per image, so about 2.3 for c* on Qwen.
- **Checks:** every gate passed; an independent re-derivation of the scoring from saved outputs matched every item.
- **User-read checks:** cold read on Haiku, fidelity check on Sonnet `high`, one fix wave, humanizer pass. The findings not applied are
  in the round log, under "User-read report".

## 4. Open points for the next step

For the user:

1. **Key, decision C:** what do these results mean? Ask before showing our reading.
2. **Key, decision A:** the report's options are: a. freeze these rows into the paper's tables, no compute; b. measure the cost against
   a teacher that encodes the image once, about 1 h, an estimate; c. emotion and style on new artists, a data search first. Option a
   may fit into r20's decide chat, or into the paper chat. Ask where.
3. **Storage:** the table below, one decision per row.

For the agent, at the wrap-up:

- The direction log entry for r19 (`~/.claude/templates/direction_log_entry.md`) and a new "Now". r20's decide chat may write its own
  entry at the same time, so read the file again right before you edit it.
- Update the four disk ledger rows dated 2026-11-04 with the user's decisions. If a row moves to another date, change the auto report's
  "Files and storage" line.
- Remove each row of `.scratch/pending_deletions.md` that the user decides. Rows of r18 stay with r18.
- `routing-check` printed three items: figure jobs cost a median 13% of the five-hour window against an estimate of 3%, and two open
  items. The user set the open items for a tooling chat after the 2026-11-09 deadline. Mention them; do not start a review here.
- A one-line pointer in project memory. Then `wrapup-check` and `/project/claude-config/sync.sh`.

**Storage rows for r19, biggest first.** The other rows on `.scratch/pending_deletions.md` also count.

| Path | Size | What it is | Recommendation |
|---|---:|---|---|
| `/data/SSD2/cosir_r19/20261103_past_picks/outputs` | 16 GB | The run's outputs; the reports rest on them | Keep |
| `node402:/local/wding/res/r19_pp` | 4.0 GB | Node results, pulled and verified | Delete, with `cluster clean` after the list |
| `/tmp/claude-0/-project-CoSiR/5dffc27d-c94f-4137-9d05-bfebeec91b0e/scratchpad/t06rev` | 1.9 GB | Ticket 06 reviewer's scratch exports | Delete |
| r19's worktrees under `.claude/worktrees/` (ten) | 1.2 GB | All merged, `r19-int` included | Delete, with their branches |
| `/data/SSD2/cosir_r19/20261103_past_picks/smoke_outputs` | 249 MB | Node smoke outputs | Delete |
| `node402:/local/wding/res/r19_pp_smoke` | 248 MB | Node copy of the smoke outputs | Delete |
| `/data/SSD2/cosir_r19/20261103_past_picks/c13_smoke` and its log | 89 MB | Codex's smoke of the re-derivation | Delete |
| `/data/SSD2/cosir_r19/20261103_past_picks/c13` | 61 MB | Every compared item | Keep |
| Eight new stores under `/data/SSD2/pre_extract/` | under 1 GB | Feature stores, shared cache | Keep |

Not checked: whether `node402:/var/scratch/wding/Dataset/semart/SemArt` (3.2 GB) came from r19.

## 5. Code and pitfalls

- Records are on main: `src/test/20261103_past_picks/records/`. Run nothing and recollect nothing: the hashed records stay as they are.
- Node deletion goes only through the cluster CLI: `cluster clean` lists first, `--yes` only for what the user approved. Look at
  `cluster du --node node402`; it takes a few minutes, so start it with `run_in_background: true`.
- `/project/claude-config` has a change that is not ours, `cluster-run/cluster.conf`. Commit by path, and leave it.
- **zsh:** an unquoted variable does not split, and `echo =====` is an error, so quote it.

## 6. State at handoff

- Running: nothing of r19 (no job, subagent, Codex job or queued job). r18's work runs in its own chats.
- Uncommitted: nothing of r19's in git.
- GPU check at 22:27: driver 595.99.02 in module, library and `nvidia-smi`. Night mode. The next chat needs no GPU.
- On disk over 1 GB: the four rows above, in the disk ledger as `deferred`.
