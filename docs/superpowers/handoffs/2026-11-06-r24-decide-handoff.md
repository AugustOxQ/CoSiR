# Handoff: r24 decide: show r23's results brief, take the user's reading, wrap up, then choose the next step

## Brief
- Done: r23 is finished. The auto report and the user-read report are committed and pushed.
- State: c* holds on ArtELingo and fails on MMArt. Late fusion holds on ArtELingo and fails on MMArt. No verdict is void.
- Next: show the results brief without its collapsed block and take the user's reading (decision C).
- Blocked: this chat waits for the user's reading.

> Written 2026-11-06 16:12 by the fourth r23 run chat. Loop step reached: step 9 is done; r24 opens at step 1, decision C. Direction log entry: [Pair-hit checks read](../direction_log.md) is the last one; r23's entry comes at this chat's wrap-up.

## 1. Read in this order

1. This handoff.
2. The [report](../../reports/auto/percept/2027-01-11_text_claim.md): the results brief, then the rest as needed.
3. The [user-read report](../../user_read/2026-11-06_text_claim.md): the user reads this one. Its advice block stays folded until the user gives a reading.
4. `~/.claude/rules/research-loop.md`, "Decision C and the results brief" and "Wrap-up and handoff". `~/.claude/templates/direction_log_entry.md` for the log entry.
5. `/project/CoSiR/docs/superpowers/direction_log.md`, "Now" and "Parked ideas", before step 2.

Skills: none for step 1. `grilling` and `domain-modeling` if the chosen direction reaches step 5.

## 2. Decided by the user, not to be reopened

- All of the [spec](../specs/2026-11-06-text-claim-design.md)'s §4, D1 to D14, and the amendment of 2026-11-06 12:49 (the 19:00 wait lifted).
- **D1:** the limit on c*'s pair numbers is worded by no agent. They stay in the report's table, with no reading, until the user has talked to the other authors. This binds every text and chat, the wrap-up and the direction log included. Late fusion's pair numbers are allowed. See the project memory "Pair-hit limit".
- c* stays frozen, and a failed verdict stands (spec §6).

## 3. Where things stand

The verdicts (`src/test/20261106_text_claim/records/verdicts.json`). The claim review confirmed them.

| Row and set | WIN over plain Qwen, of 9 | WIN over the teacher's text, of 9 | Pair bar | Verdict |
|---|---|---|---|---|
| c*, ArtELingo held | 9 | 6 | no bar | holds |
| c*, MMArt test | 8 | 3 | no bar | fails |
| Late fusion, ArtELingo held | 9 | 5 | passes | holds |
| Late fusion, MMArt test | 9 | 0 | passes | fails |

The report's brief has three options for next: a, b and c. Dates: target 2026-11-10, paper 2026-11-16.

## 4. Open points for the next step

1. **Step 1, decision C (key).** Show the user the results brief with no collapsed block. "Your reading" is empty. Ask for the reading in a sentence or two, before any view of ours. Record it. Then open "Our reading and recommendation" and say where the two differ. Do not give our reading in any earlier message.
2. **Wrap-up (`research-loop.md`).** The report and its index row are already committed. Record decision C, add the r23 entry to the direction log and rewrite "Now". Give the storage summary (item 4 below), wait for the user's decision and record items of 1 GB or more in the disk ledger. Then run `wrapup-check`, `disk-check` and `/project/claude-config/sync.sh`, and follow what they print.
3. **Step 2, decision A (key).** The direction, its type and loop size, in this same chat. The options are the brief's a, b and c. The user gives a view first.
4. **Storage summary.** One table, biggest first, with the total per disk.
   - `/data/SSD2/cosir_r23/20261106_text_claim` is 79 GB. Of this, `c13/` is 59 GB and `c13_prefix/` is 19 GB.
   - The MMArt stores are 4.3 GB; keep them.
   - The 12 build worktrees are 1.6 GB.
   - Every row of `.scratch/pending_deletions.md` belongs in the table, r18's deferred rows included. The r23 build branches are also there.

## 5. Code and pitfalls

- The auto report is 18.4 KB against the target of about 15 KB. The growth is the brief and the limits that the review added. Detail already moved to the round log. Do not cut a number a verdict needs.
- The report's name uses CoSiR's sequence date (2027-01-11). The user-read report uses the real date (2026-11-06).
- The late fusion count over the teacher's text "depends on the read-out" (4, 2, 5). Keep this wording. Query expansion alone also gives 5, so "rests on the picks" is wrong.
- Option c costs are our estimate. The forward pass is not measured.
- zsh does not split an unquoted variable: write `git add` paths out. Use absolute paths in background Bash.
- A `py_compile` check writes `__pycache__` even with `PYTHONDONTWRITEBYTECODE=1`. `src/test/20261106_text_claim/__pycache__` is on the pending list.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: `chats.tsv` and `briefs.md`, committed right after the cut. Other untracked files in the status (archive results, queue status files, one user-read asset) are older than r23.
- On disk over 1 GB: see item 4 of section 4.
- User-read report checks: cold read and fidelity check done, one fix wave done, no re-check (as the reference says). D1 held. One cold-read item stays open by choice: the link to the full report shows the sequence date.
