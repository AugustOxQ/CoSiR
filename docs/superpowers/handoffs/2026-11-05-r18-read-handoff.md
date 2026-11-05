# Handoff: r18 read: read r18's results with the user (decision C), then wrap up r18

**GPU check failed at the cut (07:17):** `nvidia-smi` gives `Failed to initialize NVML: Unknown Error`; kernel module and library both 595.99.02. A host `daemon-reload` dropped the GPU from the container. This chat needs no GPU. Other chats' GPU jobs do. Only the user restarts the container, from the host.

## Brief
- Done: the auto report is claim-reviewed, the user-read report passed both checks, and r18 is merged into main and pushed. No table number was wrong.
- State: 337 cells: 330 WIN, 6 LEVEL, 1 BELOW, 0 void. Default TURTLE scored below k-means in 203 of 232 diagnostic comparisons.
- Next: show the results brief without our reading, record the user's reading, wrap up r18.
- Blocked: this chat waits for the user's reading. The GPU is lost in the container; r18 needs none.

> Written 2026-11-05 07:20 by tab "r18 run 4". Loop step reached: r18 step 9 done, step 1 next, with decision C. Direction log entry: none yet for r18; the wrap-up writes it.

## 1. Read in this order

1. The results brief at the top of [the auto report](../../reports/auto/percept/2027-01-06_outside_methods.md). Show it to the user **without** its folded block, and ask for their reading in a sentence or two. Then open ours and say where the two differ.
2. [The user-read report](../../user_read/2026-11-05_outside_methods.md), for the user. Its advice is folded too.
3. `~/.claude/rules/research-loop.md`, "Decision C and the results brief" and "Wrap-up and handoff".
4. `~/.claude/rules/storage.md`, "After every task".

Skills: none.

## 2. Decided by the user, not to be reopened

- The six-loop plan, r17 to r22 (decision A, 2026-11-03 18:30, direction log).
- TURTLE then TEMI (D10, user's own pick), the code route (D12, user), labels only with no loop verdict (D11, user, after seeing the recommendation), spec approved 2026-11-03 20:45.

**Agent default, not the user's:** this tab, as r19 had. Nothing in the plan says where r18 is read. If the user prefers to read r18 in r20's decide chat, this chat closes after a short note there.

## 3. Where things stand

- **Merge:** main holds r18 at cb280e75, pushed. `reports_sum.md` is OK and `docs-check` prints nothing.
- **Labels:** Qwen c* against TPIPS 34 of 35 WIN. Qwen and CLIP L/14 c* against CLAY 66 of 70 WIN, the four others all style. c* against the plain image with default TURTLE on both sides 114 of 116 WIN. c* with k-means against the plain image with default TURTLE 116 of 116 WIN.
- **The BELOW cell:** MMArt CLIP L/14 style accuracy against CLAY. No claim for c*, and "CLAY better" is not licensed either.
- **The point most worth the user's eye, a diagnostic with no label:** TURTLE at its defaults scored below k-means in 203 of 232 comparisons, and 170 have the whole interval below zero. On the plain image, ARI and ACC are mixed (d above zero in 20 of 48). So the TURTLE cells do not show that c* holds against a clustering stronger than k-means.
- **Closest cells:** d is below c*'s seed range in 5 cells of c* with k-means and 6 cells with TURTLE on both sides, all emotion ARI or ACC. Ten of the eleven are WIN and one is LEVEL.
- **What the checks reach:** the independent re-derivation matched 10,762 items and recomputes CLAY's vectors. It redoes no feature extraction or training. TURTLE is our re-implementation, checked against the official code only in a short run (100 outer steps on 2,000 rows), plus one full fit that matched across machines.
- **TEMI did not run.**
- **Report checks:** claim review on Sonnet `high`, cold read on Haiku, fidelity check on Sonnet `high`, one fix wave, humanizer pass. Findings not applied are in the round log, under "Auto report and user-read report".
- **Auto report size:** 20.8 KB against the limit of about 15 KB and the handoff's aim of about 18. The extra is the caveats the claim review required.

## 4. Open points for the next step

For the user:

1. **Key, decision C:** what do these results mean? Ask before showing our reading.
2. **Key, decision A:** the report's three options: a. put these rows into the paper, writing only; b. build and run TEMI, up to 10 GPU h, results before 2026-11-16 enter the paper; c. the spec's one-off TURTLE selection on validation, then plain-image TURTLE at the chosen setting, about 134 fits, about 4 GPU h by our arithmetic, not measured. Option c goes past D25's "no selection run", which was an agent default and kept this selection for a close cell, and past D9's "no 50-setting selection". D9 and D25 differ on the size: D9 says 50 fits per cell, D25 says a one-off 50 fits. The report takes D25, so a small spec must settle it. r20's decide chat holds what is next, so ask where this goes.
3. **Storage:** the table below, one decision per row.

For the agent, at the wrap-up:

- The direction log entry for r18 (`~/.claude/templates/direction_log_entry.md`) and a new "Now". The r19 read and r20 decide chats may write at the same time, so read the file again right before you edit it.
- Record each item of 1 GB or more in the disk ledger; r18 has no rows there yet. The report's "Files and storage" leaves the summary to this chat; add its one-line pointer to the ledger.
- Remove each row of `.scratch/pending_deletions.md` that the user decides. Rows of other loops stay with them.
- `/project/claude-config/codex/jobs.md` has three new rows; they go with `sync.sh`. `cluster-run/cluster.conf` there is a change that is not ours: commit by path, and leave it.
- A one-line pointer in project memory. Then `wrapup-check` and `/project/claude-config/sync.sh`.

**Storage rows for r18, biggest first.** All rows are on `.scratch/pending_deletions.md`, with smaller ones. Run `cluster du --node node402` first; it takes a few minutes, so start it with `run_in_background: true`.

| Path | Size | What it is | Recommendation |
|---|---:|---|---|
| `/data/SSD2/cosir_r18/20261103_outside_methods/outputs`, without `turtle_pack` | 5 GB | The run's outputs; the reports rest on them | Keep |
| `OUT/turtle_pack` | 14 GB | TURTLE fit inputs, a copy of existing stores | Delete |
| `node402:/local/wding/r18_in/turtle_pack` | 14 GB | Node copy of the above | Delete, with `cluster clean` after the list |
| `node_inputs/tpips_pylibs/full` here and on node402 | 7.3 GB each | Package copy for node jobs | Delete |
| `.claude/worktrees/r18-int` and its branch | 3.2 GB | Merged; holds `res/r18_turtle`, 3.0 GB, the same fits as `OUT/parts/turtle` | Delete |
| `node402:/local/wding/res/r18_turtle` | 3.0 GB | TURTLE fits, pulled and verified | Delete, with `cluster clean` after the list |
| r18 build 6's scratchpad, `11fc0fcc-...` | 3 GB | Ticket 11's smoke outputs | Delete |
| TPIPS stores under `/data/SSD2/pre_extract/{artelingo_full,mmart}/tpips_8b/` | 2.6 and 2.3 GB | Feature stores, shared cache | Keep |
| `c13_smoke` and `c13_round_smoke` | 869 and 360 MB | Smoke outputs of the re-derivation | Delete |
| r18's ticket worktrees and branches | about 120 MB each | Merged. **Final reviews A and B each hold** files: read their rows first | Decide per row |

## 5. Code and pitfalls

- Records are on main: `src/test/20261103_outside_methods/records/`. Run nothing and recollect nothing: the hashed records stay as they are.
- `*.json` is gitignored at the repo root: a record needs `git add -f`.
- Keep the settings clause beside every TURTLE sentence: "at its released code's default settings, without its hyperparameter selection". "Better than" holds per cell only. Do not call default TURTLE stronger than k-means.
- `OUT` is `/data/SSD2/cosir_r18/20261103_outside_methods/outputs`.
- Node deletion goes only through the cluster CLI: `cluster clean` lists first, `--yes` only for what the user approved.
- **zsh:** an unquoted variable does not split, and `echo =====` is an error, so quote it.

## 6. State at handoff

- Running: nothing of r18. `res-queue` idle, no running subagent, no Codex job.
- Uncommitted: other chats' untracked files only, on main. Leave them.
- GPU check at 07:17: failed, see the top. Night mode.
- On disk over 1 GB: the table above, nothing deleted.
