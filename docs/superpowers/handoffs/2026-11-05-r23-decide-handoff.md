# Handoff: r23 decide: read the r22 exploration with the user, wrap up, then choose the next step

## Brief
- Done: the r22 exploration ran on validation, seed 0; reports written and claim-reviewed.
- State: c*'s caption side is above plain Qwen and the teacher's text in 6 of 9 label cells (bar 5); 5 of 9 with one fixed read-out. Pair hits are below plain Qwen.
- Next: show the user the results brief without our reading, record decision C, wrap up, then step 2.
- Blocked: nothing. Two user decisions wait: the pair-hit limit and the free variant.

> Written 2026-11-05 by tab "r22 explore". Loop step reached: end of the r22 small loop, before step 1 (decision C). Direction log entry:
> [New scores read](../direction_log.md) ("Decided in the r22 decide chat"); r22's own entry is written at this wrap-up.

## 1. Read in this order

1. This file.
2. The [auto report](../../reports/auto/percept/2027-01-09_text_retrieval.md): only its results brief at first. Show the brief to the user without the collapsed
   block, ask for their reading in a sentence or two, then open ours and say where the two differ.
3. The [user-read report](../../user_read/2026-11-05_text_retrieval.md), the user's own copy; its advice is folded away too.
4. The direction log's "Now".
5. The [r22 explore handoff](2026-11-05-r22-explore-handoff.md), sections 2 and 4, for what the user decided and what stays open.

Skills: none.

## 2. Decided by the user, not to be reopened

Section 2 of the [r22 explore handoff](2026-11-05-r22-explore-handoff.md) holds: directions, hit rules, text, paper role, size, scope, the pair-hit limit stays the
user's call, deadlines. The rule's words were approved there. Its limits (interval above 0 against both comparators; 5 of 9 cells) are an agent default, written
in the round log and committed before any score existed (commit 8f5933e0).

## 3. Where things stand

- **Rule:** Outcome 1, 6 of 9 cells. All 9 above plain Qwen; style's 3 and genre's 3 above the teacher; emotion's 3 below it.
- **Margin:** thin. One fixed read-out for all rows gives 5 of 9 at each of the three read-outs; cross-fit picks give 6 of 9. Style's 3 cells and genre text to
  image hold in every check. Genre image to text and genre text to text change with the read-out. `tx_checks.py` came after the first read.
- **Pair hits:** best-head R@10, text to image 0.2118 against plain 0.3102, image to text 0.3186 against 0.5068, text to text 0.1756 against 0.1823.
- **Free variant:** 3 of 9, which would give Outcome 2. The rule reads c*.
- **Timing:** ahead of the plan. The exploration read was planned for 11-06. Deadlines stay: abstract 11-10, paper 11-16, supplementary 11-23.

## 4. Open points for the next step

Key, the user's:

- Decision C, the user's reading, recorded in the direction log ("Meaning (C)").
- Whether the paper states the pair-hit limit.
- How to treat the free variant.
- Decision A: the options in the results brief (claim loop on held ArtELingo and MMArt test rows; option u, joined scores; option t, single-caption retrain). Whether the
  claim loop is r23, and its size, follow from this.

Still with the user, unchanged from the explore handoff: the Lenses PDF at `.scratch/lenses_cvpr2026.pdf`, and the MMArt numbers if they are an author.

After the exploration, the order of the other open items waits for the user: r20 figure, paper chat, genre baseline row, r21, item 17, the note to the collaborator
and supervisor.

## 5. Code and pitfalls

- Round folder `src/test/20261105_text_retrieval/`; `tx_queue.sh` shows the order of the steps. `tx_data.py` reads validation only: a claim loop needs a held path,
  and no tuning, read-out pick or selection may read held rows.
- Text not yet extracted for a claim loop: the teacher's text on held ArtELingo (61,744 by 3 instructions), MMArt test rows (11,528 by 5 texts).
- The other encoders' c* weights are under `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/`; their caption side is not checked.
- The next report takes the next sequence date, 2027-01-10.

## 6. State at handoff

- Running: nothing. No queue job, no Codex job, no subagent.
- Disk: `/data/SSD2/cosir_r22/20261105_text_retrieval/`, 419 MB, for this wrap-up's storage summary; under 1 GB, so no ledger row. No pending deletion.
- GPU: worked throughout this chat. Driver 595.99.02.
