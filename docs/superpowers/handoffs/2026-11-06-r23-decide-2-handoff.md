# Handoff: r23 decide 2: run a short grill and a short spec for the claim loop, then cut to build

## Brief
- Done: r22 read, two short checks read, and decision A taken; reports committed.
- State: at a = 0.7, joined pair R@10 is 0.3215, 0.4954, 0.1952 against plain Qwen's 0.3102, 0.5068, 0.1823. 5 of 9 label cells stay above both comparators.
- Next: short grill and short spec of the claim loop, with c* as the main row and joined scores as one extra row.
- Blocked: nothing. The pair-hit limit waits for the user's talk with the other authors.

> Written 2026-11-06 03:45 by tab "r23 decide", cut at the context meter. Loop step reached: after decisions C and A, step 3 skipped since the options are clear, before step 5.
> Direction log entry: [Pair-hit checks read](../direction_log.md).

## 1. Read in this order

1. This file.
2. The direction log's two newest entries: what the user read and decided.
3. The round log, [`20261105_text_retrieval_log.md`](../../../src/test/20261105_text_retrieval/20261105_text_retrieval_log.md): r22's rule and the "Follow-up" section.
4. The [short-check report](../../reports/auto/percept/2027-01-10_pair_hit_checks.md), then the results of the [r22 report](../../reports/auto/percept/2027-01-09_text_retrieval.md).
5. `~/.claude/references/grill.md` and `~/.claude/rules/spec-templates.md`, before the grill.

Skills: grilling, domain-modeling.

## 2. Decided by the user, not to be reopened

- Section 2 of the [r22 explore handoff](2026-11-05-r22-explore-handoff.md): directions, hit rules, text, paper role, scope, deadlines.
- r23 is the claim loop, on held ArtELingo and the MMArt test rows, with five seeds; type confirmation, medium loop (user, after seeing the recommendation, 2026-11-05).
- c* is the main row. Joined scores are one extra row, with a fixed from validation (user, after seeing the recommendation, 2026-11-06).
- c* stays frozen, with no retrain with the own-caption term (user, after seeing the recommendation, 2026-11-06).
- The loop checks why a = 0.7 helps, by which queries c* and plain Qwen each get right (user, after seeing the recommendation, 2026-11-06).
- The free variant is a side row in the supplementary, and the rule keeps reading c* (user, after seeing the recommendation, 2026-11-05).
- The pair-hit limit stays a flagged point. No agent words it until the user has talked to the other authors (user, 2026-11-05).

## 3. Where things stand

All numbers are validation, seed 0, so no claim.

- **c*:** above plain Qwen and the teacher's text in 6 of 9 label cells; 5 of 9 with one fixed read-out. Emotion's 3 cells are below the teacher. Style's 3 cells and genre text to image hold in every check.
- **Pair hits:** best-head R@10 for c* is 0.2118, 0.3186, 0.1756, against plain Qwen's 0.3102, 0.5068, 0.1823 (text to image, image to text, text to text).
- **Own-caption term:** on recipes R0-5 (on) and R6-5 (off), older than c*, it closes 59% and 57% of the gap to plain Qwen and lowers label mAP in all 9 cells. Its effect on c* is not known.
- **Joined scores:** `(1 - a) c* cosine + a plain cosine`. Only a = 0.7 passes on the grid 0 to 1 by 0.1. All 9 label cells are above plain Qwen. Style image to text clears the teacher by 0.0018. Joined cells have no intervals yet.

## 4. Open points for the next step

Key, asked answer-first:

- The claim's bar: what counts as "holds" per cell for c* on held rows, and how many cells the loop needs.
- The role of the joined row: its own bar, or a row read beside c* with no effect on the verdict. Pair R@10 is the same question: in the bar, or only reported.

Minor, an agent default is fine unless the grill shows otherwise:

- a stays 0.7 for all seeds, or is picked per seed on validation. The user was told "a stays fixed from validation".
- Which of MMArt's five texts enter the bar. r22 saw that its emotional text often names its own label, which leaks the answer.
- The why-a-helps check: its measure, and validation only (our default) or also held rows.
- Device and timing. The rule says a claim read by about 2026-11-10; plan local runs to end by 11:00.

Not for this chat, unless the user brings them up: the order of the r20 figure, paper chat, genre baseline row, r21, item 17, and the note to the collaborator and supervisor. Still with the user: the Lenses PDF at `.scratch/lenses_cvpr2026.pdf` (no fetch by script), and the MMArt numbers if they are an author.

## 5. Code and pitfalls

- `tx_data.py` reads validation only. The claim loop needs a held path, and no tuning, read-out pick or selection may read held rows.
- `tx_followup.py` holds the join (`check_joined`) and the gates: a = 0 and a = 1 must give the saved c* and plain scores. Reuse them; largest difference at r23's short checks was 3.1e-6.
- c* weights exist for seeds 0 to 4 and all three heads: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/<aspect>__s<seed>.pt`.
- Not yet extracted: the teacher's text on held ArtELingo (61,744 by 3 instructions, about 10.5 minutes at about 295 per second, a rate from short captions), and MMArt test rows (11,528 by 5 texts, no cost estimate; these texts are longer).
- c*'s saved `config.loss` text says "buddy+own-caption". That label is stale: `switches.own` is 0.0 for all three heads.
- `*.json` is gitignored, so results files need `git add -f`.
- A new round folder needs a new module prefix. The next report takes the sequence date 2027-01-11.
- Valid `codex-usage --room` kinds: figure, rederivation, fidelity, mechanical, tool, review, bug, report, implementer, other.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat. Old untracked files under `archive/` and three `queue_status.txt` files are not from this chat.
- On disk over 1 GB: nothing new. `/data/SSD2/cosir_r22/20261105_text_retrieval/` is 419 MB, kept, no ledger row. `.scratch/pending_deletions.md` keeps three deferred rows.
- Routing: `routing-check` prints three items on Codex model and job cost. The two listed items go to the tooling chat after 2026-11-09; the job-cost line goes with them (agent proposal, not yet confirmed by the user).
