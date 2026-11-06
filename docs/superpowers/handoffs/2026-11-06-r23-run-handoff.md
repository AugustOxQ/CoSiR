# Handoff: r23 run: run stages 0 to 3 and C13, verify, write the report and the user-read report

## Brief
- Done: all 10 tickets, the final review, one fix wave and its re-review. Merged into main at f43f3692 and pushed.
- State: no held score exists. 464 round tests pass. The end-to-end smoke passes from stage 0 to stage 3.
- Next: run `tcl_queue.sh val` from the main checkout. Heavy jobs that would overlap 10:00 to 19:00 wait for 19:00 by themselves.
- Blocked: nothing.

> Written 2026-11-06 09:05 by tab "r23 build 5". Loop step reached: end of step 7, steps 8 and 9 next. Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule. §9 lists the deliverables.
2. §4 and §5 of this handoff: the readings the report must state, the commands and the C13 brief lines.
3. `~/.claude/references/loop-chats.md`, "Unattended chats". `~/.claude/references/codex-jobs.md` before C13.
4. Before the reports: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`, `~/.claude/templates/results_brief.md`.
5. The review notes, when a step needs them, in `/project/CoSiR/.scratch/r23-text-claim/notes/`: `final_review_run.md` (stop table, 25-row mutation table), `final_review_numbers.md` (the C13 evidence), `final_review_rule.md` (coverage, §3.6 and gate tables), `final_rereview.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. **Amendments: none.** A change to the rule is an amendment (hard rule 5).

## 3. Where things stand

**The tickets (the record of the build; for information):**

| # | Ticket | Blocked by | Spec IDs | Verdict code |
|---|---|---|---|---|
| 01 | Ranking, means, interval, outcome word (`tcl_metrics.py`) | none | SC1-SC4, D15, §3.4 | yes |
| 02 | Data door (`tcl_data.py`) | none | D6, D9, §3.3, gate (iv), C6 | yes |
| 03 | Text extraction and store checks (`tcl_extract.py`) | 02 | D9, D12, D17, gate (i) | no |
| 04 | Forward passes, late fusion, gate (ii) (`tcl_forward.py`) | 02 | D1, D8, §3.2, gate (ii) | yes |
| 05 | Scoring runner (`tcl_score.py`) | 01, 02, 04 | D11, D12, §3.4 cells, pair cells | yes |
| 06 | Replay gate (iii) (`tcl_replay.py`) | 05 | gate (iii), C12 | yes |
| 07 | Stage 1 ranges and picks (`tcl_stage1.py`) | 01 | D11, D16, §3.4 picks, ranges | yes |
| 08 | Cells, outcomes, verdicts (`tcl_outcome.py`) | 01, 07 | SC1-SC4, D3, D4, D5, D7, §6 | yes |
| 09 | Runner, guard, queue (`tcl_run.py`, `tcl_queue.sh`) | 03, 05, 06, 07, 08 | D2, D13, §3.5 | no (guard gets a mutation check) |
| 10 | Stage 3 diagnostics (`tcl_diag.py`) | 05, 08 | D10, D12, §3.6 | no |

**The final review** (four Opus `xhigh` passes, on the review's code at 273dd4da): no code must-fix. One must-fix is the C13 brief lines in §5. Eight fixes (F1 to F8) went in by Codex; a Sonnet `high` re-review passed them.

| Check | Result | Source |
|---|---|---|
| Set sizes, from raw files | 6,152 and 30,872; 12,281, 61,744, genre 9,949; 5,769; 16,384, r15's order and hash | `final_review.md` |
| Interval against an own version | largest difference 0.0 on 4 real cells; a swapped rng order moves a bound by 6.5e-5, so the check has power | `final_review_numbers.md` |
| Row entry rules | match on 36 rankings; breaking a rule moves AP by 0.03 to 0.41 | same |
| Inputs | sides and all plain and teacher image stores bit-identical | same |
| Verdict rule | 300 rows against an own table, 0 differences | `final_review.md` |
| §6 voiding | every gate and all 39 stores do what §6 says | `final_review_rule.md` |
| §3.6 values | 763 sampled values recomputed, no mismatch | same |
| Mutants on the merged fixes | 31 of 34 caught, all C13 and fix mutants included. Survivors M15, M23, M19b each sit behind a tested second check | controller's rerun, this chat's scratchpad `mut2/` |
| All round tests | 464 passed in 4.0 minutes | controller, r23-fix |
| End-to-end smoke | `TCL_SMOKE=200`, real `tcl_queue.sh`, 34 steps, none failed; gates pass; 72 label and 4 pair cells; all 8 diagnostic parts written | controller, r23-fix. Its verdict words mean nothing: do not quote them |

Not checked by any review: the C13 lines on real held-size rows (synthetic at 61,744 only); row load time per unit; the new stores, which do not exist yet; a teacher row on MMArt in the separate pipeline; other torch or GPU versions.

## 4. Open points for the next step

None for the user. **Readings the report must state** (each was a review finding; none is an amendment):

a. **C13's numeric path.** §3.4's 1e-6 per query holds only on one numerical path. Reason: near-tied cosines differ in the last float32 bits, and the GPU's matrix product depends on the matrix shape. A separate pipeline with no `tcl_`, `tx_`, `rt_` or `mt_` imports matched on 9 validation settings, with AP within 1.33e-7 and no first rank difference. Any one departure breaks 1e-6 (worst case 968 queries; mAP moved 4.3e-7; no R@10 flip). So C13 keeps separately written code and re-derived logic, and loses numerical diversity. The effect matters to a verdict only if a lower bound lies within about 1e-6 of a bar. A move to cell level would be an amendment, and it is not needed.
b. **Gate (i)'s parity bar.** 0.999 is ticket 03's, as r15; §3.5 says "compared" and names no bar. Measured here on the run's real rows at batch 64: all 8 checked rows pass, none of 200 under the bar in any, lowest 0.99914 (`genre_word`), 0.99950 (`emotion_focus`), 0.99956 (`style_word`). The margin is thin but the rows and batch size are fixed. MMArt plain rows: lowest 0.99960.
c. **`c13_void`** marks that SC's cells and writes no `cell_claims` entry. §6 names only the verdict. Left on the safe side.
d. **RoBERTa's store** is raw, so gate (i) tests norm above 0 only; `tcl_data` normalises on read. §3.6-only.
e. **A bad score file** in a set not voided crashes `decide`, with no void route. No false verdict results. Trace and fix it.
f. **Plain cells** carry a LEVEL or BELOW word in `cells.json`. It is no claim: only WIN counts, and P counts only WIN.
g. **Unrequested items**, reported at the exploration ceiling: `label_word_share`, the within-query spread, `tcl_review_pair_estimate.py` (a rule-review helper off the run path).
h. **Ticket 01** has no review trace. The numbers review's own interval and AP pass stands in.
i. **Cost:** the report gives measured times. `evaluate` at 61,744 takes 5.2 to 12.6 seconds, `painting_interval` 0.6 to 0.8 seconds, `decide` about 1 minute (the spec says CPU about 1 h; `--est 45m` is only an estimate). `score_art_held` takes about 16 minutes of ranking, row load not measured. Every job stays under 1 hour by these numbers. The 4,002 rankings are by arithmetic.

## 5. Code and pitfalls

**Before each chain and before `verdicts`:** `env | grep '^TCL_'` prints nothing. Run from the main checkout, in the background: `read_committed_record` compares with HEAD of the repo that holds the round folder.

```
cd /project/CoSiR/src/test/20261106_text_claim
PY=/root/miniconda3/envs/CoSiR/bin/python; L=/data/SSD2/cosir_r23/20261106_text_claim/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
```

**The chain, in order:**

1. `./tcl_queue.sh val`. Then `$PY tcl_run.py status`, then run the COMMIT line in `$L/stage1_mm.log` (`records/stage1_*.json` and `status.json`, `git add -f`), and push.
2. `./tcl_queue.sh held`. Nobody reads or quotes `stage2/*` until C13 passes. `decide` writes verdict words to `outcomes.json` before C13, the same pattern as r15; `verdicts.json` is the record.
3. C13, as a Codex job, `gpt-6.1-sol` `high`, under the queue when it uses the GPU. It gets neither `stage2/` nor `scores/`. Its brief carries the block below. Check `codex-usage --room` first. If Codex cannot do it, Claude takes it back and says so.
4. Write `records/c13.json` as `{"pass": true, "sha256": {"stage2/outcomes.json": <hex>, "stage2/cells.json": <hex>}}`. `git add -f`, commit. `$PY tcl_run.py verdicts`. Run the printed COMMIT line.
5. `./tcl_queue.sh post`, then read `voided` in `stage2/outcomes.json` and in `stage3/summary.md`.

**C13 brief lines (a reading; carry them as they are):**

```
C13 numeric path. These are settings, not our outputs. They are needed for per-query agreement within 1e-6.
Same machine, conda env CoSiR (torch 2.11.0+cu130), the RTX 3090 under the queue. float32 on cuda unless a line says otherwise.
1. Store rows: float16 to float32 in numpy, then x / np.linalg.norm(x, axis=1, keepdims=True), once.
2. c* head: normalise the rows of line 1 a second time the same way, then nn.Linear or F.linear (never x @ W.T + b), GELU,
   nn.Linear or F.linear, y / y.norm(dim=1, keepdim=True). Batches of 4096 in row order, on cuda.
3. Late fusion, in numpy: np.concatenate((math.sqrt(1 - 0.7) * c, math.sqrt(0.7) * p), axis=1) as float32. No further normalising.
4. Means: x.astype(np.float64).mean(0).astype(np.float32) over every validation row. Held sets use the validation means.
5. Centering, on cuda: Q - qm and G - gm, each divided by its norm(dim=1, keepdim=True).
6. Cosine: Q[a:b] @ G.T with the whole gallery, in blocks of exactly 1024 from query row 0.
   No other block size, no swapped product, no materialised transpose.
7. Text to text: after the product set the own row to -inf, and drop it after the sort.
8. Order: S.argsort(dim=1, descending=True) on cuda. Exact ties stay in ascending gallery position.
9. Expansion: E = Q[a:b] + G[o[:, :10]].sum(1), then E / E.norm(dim=1, keepdim=True), then lines 6 to 8 with E.
10. AP may be summed in float64.
Compare: per-query AP within 1e-6 with identical NaN masks; first hit rank exactly; rows and means within 1e-6 (expected
bit-identical); every cell value within 1e-6; outcome words, P, T, Q and verdicts exactly.
A difference over 1e-6: first test rows and means for bit identity, then check lines 1 to 9. A path mismatch is fixed in the
C13 code and rerun. Only a difference that stays after that is a §6 difference; it goes to c13_void in status.json.
The C13 record states: numerics taken from the run's path, logic derived from the rule; no numerical diversity.
```

Name the pairs in the brief that are not differences: 16,782 is r15's `new` export and 16,384 its test cut. Codex stops on "if a number disagrees, stop" (ticket 4 lost a run on this).

**When something stops:**

- A stopped chain prints nothing. Read its exit code, `$L/<step>.log`, then `OUT/stage<k>/logs/<step>.log`.
- Past a stop, after trace and fix, run one queue job per step. The next step for each stop is in `final_review_run.md`, "What the run chat's brief must carry".

```
s=$($PY tcl_run.py _wait --est 20); [ "$s" -gt 0 ] && sleep "$s"     # first, for an --est job
res-queue gpu --est 20m --label r23:STEP -- $PY tcl_run.py STAGE --only STEP > $L/STEP.log 2>&1
res-queue gpu --short  --label r23:STEP -- $PY tcl_run.py STAGE --only STEP > $L/STEP.log 2>&1
res-queue cpu --est 10m --label r23:STEP -- $PY tcl_run.py STAGE --only STEP > $L/STEP.log 2>&1
res-queue cpu --est 45m --label r23:decide -- $PY tcl_run.py stage2 --only decide > $L/decide.log 2>&1
```

- **After a fix, old files stay:** a complete score file and a finished store are skipped. Before a rerun after a fix to a store, the data door, a forward path or the scorer, rename `OUT/scores/<set>`, `OUT/means/<set>` and any fixed store's folder to `*_old`, then rerun from the first step it affects. A fix on validation means stage 1 reruns and is committed again.
- **A rerun of `decide`** removes `records/verdicts.json`, so stage 3 refuses until C13 reruns on the new outputs and `verdicts` runs again.
- **If a C13 difference stays:** add `"c13_void": ["SCn"]` to `status.json`, commit it, rerun `decide`, write `c13.json` with the new hashes, commit it, run `verdicts`.
- **Once any held score exists, a missing stage 1 record stays missing,** and that set stays inconclusive. The guard cannot tell it came after a held number.
- **A set with no stage 1 record:** `held` and `post` skip its steps with exit 0, `decide` voids it, and `tcl_diag.py` now runs the other group's parts alone.
- **A §3.6-only store that cannot be fixed:** §6 voids that diagnostic, and the void list reaches `summary.md`. Its missing score files still crash parts 3, 4, 6 and 7, and `--parts` does not help. Add a narrow skip then and say so in the log.

**Working here:**

- `tcl_run.py stage2`, `verdicts` and `stage3` never rewrite `status.json`, since the guard compares bytes with HEAD. Stage 0, stage 1 and `status` do rewrite it.
- `tcl_score.py --block` exists; the chains never pass it. Leave it: block size changes per-query values by up to 2.7e-3.
- Stage 3 must run on `cuda`: part 2's self-check holds at 1e-6 only on the device that wrote the scores. CPU stops at 7.36e-6. No fallback exists.
- `*.json` is gitignored: results files need `git add -f`. `.claude/20261106_log.md` is tracked although `.claude` is ignored: `git add -f`.
- In zsh, `echo =====` fails. An unquoted heredoc eats backslashes in Python patterns: use `<<'EOF'`. A chain stops only with `&&` or `set -e`.
- The `--est` values are estimates. The D13 check runs before a job enters the queue, so a job cleared at 09:10 behind another job can start after 10:00; report it if that happens.

**Run timing (D13):** 2026-11-06 is a Friday. Latest no-wait start: 09:20 for est 40, 09:40 for est 20. So expect the chain to do the first extraction, then wait for 19:00. Spec estimate: GPU 2.5 to 3 hours, so the chains end before 11:00 on Saturday. Report by 2026-11-07 12:00, at the latest 2026-11-08.

**Deliverables (§9):** round log `src/test/20261106_text_claim/20261106_text_claim_log.md` (not written yet), report `docs/reports/auto/percept/2027-01-11_text_claim.md`, user-read report `docs/user_read/2026-11-07_text_claim.md`, C13, a claim review of the report. The pair-hit limit is not worded by any agent (D1): c*'s pair numbers go in a table with no wording.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat, apart from `chats.tsv` and `briefs.md`, committed right after the cut, and the Codex ledger row in `/project/claude-config/codex/jobs.md`, which the next `sync.sh` commits.
- On disk over 1 GB: the 12 build worktrees, 1.6 GB, on `.scratch/pending_deletions.md` with their branches, for the storage summary. `/data/SSD2/cosir_r23/` does not exist yet.
- GPU check before this unattended chat: passed. Driver 595.99.02 in the module and the library. Run mode: night, so no restart question.
- Codex at 09:01: 29% left of five hours, reset 09:53; weekly 38%.
