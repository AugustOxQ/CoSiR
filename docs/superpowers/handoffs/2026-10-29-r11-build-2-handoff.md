# Handoff: r11 build 2: review and merge tickets 01, 04 and 08, build tickets 05, 06, 07 and 09, final review

## Brief
- Done: 9 tickets written; tickets 02 (metrics) and 03 (statistics) reviewed by Opus and merged; 01, 04 and 08 built.
- State: a review found a float32 tie bug (fixed: float64 cosines). Spec amendment A1: PercepT validation outputs now come from the held mapper.
- Next: Opus reviews of 01 and 04, then tickets 05, 06, 07 and 09, the GPU exports, a smoke run and the final review.
- Blocked: nothing. The GPU is busy with the Affection extraction from 19:00 to about 20:15.

> Written 2026-10-29 18:55 by "r11 build" (cut at 200k context, at a clean point: no subagent running). Loop step
> reached: step 7 (build), tickets in progress. Direction log entry: [2026-10-29](../direction_log.md).

## 1. Read in this order

1. The ticket index and its shared interfaces: `.scratch/r11-retrieval/issues/00-index.md` (outside git), then the
   tickets `01` to `09` in the same folder. The index's "Exact JSON keys" are the contract between tickets 07 and 08.
2. The spec: [2026-10-29-retrieval-design.md](../specs/2026-10-29-retrieval-design.md), §1 onward. On the
   integration branch it carries amendment A1 (§4, §5); main gets it with the build's merge.
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/loop-chats.md`, "Unattended chats".
4. The facts brief: `.scratch/r11-retrieval/facts_brief_2026-10-29.md`.

Skills: `tdd` for any new verdict-code fix.

## 2. Decided by the user, not to be reopened

- The whole protocol, the bar, the baselines and the read-out optimization: spec §4, D1 to D8 and D15; the spec was
  approved from its brief on 2026-10-29 18:30 (r11 decide 2). No container restart before the build (user).

## 3. Where things stand

Integration branch `r11-retrieval` (worktree `/project/CoSiR-wt-r11-int`, pushed). Round folder
`src/test/20261029_retrieval/`, prefix `rt_`. Outputs: `/data/SSD2/cosir_r11/20261029_retrieval/outputs` (empty).

| # | Ticket | Built by | Review | State |
|---|---|---|---|---|
| 01 | round folder, labels, arm loaders, `rt_percept_val.py` | Codex gpt-6.1-sol `high` (+ fix by resume) | not yet (Opus `high`) | branch `r11-t01` (contains the integration tip), 95 pass, 5 skip |
| 02 | `rt_metrics` | Sonnet `high` (+ fixer) | Opus `high`: pass after fixes | merged |
| 03 | `rt_stats` | Sonnet `high` | Opus `high`: pass (15 of 15 mutations caught) | merged |
| 04 | `rt_readout` | Sonnet `high` | not yet (Opus `high`) | branch `r11-t04` on top of the old 02 (before its fix): merge `r11-retrieval` into it first, then run its tests |
| 05 | CRL lists, texts, CRL arm | | | not started (blocked by 01) |
| 06 | stage 0 | | | not started (01, 02, 03) |
| 07 | runner | | | not started (01 to 06) |
| 08 | `rt_vh` (C13 re-derivation) | Sonnet `high` | not yet (Sonnet `high`; its real test is agreement on the smoke run) | branch `r11-t08` (based on main; it must not see the main code); 3 tests pass; all six modes load real data on `--smoke 300` |
| 09 | queue script, smoke | | | not started (06, 07) |

Findings so far:
- **Float32 ties (ticket 02, fixed).** float32 products gave unequal cosines for identical rows, so exact ties broke
  differently with the worker count (mAP moved by 2.7e-6, above C13's 1e-6). Every similarity is now float64 from the
  float32 rows; ticket 08 was told the same convention.
- **Amendment A1 (agent default, before any result).** Held PercepT P3 (r10's entry, D11) comes from the sample-scale
  mapper (fit on 13,400); the saved validation P3 from the full-scale mapper (42,969). Validation is re-inferred from
  the held mapper's weights (`outputs/percept/core/stage2/k40/P3__s*.pt` of the claim-test round) by
  `rt_percept_val.py`, which needs the GPU (`infer_stage2` asserts cuda). Recorded in the spec (§4, §5) and the round
  log; the report must mention it.
- Ticket 04's choices to check in its review: covariance 1/(n-1) (no effect after L2), R0 re-normalizes, 1e-12 slack
  at the guard and tie boundaries, no guard for whitening past the data's rank, dropped queries per half not reported.
  Its stage 4 estimate: about 15 minutes (not measured on real data).
- Ticket 03's `rt_stats.SC` layout: `Sc(id, stage, aspect, readout, metric, comparisons)` with `Comparison(candidate,
  comparator, bar)`, each side `(arm, head)`. Ticket 07 iterates over it.
- Codex: 83% of the five-hour window left at 18:54 (reset 22:38); an implementer job needs 80% room. Ledger rows added
  to `claude-config` (for `sync.sh` at the next wrap-up).

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- Merges happen in `/project/CoSiR-wt-r11-int`, never in the main checkout (keep `/project/CoSiR` on main).
- The round log `20261029_retrieval_log.md` was created by two tickets; the 01 merge resolved it once. Expect it again.
- GPU steps (both light, under `flock -o -w 3600 /tmp/gpu0.lock`, after the Affection extraction ends near 20:15):
  `rt_percept_val.py` (validation PercepT) and `rt_crl.py lists` (Qwen3-VL-8B-Instruct, ticket 05). Both models and
  weights are on disk (Qwen3-VL-8B snapshot `0c351dd`, CLIP L/14 snapshot `32bd642`).
- Ticket 08 must stay independent: its briefs never name or open the `rt_` main modules. It did not merge
  `r11-retrieval` (on purpose); merge its branch into the integration branch, never the reverse.
- Ticket 08's readings that ticket 07 must match (from its report, 18:55): SC7-SC9 use the column aspect's validation
  mAP range; the tie band "within 0.001" and the guard are inclusive, among kept read-outs only; stage 4's mAP per
  read-out is the mean over the two halves, then over seeds; genre's gallery (and its QE neighbours) holds labelled
  paintings only, while W is fit on every painting; a query without a relevant item is dropped and counted (`dropped`);
  `go` and `switch_go` are bools; `stage2/switch.json["cstar"][head][column]` is the mean over seeds; CRL has one width K
  for all aspects. A smoke run must compute its ranges from the smoke validation subset (rt_vh does), not use the
  committed full ranges. Its whitening near-null directions (c* at 1,024 and 2,048) are the likeliest place for a
  1e-6 miss; picks should agree.
- rt_vh usage: `python rt_vh.py {stage0a..stage5} [--outputs DIR] [--smoke N] [--workers 8]`; exit 1 on a
  disagreement, 2 when a file is missing.
- In this shell a bare `cluster` is Graphviz; this loop needs no DAS6.

## 6. State at handoff

- Running: the Affection extraction (`queue_affection_extract.sh`, pid 1545349, worktree
  `/project/CoSiR-wt-r11-impressions`, 19:00 to about 20:15). After it ends, merge `r11-data` into main (any chat).
- Worktrees of this build: `/project/CoSiR-wt-r11-{int,t01,t02,t03,t04,t08}`; at the end they go on
  `.scratch/pending_deletions.md`, not deleted.
- Uncommitted: nothing in the project. `claude-config`: two Codex ledger rows (sync at the wrap-up).
- On disk over 1 GB: nothing new.
