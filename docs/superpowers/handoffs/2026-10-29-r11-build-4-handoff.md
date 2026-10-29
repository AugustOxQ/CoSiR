# Handoff: r11 build 4: the final whole-branch review, one fix wave, merge into main, then cut to the run chat

## Brief
- Done: all 9 tickets merged on `r11-retrieval`; a C13 mismatch in stage 4 fixed (one numeric convention).
- State: the queue smoke at 2,100 rows passes: 228 tests, every step exit 0, `rt_vh` 0 disagreements in 995 values.
- Next: the final whole-branch review (Opus `xhigh`), one fix wave and its re-review, merge into main, cut to the run.
- Blocked: nothing.

> Written 2026-10-29 21:20 by "r11 build 3" (cut at 200k context, at a clean point: nothing running). Loop step
> reached: step 7 (build), every ticket merged; the final review is left. Direction log entry:
> [2026-10-29](../direction_log.md).

## 1. Read in this order

1. The final review brief, ready to send: `.scratch/r11-retrieval/notes/final_review_brief.md` (outside git). Replace
   the scratch folder in its item 1 with this chat's own scratchpad. The smoke folder it names stays as it is.
2. `~/.claude/references/matt-chain.md`: "The final whole-branch review" (Opus `xhigh` worker, then one fix wave and
   a scoped re-review at Sonnet `high`), and "implement-spec": the merge into main and the worktree list.
3. The round log on the integration branch: `/project/CoSiR-wt-r11-int/src/test/20261029_retrieval/20261029_retrieval_log.md`,
   the sections "C13 stage 4 numeric alignment" and "Build 3: the queue smoke".
4. For the cut to the run chat: `~/.claude/references/loop-chats.md` ("How to cut", "Unattended chats") and the
   handoff template. The run chat's handoff carries the ticket list (matt-chain.md, "The record").

Skills: none (the review is a `worker`; `diagnosing-bugs` only if the review finds a bug).

## 2. Decided by the user, not to be reopened

- The whole protocol, the bar, the baselines and the read-out optimization: spec §4, D1 to D8 and D15; approved from
  its brief on 2026-10-29 18:30 (r11 decide 2). No container restart before the build (user).

## 3. Where things stand

Integration branch `r11-retrieval` (worktree `/project/CoSiR-wt-r11-int`, tip `d9a032f`, pushed).

| # | Ticket | Built by | Check | State |
|---|---|---|---|---|
| 01 to 06, 08 | see the build 2 handoff | | | merged |
| 07 | `rt_run.py`: stages 1 to 5, `diag` | Sonnet `high` worker (stopped by an API error after its smoke) | controller diff read; five-arm smoke at 300 rows | merged |
| 09 | `rt_queue.sh`: full and smoke modes | Codex `gpt-6-luna` medium | controller diff read; two fixes (commit only the ranges path; a rerun keeps the committed r_a) | merged |
| fix-c13 | one numeric convention in `rt_data`, `rt_readout`, `rt_vh` | Codex `gpt-6.1-sol` high | controller diff read; queue smoke | merged (`b06d0a7`) |

- **GPU exports done** (20:22 to 20:23): `percept_val/P3__s{0..4}.npz`, `crl/lists.json` (50 values per aspect;
  values equal to a label name: emotion 4, style 3, genre 4), `crl/{aspect}_text.npy`. All in
  `/data/SSD2/cosir_r11/20261029_retrieval/outputs`. The run's GPU steps will be skipped (their files exist).
- **The C13 mismatch (found by the queue smoke at 600 rows):** `rt_vh stage4` found 174 of 790 values outside 1e-6;
  the picks were equal. Three causes: (i) at 300 rows per half, whitening to 1,024 keeps an arbitrary basis of the
  null space (a smoke artifact); (ii) the CRL cosines came from a float32 product whose rounding depends on row count
  and threads; (iii) the two codes rounded and normalized at different steps, and the score's division by r_a (about
  0.003) magnified 1e-6 mAP differences. The fix writes one convention in the round log and applies it on both sides.
  `rt_vh` still imports no `rt_` module.
- **The queue smoke at 2,100 rows** (`SMOKE_N` in `rt_queue.sh`, so each half exceeds 1,024): every step exit 0;
  `rt_vh` agrees on 0a 24, stage 1 68, stage 2 47, stage 3 19, stage 4 790, stage 5 47 values. Outputs:
  `/tmp/claude-0/-project-CoSiR/585c1724-5f23-49f9-94a9-046b675ae2e9/scratchpad/queue_smoke2100/` (9 MB).
- **Times measured** (smoke, 2,100 rows): stage 4 11.2 minutes (c* 290 s, PercepT 139 s, the others 28 to 57 s);
  every other step under 1 minute; the six `rt_vh` checks 5 minutes. Full-scale stage 4 is not measured. Estimate:
  20 to 40 minutes, since the eigendecompositions do not grow with rows and the retrievals grow by about 8.6×.
- `r11-data` merged into main (`30ae79b`, pushed). The Affection store is complete: 63,199 images and 392,331 texts,
  check passed, 1.8 GB at `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/`.
- Codex: 67% of the five-hour window left at 20:54 (reset 22:38), 72% weekly. Ledger rows for jobs 09 and fix-c13
  added to `claude-config` (sync at the wrap-up).

## 4. Open points for the next step

None for the user.

- Optional, agent default: a full-scale check of stage 0a and stage 4 with `rt_vh` on validation only (no held data),
  in a scratch `--outputs`, alongside the review. It catches a full-scale C13 mismatch before the run chat. Note:
  a full (non-smoke) `rt_stage0.py 0a` also writes `stage0a_ranges.json` into the round folder it runs from. Run it
  from the integration worktree and never commit that file there. The run commits its own copy from main.

## 5. Code and pitfalls

- `rt_run.py --arms` exists for smoke runs only (asserted); the full run always uses all seven arms.
- The queue's full mode must start from `/project/CoSiR` on `main` with no tracked changes in the round folder. Its
  commit step commits only `stage0a_ranges.json`.
- `.claude/` is gitignored: the day's change log is `/project/CoSiR/.claude/20261029_log.md` (local file).
- Merges happen in `/project/CoSiR-wt-r11-int`; the merge into main happens in `/project/CoSiR` after the re-review.
- In this shell a bare `cluster` is Graphviz; this loop needs no DAS6.

## 6. State at handoff

- Running: nothing. GPU check at 21:16: driver 595.99.02 in all three places, the GPU idle, no lock.
- Worktrees of this build: `/project/CoSiR-wt-r11-{int,t01,t02,t03,t04,t05,t06,t07,t08,t09,fix}` and
  `/project/CoSiR-wt-r11-impressions` (`r11-data`, merged). They go on `.scratch/pending_deletions.md` at the build's
  end, with `/project/CoSiR-wt-r11-t01/.scratch/r11-t01-tests/` (11 MB) and the stray untracked
  `/project/CoSiR-wt-r11-int/.claude/20261029_log.md` (a copy of the change log). Not deleted.
- Uncommitted: nothing in the project. `claude-config`: five Codex ledger rows (sync at the wrap-up).
- On disk over 1 GB: the Affection store above (1.8 GB, made by the data side job; it needs a ledger row at the
  wrap-up). Scratch smoke folders in this chat's scratchpad: 53 MB in all.
