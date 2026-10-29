# Handoff: r11 build 3: GPU exports after 20:15, build tickets 07 and 09, smoke run with the re-derivation, final review

## Brief
- Done: tickets 01, 04 and 08 reviewed (01, 04 pass; 08 pass after one fix) and merged with 05 and 06; 216 tests pass.
- State: 7 of 9 tickets merged on `r11-retrieval`. Stage 0's one full check reproduced r10's held c* emotion AMI exactly.
- Next: the GPU exports (PercepT validation, CRL lists) after the Affection extraction (about 20:15), then ticket 07.
- Blocked: nothing.

> Written 2026-10-29 19:15 by "r11 build 2" (cut at 200k context, at a clean point: no subagent running). Loop step
> reached: step 7 (build), tickets in progress. Direction log entry: [2026-10-29](../direction_log.md).

## 1. Read in this order

1. The ticket index and its shared interfaces: `.scratch/r11-retrieval/issues/00-index.md` (outside git), then tickets
   `07` and `09`. Tickets 01 to 06 and 08 are `done`.
2. The brief for ticket 07, ready to send (check it against this handoff's §5 first):
   `.scratch/r11-retrieval/notes/t07_brief_draft.md`.
3. The spec on the integration branch:
   `/project/CoSiR-wt-r11-int/docs/superpowers/specs/2026-10-29-retrieval-design.md`, §3 to §6 (with amendment A1).
4. `~/.claude/references/matt-chain.md` (the final review) and `~/.claude/references/loop-chats.md`, "Unattended
   chats".

Skills: none for the next step (07 and 09 are not verdict code; `tdd` only for a verdict-code fix).

## 2. Decided by the user, not to be reopened

- The whole protocol, the bar, the baselines and the read-out optimization: spec §4, D1 to D8 and D15; approved from
  its brief on 2026-10-29 18:30 (r11 decide 2). No container restart before the build (user).

## 3. Where things stand

Integration branch `r11-retrieval` (worktree `/project/CoSiR-wt-r11-int`, tip `dbafc91`, pushed). Round folder
`src/test/20261029_retrieval/`, prefix `rt_`. Outputs: `/data/SSD2/cosir_r11/20261029_retrieval/outputs` (empty).

| # | Ticket | Built by | Review | State |
|---|---|---|---|---|
| 01 | round folder, labels, arm loaders, `rt_percept_val.py` | Codex | Opus `high`: pass (all joins exact, 4 of 4 planted wrong joins caught) | merged |
| 02 | `rt_metrics` | Sonnet | Opus `high`: pass after fixes | merged |
| 03 | `rt_stats` | Sonnet | Opus `high`: pass | merged |
| 04 | `rt_readout` | Sonnet | Opus `high`: pass; 3 should-fix applied (dropped counts, tie test, seed assert) | merged |
| 05 | `rt_crl.py` (lists, texts), CRL arm | Sonnet `medium` | controller diff read (text features are CLIP's projected embeddings, as the image store) | merged; `lists` not run yet (GPU) |
| 06 | `rt_stage0.py` (0a, 0b) | Codex `gpt-6.1-sol` high | controller diff read | merged; smoke 0a and 0b pass; full held c* emotion AMI 0.10658729378204021 = r10's value (difference 0) |
| 07 | runner | | | not started; brief drafted |
| 08 | `rt_vh` (C13 re-derivation) | Sonnet | Sonnet `high`: pass after one must-fix, fixed by the controller | merged |
| 09 | queue script, smoke | | | not started |

Findings of this chat:
- **08's must-fix (fixed, `0bb6add`):** rt_vh read validation PercepT from the full-scale mapper's file (the pre-A1
  source). It now reads `<inputs>/percept_val/P3__s{seed}.npz` (key `prob`), with a new `--inputs DIR` for the derived
  inputs (default the real outputs folder) and exit 2 on any missing input.
- **04's stability check:** c*'s outputs span exactly 1,024 dimensions after centering (λ₁₀₂₃ = 6.7e-9, λ₁₀₂₄ =
  6.8e-11), so whiten1024 keeps the signal space. Three routes (eigh, svd, Gram) agree to 2.7e-10 in mAP: a
  re-derivation will agree far inside 1e-6.
- **Dropped queries:** the even validation half drops 1 style query (a singleton class in that half). `half_dropped`
  counts it; 07 writes it to `scores.json`.
- Times measured: one full validation seed (mAP + nDCG) 3.1 s; full 0a about 1 minute; 0b 3 to 5 minutes at 8 workers
  (estimate from one item, 28 s for the five c* emotion seeds).
- Codex: 79% of the five-hour window left at 19:12 (reset 22:38); an implementer job needs 80%, so 07 goes to Claude
  (Sonnet `high`, as the draft says). Ledger row for job 06 added to `claude-config` (sync at the wrap-up).

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- **Inputs and outputs (decided in this chat, agent default):** derived inputs (`percept_val/`, `crl/`) live in
  `rt_paths.OUTPUTS` (env `RT_OUTPUTS`) and are read by `rt_data` and by `rt_vh --inputs`; each stage writes under its
  `--outputs`. A smoke run uses the real derived inputs and a scratch `--outputs`. So the GPU exports must run before
  the smoke, and ticket 09's "CRL from a fixed tiny list" is dropped: the smoke uses the real lists and texts.
- **GPU steps** (light, under `flock -o -w 7200 /tmp/gpu0.lock`, queued behind the Affection extraction, which holds
  the lock until about 20:15), from `/project/CoSiR-wt-r11-int/src/test/20261029_retrieval` with
  `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`: `rt_percept_val.py`, then `rt_crl.py lists`, then `rt_crl.py texts`
  (CPU is fine). Both refuse to overwrite a different existing file. Then rerun `test_rt_data.py` (its 5 PercepT skips
  should pass) and read `outputs/crl/lists.json` (count per aspect; label-name matches).
- **Ticket 07** must not open `rt_vh.py`; ticket 08's readings are in the draft brief. Its stage 4 must assert one CRL
  width K for all aspects (rt_vh asserts it too).
- **Ticket 09:** the queue's smoke mode runs `rt_stage0.py 0a/0b --smoke 600`, `rt_run.py ... --smoke 600` and
  `rt_vh.py <stage> --smoke 600 --outputs <scratch>` (default `--inputs`). The C13 test of 08 is agreement on this
  smoke run: values within 1e-6, the same outcomes and picks.
- Merges happen in `/project/CoSiR-wt-r11-int`, never in the main checkout. The round log
  `20261029_retrieval_log.md` gets the smoke result; `t06_stage0_checks.md` (Codex's notes) can be folded into it.
- `rt_percept_val.py`'s docstring shows `flock -o -w 3600`: right for a queued job, so it stays.
- The merge commits of 01, 05, 06 and 08 lack the attribution lines (harmless).
- In this shell a bare `cluster` is Graphviz; this loop needs no DAS6.

## 6. State at handoff

- Running: the Affection extraction (`queue_affection_extract.sh`, started 19:00, worktree
  `/project/CoSiR-wt-r11-impressions`, holds the GPU lock until about 20:15). After it ends, merge `r11-data` into main
  (any chat). Nothing of this chat runs.
- Worktrees of this build: `/project/CoSiR-wt-r11-{int,t01,t02,t03,t04,t05,t06,t08}`; at the end they go on
  `.scratch/pending_deletions.md`, not deleted, with `/project/CoSiR-wt-r11-t01/.scratch/r11-t01-tests/` (11 MB, old
  test fixtures).
- Uncommitted: nothing in the project. `claude-config`: three Codex ledger rows (sync at the wrap-up).
- On disk over 1 GB: nothing new.
