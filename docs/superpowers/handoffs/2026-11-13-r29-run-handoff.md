# Handoff: r29 run: stage 1 (K1 and K2, SC1, K*, SC2), then stage 2 on a pass, then the auto and user-read reports

## Brief
- Done: the r29 build: 15 tickets merged into main (007fb786), round suite 731 passed, final review pass after two fix waves and a scoped re-review.
- State: every stage 0 gate passes (gate (vii) ablation passes at try 3); gate (iv) hash part waits for stage 1a.
- Next: hold r21's next GPU job, then stage 1a (K1 and K2 seed 0), SC1, stage 1b, Table 8 (part `t8`), SC2.
- Blocked: nothing.

> Written 2026-11-13 04:05 by the "r29 build 7" tab. Loop step reached: step 7 done; this chat does steps 8 and 9. Direction
> log entry: [2026-11-12 · r29 decide](../direction_log.md). Run mode: night (`run-mode`).

## 1. Read in this order

1. This handoff, all of it.
2. The spec: `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`, from §1 (skip the brief). §6 with §3.4
   and §3.5 is the decision rule; §4 holds D1 to D25; D23 the tolerances.
3. The stage scripts' headers (round folder `src/test/20261112_siglip_only/`): `r29_run_stage0.sh`,
   `r29_run_stage1.sh {a|b|t8|c}`, `r29_run_stage2.sh {a|b}`, `r29_run_common.sh`. Each has `--dry-run`.
4. The decision lines: `src/test/20261112_siglip_only/decisions.md` (build's choices; step 9 ranks and checks them).
5. `~/.claude/references/run-planning.md` (device and end time), `~/.claude/references/loop-chats.md` (unattended
   chats), `~/.claude/references/codex-jobs.md` (the re-derivations).
6. At step 9: `~/.claude/templates/auto_report_template.md`, `~/.claude/templates/results_brief.md`,
   `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec's §4, D1 to D25 (2026-11-12, decide chats; "who decided" per row there).
- GPU runs may run in weekday daytime this round: stage 2 starts right after a stage 1 pass, with
  `R29_ALLOW_DAYTIME=1` for `r29_run_stage2.sh` (amends D7 for r29 only; user, 2026-11-13 02:19; decision line).
- r29's run goes before r21's next GPU job (user, 2026-11-13 02:00): "if you want GPU, [tell the] r21 chat to cease
  the next run and queue yours first". Before stage 1, message r21's run chat (`ListAgents`; "r21 run 4" (opened 2026-11-13) or its
  successor) to hold its next run, queue r29's first steps, then tell r21 it may go on (its jobs queue behind).

## 3. Where things stand

- Code: main at 007fb786 (integration branch `r29-siglip-only` merged). Round folder
  `src/test/20261112_siglip_only/`, outputs `/data/SSD2/cosir_r29/20261112_siglip_only/`.
- Stage 0 gate records (`records/gates/`), all pass at try 1 unless named: i_a (bit-equal, no fallback), i_c (valpick
  at try 3 of 3; decision line), ii, iii, iv_store, v_image, v_text, v_mmart, vi, v_photo, vii_photo, vii_trainer
  (try 2), vii_ablation (try 3, pass at 04:04; its files are frozen now). Gate (iv) hash part needs trained K1 and K2 (stage 1a runs it). Gates (viii) run in
  stage 2. So `r29_run_stage0.sh` skips every step (records exist); run it once to confirm.
- K0 = r14's c* on SigLIP 1 (`/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/`, seeds 0 to 4,
  node402). The new model trains locally against K0's DAS6 seeds; the machine difference sits beside the pass rule
  (D25).

The tickets (for information; spec IDs in `.scratch/r29-siglip-only/issues/00-index.md`):

| # | Title | Verdict code |
|---|---|---|
| 01 | Caption input layer: SigLIP caption join and caption-width model | yes |
| 02 | Prefixed caption store and the store check | yes |
| 03 | Caption-width trainer and exports | yes |
| 04 | Text scorer for SigLIP caption towers | yes |
| 05 | Validation groups, SC1, SC2, gate (i)(a) verdict, diagnostics | yes |
| 06 | Image test cells | yes |
| 07 | Text test cells: nine directions against P1 and Q1 | yes |
| 08 | Claim ladders and the numbers sheet | yes |
| 09 | Table 8 text timing | no |
| 10 | E2 and E3: SigLIP 2 and CLIP L/14 with own text encoders | yes |
| 11 | MMArt cells (SC5) | yes |
| 12 | Photo model F1: Affection caption store and trainer | yes |
| 13 | Photo cells (SC6) | yes |
| 14 | Ablation R1 to R4 (SC7) | yes |
| 15 | Stage runner scripts | no |

## 4. Open points for the next step

None for the user (night mode). Agent defaults to carry into the run report (each has its decision line):
- Ticket 12's D14 truncation (64 tokens; Affection 0.62 % and 0.77 % cut).
- Per-part parking of SC6 and SC7 (ticket 14's line).
- Ticket 13's choices (AMI intervals by `ns_clu.paired_cell`, units, recorded comparator picks, the `cells` gate rule,
  gate (vii) photo at 1e-9) and ticket 11's (untold pick unit = K*'s own told R0 range; teacher pick; gate (vi) at 1e-9).
- Gate (i)(c) valpick at try 3 of 3 (build 4 line).
- From the final fix waves (build 7): Table 8 times K*'s own caption form (with the prefix when K* is prefixed); SC7
  lists an inconclusive removal with its cause and no numbers and scores the others (record status `partial`; replaces
  the 00:41 line that let one removal block SC7); diagnostic (d) covers SC3 and SC4 only, and the sheet shows K0's r28
  value beside each SC5 and SC6 cell without a paired interval.

## 5. Code and pitfalls

- **The order (§3.4, §3.5, C13).** 1a ends before SC1: run `r29_decide.py sc1`, have Codex re-derive it (rule and data,
  never our numbers; `codex-jobs.md`), commit `records/prefix_pick.json` and the 1a records, then 1b. 1b ends before
  SC2: `r29_decide.py sc2`, Codex re-derives, commit `records/pass_rule.json` with the pick records, then 1c. 1c and 2b
  end before the ladders: Codex re-derives every pick, range and cell first, then run the ladders. A fail still runs
  1c (D9) and goes to the numbers sheet; stage 2 refuses without a pass.
- **Records are gitignored** (`.gitignore:154`, `*.json`): commit each with `git add -f`. The scripts check
  commits with git and refuse otherwise.
- **Queue:** every step goes through `res-queue` (the scripts call it); estimates marked "guess" in the script headers
  have no timing. Steps whose record exists are skipped (`R29_FORCE=1` overrides). Gate (vii) ablation's files are
  frozen after try 3 (`r29_ablation.py`, `test_r29_ablation.py`, `r29_trainer.py`, `r29_model.py`, `r29_captions.py`):
  any edit leaves SC7 stale.
- **Stage 2 skips a piece, not the stage:** a missing picks record or a failed gate puts that piece on the failure
  list and the others go on (§3.5). The ablation train step runs a second time on a non-zero exit (D21's same-seed
  rerun after a dead process); the journal caps each seed at two attempts.
- **Table 8 is its own part:** run `r29_run_stage1.sh t8` after 1b. It needs only the committed
  `records/prefix_pick.json` and K*'s seed 0 weights, no pass rule, so it also runs when SC2 fails (§6 SC8). 1c no
  longer runs it and only prints a pointer.
- **A gate record in state `fail`** (tries below 3) is skipped by the scripts' `have` check: a rerun after a fix needs
  `R29_FORCE=1`.
- **Free check:** R3 (remove buddy) should equal r28's buddy exports bit for bit (no caption gradient).
- **Codex** had no room during the build (resets 04:44 on 2026-11-13). Re-derivations go to Codex when
  `codex-usage --room` says room, else to a Claude worker (Sonnet `high`), as `agent-routing.md` says.
- **r29's main chat** is not open: it opens only after r28's run chat ends (a `chats.tsv` row opened from r28 run's
  session `dae700bf-e7a2-4618-ae57-fa086b1e35b9`, or r28's run reports committed). Check at the start and at each
  cut; then run `loop-next ~/.claude/templates/main_chat_start.md --label "r29 main" --main --new-workspace "CoSiR r29"
  --model sonnet --effort high --notify`.
- r28's files (`src/test/20261112_siglip_main/`) belong to r28's run chat: import, never edit. The paper plan's
  SigLIP-only row belongs to the paper chat. The repo is public: no keys or personal data in files.
- At the end: the run chat opens the next loop's decide chat with `--new-loop --notify` (`loop-chats.md`).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r29.
- On disk over 1 GB: `/data/SSD2/cosir_r29/20261112_siglip_only/gate_vi/try1/` (1.5 GB, gate (vi) export part; keep
  until the report cites it); ticket 12's test scratch `/project/CoSiR-r29-t12/.scratch/r29-t12-tests/` (4.6 GB,
  system disk). Both are on `.scratch/pending_deletions.md`, with the ticket worktrees and branches.
