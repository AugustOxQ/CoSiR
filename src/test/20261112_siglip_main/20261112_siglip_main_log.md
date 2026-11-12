# r28 run log (SigLIP 1 in every main table)

Run chat "r28 run", started 2026-11-12 19:30, night mode. Plan: `docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md`.
Decision lines: `decisions.md`. Job briefs: `briefs/`.

## Jobs
- 19:45 T1 trainer extension (knob and removal rows on SigLIP 1): Codex gpt-6.1-sol high, worktree
  `/project/CoSiR-r28-trainer` (branch `r28-trainer`), brief `briefs/t1_trainer.md`, log `codex_logs/t1_trainer.log`.
- 19:45 A encoder pick on validation (item 1): Claude worker, Sonnet high.
- 19:47 B MMArt and photos (items 3, 4): Claude worker, Sonnet high.
- 19:48 C Qwen3 text-pass time (item 6): Claude worker, Sonnet medium.
- 19:35 C done: Qwen3 text pass 3.62 ms per caption (3 repeats 3.59 to 3.64), caption towers 0.017 ms for three; `records/time_text.json`.
