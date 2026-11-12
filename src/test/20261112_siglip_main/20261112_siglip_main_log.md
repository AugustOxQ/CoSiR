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
- 19:35 E CLAY and CRL on SigLIP 1, + TURTLE MMArt check (items 7, 8): Claude worker, Sonnet high, brief `briefs/e_clay_crl.md`.
- 19:44 B part done: MMArt cells from r17/r18 records (`records/mmart.json`), SigLIP 1 + TURTLE MMArt exists in r18; photo model on EmoSet 0.432 AMI / 0.483 mAP (gate: equals r14 bit for bit). Remaining photo cells and gates queued by the controller (`photos/run_score.log`).
- 19:44 D text directions and per-seed values (items 2, 5): Claude worker, Sonnet high, brief `briefs/d_text.md`.
- 19:46 A done: validation pick SigLIP 1, mean of six 0.3863 against SigLIP 2 0.3856 and CLIP L/14 0.3756 (Qwen3 0.3697, information); SigLIP 1 wins 5 of 6 cells (SigLIP 2 wins genre AMI); margin within seed ranges. Per-seed values equal r17 and r14 records exactly. `records/valpick.json`.
