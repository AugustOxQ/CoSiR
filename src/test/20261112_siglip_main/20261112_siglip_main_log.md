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
- 19:49 T1 back: commit ed83bbb8 on r28-trainer, 129 tests pass, smokes of relational and k11=80 pass (12.7 s smoke). Codex ~11% of the 5-hour limit. Verdict-code review by an Opus high worker started. GPU stage script `r28_run_trainer.sh` (root `/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1`) waits for the review.
- 19:54 T1 review (Opus high): PASS; non-blocking: no test checks k11 reaches the graph build (smoke archives show it does). Merged (f8753886). GPU stage started: `r28_run_trainer.sh`, log `/data/SSD2/cosir_r28/20261112_siglip_main/run_trainer.log`, expected end about 22:45.
- 19:55 G figure 3 and 4 candidates on SigLIP 1 (item 11): Claude worker, Sonnet medium.
- 19:58 F done: settings and ablation scorers (21 tests). c* held replay AMI 0.0985 / 0.4408 / 0.5279 equals r17 exactly; plain-cosine mAP 0.263 / 0.340 / 0.599. Settings check uses the fallback criterion (r10's stage 3 inputs do not exist for these runs). c*'s six-cell validation mean over seeds 0-4 spans 0.3849 to 0.3879. Plain SigLIP 1 row for Table 5 asked from F.
- 19:58 D part done: validation picks (center_k10, genre image-to-text center), replay at float16 resolution, plain SigLIP 1 text rows from r23 (text to image 0.211 / 0.152 / 0.419). Held text scoring queued by the controller (`text/run_held.log`).
- 20:00 G done: candidates 3a 11 (Qwen3 9), 3b 18, Figure 4 top classes amusement / Cubism / still life; 0 of 11 and 0 of 18 overlap with the Qwen3 candidates. Page `/data/SSD2/cosir_r28/20261112_siglip_main/qual/viewer/index.html`. Row-order proof and viewer check asked.
- 20:01 G checks: c* held row order proven (name list hash equals the sidecar's list_sha256, a0b1139c...); viewer check OK (3a 11, 3b 18, cluster tabs 9 / 27 / 10).
- 20:02 F plain row: plain SigLIP 1 AMI 0.061 / 0.382 / 0.402, plain-cosine mAP 0.244 / 0.268 / 0.455; Qwen3 replay equals the paper's Table 5 plain row on all six cells (`records/ablation_plain.json`).
