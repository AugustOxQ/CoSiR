Your task: help the user write this week's weekly report for CoSiR, as a user-read report, and its slides. The user will give you their requirements in chat. Do not draft anything yet. First read the files below, then reply with a short summary (at most 10 bullets) of the week's story as you understand it, plus anything you could not find, and wait for the user's requirements. Before drafting, ask the user up to three questions about any assumption that would change what the report covers, as `~/.claude/rules/user-read-reports.md` says.

Context: this tab (weekly-report) was opened on 2026-10-07 by the method-improvements tab of the CoSiR Herdr workspace, at the user's request. The last weekly report covers 22 to 30 September (`docs/reports/weekly/2026-09-30_percept_buddy_to_v2.md`). This one covers 1 to 7 October: the CoSiR v2 CVPR line.

**1. The user's rules.** Read these first; they govern format, location and style:
- `~/.claude/rules/reports-layout.md`: weekly reports and their `*_slides.md` go in `docs/reports/weekly/`, decks (`.pptx`, gitignored) in `docs/reports/pptx/`, build scripts and figures in `docs/reports/assets/`, one row in `docs/reports/reports_sum.md` per report, then `scripts/check_reports_sum.py`.
- `~/.claude/rules/user-read-reports.md`: the user-read format (reader, level, the seven-part layout, writing rules, the two light checks). Its default location is `docs/user_read/`. Ask the user whether the weekly user-read report goes there or in `docs/reports/weekly/`.
- `~/.claude/rules/report-writing.md`, `~/.claude/rules/timestamps.md`; project instructions `/project/CoSiR/.claude/CLAUDE.md`.

**2. Last week's report and slides.** These set the style, the level and where the story left off:
- `docs/reports/weekly/2026-09-30_percept_buddy_to_v2.md` and `docs/reports/weekly/2026-09-30_percept_buddy_to_v2_slides.md`;
- the deck builder `docs/reports/assets/build_2026-09-30_weekly_slides.py`, its figures `docs/reports/assets/2026-09-30_weekly/` and `docs/reports/assets/build_2026-09-30_weekly_figures.py`;
- the rendered deck `docs/reports/pptx/2026-09-30_percept_buddy_to_v2_slides.pptx`. The `anthropic-skills:pptx` skill may help with the deck.

**3. This week's material, roughly in time order.** Folder and report dates under `docs/reports/auto/v2/` are sequence numbers, not calendar dates. The run logs and report headers give the real times.
- **The index:** `docs/reports/reports_sum.md`. Its v2 rows list every report with a one-line summary.
- **The CVPR plan:** `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`, covering the task, claims, baselines, GO bar and schedule.
- **2 to 4 October:** `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`. This stage report covers the aspect-episode task, protocol and every method attempt (E1 baselines, method A and its NO-GO, MLLM probes, quick checks).
- **5 October**, the grouping component:
  - `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`;
  - `src/test/20261114_grouping_research/`, which holds the synthesis.
- **6 October**, the reader-fix line:
  - round 1: `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md`;
  - round 2: `docs/reports/auto/v2/2026-11-19_reader_fix_round2.md`;
  - the brainstorm: `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`;
  - round 3, the GO for one-sided affect steering: `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`.
- **7 October:**
  - round 4, the kill: `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md`;
  - the CVPR readiness memo: `docs/reports/stage/2026-10-07_cvpr_readiness.md`. The user held the framing decision and added a lightweight CLIP fine-tuning comparator.
- **The user-read briefings already written this week.** They are the best plain-language sources, and their figures are in `docs/user_read/assets/`:
  - `docs/user_read/2026-10-06_reader_fix.md`;
  - `docs/user_read/2026-10-06_reader_fix_affect_steering.md`;
  - `docs/user_read/2026-10-07_round4_vetoes.md`.
- **Work in progress.** These are context for "next week", not results:
  - idea 3 (GoEmotions placement of captions), in its own tab: `docs/superpowers/handoffs/2026-10-07-idea3-goemotions-handoff.md`;
  - the lightweight CLIP fine-tuning (linear probe, last block, LoRA on DAS6), running in the method-improvements tab: `docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md` and `docs/superpowers/plans/2026-10-07-clip-lightweight-ft.md`.
- **Project memory:** `~/.claude/projects/-project-CoSiR/memory/MEMORY.md` (the top entry is the running status) and `project_v2-publication-plan-pending.md`.

**4. Rules for this tab:**
- Every number comes from a report or its stored files. Use the full reports as the authority over the briefings.
- Do not touch the running work: `src/test/20261124_clip_lightweight_ft/`, `.superpowers/sdd/2026-10-07-clip-lightweight-ft/` and the idea-3 tab's folders are read-only for you.
- Commit only your own files, by explicit path, to `main`, never push, with your session's attribution lines.
- Run every Python call with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python`.
- Give times in Amsterdam local time (`TZ=Europe/Amsterdam date`).
