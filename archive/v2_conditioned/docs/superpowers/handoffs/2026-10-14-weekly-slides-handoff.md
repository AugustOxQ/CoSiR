# Handoff: show the user weekly slides v1, take their comments, fix; build the pptx only on their word

## Brief
- Done: weekly slides v1, 12 slides in the three parts the user set; a new AFF design figure; a Codex fidelity check and a cold read, with fixes.
- State: no written report this week (user). The r9 results are in as preliminary: H6 Qwen GO (+0.63 over B′(A0)), H7 SigLIP 2 GO (+0.51), H8 CLIP ArtELingo-28 NO-GO, inconclusive.
- Next: the user reads v1 and comments (meeting 11:00); then one fix wave.
- Blocked: the user's comments.

> Written 2026-10-14 10:17 by the weekly report chat (outside the r9 loop), cut on the user's word after its context
> passed 200k. Previous handoff: [2026-10-14-weekly-report-handoff.md](2026-10-14-weekly-report-handoff.md).

## 1. Read in this order

1. The slides: `docs/reports/weekly/drafts/2026-10-14_weekly_v1_slides.md` (12 slides, separated by `---`).
2. `~/.claude/references/weekly-report.md`, the "Slides" part (titles, no dates in slide text, full names, pptx build).
3. Only if a number is questioned: `docs/reports/weekly/drafts/2026-10-14_weekly_v1_slides_fidelity.md` (Codex, run on
   an earlier state; every finding was handled, see §3).

## 2. Decided by the user, not to be reopened

- Slides only this week, no written weekly report; "pages" in the user's outline means slides (user, 2026-10-14 ~09:30).
- No context opener and no background: the deck follows last week's deck. Three parts: (1) AFF's design, simple but
  with key details such as the losses, then AFF against the baselines, saying which are true baselines and which are
  our own scores (B′(A0) and the like); (2) a wrap-up of the outside-view card 2026-10-14_aff_scope_limits, with the
  same discussion; (3) the ongoing work, one slide on backbones and one on more datasets (user).
- Only the three parts: no status, go/no-go or glossary slides (user).
- No agent-style words ("Our reading", "Our recommendation", "Verdict (reference)") (user).
- The design figure was redrawn for these slides only: simple pictures, numbered blocks 1 to 6, each block explained on
  the slide after it (user, 10:05).

## 3. Where things stand

- **Slides v1:** title; Part 1 (6): design figure, the six blocks, training (practice episodes, each part's loss),
  true baselines against our scores, held test, gain per aspect pair; Part 2 (3): where the gain comes from (shown
  aspect × 1 to 4 example pairs), style and genre, the labels discussion; Part 3 (2): backbones, datasets.
- **Figures:** new `docs/reports/assets/2026-10-14_weekly/slides/aff_design.png` (Codex, two rounds; the controller
  then widened the gate diamond and moved the "yes" label, and looked at the result: no defect left). Reused:
  `held_comparison.png` and `per_pair_fresh_held.png` (`assets/2026-10-12_affect_steering_line/`),
  `csd_style_genre_ami.png` (`assets/2026-10-14_aff_scope_limits/`).
- **Checks:** Codex fidelity: 190 numbers, none wrong; two HIGH findings fixed (an overclaim, "no evaluation label is
  used anywhere"; a deadline put on the wrong reads); the MEDIUM caveats added, mostly in the notes. Three findings
  were rejected after a check of the files: the 100-row refit rule is in
  `src/test/20261202_r9_scope/fx_siglip_a2_handback.md`; H6 was running; the CUB probe had finished
  (`results/probes/cub/probe_cub.json`). The Haiku cold read gave wording fixes. **Not checked:** the blocks slide, the
  new figure and the H6/H8 numbers came after both checks; the controller copied H6 and H8 from
  `src/test/20261202_r9_scope/results/held_verdict_H6.json` and `held_verdict_H8.json`.
- **r9 in the slides** (the rule's output, before the round report): H7 SigLIP 2 GO +0.51 [+0.36, +0.66]; H6 Qwen GO
  +0.63 [+0.49, +0.76]; H8 CLIP on ArtELingo-28 NO-GO, inconclusive (R@1 over B′(A0) +0.06 [−0.10, +0.23] failed;
  condition gain passed). H9 and H10 were running at 10:15; when they finish, the datasets slide's status cell changes.

## 4. Open points for the next step

1. **Key:** the user's comments on v1. Ask before any fix wave.
2. Minor: build the pptx (`~/.claude/templates/build_weekly_slides.py <slides.md>`, CoSiR env) only on the user's word.
3. Minor: the old report draft `docs/reports/weekly/drafts/2026-10-14_weekly_v1.md` is not used now; ask whether to
   keep it or list it for deletion.
4. After approval: commit and index per `weekly-report.md` and `docs-layout.md` (a final slide md in
   `docs/reports/weekly/` changes the image paths from `../../assets/` to `../assets/`).

## 5. Code and pitfalls

- `/project/CoSiR` is on main and shared with the r9 run chat, which commits there: stage only your own files.
- The draft's image paths are relative to `docs/reports/weekly/drafts/`.
- Auto report file names carry CoSiR's sequence-number dates, not real dates.
- The GitHub repo is public: no link to it in the slides.

## 6. State at handoff

- Running: nothing from this chat (the r9 run chat runs H9 and H10).
- Uncommitted, on purpose until the user approves: the slides md, its fidelity file,
  `docs/reports/assets/2026-10-14_weekly/` (figure and build script), the old report draft v1.
- On disk over 1 GB: nothing.
