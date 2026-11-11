# Handoff: go on with the CVPR paper in its LaTeX repo; ask the user which part comes next (likely the method) and write only what the user asks for, from the user's skeleton

## Brief
- Done: private LaTeX repo CVPR2027-CoSiR-Paper (CVPR 2027 kit; Sections, Figures, Tables); related work parts (a) and (b) in the tone of the user's CBMI paper; part (c) a heading, one sentence and two TODOs.
- State: the paper builds (3 pages with references); the related work is about 540 words; no fidelity check of it yet.
- Next: ask the user which part comes next; write a part only when the user has set its skeleton and asks for it.
- Blocked: OpenReview registration by 11-11 12:59 (the user's alone).

> Written 2026-11-11 01:05 by the "paper method" tab. Outside the research loop: no loop step, no direction log entry.
> PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means `PF/CVPR2027-CoSiR-Paper/`.

## 1. Read in this order

1. This handoff, all of it.
2. `REPO/README.md` (what lives where, the build command) and `REPO/Sections/2_related_work.tex` (the text so far;
   its TRACE comments name the v1.1 line behind each fact).
3. `PF/notes/related_work_form.md`: the cut form, the tone, the claim rules.
4. `PF/CBMI_IMP_CAMREADY.pdf`, pages 1 to 3 (introduction, related work, method): the user's own paper, the tone to
   mimic. `pdftotext` reads it.
5. `docs/superpowers/handoffs/2026-11-10-paper-method-writing-handoff.md`, sections 3 to 5 (the outline's state, open
   items 1 to 4 and 6 to 9, pitfalls), and through it `2026-11-10-paper-writing-handoff.md`, sections 2 and 5.
6. As needed: `.duck/reports/2026-11-10-outline.md` section 5 (the method outline), `PF/drafts/v1.1/03_method.md` (the
   checked source of every method fact; read by section), `PF/drafts/v1.2/04_experiments.md` (the checked results).

Skills: none by default. Ducktor's `/duck-write` only when the user types it or asks for it.

## 2. Decided by the user, not to be reopened

- **No direct writing** (2026-11-10): the paper's skeleton and ideas come from the user; everything agents wrote
  (outline report, v1.1 and v1.2 drafts, the related-work form) is reference only. When the user has set a part's
  skeleton and asks for its text, write that part into the LaTeX file, compile, and show the PDF (SendUserFile,
  `display: render`). Recorded in `PF/README.md` (Decisions) and project memory (`feedback_user-writes-paper.md`).
- **The LaTeX repo** (2026-11-10): `REPO`, GitHub `CVPR2027-CoSiR-Paper` (private), CVPR 2027 kit release
  CVPR2027-v1 (first commit = the kit as released). `Sections/` (one `.tex` per section), `Figures/` (pdf, png; on the
  graphics path), `Tables/` (one `.tex` per table, kept apart from the sections).
- **Tone** (2026-11-11): mimic the user's CBMI paper (mainly its introduction and method): a framing sentence first,
  works sorted into groups, contrasts such as "X and Y provide ..., but ..." and "On the other hand", a closing
  sentence on where our work stands. The claim rules still hold (no "significant" without a test; novelty only "to our
  knowledge"; no "SOTA"); the user may lift them.
- **Related work** (2026-11-10 and 11): run-in `\paragraph{}` parts. (a) "Conditional similarity in datasets" and
  (b) "Conditional similarity in models" as written ("looks good"); the infographic retrieval model sentence in (b)
  (user, yes). (c) "VLMs and MLLMs.": a starting sentence, then the user's `\TODO{add CLIP, SigLIP, DINO family, and
  qwen +some other models?}` and `\TODO{distillation?}`. Part (d) is not written ("that's enough for related work
  now"). `\TODO{}` prints blue.
- Item 5 of the 11-10 handoff (an agent-written related-work cut) is dropped. Every decision in section 2 of
  `2026-11-10-paper-writing-handoff.md` stands (framing, temporary title, supervision wording, D1, TURTLE, r20).

## 3. Where things stand

- REPO: 11 commits, the last `1f4b037`; pushed. `latexmk -pdf main.tex` exits 0; 3 pages (most of it references).
- Related work: (a) 197 words, (b) 341 words, (c) one sentence. With (c) and (d) filled it would pass the outline's
  0.75-page plan; trim when the section is complete.
- Not checked: no Codex fidelity check of the LaTeX related work yet. Plan: one narrow pass when the section is
  complete, against the v1.1 lines in the TRACE comments (prompt pattern: `PF/notes/fidelity_checks/`).
- `REPO/refs.bib` is a copy of `PF/refs.bib` (213 checked entries; all build with the CVPR style). `tian2020crd` has no
  year. DINOv2 is `oquab2024dinov2`; DINO v1 and DINOv3 are not in it (check before adding). The user may want DINO
  under "VLMs and MLLMs"; it is an image-only encoder, so the heading may need to widen (told the user, no answer).
- `Figures/` and `Tables/` are empty.

## 4. Open points for the next step

Key:
1. **What comes next.** Ask the user. The method is likely; the outline's section 5 is the reference, not the plan.
2. **Outline items 1 to 4** (11-10 handoff, section 4): contribution 1 wording, contribution 2 widened, other encoders
   and photos moved to Sec. 4, the user's reading of challenge 2 (answer-first). They shape the method and the closing
   sentences of related work (b). Still open.

Minor:
3. Offered, not answered: copy the r20 teaser and cluster figures (`PF/figures/fig_teaser.*`, `fig_clusters.*`) into
   `Figures/`; turn v1.2's 9 checked tables into `Tables/*.tex` (numbers only, empty captions).
4. Not answered: is "no direct writing" a global rule for all the user's papers or this one only?
5. Items 6 to 9 of the 11-10 handoff (held split sentence, photo decision sentence, comparator wording, r19 rows), for
   when the results enter LaTeX.
6. The old target "draft v1.2 by 11-11 11:00" was a joined Markdown file; the user has not said whether a LaTeX PDF
   replaces it.

## 5. Code and pitfalls

- REPO is its own git repo inside PF; the outer `cosir-paper` repo ignores it (`docs/paper/.gitignore`). Commit and
  push each finished part in REPO; notes go to `cosir-paper`. REPO has a local git identity (no global one here).
- Build in place: build files and `main.pdf` are gitignored. Send `REPO/main.pdf` after each change.
- MMArt (`wang2026mmart`) lists the user as a co-author: cite it in the third person only (double-blind).
- Define an acronym before its first use across parts (part (b) once used "MLLM" before part (c) defined it).
- Each part keeps a `% TRACE` comment block naming the source line of each fact; facts only from checked sources.
- All pitfalls of the earlier handoffs hold: anonymity, numbers from sources, D1, the TURTLE clause, the supervision
  wording, "better than" only per cell.

## 6. State at handoff

- Running: nothing.
- Uncommitted: `docs/superpowers/constitution.md` (modified before this chat, not ours). `PF/CBMI_IMP_CAMREADY (2).pdf`
  (6.1 MB, differs from the committed copy; the user's, untracked, left as it is).
- On disk over 1 GB: nothing. The scratchpad holds the kit download and test builds (2 MB); it goes with the session.
