# Handoff: paper plan 13: redesign the method overview figure (fig1_overview) with the user, while the user writes the introduction

## Brief
- Done: the introduction layout is decided (all 8 questions). The one-column teaser is built and placed on page 1. The CVPR 2027 rules are checked: an LLM Usage Statement is mandatory.
- State: the user writes the introduction. Teaser in REPO `b540d16`, caption empty. About 1.7 to 2 columns still to cut.
- Next: ask the user how the method overview figure should change, then build it with agents. After it: the neighbour-graph figure (fig2_graphs).
- Blocked: nothing.

> Written 2026-11-15 03:30 by the "paper plan 12" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `README.md`, the rows dated 2026-11-15 (this chat's decisions).
3. PF `notes/2026-11-14_intro_layout.md`: the layout the user writes from. Its "Decided" line, then Sections 5 and 6.
4. PF `user_read/2026-11-13_paper_status.md`: the status table, now numbered 1 to 17. Row 17, the LLM Usage Statement, is new.
5. `docs/superpowers/handoffs/2026-11-13-paper-plan-12-handoff.md`, sections 2 and 5. Its decisions and pitfalls still hold.
6. For the overview figure: PF `notes/after_meeting_2026-11-11.md` point 1 (the meeting's comments on its style),
   PF `figures/src/fig1_overview.py`, PF `figures/fig1_overview.png`, and its caption at REPO `Sections/3_method.tex:7-14`.

Skills: none for the first step. Ducktor's figure tool (`/duck-figure`, which runs Supervisor-Skills' `figure-designer`)
can run in a worker the way this chat ran `/duck-intro` (section 5), if the user wants it.

## 2. Decided by the user, not to be reopened

All on 2026-11-14 and 2026-11-15, each a README row in PF:
- The work starts with the introduction. The user writes it. Agents do the figures at the same time: first the method
  overview, then the neighbour-graph figure.
- Introduction layout: the user's four blocks, about 420 words. Teaser T1. No cross-modal in block 1. C2 is the aspect
  adapter, kept apart from C1. C3 is the wider evaluation plus the key results. The limits are set apart by what
  supervises the aspect and where it enters the model. "Trained without aspect labels" replaces "label-free". One clause
  says that an aspect can add its own term, as genre does.
- Teaser: query a10 (Pirosmani, "Girl with a balloon"), chosen by a rule: the most top-4 neighbours of its own class in
  the three CoSiR rows, 12 of 12. Four neighbours per row. Every row label reads "✓ = class". The style name is "Naive
  Art (Primitivism)". The repeated paintings stay. It sits at the top of the right column of page 1. Its caption is the
  user's to write.

## 3. Where things stand

- REPO `b540d16` builds clean, with only the old 0.7 pt overfull box. It has 12 pages. The Conclusion heading is on
  page 9 at y 521 of 720 pt. The qualitative and cluster figures are on page 10.
  - Compare the build just before the teaser: page 9 at y 423, with those figures on page 9.
  - The float order is fragile, so measure the length again at the cut.
- The teaser is Figure 1 now. The overview is Figure 2 in the PDF, but its file is still `fig1_overview`. "The overview
  figure" means that file.
- Six TODO notes are in the PDF. The user added one for the co-authors in REPO `40cd409`. The status report still says
  five.
- A co-author reads a semi-final version today, Sunday 2026-11-15. Paper deadline: **2026-11-17 12:59 Amsterdam**.
  Supplementary deadline: 2026-11-24 12:59.
- Codex at 03:10: five-hour window 99% left, weekly 51% used.
- The loop chats have moved on. The r32 decide handoff (2026-11-14) is about stage 2 of the SigLIP-only model, and the
  encoder pick there is a close call. Not checked: whether the paper's main model will switch. If it does, the teaser
  and Figures 4 and 5 are built on c* SigLIP 1 (r28) and need a rerun. The user's TODO of `40cd409` names this risk.

## 4. Open points for the next step

Key, answer-first: what the user wants from the overview redesign. It was laid out for four blocks and now has two, so
its right side is empty, and the meeting made comments on its style. Ask the user first, then give our proposal.

Minor, each with our recommendation, none answered yet:
- A limitations paragraph: two or three sentences at the end of the conclusion, inside the 8 pages. No ethics section.
- LLM Usage Statement: the user writes it. On request we list our real uses from the records (code, experiments,
  literature search, the restyle of the neighbour-graph figure, the wording polish and fact checks). We can also add an
  empty section heading in `main.tex` between the conclusion and the references. By Monday at the latest.
- Whether Figure 3(a), the Lotto query, stays in the main paper. Decide at the length cut.
- The title, with the co-authors.
- Check that the teaser moves to the top of the right column once the introduction text reaches it.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
  The plan-12 handoff's section 5 still holds:
  - pull REPO before each edit;
  - apply changes with an assert script;
  - no notes in `.tex` files;
  - the style rules;
  - the build command;
  - the crop steps.
- Teaser source: `PF/figures/src/fig_teaser_t1.py a10`. Copy `PF/figures/fig_teaser_t1_a10.pdf` to REPO
  `Figures/fig_teaser.pdf`. Its data comes from MAIN `src/test/20261112_siglip_main/r28_teaser.py` (MAIN `837a7f81`) and
  `/data/SSD2/cosir_r28/20261112_siglip_main/qual/teaser/teaser_t1.json`, which holds the top 6 for 11 queries in four
  spaces.
- `\suppressfloats[t]` before the teaser in `Sections/1_introduction.tex` keeps it out of the left column. Without it,
  the teaser lands above the abstract.
- Figure routing (`agent-routing.md`): design judgment goes to Opus `xhigh`, look work to Opus `high`, and Codex does
  only the plumbing. The controller opens every PNG before the user sees it.
- Ducktor's tools run in a `worker` that reads the command file, for example
  `/root/.claude/plugins/marketplaces/ducktor/skills/duck-intro/SKILL.md`. Its records go to `/project/CoSiR/.duck/`,
  which is outside git.
- The paper's main model is c* on SigLIP 1 (r28). Figure data from Qwen3-era runs is stale.

## 6. State at handoff

- Running: nothing of this chat. Other tabs (r31, r32) may use the queue.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The teaser JSON is 140 KB. The scratchpad (contact sheets, page renders, a few MB) goes
  with the session.
- Repo heads: REPO `b540d16`, PF `3633f83`, MAIN `837a7f81`.
