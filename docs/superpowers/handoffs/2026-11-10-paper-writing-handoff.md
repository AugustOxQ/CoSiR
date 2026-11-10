# Handoff: guide the user's own writing of the CVPR 2027 paper with Ducktor, from the outline to the introduction to each section

## Brief
- Done: related work and method v1.1 drafted, checked by Codex and pushed; Figure 1 shows the text path; r20's ablation decisions are in the drafting brief.
- State: v1.1 has the related work (5,861 words) and the method (7,662 words); no outline, introduction, experiments or conclusion yet. Paper due 11-17 12:59 Amsterdam.
- Next: Ducktor guides the user's writing: outline (/duck-outline), then introduction (/duck-intro), then each section (/duck-write).
- Blocked: the user registers on OpenReview by 11-11 12:59 Amsterdam; the final name and the wording of contribution 2 are open.

> Written 2026-11-10 19:55 by the paper chat (v1.1 related work and method). Outside the research loop: no loop step,
> no direction log entry. PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions table: every paper decision with who made it (rows of 2026-11-02 to 2026-11-10).
3. `PF/notes/drafting_brief_v1.md`, then `PF/notes/drafting_brief_v1.1.md` (v1.1 wins): names table, numbers, claims,
   style, anonymity, D1, TURTLE, the r20 ablation decisions.
4. `PF/submission/registration.md`: the temporary title and abstract (about 230 words).
5. The headings and TRACE blocks of `PF/drafts/v1.1/02_related_work.md` and `PF/drafts/v1.1/03_method.md` (not
   whole: they are long).
6. `docs/superpowers/handoffs/2026-11-08-paper-v1.1-handoff.md`: where the experiment numbers come from (reports, run
   records) and the plan of the one experiment section.
7. The Ducktor library skill behind each command, before the command runs, if you need its steps:
   `/root/.claude/plugins/marketplaces/ducktor/library/{tech-paper-template,intro-drafter,paper-writer}/`.

Skills: Ducktor's commands, which the user types (none is model-invoked): `/duck-outline` (tech-paper-template),
`/duck-intro` (intro-drafter), `/duck-write` (paper-writer); later `/duck-figure`, `/duck-polish`, `/duck-review`;
`/duck-why` explains a step. `/duck <sentence>` picks a command. Ducktor's companion mode (`/duck on`) stays off in
CoSiR (user, 2026-11-09). ARS is not loaded in this chat (`loop-next` cannot start both); the ARS review check needs a
later chat started with `--paper`.

## 2. Decided by the user, not to be reopened

- **This chat's job** (user, 2026-11-10; PF/README.md): Ducktor guides the user's own writing, from the outline to the
  introduction to each section.
- **Framing** (user, 2026-11-09): conditional similarity across aspects (emotion, style, genre), for images and text.
  The image side is the main evidence; the text side is the second result (option B, user after seeing the
  recommendation).
- **Temporary title** (user, 2026-11-09): "CoSiR: Learning Conditional Similarity for Emotion, Style and Genre from an
  Instructed Embedding Model". Drafts write `{Method}`; "CoSiR" only in the registration text.
- **Contributions** (user, 2026-11-09): (1) the model: image and caption towers, a trunk shared by the aspects and one
  head per aspect, reading the plain embedding once; (2) the aspect neighbour graphs as supervision. Contributions 3
  (evaluation) and 4 (photo adaptation) wait for the user's talk with the supervisor.
- **Supervision wording** (user, 2026-11-02): "trained without aspect labels; labels are used only to select settings
  on the validation split and to evaluate". Never "label-free", or "unsupervised" for the method. The instructions
  name the aspects.
- **D1** (user, 2026-11-05): no text states or implies that the method's text side alone has lower known-pair R@K
  than plain Qwen, and no text leaves it out on purpose, until the user has talked with the other authors. Late
  fusion's pair numbers may be used. `03_method.md` marks its late-fusion paragraph with `<!-- D1: ... -->`.
- **TURTLE** (user, 2026-11-05): every sentence that says we ran it carries "at its released code's default settings,
  without its hyperparameter selection". "Better than" over outside methods only per reported cell.
- **Structure** (user, 2026-11-08): related work in four groups; one experiment section (setup; clustering on Qwen
  across datasets; retrieval; other encoders; photos; ablation; qualitative); limits in the conclusion; compact
  versions of all sections at the end.
- **r20** (user, 2026-11-10): the ablation table gets a retrieval column; the instructed graphs, the content correction
  and the graph-positive terms are worded as one chain with no separate shares; teacher figures go to the
  supplementary.

## 3. Where things stand

- **Drafts v1.1** (paper repo, last commit 5a25e2d): `02_related_work.md` and `03_method.md`, each checked by Codex
  against its sources (35 and 27 findings, all handled) and pushed. v1 (`drafts/v1/`) stays as it is.
- **Figure 1**: `PF/figures/src/fig1_overview.py` (png, pdf); its caption in `figures/captions.md`.
- **refs.bib**: 213 checked entries. New ones are merged with `PF/tools/merge_bib.py`.
- **Not written yet**: an outline of the whole paper, the introduction, the experiments, the conclusion, the final
  abstract, the LLM Usage Statement, the compact versions, the LaTeX.
- **CVPR 2027 facts** (official CFP, Dates and Author Guidelines pages, checked 2026-11-09):
  - Registration closes 11-10 23:59 AoE (11-11 12:59 Amsterdam), with no extension. It needs the OpenReview title,
    abstract and all co-authors. No author can be added or removed after it; only the order can change until 11-16.
  - Title and abstract stay editable until the paper deadline (from the FAQ, read through a summarising fetch, not
    verbatim).
  - Every author needs a complete OpenReview profile with conflicts, else a desk reject is possible.
  - A mandatory LLM Usage Statement goes on an extra page; without it, a desk reject.
  - 8 pages plus references. Paper 11-16 23:59 AoE (11-17 12:59 Amsterdam); supplementary 11-23 (11-24 12:59).

## 4. Open points for the next step

- **Key, ask first:** how the v1.1 drafts enter the guided writing: as source material the user rewrites from, or as
  drafts that Ducktor's steps revise.
- **Key:** a new schedule to 11-17 12:59 for outline, introduction and each section (the old agent proposal is in
  the 11-08 handoff, lines 83 to 88: experiments 11-11 to 11-12, introduction 11-12 to 11-13, ARS review 11-13 to
  11-14, compact versions and LaTeX 11-15 to 11-16).
- **Key, the user's alone:** the OpenReview registration by 11-11 12:59 Amsterdam, with all co-authors.
- **Key, before 11-16:** the final name and anonymity: (a) another name with the supervisor, or (b) keep "CoSiR" and
  make the public repo of that name private or rename it. Our recommendation: one of the two.
- **Key:** does the contribution 2 wording ("aspect neighbour graphs as supervision: mutual nearest neighbours of the
  instructed embeddings, content-corrected for emotion and style, as contrastive positives") match what the user
  means by "buddy"? Our recommendation: yes.
- **Minor:** the compact method (talk planned); r19's rows in the ablation or the appendix; MMArt's narrative-text
  result worded as a limit; the second instructed embedder in the supplementary.

## 5. Code and pitfalls

- **Anonymity:** no paper text names the project, either repo, "buddy", paths, run tags or people (cited authors are
  fine). The GitHub repo of the project is public.
- **Numbers** come from reports or code, never from memory. Each section keeps a TRACE block. Only checked citations
  enter `refs.bib`.
- **Fidelity checks:** a Codex job per section, with a prompt like `PF/notes/fidelity_checks/method_prompt.txt` (check
  `codex-usage --room fidelity` first; ledger `/project/claude-config/codex/jobs.md`).
- **Paper repo:** `docs/paper/` is its own private repo (`cosir-paper`); commit and push each finished piece there.
  The main repo never stages `docs/paper/`.
- **Shared files:** the r20 tab owns `PF/figures/fig_clusters.*`, `PF/figures/fig_teaser.*` and their `captions.md`
  sections. In the main repo, commit only your own files.
- **Ducktor records** go to `/project/CoSiR/.duck/`, which `.git/info/exclude` keeps out of the public repo.
- **Joining:** `PF/tools/join_draft.py drafts/v1.1` joins the sections for reading; never edit a joined file.
- **Python:** `PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat.
- On disk over 1 GB: nothing.
