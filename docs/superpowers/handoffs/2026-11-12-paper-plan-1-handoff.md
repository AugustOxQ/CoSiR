# Handoff: work through the paper plan with the user (Sections 3 and 4 after the meeting), starting where the user says

## Brief
- Done: both PercepT (CLIP L/14) rows in Tables 2 and 3, below CoSiR; caption trims; registration keywords and abstract; the new related-work paragraph "Neighbour graphs as supervision".
- State: Sections 2 and 4 hold all table rows. The meeting changed the storyline: any aspects, graphs first, a SigLIP student.
- Next: ask the user which plan item to start. The plan is `notes/paper_plan.md`.
- Blocked: nothing.

> Written 2026-11-12 17:57 by the "paper sec4 tables 7" tab, cut at about 220k tokens at the user's yes. Outside the
> research loop: no loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`;
> REPO means `PF/CVPR2027-CoSiR-Paper/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/notes/paper_plan.md`: the storyline after the meeting, the user's own plan items for Sections 3 and 4 (from
   the user's `%` comments at the top of `REPO/Sections/3_method.tex` and `4_experiments.tex`), their status, and the
   open questions.
3. `PF/README.md`, the decisions table, last rows (2026-11-11 and 2026-11-12).
4. `PF/notes/abstract_registration_2026-11-11.md`, when the user works on the abstract or the introduction.
5. `PF/notes/after_meeting_2026-11-11.md`, for the pre-meeting points still to raise (see section 4).
6. `PF/notes/trace/2_related_work.md` (last section) and `4_experiments.md` (last two sections), only when those
   sections change again.

Skills: none. For a Ducktor step, follow `~/.claude/plugins/marketplaces/ducktor/skills/duck-<name>/SKILL.md` by hand
(Ducktor is off in CoSiR chats); records go to `/project/CoSiR/.duck/`.

## 2. Decided by the user, not to be reopened

All of section 2 of the earlier handoffs (`2026-11-11-paper-sec4-tables-{4,5,6,7}-handoff.md`) stands. New:

- **Storyline after the meeting** (2026-11-11, PF README): (1) CoSiR is a general method for any set of aspects named
  by instructions; (2) the instructed graphs and the graph-positive loss get more weight; (3) Qwen signal on a
  lightweight student backbone (SigLIP), at a much lower cost.
- **Captions** (2026-11-11, REPO `7c34f00`): Table 5's row sentences merged; "Qwen3" phrases cut from Table 7; a
  TODO on the underlines of Tables 6 and 7 is for the co-authors. The duplicate public-domain sentence of Figures 3
  and 4 waits for the caption decision.
- **Speed table** back in the main paper (user, Overleaf `451197d`).
- **Related work** (2026-11-12): the "VLMs and MLLMs" paragraph is gone (user). The new "Neighbour graphs as
  supervision" paragraph is in (REPO `4bd78dd`, 134 words, about 2/3 of the datasets paragraph). Guideline-driven
  clustering joins the multiple-clustering citations (REPO `cb473ff`, user yes). A short novelty search (reciprocal
  kNN graphs as training positives; correcting a neighbour graph for shared content) comes later, in the final
  related-work pass (user).

## 3. Where things stand

- REPO (pushed) at `cb473ff`; PF (pushed) at `8e5a65d`. The paper builds at 10 pages with no undefined citations.
- **Table 2, PercepT (CLIP L/14):** AMI 0.064 / 0.262 / 0.474 against CoSiR 0.102 / 0.411 / 0.510 (REPO `609a3cf`,
  by the "paper experiments 2" tab). **Table 3:** mAP 0.242 / 0.177 / 0.540 against 0.269 / 0.323 / 0.615 (REPO
  `62b51d8`; record `src/test/20261111_table3_rows/records/percept_clip.json`). 5 of 5 seeds kept, none collapsed,
  so no extra seed note in 4.1. The bold is unchanged.
- **The SigLIP 1 student against the Qwen3 teacher** (Table 6): AMI 0.099 / 0.441 / 0.528 against 0.089 / 0.406 /
  0.501; mAP 0.269 / 0.355 / 0.623 against 0.268 / 0.322 / 0.612. Speed table: 19.5 ms against 266.8 ms per image
  (13.7 times faster), 428M against 2.13B parameters on the image path.
- **Abstract registration:** keywords and a five-sentence abstract are in PF notes. `REPO/Sections/0_abstract.tex` is
  still empty; the user decides what goes in.

## 4. Open points for the next step

Key:
1. **Which plan item first** (`paper_plan.md`, Method and Experiments). The user's items: Figure 1 more general; the
   genre distillation moves to the experiments; Table 1 general, the current one to the experiments; new names for
   the teacher-student terms; the graph part split from the model; tables split by passes; one big table with two
   headlines; the SigLIP student as the lead result.
2. **The PercepT sentence in 4.2** ("the fair comparison is on the same backbone, Qwen3"): keep, or "CoSiR is above
   both versions of PercepT on every aspect"? Recommended: the second (PercepT on CLIP L/14 beats our Qwen3 version on
   style and genre). Not yet answered.
3. **Main tables and the SigLIP student:** lead Tables 2 to 4 with it? Table 4 (MMArt) has no SigLIP row. The r15
   round has MMArt on four encoders (archived report 2027-01-03, numbers not checked).

Minor:
4. Pre-meeting points (`after_meeting_2026-11-11.md`): Figure 1 comments from the co-authors, the retrieval read-out,
   the scope of the no-direct-writing rule. The pair-hit point is decided (project memory `pair-hit-limit`).
5. The introduction uses `\enumerate{...}` instead of `\begin{enumerate}...\end{enumerate}`. It builds, but the list
   never closes. Mention it when the user works on the introduction.
6. The comment beside `\input{Tables/cost}` in 4.5 still says "This goes to supplemental material".

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand (two chats share REPO and PF; commit only your own paths; no
  notes of ours in REPO; paper names only; no semicolons between clauses; "test set", never "held-out").
- **The user edits REPO in Overleaf.** Pull before editing (`git pull --ff-only`), and tell the user after each push
  that Overleaf needs a pull. When a file holds both the user's uncommitted lines and ours, stage only ours (write the
  HEAD-based version to the scratchpad, `git hash-object -w`, `git update-index --cacheinfo`, then `git commit` with no
  paths). Commit the user's own edits only when the user asks.
- The "paper experiments 2" tab may still edit REPO tables (it scores the PercepT row on the cluster-count grid for a
  supplementary figure). Check `git status` before each commit.
- Related-work facts: the citation check of 2026-11-12 (trace `2_related_work.md`) found that guideline-driven
  clustering builds its spanning tree over clusters and trains nothing, that k-reciprocal is person re-identification
  re-ranking, and that the Platonic metric is a kNN-set overlap between two representations (not "mutual").
- In zsh, `echo ====` fails (`=` expansion): quote such strings.

## 6. State at handoff

- Running: nothing of this tab. Others: "paper experiments 2" (supplementary cluster-count figure), r21 run.
- Uncommitted: in REPO, nothing. In PF, `CBMI_IMP_CAMREADY (2).pdf` (the user's, left out of git). In the main repo,
  files not ours.
- On disk over 1 GB: nothing. The session scratchpad holds a few MB of page renders and build logs.
