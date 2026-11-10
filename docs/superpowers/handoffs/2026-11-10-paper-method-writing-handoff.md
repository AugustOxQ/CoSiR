# Handoff: settle the outline's open contribution points with the user, then guide the user's method writing with Ducktor (/duck-write); draft v1.2 is due 11-11 11:00

## Brief
- Done: /duck-outline (thinking template, four checks, storyline, page plan); an early experiments section for v1.2, every table cell checked by Codex.
- State: v1.2 has the experiments only (about 2,100 words, 9 tables). Two outline checks fail on the contribution wording, not on the results.
- Next: the user answers items 1 to 5 (contributions, method layout, related work); then /duck-write guides the user's method writing.
- Blocked: the user's answers; registration by 11-11 12:59 (the user's alone).

> Written 2026-11-10 20:40 by the paper chat (outline and early results). Outside the research loop: no loop step, no
> direction log entry. PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`.

## 1. Read in this order

1. This handoff, all of it.
2. `/project/CoSiR/.duck/reports/2026-11-10-outline.md`: the filled template, the four checks, the methodology
   outline (3.1 to 3.5), the storyline and the page plan. The method writing starts from its section 5.
3. `docs/superpowers/handoffs/2026-11-10-paper-writing-handoff.md`, sections 2 and 5: the user's paper decisions and
   the pitfalls (anonymity, D1, TURTLE, numbers, paper repo). Still valid; not repeated here.
4. `PF/notes/drafting_brief_v1.md` and `PF/notes/drafting_brief_v1.1.md`: names table, D1, TURTLE, style. They bind
   any text written for the paper, by the user's guidance or by an agent.
5. `PF/drafts/v1.1/03_method.md`, section by section as needed: the checked source of every method fact and number
   (its TRACE block names the code). Not to be read whole.
6. `PF/drafts/v1.2/04_experiments.md`: headings first, then as needed.

Skills: Ducktor's `/duck-write` (paper-writer). The user types it, or says to use it; then read
`/root/.claude/plugins/cache/ducktor/duck/0.2.3/skills/duck-write/SKILL.md` with Read and follow it, and its
`library/paper-writer/SKILL.md`. Ducktor is not switched on in CoSiR, so a typed command closes as "on" (results,
a short explanation block, a decision line, a report in `.duck/reports/`). The user writes in English.

## 2. Decided by the user, not to be reopened

- **This chat's plan** (user, 2026-11-10 20:05, in the outline chat): before 11-11 11:00, a draft v1.2 with part of
  the related work, the method and the results; the introduction and the qualitative examples later. Steps: (1) the
  outline and storyline with /duck-outline, the most important; (2) Ducktor guides the user through the method
  writing, and an agent writes a simple results section (done).
- Every decision in section 2 of `2026-11-10-paper-writing-handoff.md` stands (framing, temporary title,
  contributions 1 and 2 as worded on 11-09, supervision wording, D1, TURTLE, structure, r20).

## 3. Where things stand

**Outline** (`.duck/reports/2026-11-10-outline.md`). Paper type: technique paper (AI decided, in `.duck/decisions.md`).
Integrity gate: needs user attention.
- Check 1 (limitations to key idea) and check 2 (key idea to challenges) pass.
- Check 3 fails: other encoders (v1.1 Sec. 3.7) and photos (Sec. 3.8) are method parts with no challenge.
- Check 4 fails in three places. Contribution 1 says "a trunk shared by the aspects", but at test time each aspect
  uses the trunk of its own best epoch (17% to 444% apart in norm), and removing the shared trunk changes held
  retrieval by under 0.001 label mAP. The relational distillation term carries genre (its removal loses 113% of the
  genre gain over the plain embedding) and no contribution names it. No contribution names the main result.

**Experiments v1.2** (`PF/drafts/v1.2/04_experiments.md`; paper repo commits ae46ad0 to 833e160). Written by an Opus
`medium` worker from the reports and run records; sections 4.1 to 4.7, 4.7 a placeholder.
- Three Codex fidelity passes (`gpt-6.1-terra`, medium). Every cell of Tables 1 to 9 checked; Table 3 recounted
  independently twice, all counts match. 11 findings applied (two cells "below" that are level, the bootstrap and
  seed wording, the TURTLE clause, the TURTLE count for the plain embedding alone, 94 of 116, the photo sentence, the
  EmoSet validation fact, no paths in prose). One finding rejected: Codex wanted the D1 marker replaced by a
  sentence; the marker stays (the user words D1 after the authors' talk).
- Not independently checked yet: the numbers in prose lines 17, 21, 31, 37, 73, 75 (CRL 0.0582 and 0.0642), 93, 135,
  178, 196 (all but the timing values), 211 (the class-count claim) and 213. The pass-3 prompt pattern (narrow scope,
  named lines) works; the prompt is `PF/notes/fidelity_checks/experiments_prompt.txt` plus a scope header.
- Citation placeholders: NMI (Strehl 2002), ARI (Hubert 1985), Hungarian matching (Kuhn 1955); check before refs.bib.
- The writer's flags, taken into items 6 to 9 below: the held split sentence, the photo decision sentence, the
  comparator wording, the r19 rows.

**Not written for v1.2:** the method (the user writes it, guided), a short related work (item 5), the introduction,
the qualitative part. The v1.1 related work and method stay as the checked source.

## 4. Open points for the next step

Numbered as the user saw them, so the user can answer by label. Our recommendation follows each.

Key, from the outline (answer before the method writing):
1. **Contribution 1 wording.** (a) Claim one encoder pass for all aspects with no instruction, texts in the same
   spaces, and the cost (3.0 to 51.4 times less model time than the teacher's three passes; not measured against a
   teacher that encodes the image once); the trunk "shared in training" as a detail. (b) Keep "a shared trunk" and
   use one trunk snapshot at test time (new runs). Recommendation: a.
2. **Contribution 2.** Widen it to "the instructed model's relations as supervision: content-corrected neighbour
   graphs for emotion and style, relational distillation for genre". Recommendation: yes.
3. **Other encoders and photos.** Move both to Sec. 4 as generality results, with one method paragraph each.
   Recommendation: yes; photos as contribution 4 stays open for the supervisor talk.
4. **Challenge 2.** The user's own reading of the ablation, in one line (ours: the direct teacher term changes no
   aspect clearly in retrieval; the graphs and the relational term carry the gain). Ask answer-first.
5. **Related work for v1.2.** A writer makes a 0.75-page version of v1.1's four groups (Opus `medium` worker, paper
   section role). Recommendation: yes, launched at once so it is ready by 11:00.

From the results draft:
6. **The held split sentence** in 4.1 ("scored during development before {Method} was fixed; test values known
   before some comparators ran"). The source supports it: one configuration became the recipe after its held genre
   reading (`docs/reports/auto/percept/2026-12-28_recipe_sweep.md`, "What it does not show"). (a) Keep it and base the
   headline claims on MMArt and EmoSet test; (b) cut it. Recommendation: a.
7. **The photo decision sentence** in 4.5 (adaptation decided after the Affection test scores; "EmoSet is the cleaner
   test"). Recommendation: keep.
8. **Comparator wording (minor).** PercepT and CRL are called "our re-implementation" (from the glossary's "matched
   arm" and the facts brief). Not checked whether PercepT's numbers come from its authors' code. Ask the user.
9. **r19 rows (minor).** The three extra encoders and the cost are in; SemArt new artists and the detailed teacher
   rows are out. Recommendation: fine for now.

Also: the OpenReview registration by 11-11 12:59 Amsterdam with all co-authors (the user's alone); the final name
before 11-16 (earlier handoff, section 4).

**Suggested order for the next chat:** items 1 to 5 → launch the related-work writer (if 5 is yes) → the user writes
3.1 with /duck-write's guidance, then 3.2 to 3.4 → a Codex fidelity check of the new method text → join v1.2
(`PF/tools/join_draft.py drafts/v1.2`) for the user's 11:00 read. Items 6 to 9 can be settled during the morning
read.

## 5. Code and pitfalls

- All pitfalls of section 5 of `2026-11-10-paper-writing-handoff.md` apply (anonymity, numbers from sources, TRACE
  blocks, paper repo commits, shared figure files, `.duck/` excluded from git).
- Codex fidelity checks report partial coverage on a whole section; give each pass a narrow scope with named lines
  or tables, and verify every finding at its source before applying it (all 11 applied ones were checked).
- Ducktor's writing rule: when the user asks for text, research decisions the draft has not settled are listed and
  numbered for the user, not chosen by the agent.
- `drafts/v1.2/` holds only `04_experiments.md` so far; v1.1 stays unchanged.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat. `docs/superpowers/constitution.md` in the main repo was modified before this
  chat started (not ours; left as it is).
- On disk over 1 GB: nothing. The Codex prompts, logs and findings files are in this session's scratchpad (under
  1 MB in all; they go with the session).
