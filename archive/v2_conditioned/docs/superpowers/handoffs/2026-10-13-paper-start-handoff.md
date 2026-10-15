# Handoff: CoSiR paper: start the CVPR draft from what is fixed today; the r9 results fill in later

## Brief
- Done: r9 spec approved; the paper chat opens beside the r9 build (user, 2026-10-13).
- State: AFF is the method (held GO on CLIP, +0.60 R@1 over B′(A0)); r9 tests two more backbones and ArtELingo-28.
- Next: the paper's outline and the stable sections (task, method, related work, CLIP results), with ARS review rounds.
- Blocked: the scope results (r9 about 10-16; CUB, SemArt, Affection later). Results enter the body until 2026-11-06.

> Written 2026-10-13 15:15 by `r9 decide 2`. Outside the loop's phase chats: a `claude-paper` chat (ARS plugin on).
> Direction log: [2026-10-13 · Round 8](../direction_log.md) ("Now" and the grill bullet).

## 1. Read in this order

1. The stage report for outside readers, `docs/reports/stage/2026-10-12_affect_steering_line.md`: the story of the
   AFF line so far.
2. The round 6 held-test report, `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md`: the paper's main numbers
   today (seven checks, per pair, the comparators, the disclosure of reads).
3. The r9 spec, `docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md` (§1, §3.1 and §6's licensed claims): what
   the scope results may say, and when they arrive.
4. `GLOSSARY.md` and `docs/superpowers/constitution.md` (C5: every held read reported, the number of reads per split
   disclosed; C14: claim scope).
5. Literature: `docs/reports/literature/` (index in `docs/reports/reports_sum.md`), from `2026-10-21_cvpr_literature_review.md`,
   `2026-11-28_design_l_novelty.md` and `2026-12-01_affect_caption_datasets.md`.
6. The old paper workspace `docs/paper/cosir_v2_aspect_similarity/` (an outline-only ARS run from the method-A era; its
   branch logic is out of date) and the CVPR plan spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`
   (K2, K4, §10 are partly out of date; the facts brief `.scratch/r9-scope/facts_brief.md` §6.3 lists ten points).

## 2. Decided by the user, not to be reopened

- The paper runs in a `claude-paper` chat with round-by-round ARS review and revision (2026-10-13, decision A).
- It starts now, with the stable parts; the claim sentence (abstract, intro, contributions) is drafted in two or three
  versions until the r9 reads exist (2026-10-13, grill Q4).
- A result enters the paper body if its report exists by 2026-11-06; later ones go to the supplement (user, after
  seeing the recommendation). Affection, if its data comes after 2026-10-24: future work or the appendix.
- Every backbone goes in one main table, pass or fail (spec D13).
- Ongoing papers never go public: `docs/paper/` is gitignored in this public repo, and the repo is never named in
  paper material (double-blind).

## 3. Where things stand

- Method: AFF (glossary), frozen since round 3. Held read H5 on CLIP: all seven checks pass; +0.60 [+0.46, +0.73]
  R@1 over B′(A0), +5.89 over RCA; style × genre loses (−0.43 against B′(A0)).
- Scope runs under way (r9): Qwen3-VL-Embedding-2B and SigLIP 2 on ArtELingo (H6, H7); ArtELingo-28 translated
  captions on all three backbones (H8 to H10). Stage 3 (CUB, SemArt, Affection) gets its own decide chat about
  2026-10-17.
- Dates: abstract 2026-11-10, paper 2026-11-16.

## 4. Open points for the paper chat

- The paper's title and folder under `docs/paper/` (the old folder's name is from the method-A era): agent proposes,
  user decides.
- The order of the ARS rounds against the result dates.

## 5. Code and pitfalls

- Numbers come from the reports and their result files, never from memory or summaries; a number the paper states is
  traced to its file.
- The paper is not a report: no row in `reports_sum.md`.

## 6. State at handoff

- Running: the r9 build (after the user's container restart).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: nothing new.
