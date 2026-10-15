# Handoff: CoSiR paper R1 (4): the user's answers to the review decisions, then our answer letter and one revision

## Brief
- Done: the full ARS review of v1. Verdict Major Revision: one repairable block on methodology rigor, warn on the other five dimensions, no Critical issue.
- State: 11 required and 28 suggested items. A fact check shows all 11 premises right, with 4 reviewer errors.
- Next: the user reads the review and answers the five decisions in `reviews/r1/03_open_decisions.md`.
- Blocked: waiting for the user's read.

> Written 2026-10-13 18:45 by the paper R1 (3) chat (outside the loop's phase chats). Clean cut at the context meter
> (208k), after R1 step 1. Previous handoff: [paper R1 (3)](2026-10-13-paper-r1-3-handoff.md). Runs with `claude-paper`.

## 1. Read in this order

PF = `docs/paper/cvpr2027_aspect_episodes/` (the private paper repo).

1. `PF/reviews/r1/03_open_decisions.md`: the five decisions, the reading list given to the user, the smaller
   points that take agent defaults. This is the chat's agenda.
2. `PF/reviews/r1/02_fact_check.md`: per item, whether the premise is right, what v1 already says, the cost class,
   whether it adds a held read.
3. `PF/reviews/r1/01_editorial_decision.md`, Part 2 (the roadmap: Required Revisions, Required Item Details,
   Suggested Revisions). Read Part 1 only where the user asks about it.
4. `PF/reviews/r1/run_log.md`: how the review ran (models, checkers, the one deviation).
5. The previous handoff, §3 (agent defaults and word counts) and §5 (code and pitfalls): both still apply.

## 2. Decided by the user, not to be reopened

- The panel ran as configured (`reviews/r1/00_reviewer_configuration.md`), 2026-10-13.
- The user reads v1 with the review; the open defaults settle in that one pass (2026-10-13).
- No more tuning of AFF for a larger held number; a new-method loop is on hold, undecided (2026-10-13, user, after
  seeing the recommendation; PF README).
- All decisions of the earlier handoffs stand (PF README decision table).

## 3. Where things stand

- **Review** (ARS 3.23.0, full mode, sprint contract): five seats, each with a blind Phase 1 and a paper-visible
  Phase 2; every card passed `check_phase_conformance.py` and `check_panel_synthesis.py`. The synthesis passed the
  checker (`PANEL-SYNTHESIS: PASS`). Fired conditions F2, F3, F5; the decision follows F2 (D1 block, repairable).
- **Required items, short:** R1 headline vs B′+style; R2 mechanism vs RS (+0.02); R3 a result beyond ArtELingo and
  CLIP in the body; R4 an upper reference; R5 emotion as a caption property, viewer agreement; R6 8 pages; R7 the
  held-read count; R8 training-seed spread; R9 a pre-registration record; R10 MLLM scope; R11 the B ablation.
- **Fact check, the load-bearing facts** (re-checked by the chat): the decision rule `c394b60` was committed
  2026-10-09 04:53, and the first aspect-episode held read (H5) came on 2026-10-10. The ArtELingo held paintings were
  read 4 times (H1, H2, H3, H5); H4 is CUB. B′+style was built 2026-10-06, before AFF's own rule, and was the held
  rule's secondary check S1. AFF − B′+style on development: +0.33 [+0.05, +0.62].
- **Reviewer errors** to answer: the +0.5 development bar was not used by the held rule; §3.3 already has an
  oracle-label reference (23.09, one pair); "reads the condition" is in §1, not the abstract; SemArt is a painting
  dataset. Project side: r9 spec §8 says "read five times (H1 to H5)", but H4 is CUB (spec not changed; user to say).

## 4. Open points for the next step

- **Key, answer-first** (the user answers with a line of why, then the agent shows its recommendation): the five
  decisions of `03_open_decisions.md`. Our recommendations, to show only after each answer:
  1. Headline: AFF beats B′ and every comparator of the held checks; it ties B′+style and RS on held paintings. The
     condition gain (+3.06 against the MLLM's 0.12) is the central finding. Drop "clears this bar" (§8) and turn
     "reads the condition" (§1) into a claim about the condition gain, with the RS tie stated.
  2. Run R8 (seed spread), R11 (B ablation), R5 (viewer agreement) and the R4 oracle on all three pairs, all on
     development or fresh episodes. State human raters (R4), the MLLM scope (R10) and the held random gate (R2) as
     limits.
  3. No new held reads.
  4. An anonymised copy of the decision rule in the supplement, with its date and the read date, and no commit hash.
  5. The analyses are CPU work: one small loop, beside r9's GPU stages if `uptime` and `free -g` allow.
- **Then R1 steps 2 to 4** (agent defaults, due 2026-10-19): our answer per item (ARS
  `academic-paper-reviewer/templates/revision_response_template.md`, R→A→C) into `PF/reviews/r1/04_response.md`;
  one revision of the section files, still v1; re-join, check only the changed numbers, `check_invariants.py` over
  any prose pass; commit, push and tag the revised state. Results of new analyses enter at R2 (v2) if they come later.
- **Minor:** Figure 6 (a quick CPU job: re-score the drawn episodes with B and AFF), after the revision.

## 5. Code and pitfalls

- Keep `PF/reviews/r1/` unchanged after the answer letter: the R2 re-review uses the frozen round-1 cards
  (`seats/*/phase2.md`, `00_reviewer_configuration.md`) as its yardstick.
- The review target was `reviews/r1/manuscript_v1_review_copy.md`: `paper_v1.md` with every HTML comment removed and
  the title added. Make the R2 target the same way.
- `ars_anchorize_draft.py` rejects the draft (the `[^goemo]` footnote in §5.2), so the review has no JSON roadmap.
  Turn the footnote into text before R2 if the ARS patch tools are wanted.
- The checkers live in `/root/.claude/plugins/cache/academic-research-skills/academic-research-skills/3.23.0/scripts/`.
- The handoff and every file in the public repo never carry the paper's title.
- At the next cut: `loop-next <handoff> --label "paper r1 5" --workspace "$HERDR_WORKSPACE_ID" --paper`.

## 6. State at handoff

- Running: nothing of this chat's.
- Uncommitted: nothing. The paper repo is pushed (last commit `0431c88`); tag `r1-v1-pre-review` still marks v1.
- On disk over 1 GB: nothing. `PF/reviews/r1/` is 0.3 MB; the scratchpad holds small briefs only.
