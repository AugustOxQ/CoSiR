# Handoff: discuss with the user what CoSiR v2 does next after round 5's kill (decisions C and A)

> Written 2026-10-08 21:55 by the idea3-goemotions tab. Loop step reached: after step 9 (round 5 verified and reported)
> and the user's reading of the round-5 briefing; decision C not yet recorded, decision A open. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The direction log `docs/superpowers/direction_log.md`: the "Now" paragraph and the entries since 2026-10-02 (the
   CVPR loop so far, who decided what).
2. Round 5's user-read briefing `docs/user_read/2026-10-07_idea3_goemotions.md`: what idea 3 tried, why it failed,
   our advice, and the user's three open decisions, written for the user. Then round 4's briefing
   `docs/user_read/2026-10-07_round4_vetoes.md`, whose decisions 2 to 4 (B′(A1), idea 3 and design L, style × genre)
   are partly still open.
3. Round 5's full report `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md`: Summary, §6 (why the sharper placement
   did not help) and §9 (what follows). Round 3's report `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`
   Summary and §8: the only pre-registered pass (AFF on seeds 49 to 51) and what it licenses.
4. The CVPR readiness memo `docs/reports/stage/2026-10-07_cvpr_readiness.md` and the CLIP fine-tuning comparator
   report `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md`: what a reviewer will ask, and a reported baseline
   whose role in the paper is on hold.
5. The weekly report `docs/reports/weekly/2026-10-07_aspect_task_to_affect_steering.md`: the paper story so far
   (idea and contributions, AFF bottom-up as one design, a go/no-go checklist).
6. The process: `~/.claude/rules/research-loop.md` (steps, decisions A to C, loop sizes, the grill, wrap-up),
   `docs/superpowers/constitution.md` (standing principles), the held ledger `docs/superpowers/held_ledger.md` and the
   seed ledger `docs/superpowers/episode_seed_ledger.md`.

## 2. Decided by the user, not to be reopened

- **AFF, frozen as tested in round 3, is the current best** (GO on seeds 49 to 51, 2026-10-06; round 3 report).
- **Round 4's kill stands** (no veto on AFF's gate beat AFF on seed 42; 2026-10-07; direction log). The veto direction
  on A0 is considered spent.
- **Round 5's kill stands** under its committed rule (neither GoEmotions placement candidate cleared the bar or beat
  AFF on seed 42; 2026-10-07; rule `src/test/20261123_idea3_goemotions/DECISION_RULE.md`). Idea 3 was ruled not part
  of the parked grouping redesign (design L) by the user on 2026-10-07 (spec
  `docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md`).
- **Held-read budget:** ArtELingo aspect episodes, 1 main + 1 reserve read, 0 used (handoff of round 5, §2; held
  ledger). Idea 3 ran first precisely to keep both reads.
- **Venue and dates:** CVPR, abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23 (direction log).
- **Process:** a spec the user approves, a rule committed before any code with a fresh Opus check, subagent-driven
  implementation, an independent re-derivation, the whole-branch final review before the report; scoped commits to
  main without asking, never push.

## 3. Where things stand

- **AFF** on the fresh seeds 49 to 51 (round 3): bar margin +0.591 [+0.462, +0.729] R@1 against B′(A0), the strongest
  condition-free comparator of that round; below B′(A0) on style × genre (−0.580).
- **Seed 42** (development, read very many times): AFF fused R@1 19.137 against B′(A0) 18.437 and B′(A1) 18.805
  (B′ rebuilt with the CSD style grouping, the strongest condition-free scorer we have); AFF minus B′(A1) +0.332
  [+0.048, +0.625].
- **Round 4** (vetoes on AFF's gate): net rankings against AFF −18, −18, −34 of 49,152; kill.
- **Round 5** (GoEmotions placement of captions): placement accuracy 86.17% against the CLIP caption head's 35.72%,
  yet G-T and G-TF reached bar margins of only +0.356 and +0.427 against B′(A0) and lost to AFF by 169 and 134 net
  rankings; kill. Why (descriptive): every agreement multiplies the caption side by the image head (9.81% accurate,
  emotion lift 1.04), and the CLIP agreement carried cross-painting similarity the GoEmotions agreement lacks.
- **The CLIP fine-tuning comparator** (label-free LP, LB, LoRA): about 15.0 R@1 pooled over seeds 49 to 51, against
  B′(A0) 18.29 and AFF 18.88; a reported baseline, framing on hold (CVPR memo).
- **Free:** episode seeds 52 and later; both held reads. Nothing is running.

## 4. Open points for the next step

Start by asking the user for **decision C**, their reading of round 5, in a sentence or two (answer first; show ours
after: briefing "Advice"), and record it in the direction log entry. Then discuss **decision A**. Key questions, asked
answer-first, at most three per round:

1. **(key) What comes next.** (a) The held-split paper test with AFF frozen (our view in the briefing); (b) more
   method work, chiefly the grouping redesign (design L) for a real style signal; (c) a small caption × caption
   detector experiment (it could not fix the steering term). Each with its cost against the CVPR dates.
2. **(key) B′(A1) in the paper test**, if (a): a GO check (the test could fail on it), a comparator reported beside the
   result (our view: it uses a pretrained style model AFF does not use, so it is not the matched comparison), or left
   out. It must be fixed before that test's rule is written.
3. **(key) Style × genre**: disclose the loss with design L as the route, or take up design L now.
4. **(minor, batch with recommendations)**: the CLIP fine-tuning comparator's role (reported baseline, our default);
   which held split episodes and seeds the paper test uses (the held ledger's budget rules); whether the paper test also
   reports R1 beside AFF.

Then the loop size from `spec_experiment.md`: the paper test is a claim test (full loop: literature check of the
strongest comparator at step 3, a grill, a spec); design L is a new idea (full loop).

## 5. Code and pitfalls

- **AFF's frozen code** is round 3's (`src/test/20261121_round3_affect_gate/`, `r3_*`), reused by round 4
  (`src/test/20261122_round4_aff_vetoes/`, `r4_*`) and round 5 (`src/test/20261123_idea3_goemotions/`, `r5_*`).
  Module names must not repeat across rounds (their folders end up on `sys.path`); use a new prefix.
- **Never built:** test-seed runners for rounds 4 and 5 (their kills made them unnecessary); round 5's final review
  lists what to fix before any reuse of its code (`final_review/final_review.md` §4, §5).
- **Held rows have never been read by this line.** A held-split test needs its own rule, the held ledger row, and the
  "final scripts check the ledger and refuse to run twice" guard.
- **Process lapses to avoid** (round 5 report §7): every Python call, pytest included, with
  `PYTHONDONTWRITEBYTECODE=1` (subagents forgot and left `.pyc` in read-only folders); mutation tests on copies, never
  in place; tests built on the real data shape (a guard bug passed two reviews because its tests used a wrong shape);
  never estimate a time in a log (`TZ=Europe/Amsterdam date`).
- **Codex** may now take bounded jobs (`~/.claude/rules/agent-routing.md`, 2026-10-08): figures, re-derivations,
  fidelity checks, mechanical tasks, second-opinion reviews; run `codex-usage` before and log each job in
  `/project/claude-config/codex/jobs.md`. Never for discussion, brainstorming or grills.

## 6. State at handoff

- Running: nothing (the round-5 tab ends after this handoff).
- Uncommitted: nothing of this line (`bin/`, `docs/paper/` and `.DS_Store` files are the user's and stay untracked).
- On disk over 1 GB: nothing. Round 5 left about 50 MB under `src/test/20261123_idea3_goemotions/` (gitignored
  `cache/`, `results/`, `rederive/results/`, `final_review/out/`), kept because the report rests on them.
