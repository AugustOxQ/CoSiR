# Handoff: CoSiR v2 publication plan for CVPR

Written 2026-10-02. Read this first in the next session; it points to everything else.

## Goal

Publish CoSiR v2 at **CVPR**. The deadline is about 40 days away (around 2026-11-11; confirm the exact
date with the user). Every plan item should be scheduled backwards from that date.

The user's four requirements for a good paper:

1. **Strong results.**
2. **A good storyline.**
3. **Clear baselines on a comparable benchmark.**
4. **A good problem definition** (conditional similarity) and a clear **method contribution**: what our
   model adds and why it works.

## Decisions taken on 2026-10-02

The user chose these from the options in the affect factor-learning held report:

- **Adopt SE as the v2 factor model.** Frame it as distant supervision from PercepT's affect teacher
  (GoEmotions RoBERTa `SamLowe/roberta-base-go_emotions`), with C0 as its matched control.
- **Set a held-row budget before any new pre-registration.** Held rows have already been read in three
  final tests: the repair plan, stage (d) and the affect held test. Every further read must be planned
  and counted.
- **Write the v2 publication plan.** This is the main deliverable of the next session.
- **Run the baselines the paper needs.** At least a teacher-only baseline (the naive rule on the raw
  28-d GoEmotions codes) and a PercepT comparison. The literature review will add more.
- Not chosen for now: closing out the percept line (archive tag, push, weekly-report fix). Its branch
  `experiment/percept_topic_pipeline` (head 1b70d7f) stays as it is until the user archives it.

## Where the evidence stands

Held test, from [`2026-10-19_candidate_a_affect_factor_learning_held.md`](../../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md):
naive-rule R@1 (%) at β 0.3, mean of both retrieval directions, 8,192 episodes per label, 13 candidates
per episode.

| Model | Pooled | Emotion | Art style |
|---|---:|---:|---:|
| CLIP only | 13.16 | 10.25 | 16.06 |
| original R3 (current system) | 19.45 | 15.43 | 23.47 |
| C0 (matched control) | 19.88 | 14.92 | 24.84 |
| **SE (adopted)** | **21.24** | **16.99** | **25.48** |

- SE − C0 on held: emotion **+2.08 [+1.51, +2.62]**, style **+0.65 [+0.07, +1.21]**.
- SE − CLIP only: **+8.08 pooled**.

**Weaknesses a CVPR reviewer will find. The plan must address each one.**

- **Small absolute numbers.** Emotion R@1 is 17% among 13 candidates. A label-aligned code reached 49.8%
  in the headroom probe, so the current factors are far from what is achievable.
- **Partly a shortcut.** Much of the emotion gain comes from captions that state the emotion word.
  Where no caption names it, the gain is +0.75 and not significant.
- **Weaker unsupervised claim.** Distant supervision undercuts the original differentiator,
  "self/unsupervised condition discovery". The storyline has to be rebuilt around what still holds.
- **No standard benchmark yet.** Everything so far uses in-house ArtELingo label episodes.
  [`2026-10-20_genecis_feasibility.md`](../../reports/auto/v2/2026-10-20_genecis_feasibility.md),
  written by another session and uncommitted on main, says GeneCIS (CVPR 2023, the paper CoSiR builds on)
  is usable for evaluation:
  - The object half can be evaluated today; the attribute half needs a Visual Genome 1.2 download.
  - It needs an image-to-image scoring mode and a text-condition adapter.
  - It measures zero-shot transfer, which is not v2's primary claim.
- **Missing baselines.** The paper has none of the published methods yet: GeneCIS's own model and
  composed-image-retrieval methods (identify these in the literature review; do not assume them),
  plus teacher-only and PercepT.

## What the next session should do

1. **Use the brainstorming skill (architectural path) to write the plan as a spec** in
   `docs/superpowers/specs/`. The old buddy plan is a model for the structure:
   [`docs/archive/buddy_publication_plan/`](../../archive/buddy_publication_plan/). It should contain:
   - goal and venue;
   - the problem definition;
   - contributions;
   - a claims table (claim, evidence, status);
   - baselines and benchmarks;
   - a numbered experiment plan with dates working back from the deadline;
   - the held-row budget;
   - risks.
2. **Settle these with the user early, one question at a time:**
   - **The problem definition, written precisely.** v2 scores `s(I, T | c)` across modalities. The
     condition `c` is given as support and contrast pairs (examples), not as a text phrase. Is that
     framing the paper's problem, and how does it relate to GeneCIS's image-plus-text-condition setting?
   - **The headline contribution and its "why".** Shared factors plus a support-conditioned score.
     Condition episodes from affect clusters and CLIP image clusters, which is distant supervision.
     Which differentiators against GeneCIS survive?
   - **The benchmarks.** ArtELingo label episodes as the primary evaluation, GeneCIS as a zero-shot
     comparison, or both. Any other dataset?
   - **The minimum baseline set and the compute budget.** DAS6 reservations are made by the user.
3. **Do a literature review before fixing baselines.** Delegate it to a deep-research subagent.
   Cover conditional similarity, composed image retrieval, and few-shot or support-conditioned
   retrieval, and record what each paper reports on which benchmark.
4. **Only then** write the plan (writing-plans skill) and execute it (SDD). Finish with the whole-branch
   final review on the most capable model.

## Inputs to read

- v2 foundation spec: [`2026-09-28-cosir-v2-ground-up-redesign.md`](../specs/2026-09-28-cosir-v2-ground-up-redesign.md)
- Affect factor-learning spec and held report:
  [`2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md`](../specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md),
  [`2026-10-19_candidate_a_affect_factor_learning_held.md`](../../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md)
- The v2 chain of results: everything in `docs/reports/auto/v2/` from 2026-10-04 on, in date order.
  It runs separability → ranking evaluation → CLIP-only baseline → naive-rule mechanism → factor
  collapse → repair (R3) → stage (d) → headroom probe → factor learning → affect factor learning.
- Index: `docs/reports/reports_sum.md` (the "Current work (v2)" line).
- PercepT background, for the baseline: the matched head-to-head
  [`auto/percept/2026-09-30_matched_percept_buddy_h2h.md`](../../reports/auto/percept/2026-09-30_matched_percept_buddy_h2h.md).
  It has a validated PercepT Stage 1 port (`scripts/buddy_percept_sweep/h2h_percept.py` on the percept
  branch) that reproduces §6g, with K = 40 and AUC 0.9258 against 0.9226.
- Project memory: `project_next-step-factor-learning.md`, `project_v2-publication-plan-pending.md`,
  `project_candidate-a-factor-discovery-status.md`.

## Constraints and conventions

- **Code.** Do not merge percept-branch code into main. Rebuild any PercepT baseline as a clean, tested
  v2 module, using the percept port and pilots as reference.
- **Implementation.** Use Claude Code subagents sized to the task. Use Codex only if the user asks.
  The main session launches long jobs. On DAS6, use only the `cluster-run` CLI, and keep all node-side
  work under `/local/wding/`.
- **Held rows.** Do not read them without a pre-registration and the budget above.
- **Reports.** Put them in `docs/reports/{auto/v2,stage,weekly}`, add one row per report to
  `reports_sum.md`, and run `scripts/check_reports_sum.py`. Follow the user's report rules:
  - a real baseline beside every headline number;
  - numbers with analysis;
  - figures;
  - paper-draft style, with no dashes.
- **Review.** The whole-branch final review is mandatory; it has caught a paper-facing defect every time.
- **Other sessions' work on main.** Main has uncommitted files from other sessions: the 2026-09-30 weekly
  report and its assets, the GeneCIS feasibility report, `scripts/preprocess_genecis.py`, and
  `configs/dataset/genecis.yaml`. Stage only your own files by explicit path.
  - The weekly report still calls the matched head-to-head interim. Its summary also says PercepT leads
    Stage 2 at 0.9226 vs 0.8534, which the head-to-head (master report §6k) superseded. Fix both before
    it is shared.
