# Handoff: r7 decide 2: write round 7's spec and decision rule from the closed grill, have the rule reviewed, get the brief approved, then cut to r7 build

## Brief
- Done: step 3 (facts brief, a standard and a deep literature check), decision B and the grill for design L.
- State: plan settled. A ceiling check from true style labels (stop below +0.5), CSD with its genre part removed and generic token statistics on seed 42, then fresh seeds 55 to 57 in three stages.
- Next: write the spec and decision rule (step 6), have the rule reviewed on Opus, user approves the brief, cut to `r7 build`.
- Blocked: nothing; the user approves the brief in that chat.

> Written 2026-10-12 02:05 by the `r7 decide` chat (cut at its context threshold). Loop step reached: after decisions A
> and B and the grill (step 5), before step 6. Direction log: "Now" and "Parked ideas" (updated 2026-10-12); this
> round's entry is written at its wrap-up.

## 1. Read in this order

1. This handoff's §2: the grill's record, the source of truth for the spec's §4. The visual version is the plan page
   `docs/superpowers/specs/2026-10-11-r7-design-l-plan.html` (its source `.src.html` beside it); the user did not paste
   its reply but confirmed every suggestion in chat.
2. `src/test/20261126_design_l_style_source/step3_facts.md` (our numbers with sources) and `step3_candidates.md`.
3. Results briefs only: `docs/reports/literature/2026-11-27_style_sources.md` §1 and
   `docs/reports/literature/2026-11-28_design_l_novelty.md` §1, §4 (novelty) and §5 (comparators).
4. `~/.claude/templates/spec_core.md`, `spec_experiment.md`, `decision_rule_template.md`;
   `docs/superpowers/constitution.md`; round 3's rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md` (fresh-seed
   test, D6 thresholds) and round 6's rule `src/test/20261125_artelingo_held_test/DECISION_RULE.md` (Holm, layout).
5. `GLOSSARY.md`.

## 2. Decided, not to be reopened

All 2026-10-11 to 2026-10-12, in the `r7 decide` chat.

- **Type and size:** new idea, full loop (user, picked the recommended option).
- **Idea (decision B):** first test whether CSD can be used once its genre part is removed (residual and cut-across
  variants); generic sources run in parallel (Q7); a wider screen of new style encoders is parked (user, after seeing
  the recommendation). **Generality principle** (raised by the user): encoders fixed in advance and chosen without the
  target dataset's aspect labels; the grouping step knows nothing of aspects.
- **Q1, the ceiling check:** a style grouping from the 27 true WikiArt style labels (heads place items, as for every
  grouping), genre still on the image grouping, scorer told; style × genre on seed 42 against its matched control.
  Stop if the point is below +0.5 or its 95% lower bound is at or below 0; +0.5 to +1.0 is a weak zone (the run goes
  on, the run chat flags it at the top of its report and pushes a note to the user's phone); +1.0 or more is clear
  (user; ours was +1.0, so flag it under "Check these"). References: told CSD +0.13 [−0.35, +0.60], told without a
  style grouping −0.40 [−0.64, −0.16] (step3_facts.md).
- **Q2, what picks between variants:** seed-42 episode R@1; labels only describe (AMIs) (user, after a facts table,
  before our recommendation; same as ours).
- **Q3, by Friday:** development plus a pre-registered test on fresh seeds, done by Thursday 2026-10-15 (user; same as
  ours).
- **Q4, scorers:** both R1 and extended AFF (AFF's gate open when the reader picks affect or the new style grouping),
  each on its retrained reader (user, "AFF's scores are already recorded in rounds 5 and 6"; ours was extended AFF
  alone).
- **Q5, the seed-42 gate (all three):** the pooled development bar (C10) over the strongest of B, B′(A0), B′(A1), its
  own B′ and its own matched control; style × genre at or above its matched control (point); emotion × style and
  emotion × genre each at least −0.3 R@1 against AFF, paired (user; same as ours).
- **Q6, fresh seeds, a fixed sequence** (a stage is tested only if the one before passed): stage 1, all condition-free
  comparators (B, B′(A0), B′(A1), own B′, own matched control) pooled with Holm; stage 2, a paired win over AFF (C3);
  stage 3, style × genre above B′(A1), a superiority check (lower bound above 0). **Friday 2026-10-16:** stage 1 passes
  → go for the method paper with design L; otherwise no-go → the ArtELingo paper with AFF (user; ours was stage 1 with
  AFF secondary, the staging extends it).
- **Q7:** one CSD-based and one generic scorer go to the fresh seeds, each chain at one-sided 0.0125 (user, after the
  budget estimate: a few hours of compute, 10 to 20 min of GPU).
- **After the deep check** (user, "all suggested confirmed", 2026-10-12):
  - the generic arm is intermediate-layer CLIP ViT-B/32 token statistics (per-channel mean and standard deviation,
    high-norm outlier tokens dropped) with the CLIP image direction removed; it replaces "a second CLIP grouping cut
    across the first";
  - every arm gets a size-matched random grouping as a control;
  - leftover content is measured by the grouping's AMI with the CLIP image grouping (label-free);
  - the residual arm is implemented and cited as Cui et al. 2007's orthogonal re-clustering (ClustPy Orth1/Orth2).
- **Minor answers, our recommendations, confirmed in bulk by the user** ("all suggested confirmed"; record as "user,
  after seeing the recommendation"):
  - m1 the ceiling uses probe placement; a second ceiling with true labels on the development items is descriptive.
  - m2 CSD variants: the residual, grouped by Leiden at the defaults; Leiden on CSD's neighbour graph with edges inside
    one content group down-weighted to 0 or 0.5; at most four; raw CSD as reference. The content grouping is the
    reader's 64-group CLIP image k-means grouping.
  - m3 CLIP logistic heads for images and captions; no CSD features at scoring time.
  - m4 labels only for the ceiling and descriptive AMIs; seed 42 only; no held rows; sample IDs per C6.
  - m5 each variant against its own matched control, its own B′, B′(A0) and B′(A1).
  - m6 round folder `src/test/20261126_design_l_style_source/`, module prefix `r7_`.
  - m7 the reader retrained with round 1's recipe, unchanged.
  - m8 fresh seeds 55, 56, 57; 4,096 episodes per pair per seed; C9 statistics.
  - m9 and m11 the pick: the highest pooled bar margin among the variants that pass the gate, one CSD-based and one
    generic, fixed in the rule so the run needs no decision.
  - m10 stage 3 as superiority (lower bound above 0).
  - m12 glossary terms "ceiling check" and "extended AFF" (add them with the spec).
  - s1 the seed-42 winner goes on even if its AMI against genre is worse than raw CSD's.
  - s2 if no variant passes the gate, design L stops on seed 42 and Friday is a no-go.
  - s3 if CSD fails the gate but a generic variant passes, the generic scorer goes to the fresh seeds alone at the full
    one-sided 0.025. This departs from B's order ("generic only if CSD works"): flag it under "Check these".

## 3. Where things stand

Numbers are in `step3_facts.md` with sources. Seed 42, style × genre: AFF 20.74 R@1 against B′(A0) 21.07 and B′(A1)
21.42. Held (round 6): AFF minus B′(A1) on style × genre −1.24 [−1.51, −0.96]. Style–genre label AMI 0.217 (148,956
scorer-train rows). Costs: Leiden 10 s, a head 20 to 36 s, scoring a configuration 20 s, the reader 10 min on CPU.

## 4. Open points for the spec (agent defaults unless marked)

1. **Stage types** (agent default, list under "Check these"): ceiling = exploration with a stop gate; seed-42 stage =
   new idea; fresh seeds = claim test worded per C14. The rule is checked by a fresh reviewer on Opus `xhigh` and
   committed before any code (C7).
2. **Extended AFF's thresholds:** round 3's τ were percentiles of seed-42 reader margins (its rule D6); recompute by the
   same procedure for each retrained reader, and say so in the rule.
3. **Regression check (C12):** with the reused round 6 pipeline, reproduce AFF's seed-42 pooled 19.137 (+0.700 over
   B′(A0), +0.332 over B′(A1)) and told A1 style × genre 21.82 before any new number.
4. **Selection disclosure (C8):** about 50 variants on seed 42 before AFF, about 10 in rounds 4 and 5, and this round's
   variants × 2 scorers.
5. **Claim scope:** the non-redundancy step is not new (Gondek and Hofmann 2004, Cui et al. 2007, FCMI 2023); the
   combination is (deep note §4). A fresh-seed GO is worded per C14; paper claims need a held read later (C5).
6. Seeds 55 to 57 hash-checked against every used seed and added to `docs/superpowers/episode_seed_ledger.md`.
7. At approval, paste the brief; then cut to `r7 build` (`loop-next ... --label "r7 build" --unattended`).

## 5. Code and pitfalls

- Round folders go on `sys.path`: module prefix `r7_`. Development on seed 42; no held row is read (C5); joins through
  `row_to_image` per C6, never `range(n)`; every Python call with `PYTHONDONTWRITEBYTECODE=1`.
- The reusable pipeline is round 6's `r6_heads`, `r6_bundle`, `r6_score`; its reader is round 1's A0 half-readers, so
  readers for new sets must be retrained.
- New features (CLIP token statistics) go under `/data/SSD2/pre_extract/artelingo/<name>/`; extraction on the local GPU
  under the lock (`flock -n -o -E 75 /tmp/gpu0.lock`). CPU work at 8 threads.
- The DAS6 reservation (node401, node402, node408) ends about 2026-10-14 02:40; this round needs no cluster.
- Codex's allow rule matches only a command that starts with `codex e --dangerously-bypass-approvals-and-sandbox`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing.
- On disk over 1 GB: `res/cluster_jobs/` 3.1 GB (round 6's pulled outputs; keep, ledgered).
