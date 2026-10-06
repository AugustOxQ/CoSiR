# Fixing the aspect reader with a CSD style grouping in the set: a pre-results plan for review (CoSiR v2, plan (a))

Date: 2026-10-06 (Amsterdam). Status: a plan for review. Nothing in it has been run: no reader code, no bank, no
episodes, no decision-rule file. The plan itself is Appendix A, §5 (the handoff of 2026-10-06, verbatim). Every number
quoted from earlier work comes from Appendices B to E, whose controller reviews re-derived them from stored per-anchor
arrays.

## 0. The decision requested

CoSiR v2 scores an image and a caption under an unnamed aspect shown by 4 support pairs and 4 contrast pairs. A
label-free pipeline places items into pseudo-aspect *groupings* through cross-modal logistic *heads*, and a *reader*
picks which grouping the support pairs share. On 5 October a fourth grouping built from CSD style embeddings raised the
told margin (the scorer given the right grouping) to +2.23 [1.93, 2.56] R@1 over its matched condition-free
counterpart, while the label-free reader fell to +0.06 [−0.15, 0.28] (Appendix B §9). The user chose plan (a): replace
the reader's "largest raw Δ" rule with readers that can tell the groupings apart, pick one on the seed-42 development
episodes against a pre-written rule, and test it once on fresh episode seeds 49, 50 and 51 before a go/no-go on
Friday 9 October 2026 (CVPR abstract 10 November).

The question for this review is **whether the plan in Appendix A §5, including its draft decision rule (§5.4), can
discriminate what it claims to and supports the decision it feeds**, specifically:

1. whether each reader candidate (R-a scaled Δ, R-b learned reader, R-c confidence gate) has a matched counterpart and
   B′ that remove only the condition;
2. whether R-b's training can leak evaluation information or in-sample head posteriors into the reader;
3. whether the selection among several candidates on a reused development seed, carried by the maximum, keeps the
   fresh-seed test meaningful;
4. whether R-a's spread should be estimated on the development episodes or elsewhere;
5. whether the kill thresholds, the development bar and the GO rule are coherent with each other and with the label
   policy (the told mapping is a diagnostic only);
6. whether the timeline (Tue 6 to Fri 9 October) is realistic.

The author's order of work: this review first. If it is smooth, the plan runs as written; if it raises issues, the plan
and its §5.4 rule are fixed before any code, and the fixed rule is committed as
`src/test/20261117_reader_fix_csd/DECISION_RULE.md` before any number it governs exists.

## 1. Task, metrics and bars in brief (details: Appendix E)

- **Episode.** A query (one image or one caption of a selection painting), 4 support pairs that each agree within the
  pair on a value of aspect A (four different values, never the query's), 4 contrast pairs that do the same for aspect B,
  and 13 candidates in the other modality: p_A shares the query's value of A, p_B its value of B, 11 negatives share
  neither. Swapping supports and contrasts (condition b) makes p_B the target. Aspects: emotion (8 values, labelled per
  viewer row), style (23) and genre (10) (per painting). Pairs: emotion × style, emotion × genre, style × genre; 4,096
  episodes per pair per episode seed. Each episode yields four rankings (2 conditions × 2 directions).
- **Metrics.** R@1 (target strictly first; ties miss), other-aspect rate, condition gain = R@1 − other-aspect rate,
  either rate = R@1 + other-aspect rate, so R@1 = (either + gain) / 2. A condition-free scorer has gain exactly 0.
- **Comparators.** Cosine (12.96 on seed 42), RCA (13.38, the GO bar), B (the best condition-free score, 18.34), B′ (B
  rebuilt with a configuration's own averaged-heads term) and the configuration's matched counterpart.
- **Margin** = fused reader minus its matched counterpart. **Bar margin** = fused reader minus whichever of B′ and the
  counterpart has the larger R@1. **Development bar** (inherited): bar margin ≥ +0.5 with a 95% lower bound above 0,
  set because every fresh-seed test so far roughly halved the development margin.

## 2. Fixed by the user and not to be reopened (Appendix A §3)

- Plan (a) itself; design L and a change of course are not pursued now.
- The groupings: affect = Leiden on GoEmotions probabilities (41 groups); image and caption = k-means 64 on CLIP
  features; style = CSD Leiden (17 groups). No grouping is re-chosen.
- No grouping choice reads evaluation labels; reader variants are method choices made on development episodes; the told
  mapping is a diagnostic only.
- Matched controls for every configuration; B extended to B′ by every new condition-free ingredient.
- Seed handling (user preference, recorded 2026-10-04): develop and pick on episode seed 42; test the one carried
  configuration on fresh seeds 49, 50 and 51, each reported and pooled. The user considers several fresh test seeds
  sufficient protection and does not want single-look or lucky-seed ceremony; pre-registering the rule before the test
  still stands (Appendix F).

## 3. Implementation facts the plan relies on

Read from the code on 2026-10-06; stated here because the plan refers to these functions by name.

- **Development episodes (seed 42):** 12,288 episodes on 4,602 anchor paintings (anchors drawn with replacement). The
  two cross-fit halves are the episodes' index parity. Fresh episode seeds are new draws on the same 6,451 selection
  paintings.
- **Reader statistic** (`aspect_deltas`): for grouping h, Δ_h = mean over the 4 support pairs of p_h(image) · p_h(caption)
  minus the same mean over the 4 contrast pairs, with p_h the head posteriors. Under condition b, Δ_h is exactly −Δ_h
  under condition a. The current hard reader takes the arg-max of Δ over the configuration's groupings (ties to the
  first) and scores candidates by p_h(query) · p_h(candidate) on the picked grouping.
- **Fused reader** (`crossfit_nested(B, B, T, parity)`): every term is z-scored per ranking row; the score is
  (1 + λ_u)·z(B) + λ_a·z(T) over a 7 × 8 grid (λ_u, λ_a ∈ {0, …, 16}, 56 cells). On each parity half, the cell that
  maximises min(R@1 − R@1 of B, condition gain) is chosen and applied to the other half.
- **Matched counterpart:** T_cf = (T under condition a + T under condition b) / 2, fused by
  `crossfit_condition_free(B, B, T_cf, parity)` over the same 56 cells with the maximum-R@1 rule. That function raises an
  error unless each of its terms is identical under both conditions.
- **B and B′:** B = `crossfit_condition_free(cos, T_N1u, T_6u, parity)`, with T_N1u the centered factor term of A3 and
  T_6u the head agreement averaged over the three E2 groupings (R@1 18.34). B′ is the same call with T_6u averaged over
  the configuration's own groupings (A0 18.44, A1 18.80), so B′ is B rebuilt, not B plus a term. B's own cross-fit
  picks were tuned on the same parity halves that later fusions reuse.
- **Bar margin:** the comparator (B′ or the counterpart) is the one with the larger mean R@1 over all episodes; the
  margin is the paired per-anchor difference; the same comparator is used per aspect pair.
- **Pick accuracy** (diagnostic): per episode, the mean over the two conditions of 1[picked grouping = told grouping];
  A1's told mapping is emotion → affect, style → CSD, genre → image. With four groupings chance is 25%, with three 33%.
- **Heads** (`fit_one_head`): one image head and one caption head per grouping, `LogisticRegression(C=1, max_iter=300)`
  on unit-normalised CLIP ViT-B/32 features of a 60,000-row draw from the 183,694 scorer-train rows (the draw touches
  31,287 of 36,518 scorer-train paintings); posteriors are computed on selection rows. Held-out accuracies, image /
  caption head: affect 9.81 / 35.72, image 92.7 / 22.9, caption 21.6 / 89.7, CSD 85.10 / 40.41.
- **Bank** (`build_episode_bank(partitions, groups, rows, n_per_pair, seed)`): one block per unordered pair of groupings
  (six blocks for four groupings), each built by the same `build_aspect_episodes` that builds evaluation episodes, with
  groupings in place of aspects and group ids in place of values (cross-item pairs, value-disjoint, 13 candidates, at
  least 30 paintings per value). When exactly three groupings are given, the third is controlled on the candidates; with
  four groupings no third grouping is controlled. Bank supports share a group of the conditioned grouping exactly. The
  affect, image and caption groupings assign groups per row; the CSD grouping assigns one group per painting.
- **Intervals:** 5,000 painting resamples of the per-anchor arrays; cross-fit picks are fixed before resampling.
- **Δ scale:** for the E2 groupings, the per-episode spread of Δ was about 0.017 (Appendix D §3); coarse groupings (17
  groups) give larger agreement values than 64-group ones (Appendix B §9).

## 4. What the review should look at (Appendix A §5.6, verbatim)

> Matched counterparts and B′ for each candidate (gating included); leakage in R-b (bank rows, cross-fitted heads,
> features only from out-of-sample posteriors); multiplicity (several candidates on a reused seed 42, carried by the
> maximum; every fresh-seed test so far roughly halved the development margin); whether R-a's spread estimated on the
> development episodes is acceptable or should come from bank episodes; the kill thresholds; the label policy (told
> mapping only as a diagnostic); whether the timeline is realistic.

The object of review is Appendix A §5 (5.1 to 5.6). Appendix A §1 to §4 and §6 to §7 are context. Appendices B to F are
evidence and are not themselves under review.

## Appendices (verbatim)

- **Appendix A.** Handoff of 2026-10-06, `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md` (the
  plan).
- **Appendix B.** Draft report of 2026-10-05, `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`
  (groupings, Leiden, step 0, step 1).
- **Appendix C.** Step-1 log, `src/test/20261116_grouping_step1_style/20261116_grouping_step1_style_log.md`, sections
  Results to Controller review.
- **Appendix D.** The earlier reader handoff of 2026-10-04, `docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`
  (the templates this plan adapts).
- **Appendix E.** Stage report of 2026-10-04, `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`,
  §2 (task, data, metrics, protocol) and §14 (oracle layers and diagnosis).
- **Appendix F.** Two project lessons recorded by the user: the matched-control lesson and the seed-handling preference.
