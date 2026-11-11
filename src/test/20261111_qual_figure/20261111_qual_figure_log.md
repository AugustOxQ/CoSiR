# Paper 4.6: qualitative figure candidates (panels a and b, cluster figure at the class count)

Outside the research loop (paper work). Started 2026-11-11 09:17 in the "paper sec4 tables 5" tab, from the
[sec4 tables 5 handoff](../../../docs/superpowers/handoffs/2026-11-11-paper-sec4-tables-5-handoff.md). Built as r20
(`src/test/20261109_r20_figures/`): candidates by a stated rule, a review page, the user picks, then the figure.
Outputs: `outputs/` -> `/data/SSD2/cosir_paper/20261111_qual_figure/`.

## Decisions (2026-11-11 09:16)

| Point | Decision | Who |
|---|---|---|
| Figure plan | one full-width figure, panel (a) same subject, different feeling, panel (b) close in one aspect, far in another; the cluster figure rebuilt at the class count (9, 27, 10) | user, after seeing the recommendation (PF README, 2026-11-11) |
| Shared | ArtELingo test set; c* seed 0 against the plain Qwen3 image only; public-domain paintings only; near copies skipped (plain cosine 0.95 or more); the teaser's query (Childe Hassam, "Old Lyme Bridge") left out | rules proposed by the agent, user yes |
| Panel (a) rule | top 5 in the plain image and in our emotion head; keep when at least 4 of ours share the query's emotion, at most 1 of plain's do, and at least 3 of plain's share its genre; rank by our emotion matches minus plain's, then fewer of our neighbours in the query's genre; at most 2 per emotion class, up to 18 | rule: agent; user yes |
| "Same subject" = the same genre label | kept | user, after seeing the recommendation |
| Panel (b) rule | six ordered aspect pairs; a pair is near in A (in the top 5 of our A head, same A label) and far in B (different B label, ranked by its rank percentile in our B head, farthest first); top 3 per ordered pair, no painting twice; plain cosine stored beside | rule: agent; user yes |
| Cluster figure rule | r20's rule at the class count: the class with the largest share gain of c* over plain, the 4 paintings nearest the centre (picks contentment 0.73/0.59, Ukiyo-e 0.94/0.17, still life 0.91/0.36); the page shows every class | user kept the rule, after seeing the recommendation |

The rules were fixed before any painting of panels (a) and (b) was seen. The user picks among the candidates, so a
caption can state the rule and that the example was picked among its candidates.

## Jobs

- 09:18 a Sonnet `high` worker writes `qf_cand.py`, `qf_view.py`, `qf_figs.py`, `test_qf.py` (imports r20's code),
  runs the smoke and the full steps (CPU, `res-queue cpu --short`), and builds the review page.
