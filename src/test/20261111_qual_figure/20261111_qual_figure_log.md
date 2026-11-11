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
- 09:23 worker done. Code: `qf_cand.py` (rules), `qf_view.py` (page), `qf_figs.py` (CLI), `qf_viewer_template.html`,
  `qf_viewer_check.js` (node check, stub from r20), `test_qf.py` (12 passed; panel (b) checked against brute force).
  Smoke (800 rows) 8 s; full `candidates` 33 s through `res-queue cpu --short`; full `viewer` 1 s; page 3.95 MB.
  Commands, from `/project/CoSiR` with the env python: `qf_figs.py candidates [--smoke]`, `qf_figs.py viewer
  [--smoke]`, `node src/test/20261111_qual_figure/qf_viewer_check.js <outputs/viewer>`; serve the page with
  `python3 -m http.server` inside `outputs/viewer` (it loads its data with fetch).
- Counts (ArtELingo test): 9,150 public-domain paintings, 9,149 without the teaser painting.
  - Panel (a): 7,565 queries; 21 pass the filter (amusement 3, awe 8, contentment 7, fear 2, sadness 1); the cap of 2
    per class keeps 9 (a1 to a9), since only five emotion classes pass.
  - Panel (b): 18 pairs (b1 to b18), 3 per ordered pair; candidates per ordered pair after the (i, j) / (j, i) dedup:
    emotion/style 6,874, emotion/genre 2,762, style/emotion 13,030, style/genre 4,638, genre/emotion 14,225,
    genre/style 10,625.
  - Clusters: first pick by gain contentment (+0.141), Ukiyo-e (+0.772), still life (+0.546).
- Worker's interpretations: a rank tie in (b) does not count as nearer; the gain uses r20's stored shares (4
  decimals). Page marks: in (a) emotion and genre labels green when they match the query; in (b) the A label green
  when it matches the partner, the B label orange when it differs.

## Picks (2026-11-11, after the review page)

| Point | Decision | Who |
|---|---|---|
| Panel (a) | a5 (Metsu, "The Sick Girl", sadness): plain neighbours are sickbed scenes with mixed emotions, ours are sad works on other subjects (a Pietà, a farewell scene). Considered: a8 (ours stay stormy landscapes, 2 of 5 shared with plain) | user, after seeing the recommendation |
| Panel (b) | three pairs from three ordered aspect pairs: b5 (near in emotion, far in genre: Renoir study and Moser illustration, amusement), b7 (near in style, far in emotion: Borisov-Musatov boy and Renoir Cagnes, Impressionism), b15 (near in genre, far in emotion: Rembrandt laughing young man and Manet Victorine Meurent, portraits). b5 over b6: Titian's Charles V labelled amusement would look wrong | b5: user; b5 over b6, b7 and b15: user, after seeing the recommendation |
| Cluster figure | keep the rule's picks at the class count: contentment, Ukiyo-e, still life | user |

The agent looked at a4, a5, a6, a8 and at b1 to b3, b5 to b9 and b13 to b15 at full size before recommending.
