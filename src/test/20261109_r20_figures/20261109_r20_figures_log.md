# r20: qualitative cluster figure, page-1 teaser figure, retrieval check of the eight ablation removals

Small loop (design in chat, run, short auto report and half-page user-read report). Started 2026-11-09 00:10 from the
[r20 decide handoff](../../../docs/superpowers/handoffs/2026-11-08-r20-decide-handoff.md).
Outputs: `outputs/` -> `/data/SSD2/cosir_r20/20261109_r20_figures/`.

## Decisions (2026-11-09 00:20)

| Point | Decision | Who |
|---|---|---|
| Cluster figure: set, aspects, cluster count, cluster and painting rules | build every candidate, the user picks after seeing them | user |
| Teaser: painting, neighbours per space, plain-image row | build every candidate, the user picks after seeing them | user |
| Retrieval check | image to image, label mAP, plain-cosine read-out for all nine models, held ArtELingo, seeds 0 to 4, no bar | user ("maybe q", unsure) |
| Image rights | only public-domain paintings: artist died 1955 or earlier (Wikidata); a failing painting is replaced by the next one under the same rule | user, after seeing the recommendation |
| Who builds | all figure work with Claude, not Codex | user |
| Candidate details | c* seed 0; k-means state 0 of r17's `ns_metrics.partitions`; 6 clusters for the purity and random rules (at least 20 members); 6 paintings per row; near copies (plain cosine 0.95 or more) skipped; teaser rule: queries with at least 3 of 5 label matches in each head, ranked by distinct neighbours | agent default |

Note: the user picks the rule after seeing the candidates, so the caption can state the rule but not that it was
fixed before the paintings were seen.

## Jobs

- Retrieval check: Codex (gpt-6.1-sol, medium) writes `r20_ret.py` and a smoke run; the controller runs the full job
  through `res-queue cpu`.
- Candidates and review page: a Sonnet `high` worker writes `r20_figs.py` (steps `pd`, `partitions`, `candidates`,
  `viewer`) and smoke runs; the controller runs the full steps through `res-queue cpu`.

## Progress
- 00:45 Codex job done: `r20_ret.py`, `test_r20_ret.py` (4 passed), smoke in 5 s. Controller read the code: project AP
  (`rt_metrics.retrieval`, qe_k 0) and bootstrap (`ns_ret.paired_ret`, shared seeds, 10,000 draws), rows joined by name.
- 00:47 to 00:55 full run through `res-queue cpu` (7.75 min): `ret_full/ablation_ret.{json,md}`, log `ret_full.log`.
  c*'s mAP matches r11's recorded plain-cosine held mAP (stage 1 verdict) exactly in all three aspects.
- 00:45 Figures worker (Sonnet): wrote `r20_data.py`, `r20_pd.py`, `r20_part.py`, `r20_cand.py`, `r20_view.py`,
  `r20_figs.py` (CLI), `r20_viewer_template.html`, `r20_viewer_check.js` (node check with a DOM stub),
  `test_r20_figs.py` (13 passed). `pd` full done (`outputs/pd/`); other steps smoke only (`outputs/smoke/`). The
  full `partitions`, `candidates` and `viewer` steps are for the controller (commands in the hand-back).
- 00:40 worker done: `r20_figs.py` and its step modules, `test_r20_figs.py` (13 passed), `pd` full (Wikidata P6002 and
  P570: ArtELingo held 1,003 of 1,013 artists matched, 9,150 of 12,281 paintings public domain; MMArt test 1,026 of
  1,031, 11,989 of 16,384). Worker's addition: a second match pass reading dots as dashes (3 ArtELingo artists).
- 00:46 first full-steps launch failed at once (controller's quoting: zsh did not split the env prefix); rerun fine.
- 00:50 full `partitions` (18 fits, 0.7 min), `candidates` (0.35 min; 12 teaser candidates per set), `viewer` (17 MB,
  2,736 thumbnails) through `res-queue cpu --short`. Page at `outputs/viewer/index.html`, served on localhost:8766,
  checked by headless screenshots (Playwright of the DustDevilVis env, run isolated, screenshots only): both tabs
  render, no page errors.

## Design picks (2026-11-09 01:05)

| Point | Decision | Who |
|---|---|---|
| Review page | opened, read | user |
| Teaser | candidate #2 of ArtELingo held (Childe Hassam, "Old Lyme Bridge"); 5 neighbours; plain-image row kept | user picked #2; count and row agent default |
| Cluster figure | ArtELingo held, 20 clusters, per-class rule, one class per aspect by the largest share gain of c* over the plain image (sadness 0.45/0.25, Ukiyo-e 0.93/0.14, sketch and study 0.51/0.28), 4 paintings nearest the centre; all classes to the supplementary | agent, on the user's request ("you give your selected ones") |
| Dot-as-dash artist match | keep | user |

The class rule favours c*; the caption says so. An Opus `high` worker builds `fig_clusters` and `fig_teaser` in the
paper folder (`docs/paper/cvpr2027_aspect_spaces/figures/`) with a look-and-fix loop.
- 01:20 figures built (Opus `high` worker, three renders each): `fig_clusters` 6.875 x 1.2 in (1.5 MB PDF),
  `fig_teaser` 6.875 x 2.3 in (2.3 MB PDF), scripts in the paper folder's `figures/src/`, two draft captions in
  `captions.md`. Controller viewed both PNGs. Paper repo commit eb4bb74.
- 04:20 user: keep both figures as they are (user); the all-classes supplementary figure later (user); cut to a fresh
  chat for the reports and the wrap-up (user, after seeing the recommendation). Open: compare also with the teacher?
  Agent's recommendation: main figures stay plain image only; a teacher version goes to the supplementary.

## Report and reading (2026-11-09 04:25 to 2026-11-10 19:50)

- 2026-11-09 04:40 short auto report [2027-01-13](../../../docs/reports/auto/percept/2027-01-13_ablation_retrieval.md) and the
  user-read report, after a Codex cold read (gpt-6-luna) and a Codex fidelity check (gpt-6.1-sol) and one fix wave.
- Decision C (user, 2026-11-10, before ours): the content correction and the instructed graphs matter much for style and
  possibly emotion, not for genre; the buddy terms also help style and emotion; the relational term carries genre; all
  four parts contribute, with no redundancy.
- Teacher version of the figures (user, after seeing the recommendation): the main figures stay as they are (plain
  image only); a teacher version goes to the supplementary with the all-classes figure.

