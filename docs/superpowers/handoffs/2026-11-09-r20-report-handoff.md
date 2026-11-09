# Handoff: r20: write the short report and the half-page user-read report, ask for the user's reading of the ablation retrieval table, then wrap up

## Brief
- Done: the ablation retrieval check (9 models, held, 5 seeds) and both paper figures (cluster figure, page-1 teaser), kept by the user.
- State: the retrieval removals follow the clustering ablation's pattern; c*'s values match r11's exactly.
- Next: write the short auto report and the user-read report, ask for the user's reading (decision C), then the wrap-up.
- Blocked: nothing; one open question (a teacher version of the figures).

> Written 2026-11-09 04:25 by tab "r20 decide". Loop step reached: r20 small loop, after the run (step 8), before the
> report (step 9) and decision C. Direction log entry: none yet for r20 (the wrap-up writes it).

## 1. Read in this order

1. The round log, [`src/test/20261109_r20_figures/20261109_r20_figures_log.md`](../../../src/test/20261109_r20_figures/20261109_r20_figures_log.md):
   every decision with who made it, the jobs, the checks.
2. The retrieval results, [`ret_full/ablation_ret.md`](../../../src/test/20261109_r20_figures/ret_full/ablation_ret.md)
   (JSON beside it, with per-seed values and input hashes).
3. The two figures and their draft captions: `docs/paper/cvpr2027_aspect_spaces/figures/fig_clusters.png`,
   `fig_teaser.png`, `captions.md` (last two sections, with TRACE blocks).
4. For the report's context: the "Ablation" section of
   [the recipe sweep report](../../reports/auto/percept/2026-12-28_recipe_sweep.md) (held AMI of the same eight
   removals), and the templates `~/.claude/templates/auto_report_template.md` and `results_brief.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- The retrieval check: image to image, label mAP, plain cosine for all nine models, held ArtELingo, seeds 0 to 4, no
  bar (2026-11-09, round log).
- Only public-domain paintings in the figures (artist died 1955 or earlier, Wikidata); the dot-as-dash name match
  stays (2026-11-09, round log).
- The teaser is candidate #2 of ArtELingo held (user's pick); the cluster figure is the agent's pick on the user's
  request (ArtELingo held, 20 clusters, one class per aspect by the largest share gain of c* over the plain image,
  4 paintings nearest the centre). Both kept as they are (user, 2026-11-09 04:20).
- The all-classes supplementary figure comes later, with the supplementary work before 11-23 (user).

## 3. Where things stand

- **Retrieval check** (label mAP, removal minus c*, 95% interval; 12,281 queries, genre 9,949):
  - largest changes: instructed graphs style -0.1223 [-0.1246, -0.1200], emotion -0.0208; buddy terms style -0.0981,
    emotion -0.0181; content correction style -0.0577, emotion -0.0068; relational term genre -0.1287
    [-0.1328, -0.1241] (genre 0.4610 against the plain image's 0.4760).
  - caption endpoints raise style +0.0044 [+0.0034, +0.0055]; sharper teacher row lowers genre -0.0077.
  - reference rows: c* 0.2642 / 0.3110 / 0.5896; plain Qwen image 0.2424 / 0.1801 / 0.4760; teacher 0.2665 / 0.3091 /
    0.5860 (emotion / style / genre).
  - c*'s three values equal r11's recorded plain-cosine held mAP (stage 1 verdict) exactly.
- **Figures:** `fig_clusters` 6.875 x 1.2 in, `fig_teaser` 6.875 x 2.3 in, in the paper repo (commit eb4bb74). The
  plain image's Ukiyo-e cluster (14% Ukiyo-e) is mostly Art Nouveau; that cluster and the plain sketch-and-study
  cluster also serve another class (TRACE block, not the caption).
- **Review page** (all 72 cluster candidates and 12 teaser candidates per set):
  `src/test/20261109_r20_figures/outputs/viewer/index.html`; serve with
  `python3 -m http.server 8766 --bind 127.0.0.1 --directory /data/SSD2/cosir_r20/20261109_r20_figures/viewer`.

## 4. Open points for the next step

- **Key, decision C:** the user's reading of the retrieval table, given before ours (show the results brief with our
  reading folded away).
- **Minor:** a teacher version of the figures. The user asked whether the figures compare with the teacher too; they
  compare with the plain image only. Our recommendation: main figures stay as they are, since the plain image has the
  same input as c*; a teacher version goes to the supplementary with the all-classes figure (CPU minutes; the
  teacher's held outputs are saved). Ask for the user's decision and record it.
- **Minor:** the cluster caption's first sentence is clumsy; the paper chat may reword it.

## 5. Code and pitfalls

- The report is short (small loop): results brief, what ran, results table, analysis, deviations, files and storage;
  the user-read report half a page (`~/.claude/references/user-read-report.md`), in `docs/user_read/`.
- Report name and date: CoSiR keeps its sequence-number dates in `docs/reports/auto/percept/` (the last is
  2027-01-12); add the index row in `docs/reports/reports_sum.md` and run `python scripts/check_reports_sum.py`.
- Deviations to state: the user chose the figure designs after seeing all candidates (the caption states the rule,
  not that it was fixed in advance); the class rule favours c*; the controller's first full-steps launch failed on a
  quoting error and was rerun unchanged; screenshots used the Playwright of another project's env, isolated, for
  screenshots only.
- Storage at the wrap-up: `/data/SSD2/cosir_r20/20261109_r20_figures/` is 32 MB (viewer 17 MB, smoke 12 MB); the
  untracked `src/test/20261109_r20_figures/ret_smoke/` is 120 KB. Nothing over 1 GB.

## 6. State at handoff

- Running: the review page's server on localhost:8766 (a background job of the old tab; it ends when that tab
  closes). Nothing else.
- Uncommitted: nothing after this cut's commit (the untracked `ret_smoke/` is a smoke output, left for the storage
  summary).
- On disk over 1 GB: nothing.
