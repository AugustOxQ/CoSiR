# Whole-branch final review: lightweight CLIP fine-tuning comparator

> 2026-10-07 10:09 to 10:30 (Amsterdam). Reviewer: Opus 5.5 (most capable model), brief
> `.superpowers/sdd/2026-10-07-clip-lightweight-ft/final-review-brief.md`. CPU only, at most 2 processes, no cluster,
> no held row indexed, no repo file changed outside this folder, no commits, no subagents.
> Scripts and outputs: this folder (`fr_*.py`; `fr_*.json` and `fr_*.log` are gitignored by the folder's `.gitignore`).

## Verdict: CONFIRMED WITH FIXES

The claim holds. Three label-free lightweight fine-tunes of CLIP ViT-B/32, each selected on val image–caption
retrieval only, raise plain CLIP's pooled cosine R@1 on seeds 49 to 51 by +1.93 to +2.10 (13.04 to 14.97 to 15.14),
stay 3.15 to 3.32 below B′(A0) (18.29) and 3.74 to 3.91 below AFF (18.88), with every interval excluding 0 on every
seed and every pair. Every number in the draft's tables matches my own derivation, and both notes from the report
writer are correct. No finding is blocking.

There are four should-fix items, all in the report's text. The main one is a wrong conclusion sentence in the
Summary: the fine-tune did raise the condition-free floor on emotion × genre, where AFF gains most. The other three are
a missing caveat about B's example-centred term, a cited source that git ignores, and an inaccurate sentence about
which tests guard held rows. There are 13 nits.

## What I re-derived (own code)

| # | Check | Result |
|---|---|---|
| 1 | **Data integrity** (`fr_data.py`). Splits are painting- and leakage-group-disjoint, held included (all 6 overlaps 0). The cache SHA-256 matches the record. The cache's 49,121 paintings are exactly the paintings of scorer-train ∪ val ∪ selection, with 0 held paintings. The index is in sorted order. For all 246,979 non-held rows, the annotation `image` and `painting` equal the cache index and `data.paintings`. Rebuilding 48 random paintings from the WikiArt JPEGs with the CLIP processor gives max uint8 difference **0** (neighbour-row control: 255). | pass |
| 1b | **Training pairs through the trainer's own path** (`make_rowset("LB", …)` on all 183,694 scorer-train rows, then `train_dataset`, 64 random items). The image equals the painting's cache row (0 mismatches) and a fresh WikiArt decode (16 items, 0 mismatches). The tokens equal those of `annotations[sample_ids[row]]["caption"]` (0 mismatches). The sampler over epochs 1 to 10 on the real index never puts a painting twice in a batch, and there are 0 scorer-train leakage groups with several painting ids. | pass |
| 1c | **Caption join, checked independently.** CLIP text features of `annotations[sample_ids[i]]["caption"]` for 72 random non-held rows equal the cached `txt_features[i]`: min cosine 0.9999998, max abs 7.6e-6. Shifted-row control: max cosine 0.874. Image features from the stock processor against the cached ones: min cosine 0.99961 over 24 rows. | pass |
| 2 | **Selection** (`fr_selection.py`), spec §4 rule applied to the integer counts of the nine `metrics.json`. Picks: LP 3e-4 epoch 9 (11.683), LB 3e-5 epoch 8 (13.336), LoRA 1e-4 epoch 10 (15.152). No ties; margins to the runner-up are 0.036, 0.021 and 0.029 val points. Val retrieval recomputed with my own code from **every** run's `features.npz` reproduces its best epoch's i2t and t2i counts exactly (0 difference, all 9 runs). Every `features_epoch0.npz` reproduces epoch 0 exactly. All `rows` equal sorted val ∪ selection. All 9 run records are at commit 65bb2f4 with a clean code folder. | identical |
| 3 | **Episode numbers** (`fr_episodes.py`). Own placement, float32 cosine, strict first place, per-anchor metrics and painting-cluster bootstrap; only `load_artelingo`/`artelingo_splits` are shared. I compared **3,595 numeric leaves** of `eval.json`: every scope (seeds 42, 49, 50, 51; pooled; pooled per pair; seed-42 per pair), every scorer and every comparison, for r1, either, gain and other. That covers points, both CI bounds and cluster counts. Max \|diff\| **0.0**. Scoring in float64 flips no anchor. The episode SHA-256s, the stored `cl`/`pair_index`, and the stored `cosine__*` next to AFF, B and B′ all match. The AFF-minus intervals Table S1 quotes from round 3 recompute to +5.84 [+5.61, +6.07], +0.81 [+0.69, +0.93] and +0.59 [+0.46, +0.73]. | identical |
| 4 | **End to end on CPU** (`fr_e2e.py`). Each selected `best_params.pt` was loaded into a fresh model with my own assembly (LB by name; LoRA by my own `peft.inject_adapter_in_model` with the spec's config; LP as two matrices on the frozen features). I encoded 12 selection rows from the cache and their captions. Against `features.npz`, max relative error was 1.7e-6 and min cosine 0.9999995 for all three variants and both modalities (tolerance 1e-4, which allows for GPU fp32 vs CPU fp32). The untrained cache path matches `features_epoch0.npz` to 2.2e-6. LB's saved keys equal the spec's set (10,898,177 parameters), and LoRA's equal the 192 adapters plus `logit_scale` (1,966,081). The trained features differ from untrained ones (min cosine 0.80 for LB, 0.84 for LoRA). | match |
| 5 | **Report tables** (`fr_report_tables.py`). I rebuilt Tables S1 (fine-tuned rows), 3, 4, 5, 6, 8, 9 and 10 from my derivation at 2 decimals: **58 of 58 rows match** (2 differ only in whitespace). I also checked the text numbers individually: 37 to 40% of 5.25; 2.94 to 3.10; 3.60 to 3.78 (B′(A1), seed 42); 37.76 and 7.5 to 7.8; +0.43 to +0.59; −0.01 [−0.02, +0.00]; per seed −0.02 to 0.00; grid-edge arithmetic; the Table 2 counts; epoch times; and "i→t gained more than t→i" (true for every epoch of every run). | match |
| 6 | **Figure script.** `build_figures.py` was re-run from a scratch copy (only ROOT patched): "OK: 896 points, 50 intervals". Its `figure_data.json` is byte-identical to the asset. I also inspected the figures. | pass |

**The report writer's two notes are both confirmed.**

- **The lift goes to the genre candidate.** Pooled, the genre candidate's first-place rate rose +4.22 to +5.34, and
  the emotion candidate's rose +0.02 to +0.98 (on emotion × genre only +0.02 to +0.28). This holds on every seed. Per
  seed and per pair, the genre side rose +3.6 to +6.1, while on emotion × genre the emotion side moved −0.65 to +0.76.
- **The cache-path minimum image cosine.** Over all rows it is 0.98236 on val (30,872 rows) and 0.99561 on selection
  (32,413). The means are 0.999987 and 0.999991, and 29 rows (6 paintings) and 21 rows (4 paintings) fall below 0.999.
  The six LB/LoRA epoch-0 files hold identical arrays, and so do the three LP files (which equal the frozen cache). The
  low-cosine rows are not rows whose frozen features differ within a painting: within a painting, rows agree to a
  cosine of 1 − 1e-11.

**Descriptive robustness check** (`fr_allruns.py`; decides nothing, and selection was fixed beforehand). The
best-epoch features of all nine runs score 14.97 to 15.14 pooled R@1, which is −3.15 to −3.32 against B′(A0). Within the
grid, episode R@1 did not follow val retrieval: LB 3e-6 had val 11.79 and R@1 15.05, against val 13.34 and R@1 14.97
for LB 3e-5. So the conclusion does not depend on which grid point selection picked.

## Findings

### Blocking

None.

### Should-fix

**S1. The Summary misstates the pair-level result ("it did the opposite").**
- Location: report lines 57 and 58 (Summary bullet), line 431 (§5.4) and lines 629 and 630 (§9).
- Evidence: the memo expected the fine-tune to raise the floor most on emotion, "exactly where all of AFF's gain
  sits", meaning the two emotion pairs. On emotion × genre the fine-tuned floor rose +2.31 to +2.68 [lower bounds
  2.00 to 2.36]. That is above the pooled lift (+1.93 to +2.10) and close to style × genre (+2.57 to +2.82); only
  emotion × style rose little. The emotion **candidate** barely moved, but a condition-free R@1 is the mean of both
  sides, so the floor on emotion × genre still rose, through the genre candidate. AFF's lead there stayed +5.53 to
  +5.90. "It did the opposite" and "it raised it least there" are therefore wrong at the pair level.
- Replacement for lines 57 and 58: "…and barely moved emotion × style (+0.78 to +0.90). The memo had expected the
  floor to rise most on emotion. The emotion candidate barely moved, but emotion × genre still rose by +2.31 to +2.68,
  through its genre candidate, and AFF stayed +5.53 to +5.90 ahead there."
- Replacement for line 431 ("On these episodes it raised it least there."): "On these episodes it raised the emotion
  candidate least. At the pair level, the floor rose least on emotion × style (+0.78 to +0.90), but on emotion × genre
  it rose +2.31 to +2.68, about as much as on style × genre, almost entirely through the genre candidate."
- Replacement for lines 629 and 630: "Its gain came through the genre candidate, not the emotion candidate. On
  emotion × genre that still raised the floor by +2.3 to +2.7, yet AFF stayed +5.5 to +5.9 ahead there. In our reading,
  more than half of AFF's margin over it comes from either rate carried by B's condition-free terms (§5.3, §5.5)."

**S2. A caveat is missing: B and B′ use the episode's example items, and a cosine cannot.**
- Location: §5.5 reason 3 (lines 453 to 460), §5.3 (lines 366 and 367), and §8.
- Evidence: B's factor term T_N1u centres the query on the episode's own 8 example items. The 2026-11-08 report (§5)
  says so, and its final reviewer's probe found 17.95 with centring on the episode's own examples, against 16.85 on 8
  random items and 17.14 on the global selection mean. That is "about 0.8 of the 1.4-point lift needs the episode's own
  examples, so part of it may come from the value-disjoint episode construction". So B and B′ are condition-free but
  use the examples, while any cosine, fine-tuned or not, sees only the query and the candidates. The report explains
  B's lead only as "trained on ArtELingo and built for cross-item sharing", and it attributes AFF's margin to "B's
  condition-free terms" as a fact.
- Add to reason 3, after "…T_6u 16.96.": "The factor term also centres the query on the episode's own 8 example items,
  which a cosine never sees. The 2026-11-08 final review found 17.95 with that centring, against 16.85 on 8 random items
  and 17.14 on the global selection-row mean, so part of B's lead over any cosine comes from using the episode's
  examples (possibly through the value-disjoint construction), not only from better features."
- Line 366 ("AFF's either rate comes from B, its fused base, so"): "In our reading, AFF's either rate comes from B,
  its fused base (including B's example-centred factor term), so"
- New §8 bullet: "**Example-aware comparators.** B and B′ ignore the condition but use the episode's example items
  (the factor term is centred on them). A cosine, fine-tuned or not, uses only the query and the candidates, so the gap
  to B and B′ mixes better features with example-aware scoring."

**S3. A cited source would be missing from git.**
- Location: `docs/reports/assets/2026-11-24_clip_lightweight_ft/figure_data.json`.
- Evidence: the root `.gitignore:154` (`*.json`) ignores it. The report cites it as the source of Table 10, Table 7,
  Table 11, the grid-edge ratios and the gap shares. The earlier reports force-added theirs (8 `*.json` are tracked
  under `docs/reports/assets/`, e.g. `2026-11-12_partition_quality_leiden_communities/figure_data.json`).
- Fix (at commit): `git add -f docs/reports/assets/2026-11-24_clip_lightweight_ft/figure_data.json`.

**S4. The held-row sentence attributes the guards inaccurately.**
- Location: §3.1, line 158.
- Evidence (mutation table below):
  - The cache builder's painting-level guard is tested by Task 1's test (H3 caught).
  - The trainer's `check_rows` is tested by a Task 2 unit test (H1 caught). Inside `run()` it can never fire, because
    the rows come from `split_index` itself (H4 survived).
  - `split_index`'s own held-overlap check is untested (H2 survived; Task 1 deferred M2).
  - The real data is clean (check 1).
- Replacement: "Held rows stay untouched. Every row comes from the three non-held splits of `artelingo_splits`. The
  cache builder refuses a painting that also has held rows (Task 1's test), and the trainer's row check refuses a held
  row (Task 2's test). The final review confirmed on the real data that the cache's 49,121 paintings are exactly those of
  scorer-train, val and selection, with no held painting or leakage group among them."

### Nits

1. **Title** (line 1). It quotes the best variant (LoRA: 3.15 and 3.74) while the brief's claim says "about 3.8 below
   AFF". Replacement: "…: about +2 R@1 over plain CLIP, still 3.2 to 3.3 below B′(A0) and 3.7 to 3.9 below AFF".
2. **"never loading a held row"** (line 31) and **"No … label is read"** (line 156). The shared loaders read the
   whole feature and annotation files, held rows and label fields included, and index only non-held rows. Replace
   "never loading a held row" with "never using a held row (no held image loaded, no held feature extracted)", and
   "is read by" with "is used by".
3. **LP in fp32 is a spec departure.** Spec §3 says "mixed precision", and LP trained in fp32 under Task 2's review
   M3. Line 177 should say "LP in fp32 (the spec said mixed precision; Task 2's review moved the 512 × 512 map to fp32,
   at no cost)".
4. **The temperature never left its cap in LB and LoRA.** CLIP's `logit_scale` starts at ln 100, the cap. LB and LoRA
   ended at 4.599 to 4.605, and LP at 4.52 to 4.59. Add one clause after "(capped at ln 100)": "CLIP's initial value
   equals the cap, so in LB and LoRA the temperature stayed within 0.01 of it".
5. **Grid edges** (§4 lines 246 to 257, Summary line 62, §9 line 634). Add the descriptive check: "Descriptively (final
   review, after selection), the best epochs of all nine runs scored 14.97 to 15.14 pooled R@1 (−3.15 to −3.32 against
   B′(A0)); within the grid, a higher val score did not mean a higher R@1 (LB 3e-6: val 11.79, R@1 15.05; LB 3e-5: 13.34,
   14.97)." In §9, change "edges worth at most about 0.2" to "edges worth about 0.2 by linear extrapolation, and a nine-run
   spread of 0.17".
6. **§5.5 reason 1** (line 446), "the variant order on val … did not carry over". Replacement: "LoRA's 3.5-point val
   lead over LP bought 0.14 R@1, and LB's lead over LP on val reversed on the episodes".
7. **§6 cache path** (lines 491 and 501).
   - "All six LB and LoRA epoch-0 files give identical episode results" → "The six LB and LoRA epoch-0 files hold
     identical arrays, and so do the three LP files".
   - Add: "The uint8 cache equals a fresh decode exactly (48 paintings, final review). A fresh stock-processor feature
     differs from the frozen cache by a minimum cosine of 0.9996 (24 rows). So the shift lies between the original
     extraction's preprocessing and today's processor, not in the cache."
8. **Naming of B** (lines 45 and 111). "B (best condition-free score of the project)" sits next to a stronger B′(A0).
   Add "(named in round 3; B′(A0) is now stronger)".
9. **Ladder figure.** The "chance 7.69" label is drawn over the AFF bar. Place it at the left
   (`ax.text(-0.45, 100/13 + 0.2, …, ha="left")`).
10. **`eval.json` provenance.** `eval.json` and the run log do not name the input `features.npz` files; the recompute
    had to infer them. Add the `ft_eval.py` command line (LP, LB and LoRA `features.npz` and the LB 3e-5
    `features_epoch0.npz` as CLIPcache) to the run log's 09:35 row.
11. **Storage** (§7.6). Of the nine `features_epoch0.npz`, eight hold arrays identical to another copy: about 2.1 GB of the 4.6 GB
    on the system disk. Keeping only the selected LB run's copy (the one eval and the figures read) would free that
    space. But `rederive/rd_ft.py` reads all nine, so this is the user's call. State it in §7.6.
12. **§7.7 placeholder.** Proposed text: "A whole-branch final review (Opus 5.5) re-derived with its own code the data
    integrity on the real cache, the selection, val retrieval from every `features.npz` (exact counts), all 3,595
    numeric leaves of `eval.json` (points, intervals, cluster counts; max difference 0.0), an end-to-end CPU
    re-encoding of each selected model (max relative error 1.7e-6) and every table of this report (58 rows). 23 of 26
    guard mutations were caught. Verdict CONFIRMED WITH FIXES: 4 should-fix and 13 nits, all in the report text and
    commit step, applied in one fix wave and re-reviewed (`final_review/final_review.md`)." Add the same outcome to the
    `reports_sum.md` row and the run log.
13. **Process.** The first diff package was empty (a controller shell slip) and was regenerated during the review.
    The CVPR memo §2 update (plan Task 6) is still open.

## Deferred-minor triage

| Minor (origin) | Triage |
|---|---|
| `split_index` held check untested (Task 1 M2) | **Keep open; fix before the held test reuses this code.** Mutation H2 survives. The real data is clean (check 1). One test, `split_index` with `held` overlapping `scorer_train` → `pytest.raises(ValueError, match="non-held")`. |
| Docstring "never read" overstates (Task 1 M1) | Same wording issue as nit 2; fold into that fix. |
| Toy crop test cannot see a wrong crop (Task 1 M4) | Closed for this experiment: the real cache equals a fresh CLIP-processor decode on 48 paintings (max difference 0). |
| Processor backend not in `cache_record.json` (Task 1 M5) | Low. The backend matters (PIL vs torchvision is one uint8 level). Record `transformers` 5.6.2 and the torchvision `CLIPImageProcessor` in the run log. |
| `--limit` cache accepted as full (Task 1 M6) | Closed for this experiment: the real cache has `limit: null` and 49,121 images. Keep it in mind for reuse. |
| Cache build oversubscribed CPU (Task 1 M7) | Process note only; no effect on results. |
| Relaunch into an existing `--out` refuses (Task 2 M6) | Intended. No action. |
| NFS cleanup traceback at exit (Task 5) | Harmless (outputs written, exit 0); the `TMPDIR` fix can wait for the next cluster use. |

## Mutation table

Each mutation was applied to a scratch copy of the six code and test files and run against its test file
(`fr_mutations.py`; the unmutated suite gives 41 passed in 18 s).

| ID | Guard and mutation | Result | Caught by |
|---|---|---|---|
| H1 | held: `check_rows` never raises | caught | `test_check_rows_rejects_held` |
| H2 | held: `split_index` held-overlap check removed | **survived** | none (Task 1 M2) |
| H3 | held: cache builder's painting-level held check removed | caught | `test_painting_shared_with_held_raises` |
| H4 | held: `run()` no longer calls `check_rows` on its row sets | **survived** | none; the call cannot fire because its rows come from `split_index` |
| S1 | sampler: every train row instead of one per painting | caught | `test_sampler_one_caption_per_painting_per_epoch` |
| S2 | sampler: rng ignores the epoch | caught | same |
| S3 | sampler: runtime same-painting batch assertion removed | **survived** | none; defence in depth, since the sampler itself is tested (S1, S2) |
| P1 | trainable set: LB without the projections | caught | `test_lb_trainable_set` |
| P2 | trainable set: LB on the second-to-last vision layer | caught | `test_lb_trainable_set` |
| P3 | trainable set: LoRA without `out_proj` | caught | `test_lora_trainable_set` |
| P4 | trainable set: LoRA temperature frozen | caught | `test_lora_trainable_set` |
| P5 | trainable set: LoRA rank 8 | caught | `test_lora_trainable_set` |
| T1 | tie rule: earlier epoch before smaller lr | caught | `test_best_epoch_and_cross_run_selection` |
| T2 | tie rule: larger lr wins | caught | same |
| T3 | tie rule: `best_epoch` ties to the later epoch | caught | same |
| T4 | selection: epoch 0 selectable | caught | same |
| F1 | placement (eval): rows sorted before values are written | caught | `test_place_features_nan_outside_selection_and_val_ignored` |
| F2 | placement (trainer): features not reordered with rows | caught | `test_run_outputs_and_row_order[LP]` |
| F3 | placement (eval): val rows kept | caught | `test_place_features_nan_outside_selection_and_val_ignored` |
| F4 | pairs: training image by painting position instead of cache row | caught | `test_training_items_hold_their_rows_image_and_caption` |
| F5 | pairs: captions by feature row instead of `sample_ids` | caught | same |
| F6 | eval: painting image broadcast to the wrong rows (LB/LoRA) | caught | `test_run_outputs_and_row_order[LB]` |
| E1 | either = other-aspect rate | caught | `test_difference_signs_and_definitions` |
| E2 | either = 2 × R@1 (right only for condition-free scorers) | caught | same |
| E3 | comparison sign flipped (plain minus ft) | caught | same |

23 of 26 were caught. All three survivors are defence-in-depth checks; the real-data check 1 shows the property they
protect holds.

## Process, rulings, storage

- **Rulings.** All eight stayed within the spec, except LP's fp32 (nit 3). The epoch-0 features ruling adds outputs
  only. Best epoch ≥ 1 is a reasonable reading of spec §4 and is disclosed in §3.3.
- **Push.** `origin/main` is at 65bb2f4, the runs' commit. Local `main` is 4 commits ahead (e98f21d, c5708ab, d98407b
  and another tab's 750e06f), so nothing was pushed after the user-approved deploy pushes. Disclosed in §7.5.
- **Pull incident.** No `test_backbone_embeddings.pt` under `res/` was born on 2026-10-07; every copy has a birth time
  from September. The system disk has 221 GB free, as reported.
- **Storage left behind.** By this experiment: `res/cluster_jobs/20261007-07*` (4.6 GB, system disk; about 2.1 GB of
  it is duplicate epoch-0 features, nit 11) and the image cache `/data/SSD2/pre_extract/artelingo_clip224/` (7.39 GB,
  data SSD). By this review: about 0.4 MB in this folder. My scratch copies (mutations, figure re-run, about 0.4 MB)
  are in the session scratchpad.
- **`check_reports_sum.py`.** This experiment's row passes: the report is indexed and its link resolves. The three
  remaining problems are all `weekly/drafts/` files from another tab.
