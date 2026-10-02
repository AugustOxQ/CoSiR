# Reports guide

This file is the index to every report in `docs/reports/`. Keep it current: every new report gets one row here (see [Adding a report](#adding-a-report)). All dates are 2026 unless marked otherwise.

## Layout

| Folder | What goes there |
|---|---|
| `auto/<line>/` | **Automatic reports.** One per experiment, diagnostic, review or brainstorm, written when the task finishes. Grouped by research line: `buddy`, `percept`, `v2`. |
| `auto/<line>/pilots/<dir>/` | Pilot reports and debug logs copied from `src/test/<dir>/` on branches whose code is not on main. |
| `stage/` | **Stage reports.** Syntheses across several experiments, including progress and comprehensive reports, plus their slide markdown. |
| `weekly/` | **Weekly reports** and their slide markdown. |
| `pptx/` | **Slide decks (.pptx).** Built from slide markdown by `assets/build_*_slides.py`. `*.pptx` is gitignored, so decks exist only locally; rebuild them from the slide markdown. |
| `assets/` | Figures, figure data and build scripts used by the reports. |

Research lines:
- **buddy:** conditional-buddies publication track (Exp 0–17), Jun to Sep 15, plus earlier CoSiR debug logs.
- **percept:** Exp 18 prototype conditioning, the ArtELingo affect/fusion work and the PercepT topic pipeline, Sep 15–28.
- **v2:** the CoSiR v2 redesign, from Sep 28. This is what main contains.

## Start here

- **Current work (v2):** [affect factor-learning held test](auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md), the latest result (SE, factor codes trained with condition episodes from GoEmotions affect clusters of the captions and CLIP-image clusters, is confirmed against the matched control on fresh held episodes: emotion +2.08 [+1.51, +2.62] and style +0.65 [+0.07, +1.21] R@1, +1.79 pooled over original R3, seeds 43/44 agree), after the [affect factor-learning selection](auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md) (SE picked with emotion +1.31 and style +1.12 over the matched control; E, affect alone, gains +3.99 on emotion but fails the style guard), the [factor-learning selection](auto/v2/2026-10-16_candidate_a_factor_learning_selection.md) (2×2 of painting-level agreement and CLIP-image-cluster condition episodes against a matched control: no cell qualifies; the style episodes raised art-style R@1 but cost emotion and broke the sparsity gate, and painting-level agreement lowered both labels), the [factor headroom probe](auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md) (frozen CLIP carries far more emotion and style than R3's factors express, so factor learning is the next lever), the [stage (d) final held-out test](auto/v2/2026-10-14_candidate_a_stage_d_final.md) (trained scorer G3 vs naive: condition-use gain not met, inconclusive because underpowered; human swap test met as pre-registered, but reproduced by the naive rule at G3's learned β, so it reflects the lower CLIP weight; the stage (e) decision is the user's), following [stage (d) selection](auto/v2/2026-10-13_candidate_a_stage_d_selection.md) and building on [condition eval on repaired factors](auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md) and [factor repair](auto/v2/2026-10-11_candidate_a_factor_repair.md), the current recipe R3.
- **Percept line summary:** [buddy vs PercepT master report](auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md), [comprehensive analysis](stage/2026-09-27_comprehensive_analysis.md) and [key advantages](stage/2026-09-27_key_advantages.md); the final result is the [matched head-to-head](auto/percept/2026-09-30_matched_percept_buddy_h2h.md) (master report §6k).
- **Buddy line summary:** the [archived publication plan](../archive/buddy_publication_plan/2026-08-04-buddy-publication-plan-design.md) (claims C1–C12; no longer updated) and the [last buddy-line weekly report](weekly/2026-09-02_conditional_buddies.md).
- **Latest weekly report:** [22 to 30 Sep](weekly/2026-09-30_percept_buddy_to_v2.md), covering the buddy model against the PercepT baseline and CoSiR v2 up to stage (d), with its figures in `assets/2026-09-30_weekly/`.
- **Publication plan for v2:** not written yet. It will replace the archived buddy plan, with v2 as the main body.

## Where the code lives

v2 kept only a reusable foundation, so most buddy and percept reports cite code that is not on main. Links to that code are left as written.

| Line | Code |
|---|---|
| buddy | tag `archive/experiment/condition_drift_retrieval_correlation`. Exp 13 is on tag `archive/experiment/two_side_conditioning`. Earlier branches have their own `archive/<branch>` tags. |
| percept | branch `experiment/percept_topic_pipeline` (active; folder `/project/CoSiR-buddy_prototype_conditioning`). Exp 18 is also on tag `archive/experiment/buddy_prototype_conditioning`. |
| v2 | main |

To open an archived line: `git worktree add ../CoSiR-<name> archive/<branch>`.

## Adding a report

1. **Name:** `YYYY-MM-DD_<topic>.md`. Leave out words the folder already says: no `weekly_`, `stage_report`, `_report` or `cosir_v2_`.
2. **Place:** put it in `auto/<line>/`, `stage/` or `weekly/`. Slide markdown sits next to its report. Decks go in `pptx/`, and their build script goes in `assets/build_<date>_<topic>_slides.py`.
3. **Index:** add one row to the matching table below, oldest first, with a one-line description. Starting a new research line means adding `auto/<line>/`, a table here and a row in the tables above.
4. **Reports on other branches:** these stay on their branch until a stage report or comparison needs them. Gather them with `python scripts/promote_reports.py <branch>`, adding `--line <name>` for a branch it doesn't know yet, and `--dry-run` to preview. It copies new or changed reports, pilots and local decks into this layout, adds rows here using each report's title, and runs the check. Polish the new rows, then commit. `.promoted.json` records what came from where.
5. **Check:** run `python scripts/check_reports_sum.py`. It fails if any report, deck or pilot folder is missing here, if a link here is broken, or if a file sits loose in `docs/reports/`.

## auto/v2: CoSiR v2

| Date | Report | What it is |
|---|---|---|
| 09-28 | [architecture_rethink_literature_brainstorm](auto/v2/2026-09-28_architecture_rethink_literature_brainstorm.md) | Literature review and design brainstorm behind the v2 redesign |
| 09-28 | [block1_stage1_validation](auto/v2/2026-09-28_block1_stage1_validation.md) | Block 1: content-only Stage 1 on ArtELingo |
| 09-28 | [stage1_genecis_synthesis_brainstorm](auto/v2/2026-09-28_stage1_genecis_synthesis_brainstorm.md) | From validated topics to open conditional image–text similarity |
| 09-28 | [candidate_a_factor_discovery_validation](auto/v2/2026-09-28_candidate_a_factor_discovery_validation.md) | Candidate A: factor discovery on ArtELingo |
| 09-29 | [block1_cross_encoder_ablation](auto/v2/2026-09-29_block1_cross_encoder_ablation.md) | Block 1: cross-encoder ablation with e5 text |
| 09-29 | [candidate_a_factor_balance_fix](auto/v2/2026-09-29_candidate_a_factor_balance_fix.md) | Candidate A: factor balance fix |
| 09-29 | [code_review](auto/v2/2026-09-29_code_review.md) | Code review: does each component serve Candidate A's goal? |
| 09-30 | [candidate_a_reduced_rank_whitening](auto/v2/2026-09-30_candidate_a_reduced_rank_whitening.md) | Candidate A: reduced-rank whitening |
| 09-30 | [candidate_a_usage_balance_no_whitening](auto/v2/2026-09-30_candidate_a_usage_balance_no_whitening.md) | Candidate A: usage balance without whitening |
| 10-01 | [candidate_a_condition_interface_validation](auto/v2/2026-10-01_candidate_a_condition_interface_validation.md) | Condition interface and episode mining |
| 10-02 | [candidate_a_condition_confusion_diagnostic](auto/v2/2026-10-02_candidate_a_condition_confusion_diagnostic.md) | Held-out condition confusion |
| 10-03 | [candidate_a_condition_recovery_balanced_retrain](auto/v2/2026-10-03_candidate_a_condition_recovery_balanced_retrain.md) | Class-balanced condition-recovery re-test |
| 10-04 | [candidate_a_factor_separability_diagnostic](auto/v2/2026-10-04_candidate_a_factor_separability_diagnostic.md) | Factor separability versus condition recovery |
| 10-05 | [candidate_a_condition_ranking_evaluation](auto/v2/2026-10-05_candidate_a_condition_ranking_evaluation.md) | Item-disjoint condition ranking evaluation |
| 10-06 | [candidate_a_clip_only_baseline](auto/v2/2026-10-06_candidate_a_clip_only_baseline.md) | Exact CLIP-only ranking baseline |
| 10-07 | [candidate_a_naive_rule_mechanism](auto/v2/2026-10-07_candidate_a_naive_rule_mechanism.md) | Why the naive rule works |
| 10-09 | [candidate_a_factor_collapse_diagnosis](auto/v2/2026-10-09_candidate_a_factor_collapse_diagnosis.md) | Where the factor-space collapse comes from |
| 10-11 | [candidate_a_factor_repair](auto/v2/2026-10-11_candidate_a_factor_repair.md) | Factor-repair grid, pre-registered gates, goal-based selection |
| 10-12 | [candidate_a_condition_eval_repaired_factors](auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md) | Condition interface on the repaired factors, held-out |
| 10-13 | [candidate_a_stage_d_selection](auto/v2/2026-10-13_candidate_a_stage_d_selection.md) | Stage (d) selection: five self-generated-condition scorers vs the naive rule (G3 selected; post-hoc: about a third of the gain is the β drop, the per-episode ceiling is uninformative, and a per-label oracle does not clearly beat naive) |
| 10-14 | [candidate_a_stage_d_final](auto/v2/2026-10-14_candidate_a_stage_d_final.md) | Stage (d) final held-out test: G3 vs naive (condition-use gain not met, inconclusive; human swap test met as pre-registered but reproduced by naive at G3's learned β) |
| 10-15 | [candidate_a_factor_headroom_probe](auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md) | Headroom probe before factor learning: a label-aligned code on frozen CLIP reaches 49.8% label-oracle R@1 vs R3's 20.5% (baseline naive on R3 19.4%), so R3's code, not CLIP, is the limit; style is visual and emotion textual; self-generated sources line up with style only |
| 10-16 | [candidate_a_factor_learning_selection](auto/v2/2026-10-16_candidate_a_factor_learning_selection.md) | Factor-learning 2×2 selection (C0 / A / S / AS on selection rows): stop, no cell qualifies; style episodes (S) +1.44 R@1 over the matched control, all from art style, but the emotion guard and the sparsity gate fail; painting-level agreement (A) −3.03 |
| 10-18 | [candidate_a_affect_factor_learning_selection](auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md) | Affect-signal factor learning, selection (E / SE vs the matched control C0 on selection rows): SE (GoEmotions affect clusters and CLIP image clusters) picked, emotion +1.31 [+0.54, +2.06] and style +1.12 [+0.29, +1.93] R@1; E (affect clusters alone) emotion +3.99 but style −1.23, fails the guard; replication seeds 43/44 point the same way (emotion clear at 44, not at 43) |
| 10-19 | [candidate_a_affect_factor_learning_held](auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md) | Affect-signal factor learning, held test (SE vs the matched control C0 on 8,192 fresh seed-43 held episodes per label, run once): confirmed, emotion +2.08 [+1.51, +2.62] and style +0.65 [+0.07, +1.21] R@1; over original R3 +1.79 pooled and +1.56 on emotion; seeds 43/44 agree |
| 10-20 | [genecis_feasibility](auto/v2/2026-10-20_genecis_feasibility.md) | Can v2 use the GeneCIS benchmark? Yes for evaluation (CC BY-NC 4.0): the object half (3,920 templates) runs from local COCO 2014 images, the attribute half needs a Visual Genome 1.2 download; GeneCIS is image+text→image, so it needs an image-to-image mode and a text-phrase condition adapter and tests zero-shot transfer, not the s(I,T\|c) claim; local CC3M triplets unusable without the scene graphs |
| 10-21 | [cvpr_literature_review](auto/v2/2026-10-21_cvpr_literature_review.md) | CVPR literature review: no prior example-conditioned cross-item image↔text similarity found (to our knowledge), but the naive rule is Rocchio relevance feedback and the factor term a CSN mask; "unsupervised condition discovery" is taken (SCE-Net, DiscoverNet, EmotionCLIP distant supervision); ranked baselines (support prototype/Rocchio, Tip-Adapter, probe, text-named condition, instruction embedders) and benchmarks (CUB + Reed captions, GeneCIS, SemArt); PercepT itself uses ModernBERT-GoEmotions and CLIP ViT-L/14 |
| 10-22 | [support_baseline_spike](auto/v2/2026-10-22_support_baseline_spike.md) | Simple support-set baselines on raw CLIP beat SE pooled (best: both-modality logistic probe 24.10 vs SE 21.22, +2.87 [+2.14, +3.64], all in art style; emotion matched), the query adds only +0.57 over a condition-only prototype so the label episodes mostly test few-shot recognition, and SE wins only cross-modally (style i2t, emotion t2i) |
| 10-23 | [aspect_episode_spike](auto/v2/2026-10-23_aspect_episode_spike.md) | On cross-modal aspect episodes no factor model selects the aspect (SE agree 11.32 vs CLIP only 11.13, +0.20 [−0.12, +0.50]; C0, R3, raw agree and the value prototype also at CLIP level; only privileged names gains, +1.51), although label-supervised probes reach about 24 (emotion) and 22 (style); emotion lives in captions and style in images, so cross-modal matching is capped by the weaker modality |

## auto/percept: Exp 18, ArtELingo and PercepT

| Date | Report | What it is |
|---|---|---|
| 09-15 | [buddy_prototype_conditioning](auto/percept/2026-09-15_buddy_prototype_conditioning.md) | Exp 18: buddy-graph prototype conditioning |
| 09-26 | [artelingo_buddy_vs_percept_stage1](auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md) | **Master report:** buddy graph replacing PercepT's autoencoder+DEC in Stage 1. §6i sweep, §6j confirmation checks, §6k matched head-to-head. |
| 09-26 | [buddy_silhouette_gap_brainstorm](auto/percept/2026-09-26_buddy_silhouette_gap_brainstorm.md) | Ranked next directions for the Stage 1 silhouette gap |
| 09-27 | [agy_independent_percept_review](auto/percept/2026-09-27_agy_independent_percept_review.md) | Independent adversarial review of the PercepT baseline and the evaluation |
| 09-27 | [method_improvement_and_redcaps_brainstorm](auto/percept/2026-09-27_method_improvement_and_redcaps_brainstorm.md) | Next ArtELingo tests and a RedCaps transfer pilot |
| 09-27 | [redcaps_b1_result_and_diagnosis](auto/percept/2026-09-27_redcaps_b1_result_and_diagnosis.md) | RedCaps B1 pilot result and root-cause diagnosis |
| 09-28 | [buddy_percept_sweep_handoff](auto/percept/2026-09-28_buddy_percept_sweep_handoff.md) | Handoff for the buddy-percept comprehensive sweep |
| 09-30 | [matched_percept_buddy_h2h](auto/percept/2026-09-30_matched_percept_buddy_h2h.md) | **Matched PercepT-vs-buddy head-to-head:** one validated harness, matched K, equal 300-trial searches. Plain AUC: buddy leads but with less emotion structure; at an emotion floor no AUC difference detected. Supersedes §6g |

## auto/buddy: conditional buddies

| Date | Report | What it is |
|---|---|---|
| 06-09 | [buddies_dim_hparam_study](auto/buddy/2026-06-09_buddies_dim_hparam_study.md) | Dimensionality and hyperparameter study |
| 06-22 | [buddy_analysis](auto/buddy/2026-06-22_buddy_analysis.md) | Are condition buddies meaningful? (Phase 1) |
| 06-23 | [redcaps_buddy](auto/buddy/2026-06-23_redcaps_buddy.md) | Do buddies generalize to RedCaps? |
| 07-08 | [heldout_grid](auto/buddy/2026-07-08_heldout_grid.md) | Held-out encoder grid across vision and language models |
| 07-16 | [buddy_cross_vlm_survival](auto/buddy/2026-07-16_buddy_cross_vlm_survival.md) | Do buddies survive changing the VLM? |
| 08-10 | [prior_art_note](auto/buddy/2026-08-10_prior_art_note.md) | Prior-art grounding note |
| 08-16 | [buddy_init_ablation](auto/buddy/2026-08-16_buddy_init_ablation.md) | Buddy-graph init versus generic image-text init |
| 08-19 | [buddy_init_ablation_redcaps_300k](auto/buddy/2026-08-19_buddy_init_ablation_redcaps_300k.md) | At RedCaps-300k, buddy init beats generic init, but neither beats raw CLIP |
| 08-24 | [buddy_distance_mode_ablation](auto/buddy/2026-08-24_buddy_distance_mode_ablation.md) | Fixing the modality-dilution bug doesn't move retrieval |
| 08-24 | [buddy_init_encoder_ablation](auto/buddy/2026-08-24_buddy_init_encoder_ablation.md) | Does the graph's source encoder pair matter? |
| 08-24 | [redcaps_subreddit_signal_correlates](auto/buddy/2026-08-24_redcaps_subreddit_signal_correlates.md) | What predicts buddy-signal strength across subreddits (Exp 9) |
| 08-25 | [condition_freeze_ablation](auto/buddy/2026-08-25_condition_freeze_ablation.md) | Does post-init condition training help? (Exp 11.1) |
| 08-26 | [polysemy_bridge_diagnostic](auto/buddy/2026-08-26_polysemy_bridge_diagnostic.md) | Cross-modal bridge nodes and false transitivity (Exp 12) |
| 08-27 | [combine_side_txt_replication](auto/buddy/2026-08-27_combine_side_txt_replication.md) | Does `combine_side="txt"` flip the i2t/t2i asymmetry? |
| 08-27 | [symmetric_conditioning_exp13](auto/buddy/2026-08-27_symmetric_conditioning_exp13.md) | Exp 13: symmetric conditioning (code on the two_side archive tag) |
| 08-31 | [closed_triangle_bridge_diagnostic](auto/buddy/2026-08-31_closed_triangle_bridge_diagnostic.md) | Closed-triangle positive control for false transitivity (Exp 14) |
| 09-01 | [buddy_k_scaling_stage_a](auto/buddy/2026-09-01_buddy_k_scaling_stage_a.md) | Does mutual-kNN K matter and scale with N? (Exp 16.1) |
| 09-01 | [combiner_architecture_brainstorm](auto/buddy/2026-09-01_combiner_architecture_brainstorm.md) | Injecting a tiny buddy vector into frozen CLIP space |
| 09-02 | [buddy_k_ablation_stage_b](auto/buddy/2026-09-02_buddy_k_ablation_stage_b.md) | Does K move real retrieval numbers? (Exp 16.2) |
| 09-02 | [combiner_architecture_ablation](auto/buddy/2026-09-02_combiner_architecture_ablation.md) | Combiner family matters, not dimension |
| 09-03 | [combiner_architecture_fullscale_validation](auto/buddy/2026-09-03_combiner_architecture_fullscale_validation.md) | Does the combiner win survive at 500k scale? |
| 09-15 | [condition_space_audit](auto/buddy/2026-09-15_condition_space_audit.md) | Condition-space steerability audit (Exp 17.1) |

## Pilots

| Folder | Files | Contents |
|---|---|---|
| [percept/pilots/20260922_percept_topic_pipeline](auto/percept/pilots/20260922_percept_topic_pipeline/) | 23 | PercepT Stage 1/2 pilot reports |
| [percept/pilots/20260922_percept_brainstorm](auto/percept/pilots/20260922_percept_brainstorm/) | 1 | Codex brainstorm findings |
| [percept/pilots/20260923_artelingo_buddy_analysis](auto/percept/pilots/20260923_artelingo_buddy_analysis/) | 36 | ArtELingo buddy analysis: affect, fusion and attention-h1 pilots, reviews |
| [percept/pilots/20260927_deep_stage_analysis](auto/percept/pilots/20260927_deep_stage_analysis/) | 13 | Deep-stage analysis: candidate-fix pilots and stress tests |
| [percept/pilots/20260927_redcaps_topic_formation](auto/percept/pilots/20260927_redcaps_topic_formation/) | 4 | RedCaps transfer: B1 single-teacher, repaired graph, Leiden resolution, subreddit proxy |
| [percept/pilots/20260928_buddy_percept_sweep](auto/percept/pilots/20260928_buddy_percept_sweep/) | 2 | Sweep SDD log and stress-test summary |
| [buddy/pilots/20250911_embedding_performance_debugging](auto/buddy/pilots/20250911_embedding_performance_debugging/) | 5 | 2025 embedding-performance debugging notes |
| [buddy/pilots/20250930_config_debug](auto/buddy/pilots/20250930_config_debug/) | 1 | 2025 config debug log |
| [buddy/pilots/20251111_condition_space_evaluator](auto/buddy/pilots/20251111_condition_space_evaluator/) | 1 | 2025 condition-space evaluator debug log |
| [buddy/pilots/20251119_text_to_image_retrieval](auto/buddy/pilots/20251119_text_to_image_retrieval/) | 1 | 2025 text-to-image retrieval log |
| [buddy/pilots/20260519_network_analysis](auto/buddy/pilots/20260519_network_analysis/) | 1 | Network-analysis improvement suggestions |
| [buddy/pilots/20260609_conditional_buddy](auto/buddy/pilots/20260609_conditional_buddy/) | 1 | Conditional-buddy debug log |
| [buddy/pilots/20260716_buddy_cross_vlm](auto/buddy/pilots/20260716_buddy_cross_vlm/) | 1 | Cross-VLM survival log |
| [buddy/pilots/20260915_condition_space_audit](auto/buddy/pilots/20260915_condition_space_audit/) | 1 | Condition-space audit log |
| [percept/pilots/20260930_harness_confirmation](auto/percept/pilots/20260930_harness_confirmation/) | 1 | Confirmation checks log (master report §6j) |
| [percept/pilots/20260930_matched_h2h](auto/percept/pilots/20260930_matched_h2h/) | 4 | Matched head-to-head: debug log, SDD ledger, val stress and test summaries |

## stage: stage reports

| Date | Report | What it is |
|---|---|---|
| 06-24 | [buddy_progress](stage/2026-06-24_buddy_progress.md) | Conditional-buddies progress report. §8a gives baseline numbers the analysis scripts cite. |
| 09-16 | [prototype_conditioning](stage/2026-09-16_prototype_conditioning.md) · [slides](stage/2026-09-16_prototype_conditioning_slides.md) | Exp 18 stage report: prototype-pooled conditioning |
| 09-22 | [artelingo_affect_investigation](stage/2026-09-22_artelingo_affect_investigation.md) | ArtELingo affect investigation |
| 09-22 | [artelingo_fusion_mechanism_investigation](stage/2026-09-22_artelingo_fusion_mechanism_investigation.md) | ArtELingo fusion-mechanism investigation |
| 09-22 | [artelingo_fusion_diagnostics](stage/2026-09-22_artelingo_fusion_diagnostics.md) | Visual companion to the fusion-mechanism report |
| 09-23 | [artelingo_percept_stage1](stage/2026-09-23_artelingo_percept_stage1.md) | PercepT Stage 1: P-topic formation |
| 09-23 | [artelingo_percept_stage2](stage/2026-09-23_artelingo_percept_stage2.md) | PercepT Stage 2: P-topic mapping |
| 09-27 | [comprehensive_analysis](stage/2026-09-27_comprehensive_analysis.md) | Comprehensive buddy versus PercepT analysis on ArtELingo |
| 09-27 | [key_advantages](stage/2026-09-27_key_advantages.md) | Key advantages of buddy-graph topic formation |

## weekly: weekly reports and slides

| Week of | Report | Slides md | Topic |
|---|---|---|---|
| 06-09 | [conditional_buddies](weekly/2026-06-09_conditional_buddies.md) | | Conditional-buddies initialization |
| 07-08 | | [buddy_slides_guide](weekly/2026-07-08_buddy_slides_guide.md) | Slide guide: what worked, what didn't, and why |
| 08-19 | | [conditional_buddies_slides](weekly/2026-08-19_conditional_buddies_slides.md) | Progress update |
| 08-26 | [conditional_buddies](weekly/2026-08-26_conditional_buddies.md) | [conditional_buddies_slides](weekly/2026-08-26_conditional_buddies_slides.md) | What matters after initialization? |
| 09-02 | [conditional_buddies](weekly/2026-09-02_conditional_buddies.md) | [conditional_buddies_slides](weekly/2026-09-02_conditional_buddies_slides.md) | Why does training sometimes hurt, and what should the defaults be? |
| 09-23 | [percept_topic_pipeline](weekly/2026-09-23_percept_topic_pipeline.md) | [percept_topic_pipeline_slides](weekly/2026-09-23_percept_topic_pipeline_slides.md) | PercepT topic formation; reproducing CoSiR in PercepT |
| 09-30 | [percept_buddy_to_v2](weekly/2026-09-30_percept_buddy_to_v2.md) | [percept_buddy_to_v2_slides](weekly/2026-09-30_percept_buddy_to_v2_slides.md) | 22 to 30 Sep: the buddy model against the PercepT baseline on ArtELingo, and the CoSiR v2 redesign up to stage (d) |

## pptx: rendered decks

Rebuild a deck with `python docs/reports/assets/<build script>`, which needs `python-pptx`. Decks without a build script were made by hand.

| Deck | Source | Build script |
|---|---|---|
| [2026-08-19_buddy_init_ablation_slides.pptx](pptx/2026-08-19_buddy_init_ablation_slides.pptx) | none | none |
| [2026-08-19_conditional_buddies_slides.pptx](pptx/2026-08-19_conditional_buddies_slides.pptx) | weekly/2026-08-19_conditional_buddies_slides.md | none |
| [2026-08-26_conditional_buddies_slides.pptx](pptx/2026-08-26_conditional_buddies_slides.pptx) | weekly/2026-08-26_conditional_buddies_slides.md | `build_2026-08-26_weekly_slides.py` |
| [2026-09-02_conditional_buddies_slides.pptx](pptx/2026-09-02_conditional_buddies_slides.pptx) | weekly/2026-09-02_conditional_buddies_slides.md | `build_2026-09-02_weekly_slides.py` |
| [2026-09-16_prototype_conditioning_slides.pptx](pptx/2026-09-16_prototype_conditioning_slides.pptx) | stage/2026-09-16_prototype_conditioning_slides.md | `build_2026-09-16_stage_report_slides.py` |
| [2026-09-23_percept_topic_pipeline_slides.pptx](pptx/2026-09-23_percept_topic_pipeline_slides.pptx) | weekly/2026-09-23_percept_topic_pipeline_slides.md | `build_2026-09-23_weekly_percept_topic_pipeline_slides.py` |
| [2026-09-30_percept_buddy_to_v2_slides.pptx](pptx/2026-09-30_percept_buddy_to_v2_slides.pptx) | weekly/2026-09-30_percept_buddy_to_v2_slides.md | `build_2026-09-30_weekly_slides.py` (figures: `build_2026-09-30_weekly_figures.py --slides`) |
