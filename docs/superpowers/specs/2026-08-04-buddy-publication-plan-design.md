# Conditional Buddies — Publication Plan

**Date:** 2026-08-04
**Motivates from:** `docs/reports/2026-06-24_buddy_progress_report.md` (full narrative to date), `docs/reports/2026-07-16_buddy_cross_vlm_survival.md` (latest completed result), `docs/reports/2026-07-08_buddy_slides_guide.md` (existing narrative skeleton)
**Status:** proposed — pending user sign-off before `writing-plans` produces the task-by-task execution plan
**Branch:** `experiment/conditional_buddy_cross_vlms` (docs only; experiments below may run on their own branches per existing repo convention)

---

## 1. Goal

Turn ~2 months of "conditional buddies" work (a cross-modal mutual-kNN graph used to give each sample's trainable condition vector a content-aware geometric initialization) into a submission-ready paper within **1–2 months**, at the highest venue tier the evidence honestly supports.

This spec covers: the venue/framing decision, the exact set of new experiments needed (in priority order, with success criteria and decision rules), the statistical standards to hold them to, and the timeline. It does **not** re-derive anything already validated — see §2 for what is already load-bearing.

---

## 2. What's already established (no new work needed)

From the existing reports, three claims are validated with independent evidence and do not need re-running:

| Claim | Evidence | Report |
|---|---|---|
| **C1** — the buddy signal (cross-modal mutual-kNN edges) is real and content-specific, not a near-duplicate or type-matching artifact | category/subreddit lift (1.5–3× Impressions, ~20× RedCaps), held-out DINOv2 distance, VLM-judge caption-match (74–81% vs ≤7% negatives) — 2 datasets | `2026-06-22_buddy_analysis.md`, `2026-06-23_redcaps_buddy.md` |
| **C2** — the signal is not a CLIP/single-encoder/modality artifact | 6 held-out encoders (3 vision paradigms + 3 text embedders) × 2 graphs × 2 datasets = 24/24 cells: buddy pairs closer than random | `2026-07-08_heldout_grid.md` |
| **C3** — the signal survives rebuilding the graph itself under 16 different (vision × text) encoder pairs, with a semantically coherent consensus core | ~20% exact edge recurrence at 10³–10⁵× chance; core of 2,915(B)/174,161(E) edges surviving all 16 cells, 12–23× subreddit-lift | `2026-07-16_buddy_cross_vlm_survival.md` |
| **C4** — of three natural training-time uses of the signal, two (smoothness regularizer, self-refreshing graph) are seed-replicated nulls; the third (contrastive supervision) wins on Impressions but the win is substantially attributable to a measured near-duplicate confound (40.6% of optimized edges are same-photo, 279× enriched) and does not transfer to RedCaps | seed-paired ablation (§8 of the progress report) | `2026-06-24_buddy_progress_report.md` §5–8 |

**C1–C3 are the strongest part of the paper as it stands: rigorous, multi-probe, multi-dataset, multi-encoder robustness evidence for a real signal.** C4 is a well-instrumented negative/nuanced result, not a weakness — but on its own it does not answer whether the signal is *useful*, only that one way of using it during training mostly isn't (and the one way that seemed to be, wasn't really). That gap is what §4's experiments close.

---

## 3. Venue & framing decision

### 3.1 Assessment

The project's methodological rigor (seed replication, paired significance, analytic nulls, an explicit confound quantified and traced to its cause) is well above the bar for most venues. The gap is not quality — it's that **no experiment yet shows the buddy signal produces a retrieval win**, which is the load-bearing claim a "novel initialization method" paper needs. `initialization_strategy: buddies` is already the config default (`configs/train/default.yaml:9`) but no report compares it to `imgtxt` at matched settings with all training-time terms off.

### 3.2 Tiering

| Tier | Venues | Fit | Condition |
|---|---|---|---|
| **Primary** | **TMLR** (Transactions on Machine Learning Research) | Best fit. Reviewed on correctness/soundness/community value, not novelty-or-SOTA hype — matches a confound-controlled analysis with partly-null results exactly. No conference deadline pressure, which suits an experiment-dependent timeline. | **Not unconditional.** If Exp. 1 is positive, TMLR is a clean submit. If Exp. 1 is null/negative, TMLR is still reachable but only by reframing explicitly as a methodology/cautionary paper ("we checked whether a validated, robust graph signal is useful anywhere, including the most basic use, and report exactly what we found") — TMLR's "of interest to a substantial part of the community" bar is harder to clear on pure robustness-plus-nulls than the original framing implies. Also gated on Exp. 0 (below) turning up no fatal prior-art overlap. |
| **Stretch** | **ICLR / NeurIPS main track** | Plausible only with a genuine positive headline: buddy-init beats generic init cleanly (Exp. 1) on ≥2/3 datasets, ideally with COCO (Exp. 5) broadening the generalization claim beyond RedCaps/Impressions. Without a positive Exp. 1 result, a submission here is a likely reject — these venues weight "does it work" heavily even for analysis-flavored papers. | Go/no-go at the **Week-3 checkpoint** (§6), gated on Exp. 1 + Exp. 4 results. Also check current CFP/deadline dates against the timeline before committing — not verified as part of this spec. |
| **Hedge** | **CVPR/ICCV/NeurIPS workshop** (multimodal representation learning / data-centric AI / robustness) | High accept likelihood given the existing rigor; fast turnaround (weeks, not TMLR's months). Submit in parallel as insurance so the project has *a* concrete publication in-window even if TMLR review runs long. | Prepare alongside the TMLR submission at reduced length; no additional experiments required beyond §4. |
| **Not recommended** | CVPR/ICCV/ECCV main track | Vision main-track reviewers weight SOTA benchmark tables heavily; a graph-analysis paper with mostly-null training-time results is a weak fit without a decisive win. | — |
| **Not recommended** | ACL/EMNLP main/Findings | No core NLP contribution (CLIP text tower + off-the-shelf sentence encoders used only for validation); off-target audience. | — |

### 3.3 Framing decision (approved)

**Primary spine: rigorous analysis/robustness paper.** *"A content-adjacency graph signal in image–text data is real and encoder-agnostic — but exploiting it as auxiliary training supervision is a cautionary tale: only one of three natural mechanisms shows any effect, and that effect is a measured near-duplicate confound, not transferable signal."* This framing is defensible **regardless of Exp. 1's outcome** — if Exp. 1 is positive, it upgrades from "signal validated, training-use mostly doesn't work" to "signal validated AND useful at init, training-use mostly doesn't help further" (strictly stronger, same spine). If Exp. 1 is negative, the paper still stands on C1–C4 plus the closed loop of "we checked whether the validated signal is useful anywhere, including the most basic use, and report exactly what we found."

This decouples the *submission decision* from experimental risk — the stretch tier is upside, not a requirement.

### 3.4 Prior-art risk (not yet checked)

Neither this spec nor the existing reports cite or differentiate against the existing neighbor/graph-based contrastive learning literature — NNCLR (nearest-neighbor positives in SSL), mean-shift/prototype-based SSL (SwAV, MSF), or graph-Laplacian/spectral initialization as used in recsys and node2vec-style embedding init. A reviewer at any tier, TMLR included, will map "mutual-kNN graph used as auxiliary signal for a per-sample embedding" onto this lineage within the first paragraph. This is answerable — the project's contrastive-supervision mechanism (Family #2, C4) is architecturally close to NNCLR's neighbor-as-positive idea, and the finding that it's confound-driven rather than transferable is a legitimate, citable point of differentiation from NNCLR-class claims — but the positioning work has not been done, and finding out late (during weeks 6–8 writing) risks discovering an overlap problem after the experiment budget is already spent. See Experiment 0 below.

---

## 4. Experiment plan

All experiments reuse existing infrastructure (feature caching, buddy-init templates, `analyze_buddy_families.py`-style paired analysis, the seed-replication pattern already used throughout `docs/reports/2026-06-24_buddy_progress_report.md` §8). None require new model architecture or new encoders.

**Statistical standard for every new experiment (§5 governs in detail):** ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM, using the same significance convention already used in the progress report (`mean/SEM` roughly ≥ 2 read as significant; compare to the measured noise floor of ~0.1–0.7 R1 from a duplicate-config run, not to zero).

### Experiment 0 — Related-work grounding & prior-art differentiation check (do first, in parallel with Exp. 1)

- **What:** A focused literature pass (not an experiment — reading/writing only) against the neighbor/graph-based SSL lineage flagged in §3.4: NNCLR, mean-shift/prototype SSL (MSF, SwAV), and graph-Laplacian/spectral init precedent. Produce a short internal note: what's been done before, what's genuinely different here (candidate answer: per-sample *trainable condition vector initialization* in a frozen-CLIP + gated-combiner architecture, validated via encoder-robustness rather than proposed as a new SSL training method), and whether the C4 confound-diagnosis result reframes as a useful counterpoint to NNCLR-style neighbor-as-positive claims.
- **Why:** No related-work grounding exists anywhere in the project currently (confirmed by grep across `docs/`) despite it being scheduled only in weeks 6–8 (§6) as part of paper drafting. Discovering a fatal prior-art overlap during writing means the experiment budget is already spent; discovering it in week 1 means the framing (or, worst case, the venue tier) can still adapt.
- **Success criteria:** Not pass/fail. Deliverable is the internal note itself, feeding directly into the paper's eventual related-work section. A "red" outcome (a very close prior match with no clear differentiation) is itself decision-relevant — it would be grounds to revisit the TMLR framing before further compute is spent, not just before writing.
- **Cost:** Days, not compute. No infrastructure. Runs in parallel with Experiment 1, not sequentially before it.

### Experiment 1 — Buddy-init-only vs. imgtxt-init-only (critical path, do first)

- **What:** `initialization_strategy ∈ {imgtxt, buddies}`, all training-time terms off (`lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False` — i.e. the existing baseline arm of every family sweep, split by init strategy instead of held fixed at `buddies`), 3 seeds, on Impressions and RedCaps (COCO if Exp. 5 is in scope by then).
- **Why:** This is the paper's foundational claim and is currently unmeasured anywhere in the repo (confirmed by grep across `docs/`, `.claude/`, `scripts/` — every existing sweep fixes `initialization_strategy=buddies` and varies only loss terms; see `scripts/run_buddyreg_full.sh:49` etc.). Every other family-ablation number in the progress report is a delta *from* buddy-init, not a comparison *to* the prior init.
- **Tooling:** New sweep script mirroring `scripts/run_buddy_seeds.sh`'s pattern (fixed operating point, seed as the replication axis) but with `initialization_strategy` as the swept axis instead of a lambda; new analysis mode in `analyze_buddy_families.py` (or a sibling script) keyed on init strategy.
- **Success criteria / decision rule:**
  - **Positive** (buddy-init beats imgtxt-init, seed-replicated, mean/SEM ≥ 2, on ≥2/3 datasets) → paper leads with this as the headline positive result; stretch-tier venue becomes live (§3.2).
  - **Null** (no reliable difference) → paper's contribution reframes as "the signal is real and robust, but content-aware geometric initialization alone does not measurably improve retrieval over a generic PCA init" — still a legitimate, useful negative result, stays on the TMLR/workshop track.
  - **Negative** (imgtxt beats buddies) → same reframe as null, with an added discussion point (worth investigating briefly, not a blocking sub-study).
- **Cost:** ~2×3×2 = 12 runs (2 init strategies × 3 seeds × 2 datasets), reusing the standard 250-epoch Impressions / 100-epoch RedCaps schedules already established. Cheapest and highest-priority item in this plan.

### Experiment 2 — Gentler λ_con retune on RedCaps

- **What:** `lambda_buddy_con ∈ {0, 0.1, 0.3}` on RedCaps, 3 seeds. Already the field's own flagged next step (`2026-06-24_buddy_progress_report.md` §9): *"Decide #2's fate on clean data... one sweep decides it."*
- **Why:** Closes whether Family #2 (the one training-time mechanism with any measured effect) is net-positive at *some* dose off near-duplicates, or whether the mechanism itself doesn't transfer regardless of strength.
- **Tooling:** `LAMBDA_BUDDYCON_SWEEP=0,0.1,0.3 SEED_SWEEP=1,2,3 bash scripts/run_buddycon_redcaps.sh` (script already supports this override; confirmed at `scripts/run_buddycon_redcaps.sh:57`).
- **Success criteria:** Net-positive in both t2i and i2t directions at some dose → Family #2 survives as a real (if modest) contribution on clean data, strengthening C4 into a positive result at the right operating point. Null/negative at every dose → C4 stands as written; Family #2 is init-only-adjacent, not a training contribution.
- **Cost:** 3 doses × 3 seeds = 9 runs, RedCaps-scale (~100 epochs each, per the script's documented conservative schedule).

### Experiment 3 — B-lean init (`b_weight`) validated on RedCaps

- **What:** Sweep `train.buddies.b_weight ∈ {1, 2, 4, 8}` (the suggested grid from `docs/superpowers/specs/2026-07-08-b-lean-init-design.md`) on RedCaps, holding training-time terms off (init-geometry-only effect, per that spec's scope decision). 3 seeds.
- **Why:** The held-out grid (`2026-07-08_heldout_grid.md`) found strict-intersection graph B consistently cleaner than union E — but the progress report's §8d flags that B is 79.9% same-photo on Impressions, i.e. "B is cleaner" may itself be a near-duplicate artifact. This was never run on RedCaps, where B has *nothing* to lean on (82.5% of RedCaps samples have no strict buddy at all, per `2026-06-23_redcaps_buddy.md`).
- **Success criteria:** If β>1 shows no benefit (or harm) on RedCaps while showing benefit on Impressions, that's a second confirmed instance of the same near-duplicate-confound pattern as C4 — strengthens the paper's general warning rather than being a standalone result. If β>1 helps on RedCaps too, it's a genuine (if narrow) positive finding worth its own paragraph.
- **Cost:** 4 β values × 3 seeds = 12 runs, RedCaps-scale. `b_weight` is not an init-template key change requiring re-derivation logic beyond what's already implemented (config default confirmed at `configs/train/default.yaml:23`).

### Experiment 4 — CLIP fine-tuning baselines, run and reported

- **What:** Run `src/baseline/train_baseline.py` (modes `linear`, `last_blocks`, `lora`; `full` optional/lower priority given cost) on every dataset used in the paper, 3 seeds where feasible.
- **Why:** No baseline numbers exist anywhere in the repo (confirmed: no `res/baseline/` outputs, no report references `CLIPBaseline`/`train_baseline`). A main-venue reviewer's first question is "how does this compare to just fine-tuning CLIP" — currently unanswerable. This is infrastructure that was built (per `docs/superpowers/specs/2026-05-26-clip-baselines-design.md`) and never exercised for results.
- **Success criteria:** Not pass/fail — this is a required table for any venue, independent of what it shows. If CoSiR (any init) underperforms `lora`/`last_blocks` fine-tuning, that's a caveat to state plainly, not a blocker (the paper's contribution is about the initialization/graph-signal question, not "CoSiR beats fine-tuning").
- **Cost:** 3 modes × 3 datasets × (1–3 seeds) — the largest compute item in this plan if run at full seed count; acceptable to run `linear`/`last_blocks`/`lora` at 1 seed first and only replicate the numbers that matter for a close comparison.

### Experiment 5 — MS-COCO extension (stretch, contingent on 1–4)

- **What:** Repeat the signal-validation probes (lift/held-out-encoder check) and Experiment 1 (init-only comparison) on COCO. The pipeline is already dataset-agnostic (buddies init supports COCO per `2026-06-09_weekly_conditional_buddies.md` §3 "Impressions, MS-COCO, RedCaps").
- **Why:** COCO is the one dataset every reviewer already has priors about; a third, standard-benchmark data point substantially strengthens any generalization claim and is close to required for the stretch (main-track) tier.
- **Gate:** Only pursue if Experiments 1–4 land on schedule by the Week-3 checkpoint (§6) and (ideally) Exp. 1 is positive — otherwise the marginal value of a third data point on an already-established negative/nuanced result is lower than spending the time on writing.

### Experiment 6 — Approach C: does cross-VLM survival predict downstream usefulness? (stretch, optional)

- **What:** The 16-cell cross-VLM survival study (`2026-07-16_buddy_cross_vlm_survival.md`) explicitly deferred this: *"whether higher cross-VLM survival translates into better downstream condition-vector initialization or training... deliberately left as future work."* This would connect the paper's two halves (signal generalizes / does using it help) directly, e.g. by testing whether samples whose buddy edges are in the high-consensus core benefit more from buddy init than samples whose edges are low-consensus.
- **Gate:** Only if ahead of schedule after Experiments 1–4 (and 5 if pursued). This is a new sub-study, not a rerun of existing tooling, and carries the highest risk of scope creep in this plan.

### Experiment 7 — Causal tightening of the near-duplicate confound (cheap, high value — do if time allows after 1–4)

- **What:** Re-run Family #2's confirmed Impressions setting (`λ_con=1.0`) with same-source-photo edges explicitly excluded from the buddy graph, and check whether the win (+2.3 t2i / +3.2 i2t) shrinks toward the RedCaps number (+0.4 / −0.9).
- **Why:** The current confound evidence (§8d of the progress report) is correlational — 40.6% same-photo enrichment coincides with the win. Directly removing that structure and watching the effect shrink turns "we believe this explains it" into "we removed it and the effect disappeared," a substantially stronger claim for reviewers.
- **Cost:** Cheap — reuses the existing `identity_stats` probe (`src/test/20260622_buddy_analysis/buddy_analysis.py`) to build an edge mask, one training arm (λ_con=1.0, same-photo-excluded graph) × 3 seeds, compared against the already-measured λ_con=1.0 full-graph result. No new infrastructure.

---

## 5. Statistical methodology (standardize across all new experiments)

These are not new conventions — they're what the project already does in `2026-06-24_buddy_progress_report.md` §8 and `2026-07-16_buddy_cross_vlm_survival.md`; this section exists so every new experiment (1–4, 7) is held to the same bar as the existing ones, since a reviewer will read them side by side.

1. **≥3 seeds for every comparison that feeds a paper claim.** The single-seed grid in the original Family #1/#3 ablation was demonstrably misleading (see §8a of the progress report) — this is not a hypothetical risk, it already happened once in this project.
2. **Paired-within-seed reporting**: mean Δ ± std across seeds, plus `mean/SEM` as the significance read, consistent with the existing convention (e.g. `mean/SEM = 7.0`, `z = +11`).
3. **A measured noise floor, not a zero baseline.** The progress report established the noise floor (~0.1–0.7 R1) from a duplicate-config run; any new experiment claiming a real effect should be checked against that floor, not against Δ=0.
4. **Analytic nulls over Monte Carlo where a closed form exists** — the project already made this switch once (`7226e5b perf: replace Monte-Carlo permutation null with closed-form analytic null`); no new experiment in this plan needs a new null construction, but if one comes up, prefer the closed form.
5. **No silent cherry-picking of operating points.** Where a sweep is involved (Exp. 2, 3), report the full curve, not just the best cell — the progress report's own peak-finding (λ_con dose-response, §8b) is the template: show the rollover, not just the peak.

---

## 6. Timeline (8 weeks, with an explicit decision checkpoint)

| Weeks | Work |
|---|---|
| **1** | Experiment 0 (related-work/prior-art check) run in parallel with the start of Experiment 1 — cheap and highest-value-per-day, resolved before deeper compute commitment. |
| **1–2** | Experiments 1, 2, 3 run in parallel (independent, share infra); Experiment 4 (baselines) queued alongside on separate compute. |
| **3** | Analyze all four (plus Exp. 0's note). **Checkpoint:** decide (a) TMLR/workshop-only vs. also targeting stretch-tier (gated on Exp. 1 being positive, per §3.2), (b) whether Exp. 0 requires a framing adjustment, and (c) whether Experiment 5 (COCO) is in scope given remaining time. |
| **4–5** | Experiment 5 (COCO) if in scope; Experiment 7 (causal confound tightening) — cheap, do regardless of the Week-3 outcome. |
| **6** | Experiment 6 (Approach C) only if meaningfully ahead of schedule. |
| **6–8** | Writing: paper draft (skeleton reuses the existing slide guide's Part I / Part II structure, `2026-07-08_buddy_slides_guide.md`), related-work section (drafted from Exp. 0's note, not started from scratch here), baseline table, figures (adapt `scripts/make_slide_figs.py`), internal review buffer, prepare both TMLR and workshop submission packages. |

---

## 7. Deliverables

1. **This spec** (design/decision record).
2. **Task-by-task execution plan** — to be authored by the `writing-plans` skill once this spec is approved, one plan per experiment cluster (mirrors the existing `docs/superpowers/plans/2026-07-16-buddy-cross-vlm-survival.md` pattern: checkboxed steps, exact commands, expected output).
3. **Results reports** for each experiment, in the existing `docs/reports/YYYY-MM-DD_*.md` format, as they complete.
4. **Research proposal** (separate narrative document, `docs/proposals/2026-08-04-conditional-buddies-publication-proposal.md`) — for external/advisor-facing communication of this plan; see that document for the polished pitch version of everything in this spec.
5. **Paper draft** (out of scope for this spec; produced in weeks 6–8 per §6).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Experiment 1 comes back null/negative, undercutting the "novel useful initialization" framing | Framing (§3.3) is explicitly designed to survive this — the spine is the robustness/analysis story, not a positive-init claim. |
| TMLR review latency extends past the 2-month window | Workshop hedge (§3.2) submitted in parallel guarantees an in-window publication regardless of TMLR's timeline. |
| Compute/time overruns on Experiments 2–4 | Priority order (§4) puts the cheapest, highest-information experiment (1) first; stretch items (5, 6) are explicitly gated on schedule, not committed up front. |
| Baseline runs (Exp. 4) reveal CoSiR underperforms simple fine-tuning | Not a blocker — state as a caveat; the paper's claims are about the graph signal and initialization question, not about beating fine-tuned CLIP. |
| Scope creep into Experiment 6 (Approach C) crowds out writing time | Explicit Week-6-or-later gate; only pursued if 1–5 finish ahead of schedule. |
| Prior-art overlap (NNCLR-class neighbor-as-positive methods, graph-Laplacian init precedent) discovered late, undercutting the differentiation story | Experiment 0 runs in week 1, in parallel with Experiment 1, specifically to surface this before further compute or writing investment (§3.4). |

---

## 9. Out of scope

- New backbones/architectures beyond CLIP ViT-B/32 (baseline sweep uses the same backbone family per its existing design).
- Datasets beyond COCO/Impressions/RedCaps.
- Any new training-time mechanism beyond the three already-implemented families (no "Family #4").
- Verifying exact conference CFP/deadline dates for the stretch tier — flagged in §3.2 as a to-check item before committing to that track, not resolved here.

---

## Self-review

- **Placeholder scan:** no TBD/TODO left unresolved; every experiment has a stated success criterion and cost estimate.
- **Internal consistency:** the framing decision (§3.3) is checked against both possible outcomes of Experiment 1 (§4, Exp. 1) and does not contradict the venue tiering (§3.2); the deliverables list (§7) matches the timeline (§6).
- **Scope check:** stretch experiments (5, 6) are explicitly gated rather than committed, keeping the critical path (0–4, 7) achievable alone within the window even if stretch items are dropped.
- **Ambiguity check:** every experiment names its exact tooling (existing script + flag, or "new script mirroring X") and its decision rule, so "what counts as done" is unambiguous per item.
