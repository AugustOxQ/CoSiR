# ArtELingo Stage 1: buddy-graph replacement for PercepT's autoencoder+DEC — overnight investigation report

Generated 2026-09-26. Branch `experiment/percept_topic_pipeline`.

---

## ⭐ RECOMMENDED SOLUTION (read this section first)

> **Status note, 2026-10-01 (after §6j and §6k).** This recommendation was
> written on 2026-09-27 and predates both checks. §6j found that the
> unmodified Attention-h1 student clears the emotion bar on only 2 of 4 DAS6
> seeds (3 of 4 on the local GPU), with mean independent emotion AMI 0.1230,
> just below the 0.1236 threshold. §6k's matched head-to-head removed its
> Stage 2 case: ranked by Stage 2 AUC, buddy leads PercepT only by giving up
> emotion structure; at a common emotion floor we detected no Stage 2
> difference; and none of the eight head-to-head winners used this pilot
> Stage 1 (all used the §6i harness Stage 1, which was not validated against
> the pilot). At K = 16 one pilot-Stage-1 finalist showed no detectable
> difference from PercepT on the val half (Δ +0.002, 95% CI [−0.023, +0.028], 4 seeds). The Stage 2 bullets below (§6a to §6f) were superseded by
> §6g and §6k. Read this section as the 2026-09-27 recommendation, not a
> current one.

**Adopt the unmodified Attention-h1 buddy-graph student, optionally with the
cosine-annealed learning rate, as the Stage 1 topic-formation mechanism, in
place of PercepT's autoencoder+DEC — and use `assign_to_train_communities`
(k-NN transfer, k=20) to solve the Stage 2 shared-vocabulary blocker.**

Concretely:
- **Encoder**: `run_learned_student_arch_sweep_pilot.py`'s `LearnedStudent("attn1")`
  architecture — two teacher-graph (content, affect) symmetric InfoNCE losses,
  one-head self-attention fusion, LayerNorm + L2-normalized 32-D output. No
  code change needed; it already performs this well.
- **Optional, low-risk refinement**: replace its fixed Adam learning rate with
  `CosineAnnealingLR(1e-3 → 1e-5, T_max=200)`
  (`run_attention_h1_noise_schedule_pilot.py`, `noise_std=0`). This gives a
  small but consistent held-out silhouette gain (+0.0069 mean, positive in
  **4/4** paired seeds) at no AMI cost worth worrying about. It does **not**
  change how many seeds clear the AMI Pareto bar (3/4 either way) — this is a
  minor, free improvement, not the headline result. Adopt it because it costs
  nothing, not because it is transformative.
- **Stage 2 shared vocabulary**: use `assign_to_train_communities`
  (`run_heldout_label_transfer_pilot.py`, k-NN majority vote, k=20) to assign
  held-out images onto the frozen train Leiden vocabulary. This is fully
  validated: it **improves** both held-out AMIs slightly over independent
  re-clustering (emotion 0.1249→0.1364, genre 0.2404→0.2530) and gives every
  held-out point a label in the same vocabulary train communities use — the
  one structural blocker standing between this Stage 1 and PercepT's Stage 2
  pattern is now resolved.
- **Stage 2 wiring, done and validated** (`buddy_stage2_pilot_report.md`,
  §6a): PercepT's own unmodified image-only attention-pooling mapper,
  retrained against buddy's frozen topics instead of DEC's, reaches macro
  AUC **0.5978** — beating its own train-marginal baseline by +0.098.
  **CORRECTED 2026-09-27** (see §6b): the original comparator, PercepT's
  own cited Stage 2 result of 0.5690, came from a PercepT implementation
  with two verified bugs (backwards center-pruning direction, and a
  reconstruction loss ~2,816x too weak) — see
  `docs/reports/auto/percept/2026-09-27_agy_independent_percept_review.md`. With both
  fixed, PercepT's Stage 2 macro AUC is **0.5925**, not 0.5690, making
  buddy's unmodified 0.5978 a near-tie (+0.0053), not the +0.0288 margin
  originally reported. **IMPROVED 2026-09-27** (see §6c): a deep analysis
  found buddy's macro AUC was significantly held down by its three
  smallest, sparsest topics; merging them (K=19→16) and reweighting the
  Stage 2 loss by inverse topic frequency, 4-seed validated, raises
  buddy's Stage 2 macro AUC to 0.6334 (mean, std 0.0026). **IMPROVED
  FURTHER 2026-09-27** (see §6d): buddy's mapper LR/epoch count had never
  been tuned; a sweep found `lr=1e-2, epochs=400` (vs. the original
  `1e-3, 100`) raises it again to **0.8461** (4-seed mean, std 0.0001) —
  **a +0.2536 margin over PercepT's corrected 0.5925**, stated with the
  explicit caveat that only buddy's mapper was tuned this hard (matching
  this investigation's own stated priority of not chasing PercepT's best
  number, only a reliable one). **IMPROVED ONCE MORE 2026-09-27** (see
  §6f): richer multi-label Stage 2 targets (k=20 cosine-vote fractions,
  0.15 relative cutoff) raise it to **0.8534** (4-seed mean, std 0.0001)
  — **+0.2609 over PercepT's corrected 0.5925**. This number required
  catching and fixing a flawed comparison in the first attempt (an
  independent review caught it; Codex itself declined to act on the
  finding) — see §6f for the full account. **REVERSED 2026-09-28** (see
  §6g): every prior comparison above tuned buddy's mapper but left
  PercepT's at its original untuned `lr=1e-3, epochs=100` — an asymmetry
  flagged repeatedly (§6d, §6f) but never closed. Closing it (the same
  LR/epoch sweep, applied to PercepT's mapper instead) finds PercepT had
  far more headroom: its Stage 2 mapper reaches **0.9226** (4-seed mean,
  std 0.0001) at `lr=1e-2, epochs=400` — **beating buddy's tuned 0.8534
  by -0.0692.** This is not an evaluation artifact (checked: held-out
  per-topic positive counts range 36-1097, no degenerate tiny classes;
  same single-label target convention, same scoring code, reused
  directly). **This reverses the investigation's Stage 2 headline
  conclusion** — see §6g for the full account and what it does and does
  not imply for Stage 1. **FOLLOW-UP 2026-09-30** (see §6i): a
  ~2,267-trial joint Stage 1 + Stage 2 sweep, followed by a 4-seed
  stress test of its top 10 finalists, found one configuration
  (`m8x7ifx4`) that clears the sweep's gate on 4/4 seeds, with Stage 2
  macro AUC 0.9355 ± 0.0046 at 13–16 topics. That result is internal to
  a re-implemented harness. Its Stage 1 differs structurally from the
  pilots' (teacher graph, InfoNCE negatives, optimizer). Its AMIs use
  k-NN transfer, which reads ~0.01 higher on emotion than §3's method.
  It uses a different topic count and target construction, and it had a
  much larger tuning budget than PercepT got. So it is **not**
  comparable to 0.9226, to 0.8534, or to §3's "3/4 seeds". The §6g
  headline stands until a matched head-to-head is run. The stress test
  also showed that the sweep's two top-ranked configs, and 4 of its top
  6, pass the gate on only 1/4 seeds (the winner's curse near the gate).
  Among the 10 finalists, emotion AMI and Stage 2 AUC also traded off
  (r = −0.85), a frontier observation rather than a general law.
  **MATCHED HEAD-TO-HEAD 2026-09-30** (see §6k): with both systems in one
  harness at matched K (16, 40), matched labels and an equal 300-trial
  search per cell, selected on a val half and reported on a test half at 5
  seeds. *Primary (approved) comparison:* ranked by val Stage 2 AUC,
  buddy's winners lead PercepT on test (0.993 vs 0.966 at K = 16, 0.994 vs
  0.960 at K = 40; +0.027 and +0.033, p ≤ 0.005, replicated on val), but
  keep less emotion structure (independent AMI 0.060 vs 0.079 and 0.110).
  *Secondary comparison (designed after the interim leaderboard; emotion
  floor fixed on val before test):* among trials above a common emotion
  floor, no Stage 2 difference was detected at n = 5 seeds on either half
  (test Δ +0.003 [−0.013, +0.018] and −0.005 [−0.013, +0.004]). At equal
  AUC buddy's constrained winners never showed less emotion than PercepT's
  (level on test, about 0.02 more on val); a test-half genre gap in
  PercepT's favour did not replicate on val. Which comparison leads is the
  user's open choice. Either way the §6g headline ("PercepT wins Stage 2")
  does not hold under matched conditions.
  **CONFIRMATION CHECKS 2026-09-30** (see §6j): on the investigation's own
  independent re-clustering yardstick, the sweep winner clears the emotion
  bar on only 1/8 seeds (mean 0.1189). The untuned pilot baseline scores
  higher on emotion on both yardsticks (0.1230 independent, 0.1341 gate).
  The winner's gains are genre AMI and Stage 2 AUC, not emotion.

**Do not pursue the DEC-style clustering-loss hybrid further** (see §4) — it
was tried four times (literal transplant, geometry-corrected vMF kernel, a
decoupled unconstrained clustering head, and a gradient-isolated retry of
that head), each with its own diagnosed mechanism and targeted fix for the
previous attempt's failure, and failed all four times, underperforming
every simpler buddy variant. The decoupled-head attempts (3rd and 4th) were
the worst of the four — see §4, which also corrects the 3rd attempt's
original diagnosis after the 4th attempt's controlled result ruled it out.
This is a settled negative result for this architecture, not an open
question.

**Honest bottom line on "does buddy beat PercepT":** on external human-label
agreement (AMI), buddy-based Stage 1 is genuinely competitive with — not
decisively behind — our own PercepT replication, and gets there with a
simpler, un-hacked mechanism (see §3's Pareto comparison). On raw geometric
cluster separation (silhouette, the paper's own headline metric), PercepT's
autoencoder+DEC remains far ahead **even under matched sampling** (0.0416 vs
0.4973 — see §3's matched-audit row) and that gap did **not** close despite
four dedicated, well-motivated attempts tonight. But a same-night matched-
protocol audit (§3, §6) found this is not simply "PercepT has better
topics": PercepT's 67 surviving centers are severely occupancy-collapsed on
held-out data (21/67 get **zero** held-out points at all, 50/67 are below
1% occupancy, median center holds only 13 of 9,365 points), while buddy's
19 communities are all populated and well-balanced (0 empty, only 3/19
below 1%, median 523 points). A handful of PercepT's centers absorb most of
the held-out data; its high silhouette is consistent with scoring a small
number of coarse, imbalanced macro-clusters rather than genuinely
fine-grained, well-populated topics. Report this as a real, unresolved
trade-off, now with much better evidence about *why* it exists — not as
"buddy wins" and not as "PercepT's number is fake." A same-night follow-up
downstream-probe pilot (§3, §6) then tested the natural next hypothesis —
that occupancy collapse must mean less usable topics — **and did not
confirm it**: on the one human-label outcome with real signal (genre), a
matched image-only classifier retains slightly *more* predictive
information from PercepT's collapsed topic space than from buddy's
balanced one (87.5% vs. 77.2% of a raw-features control's signal); on the
other (emotion) the two systems are statistically tied. So: silhouette
favors PercepT, occupancy/balance favors buddy, and downstream predictive
utility is a wash-to-slight-PercepT-edge on the one real signal available
— three different axes, three different (or mixed) answers. This is the
most honest summary this investigation can currently offer: there is no
single axis on which one system cleanly wins, and "PercepT wins on
silhouette" should not be read as "PercepT produces less useful topics
than its occupancy numbers might suggest" — the downstream evidence points
the other way.

---

## 1. What this investigation covers

Following from an earlier verification pass on PercepT's own paper (arXiv
2606.03345 — full text confirmed via direct reading, not summary) and this
project's own PercepT-replication pilots, tonight's work pursued a different,
explicitly-requested goal: **not** further tuning of the PercepT replication
itself, but building and evaluating a Stage-1 alternative built on this
project's own "buddy graph" (cross-modal mutual-kNN over CLIP image/text
embeddings; `src/conditional_buddy/buddy_graph.py`), benchmarked directly
against PercepT as the baseline to beat. All new work in this report lives in
`src/test/20260923_artelingo_buddy_analysis/`.

## 2. Reference numbers used throughout

| Result | Held-out emotion AMI | Held-out genre AMI | Held-out silhouette | Robustness |
|---|---:|---:|---:|---|
| PercepT-replication standing balance-hack (K=60/40, invented balance term) | 0.1252 | 0.2486 | never measured | 4/4 seeds |
| PercepT-replication faithful recipe (paper's own noise+LR mechanisms, K=100/67, no balance term) | 0.1092 | 0.3288 | **0.5120** | single seed, narrowly misses emotion bar |
| **Attention-h1 baseline, newly seed-stressed tonight** | 0.1241 (mean) | 0.2406 (mean) | 0.0397 (mean) | **3/4 seeds** (new finding — never tested before tonight) |

Held-out Pareto bar used throughout this whole investigation: emotion AMI >
0.1236 **and** genre AMI > 0.1954, simultaneously.

## 3. Tonight's five new buddy-side experiments

All experiments below start from Attention-h1's architecture and change
exactly one identified, previously-untested mechanism at a time. Full
training/collapse trajectories, per-seed numbers, and exact methodology are
in each experiment's own report file (paths given below); this table is a
summary, not a replacement for reading them.

| Experiment | Report | Held-out emotion AMI | Held-out genre AMI | Held-out silhouette | Verdict |
|---|---|---:|---:|---:|---|
| **Attention-h1 baseline, 4-seed stress** (first time ever tested) | `attention_h1_baseline_seed_stress_pilot_report.md` | 0.1241 mean (0.1213–0.1264) | 0.2406 mean (0.2289–0.2544) | 0.0397 mean (0.0375–0.0438) | 3/4 seeds clear the Pareto bar — the baseline was already solid |
| Cosine LR schedule alone (no noise) | `attention_h1_noise_schedule_pilot_report.md` | 0.1276 mean | 0.2396 mean | 0.0466 mean | 3/4 seeds clear; silhouette up in 4/4 **paired** seeds vs. baseline, AMI not reliably better |
| + Leiden pseudo-contrastive third loss (schedule + earlier pseudo-label idea combined), 4-seed | `attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md` + `attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md` | 0.1160 mean (0.1107–0.1210) | **0.3011 mean (0.2623–0.3487) — best genre AMI of any buddy variant** | 0.0739 mean (0.0692–0.0789) — second-best silhouette | **0/4 seeds clear the bar** — a real, stable trade-off (strong genre+silhouette, weak emotion), not a seed-42 fluke |
| DEC-style clustering loss, Euclidean/Student's-t (literal PercepT mechanism transplant) | `attention_h1_dec_hybrid_pilot_report.md` | 0.1160 mean | 0.1321 mean | **-0.0288 mean (negative)** | Decisive failure, 4/4 seeds — geometry mismatch (unconstrained-latent kernel on a unit-hypersphere embedding) |
| DEC-style clustering loss, cosine/vMF-style (geometry-corrected retry) | `attention_h1_vmf_dec_hybrid_pilot_report.md` | 0.1215 mean | 0.1504 mean | 0.0298 mean (**positive** — sign fixed) | Confirms the geometry diagnosis but still underperforms every simpler buddy variant; 0/4 seeds clear the bar |
| DEC-style clustering loss, decoupled unconstrained clustering head (structural fix attempt) | `attention_h1_decoupled_cluster_head_pilot_report.md` | 0.0845 mean | 0.0423 mean | -0.1568 mean | Decisive failure, 4/4 seeds — initially attributed to un-detached gradient leakage; see next row for why that diagnosis was wrong |
| DEC-style clustering loss, decoupled head with detached gradient path (isolates the leakage variable) | `attention_h1_decoupled_cluster_head_detached_pilot_report.md` | 0.0828 mean | 0.0481 mean | **-0.1529 mean (statistically unchanged from un-detached, +0.0039)** | Decisive failure, 4/4 seeds, **and it falsifies the gradient-leakage diagnosis** — the cluster head's own fully-isolated latent still only reaches 0.0018 mean silhouette, so the failure is not InfoNCE interference at all |

Also completed: **`heldout_label_transfer_pilot_report.md`** — the Stage-2
shared-vocabulary blocker (train and held-out previously used unrelated
Leiden label spaces) is now solved via k-NN transfer (k=20 matches this
project's own buddy-graph convention), with a validated, reusable
`assign_to_train_communities` function.

**Also completed (after this report was first written):
`buddy_percept_matched_silhouette_audit_pilot_report.md`** — the matched-
protocol silhouette/occupancy audit recommended by the brainstorm memo's
top-ranked candidate (§6). Both systems' deployable held-out labels (buddy:
k=20 transfer onto 19 train communities; PercepT: assignment to its 67
surviving train-fitted DEC centers) were scored under an identical
sampling protocol (same seed-42 6,000-point draw, same `silhouette_score`
call, matched held-out painting population verified by exact set
equality). Result: PercepT's silhouette advantage is real, not a sampling
artifact (0.4973 vs. buddy's 0.0416 under matched sampling) — but PercepT's
own centers are severely occupancy-collapsed on held-out data (21/67 get
zero held-out points, 50/67 are below 1% occupancy, median center holds 13
of 9,365 points), while buddy's 19 communities are all populated and
reasonably balanced (0 empty, 3/19 below 1%, median 523 points). This
strongly supports the memo's hypothesis that part of PercepT's advantage
reflects a small number of coarse, imbalanced macro-clusters rather than
uniformly better topic quality. One methodological caveat, reported
transparently: refitting the faithful recipe for this audit did not
exactly reproduce the original published numbers (fresh: emotion AMI
0.1077/genre AMI 0.3453/full-split silhouette 0.4840, vs. published
0.1092/0.3288/0.5120) — the pipeline was never pinned for bit-exact
determinism, so some run-to-run drift is expected; the qualitative picture
(very high silhouette, genre-strong/emotion-weak, still missing the AMI
bar) is unchanged.

## 4. Why the DEC-loss hybrid direction was tried, and why it's now closed

The comprehensive lineage research (read in full before any new pilot was
designed tonight; see `buddy_topic_formation_status_memo.md` in this
session's scratchpad, or regenerate the same reading pass if that file is no
longer available) established that **no prior buddy/attention pilot had ever
used DEC's actual continuous self-sharpening clustering loss** — only a
discrete, one-shot Leiden partition (used either post-hoc or as a pseudo-
label InfoNCE signal). Since DEC's own loss is the literal mechanism behind
PercepT's 0.97-silhouette claim, and behind our faithful-recipe replication's
0.51, transplanting it directly onto the buddy encoder was the most
principled, best-motivated remaining idea — not a guess.

It failed, decisively and reproducibly (4/4 seeds, negative silhouette). The
diagnosis: Attention-h1's embedding is LayerNorm'd and L2-normalized onto a
32-D unit hypersphere; PercepT's own latent is a free, unconstrained 128-D
Euclidean space. Cramming 100 K-means centers and a Euclidean/Student's-t
self-sharpening kernel onto a low-dimensional sphere, while InfoNCE
simultaneously reshapes the same coordinates for a different objective,
produced exactly the kind of pathological interleaving a negative silhouette
measures.

A geometry-corrected retry (cosine-similarity/von-Mises-Fisher-style kernel,
matching the embedding's actual geometry) **did** fix the sign — silhouette
became positive in 4/4 seeds — which is real, useful confirmation that the
diagnosis was correct. But it still underperformed every simpler buddy
variant on every axis and cleared the AMI Pareto bar in 0/4 seeds.

A third attempt tested the structural fix flagged after the second: a
separate, unconstrained (non-unit-norm) `ClusterHead` MLP on top of the
fused embedding, so DEC's Student's-t kernel gets genuine Euclidean spread
to work with instead of a bent kernel on a sphere
(`attention_h1_decoupled_cluster_head_pilot_report.md`). This was the worst
result of the three, not the best: four-seed mean held-out fused silhouette
-0.1568 (versus -0.0288 for the first attempt and +0.0303 for the second),
genre AMI collapsed to 0.0423 (mean), and even the clustering head's own
latent space — the space the new loss actually optimized — only reached a
near-zero mean silhouette (-0.0062). The pilot's own diagnosis at the time
was gradient leakage: `cluster_head`'s input was never detached, so DEC's
gradient could flow back through the shared trunk and reshape the
InfoNCE-optimized embedding.

**A fourth attempt tested that diagnosis directly** by detaching the head's
input (`attention_h1_decoupled_cluster_head_detached_pilot_report.md`) —
and it **falsified the diagnosis**. Four-seed mean held-out fused
silhouette was -0.1529, statistically unchanged from the un-detached
version's -0.1568 (a +0.0039 difference, noise-level). More tellingly, the
cluster head's own latent space — now fully protected from any InfoNCE
influence, gradient or otherwise — still only reached a 0.0018 mean
silhouette, nowhere near separable. Gradient leakage was not the problem;
the DEC loss was never meaningfully separating anything in the first place,
detached or not.

The better-supported explanation, visible in hindsight: PercepT's own DEC
loss is never run alone. Its recipe (verified via full-text reading at the
start of this investigation) jointly trains the self-sharpening KL loss
*with a fixed-weight (λ_R=1) reconstruction loss*, and this project's
faithful-recipe replication reused both terms together. DEC's KL loss is
known (independent of this project) to be prone to degenerate, uninformative
solutions without an anchoring signal like reconstruction — self-sharpening
alone has no pressure to preserve any real structure, only to make whatever
partition it currently has more confident. This project's `ClusterHead` was
trained with the KL loss **alone**, no reconstruction term, in both the
third and fourth attempts — so the near-zero cluster-space silhouette in
both is not a bug, it is DEC behaving exactly as the broader literature
would predict without its anchor. This reframes the whole DEC-hybrid
direction: the missing piece was never the embedding's geometry (kernel
choice) or the gradient path (detaching) — it was the reconstruction
objective itself, which no attempt tonight included. Four attempts at
variations on "attach a DEC-style loss to Attention-h1's own embedding"
have now failed, each with a more specific diagnosis than the last, and the
fourth attempt's role was specifically to rule out its predecessor's
explanation — which it did. A fifth attempt (add a real reconstruction loss
to the cluster head, i.e., make it a proper autoencoder matching PercepT's
actual recipe rather than borrowing only its clustering half) is a bigger
architecture change than anything tried tonight and is flagged in §6 as an
explicitly *undiscussed* option — four consecutive negative results on the
same underlying idea is the point to check in with a person, not keep
iterating alone.

## 5. What was not needed tonight

- The DAS6 cluster (node403, confirmed reserved and ready) was not used.
  Every pilot tonight fit comfortably on this container's local GPU (each
  full run: a few minutes; the largest pilot, 9 runs, well under an hour).
  This project's ArtELingo pilots use ad-hoc absolute data paths
  (`/data/SSD2/pre_extract/artelingo*`, `/data/PDD/artelingo/*`, ~1.5GB
  total) that sit outside the `cluster-run` skill's `_cluster.yaml`
  data-sync convention — porting them would have cost more setup time than
  any single pilot tonight, so it was deferred rather than forced. If a
  genuinely large-scale run is wanted later (e.g. the eventual Stage 2
  image-classifier retraining, which consumes much larger `[N,50,512]`
  patch tensors), that is a much better-justified reason to invest in
  syncing this data to a node.
- CoSiR's own trainable label-embedding retraining pipeline was explicitly
  out of scope tonight, per direct instruction: the goal was PercepT vs.
  buddy on PercepT's own topic-formation task, not CoSiR's retrieval
  benchmark.

## 6. Follow-up candidates from the brainstorm memo (all five now run)

**See also**: `2026-09-26_buddy_silhouette_gap_brainstorm.md`, a Codex-produced
research memo ranking five candidate next directions, requested after this
report was first written. Its top-ranked candidate (a matched-protocol
silhouette/occupancy audit) **has now been run** — see §3's new entry and
`buddy_percept_matched_silhouette_audit_pilot_report.md`. That audit
confirmed PercepT's silhouette lead is real under matched sampling, but
also confirmed the memo's collapse hypothesis with hard numbers: PercepT's
held-out centers are severely occupancy-collapsed (21/67 empty, 50/67
below 1%) while buddy's are not (0 empty, 3/19 below 1%). **Candidate 1's second half has now also been run** —
`buddy_percept_downstream_probe_pilot_report.md` — and its result is the
most surprising finding of the night: it does **not** confirm the
occupancy story. Two matched image-only classifiers (patch features →
frozen train topic, single-label cross-entropy, otherwise identical
architecture/training budget) were trained per system, then each system's
held-out topic-softmax prediction was fed into the same fixed-capacity
logistic-regression probe against real held-out human labels (emotion:
50/50 split, n=9,365; genre: 5-fold CV, n=159). Result: **on genre — the
one outcome with a real signal — PercepT's occupancy-collapsed topic
bottleneck is slightly *more* informative than buddy's balanced one**
(genre AMI 0.2973 vs. 0.2625, retaining 87.5% vs. 77.2% of a raw-features
control's signal), not less. On emotion, both bottlenecks lose most of the
signal a raw-features control captures and are statistically tied with
each other (AMI 0.0221 vs. 0.0231 — a coin flip). A raw mean-pooled-patch
control (no topic bottleneck at all) beats both systems on every metric,
which is expected (any discrete bottleneck is lossy) but confirms the
comparison isn't measurement noise. **This means severe occupancy
collapse does not straightforwardly imply worse downstream utility** —
this single seed/single-split result should not be over-read, but it is a
real, honest complication of the matched audit's own framing, not a
confirmation of it. **Candidate 2 (buddy-embedding-as-PercepT-input) has now also been run**
— `percept_on_buddy_embedding_pilot_report.md` — and it is a decisive
failure, distinct from every prior negative result tonight. Feeding
buddy's frozen 32-D embedding into PercepT's own unmodified autoencoder+DEC
recipe misses the Pareto bar (genre AMI 0.1765 vs. the 0.1954 bar),
collapses worse than anything else measured tonight (44/67 held-out
centers get **zero** points, 57/67 below 1%), and — most tellingly — the
resulting DEC partition scores **negative** silhouette (-0.0252) when
evaluated back in buddy's own native 32-D space, even though it scores a
deceptively high 0.4886 in PercepT's own 128-D latent. The loss-scale audit
explains why: the reconstruction/KL ratio shifted 433× from the original
recipe, because a 32-D input into an architecture built for 2,816-D makes
the 128-D latent **overcomplete** (4× wider than the input) — the decoder
can reconstruct almost anything without the encoder needing to find
well-separated clusters, so reconstruction pressure does not anchor
against collapse here the way it seemed to elsewhere tonight. This is
closed on a single seed; the failure is decisive enough across every axis
that a multi-seed stress would not change the verdict.

**Candidate 3 (`D_SHARED` capacity sweep) has now also been run** —
`dshared_capacity_sweep_pilot_report.md` — and confirms the memo's own
prior expectation: capacity is not a binding constraint. Widening
D_SHARED from 32 to 64 or 128 (module-attribute override only, zero edits
to the shared architecture file), with no other change, made things worse,
not better: emotion AMI fell monotonically (0.1249→0.1153→0.1089),
silhouette fell monotonically (0.0377→0.0367→0.0250, i.e. **wider capacity
produced a lower silhouette**), while genre AMI rose (0.2404→0.2918→0.3319)
— a real trade-off, but neither width cleared the AMI Pareto bar or came
close to the predeclared meaningful-silhouette-gain threshold, so neither
was stressed further. Effective rank at 95% variance did grow with width
(20→33→46), confirming the model uses the extra capacity when given it —
it just doesn't spend it on better held-out cluster separation.

**Candidate 4 (a properly reconstruction-anchored cluster head) has now
also been run** — `reconstruction_anchored_cluster_head_pilot_report.md`
— and it closes the entire DEC-hybrid line decisively. This attempt was
deliberately designed to control for every failure mode diagnosed in the
five attempts before it: a genuine reconstruction anchor (unlike the
decoupled-head attempts), an undercomplete 16-D cluster latent (unlike
PercepT-on-buddy-embedding's overcomplete 128-D one), and a matched
input/output scale of 32-D on both sides (avoiding that same pilot's
433× loss-scale distortion). Even with all three fixes applied at once,
it failed just as decisively as everything before it: four-seed mean
held-out fused silhouette -0.0793 (negative, worse than the best prior
DEC-hybrid attempt's +0.0298), both AMI Pareto bars clearing in **0/4**
seeds, and even the cluster's own protected latent space barely reaching
zero (mean 0.0033) — no meaningful separation anywhere, in any space,
under any of the six mechanisms tried tonight.

**This closes the DEC-hybrid research direction.** Six attempts —
Euclidean DEC, vMF-corrected DEC, a decoupled cluster head (un-detached
and detached), PercepT's own autoencoder fed buddy's embedding, and a
properly reconstruction-anchored undercomplete cluster head — have now
tested every mechanism the brainstorm memo identified for attaching a
DEC-style clustering objective to Attention-h1's embedding, and every one
has failed. This is a well-evidenced, settled negative result, not an
open question requiring a seventh attempt.

**Candidate 5 (a soft/uncertainty-preserving Stage 2 target) has now also
been run** — `soft_stage2_target_pilot_report.md` — completing the full
brainstorm memo sweep. Two alternatives to the hard one-hot Stage 2 target
were tested: a k=20 self-referential vote-frequency distribution (soft
cross-entropy) and a sparse multi-hot threshold of that same distribution
(mirroring PercepT's own "at least 2 votes" convention). Neither is a
clear win: the soft target ties the hard label (2/4 metrics better, tiny
differences — emotion AMI +0.0008, genre accuracy identical to 4 decimals),
and the sparse multi-hot target is worse on all 4 metrics (emotion AMI
0.0077 vs. 0.0231, genre AMI 0.2430 vs. 0.2625). Softening the Stage 2
target is not a high-value lever on this evidence — consistent with the
downstream probe's own earlier finding that buddy's hard labels already
retain most of the available signal, so there was little uncertainty left
for a softer target to recover.

**All five brainstorm candidates have now been run.** None reversed the
recommended solution; two (the matched audit and the downstream probe)
substantially refined the honest framing of what "PercepT wins on
silhouette" actually means, and the DEC-hybrid line (candidates driving
six total attempts) is now a settled, well-evidenced negative result
rather than an open question.

## 6a. Stage 2 wiring — done, and buddy's full pipeline beats PercepT's

The one remaining open item from §6 (Stage 2 wiring) has now also been
completed — `buddy_stage2_pilot_report.md`. Buddy's frozen train Leiden
communities (19-way) plus held-out k=20 transfer feed PercepT's own
unmodified `AttentionPoolingMapper` architecture, trained with the same
`BCEWithLogitsLoss`-against-one-hot-targets convention PercepT's own Stage
2 pilot uses (buddy's targets are naturally single-label; PercepT's own
report shows its multi-hot targets also collapsed to single-label in
practice, 0.00% multi-labeled, so this is a structurally faithful
comparison). Result: **buddy's image-only Stage 2 mapper reaches macro AUC
0.5978 (min 0.2541, median 0.5605, max 0.9043), beating its own
train-marginal baseline by +0.0978 (well past the 0.01 practical margin)
and beating PercepT's own cited Stage 2 macro AUC of 0.5690 by +0.0288.**
This is the first genuine end-to-end (Stage 1 + Stage 2) head-to-head
between the two systems, not just a Stage-1-in-isolation comparison, and
buddy's full pipeline wins it. This closes out the investigation's
concrete deliverable: a working, validated, better-performing buddy-based
replacement for PercepT's entire two-stage topic-formation pipeline.

1. ~~Stage 2 wiring — the one item still genuinely open...~~ **Done, see
   §6a**: buddy's own Stage 2 mapper (macro AUC 0.5978) beats both its own
   baseline and PercepT's cited Stage 2 result (0.5690). No further open
   items remain from this investigation's own scope.
2. ~~The combined "schedule + pseudo-contrastive" pilot was only run at a
   single seed...~~ **Resolved tonight**: 4-seed stress
   (`attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md`)
   confirms this is a genuine, stable trade-off, not a seed-42 fluke — 0/4
   seeds clear the bar, held by a consistently weak emotion AMI (0.111–0.121,
   all below the 0.1236 bar) alongside the best genre AMI (0.30 mean) and
   second-best silhouette (0.074 mean) of any buddy variant tried. If genre-
   AMI/silhouette quality is ever weighted above emotion-AMI Pareto
   clearance for a future use case, this is the configuration to revisit;
   as scored by this investigation's own predeclared bar, it is a plain
   miss.

## 6b. Independent review found two real PercepT bugs — corrected, and the margin shrinks

**2026-09-27.** An independent adversarial review (agy) audited the PercepT
implementation and this report's own comparisons. Its two most severe
findings were verified directly against the actual PercepT paper
(arXiv:2606.03345, fetched and quoted) rather than taken on trust — full
derivation in
[`docs/reports/auto/percept/2026-09-27_agy_independent_percept_review.md`](2026-09-27_agy_independent_percept_review.md):

1. **`prune_centers` kept the wrong centers.** The paper's Algorithm 1
   (§4.3) states that underused DEC centers drift to *large* norms during
   training and should be *discarded* on that basis. Every copy of
   `prune_centers` in this codebase (`run_percept_stage1_pilot.py`,
   `run_percept_stage1_faithful_recipe_pilot.py`,
   `run_percept_stage1_cluster_count_sweep_pilot.py`, and both DEC-hybrid
   buddy-side attempts in §4) sorted by norm **descending** and kept the
   *highest*-norm centers — the opposite of the paper's rule, and a direct,
   mechanistic explanation for the severe occupancy collapse cited
   throughout this report.
2. **Reconstruction loss was ~2,816x too weak.** The paper defines
   `L_R = ||h - h_hat||^2` (an unreduced per-sample squared error). Every
   copy used `F.mse_loss(reconstruction, inputs)` with PyTorch's default
   `reduction='mean'`, which additionally divides by the 2,816-D feature
   dimension. At `lambda_R=1`, the reconstruction anchor was therefore
   applying roughly 1/2816th of its intended weight relative to the KL
   term.

**Re-run with both bugs fixed** (new files, originals left unmodified for
provenance):

- **Stage 1, faithful recipe (K=100/67):**
  [`run_percept_stage1_faithful_recipe_fixed_pilot.py`](../../src/test/20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_fixed_pilot.py) /
  [`percept_stage1_faithful_recipe_fixed_pilot_report.md`](pilots/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_fixed_pilot_report.md).
  Held-out (seed 42, Variant A): emotion AMI 0.1097 (was 0.1092), genre AMI
  0.3764 (was 0.3288), silhouette **0.2224 (was 0.5120 — a large drop)**,
  occupancy 45/67 below 1% (was 50/67). **Still does not clear the Pareto
  bar and is still collapsed** — the core Stage 1 conclusion (buddy clears
  the bar, PercepT's faithful recipe does not) survives the fix, but
  PercepT's silhouette "lead" is much smaller than previously reported
  (0.22 vs buddy's ~0.04, not 0.51 vs 0.04) and its genre AMI lead over
  buddy widened (0.3764 vs buddy's 0.2404).
- **Stage 2 (K=60/40), the actual headline comparator:**
  [`run_percept_stage2_fixed_pilot.py`](../../src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py) /
  [`percept_stage2_fixed_pilot_report.md`](pilots/20260922_percept_topic_pipeline/percept_stage2_fixed_pilot_report.md).
  Fixed PercepT Stage 2 macro AUC: **0.5925** (was 0.5690), same single-
  label target convention as buddy's own 0.5978, so this is the correct
  apples-to-apples number. **Buddy still wins, 0.5978 vs 0.5925 — but by
  +0.0053, not +0.0288.** This is a near-tie, not the clear win originally
  reported.
- **Side finding, single-seed only, not yet stress-tested:** the same
  fixed Stage-1 K=60/40 re-fit that feeds the Stage 2 number above,
  *with* the existing `LAMBDA_BALANCE=1000` hack still active, gets held-
  out emotion AMI 0.1094 — under the 0.1236 Pareto bar. The *original*,
  buggy K=60/40+hack configuration was this report's only PercepT
  configuration ever cited as clearing the bar (`STANDING_EMOTION_MEAN
  0.1252`, 4/4 seeds, per §"Key advantages"). Whether that "one win"
  survives the fix under a proper 4-seed stress test is not yet known —
  this is one seed, and is flagged here rather than asserted.

Also addressed: agy flagged that a materially better PercepT Stage 2
result (macro AUC 0.8290, 14-seed mean) existed in this repo
(`percept_stage2_extended_seed_pilot_report.md`) and was never cited
alongside the 0.5690/0.5925 numbers above. That number is **not**
comparable to buddy's 0.5978 — it uses a deliberately richer multi-label
target threshold (76.9% multi-labeled) rather than the single-label
convention both buddy and the corrected 0.5925 use — but the master
report should have surfaced and explained this instead of silently
omitting it, which is corrected here.

**2026-09-28 update — now re-examined for the one attempt where it was
cheap to test.** Of the ~6 DEC-hybrid attempts in §4, only the
reconstruction-anchored cluster head (the most recent, frozen-buddy-trunk
attempt) has both a genuine reconstruction loss term and a cheap re-run
path (no InfoNCE retraining needed). Re-run with both bugs fixed
([`run_reconstruction_anchored_cluster_head_fixed_pilot.py`](../../src/test/20260927_deep_stage_analysis/run_reconstruction_anchored_cluster_head_fixed_pilot.py) /
[`reconstruction_anchored_cluster_head_fixed_pilot_report.md`](pilots/20260927_deep_stage_analysis/reconstruction_anchored_cluster_head_fixed_pilot_report.md)):
4-seed mean held-out fused silhouette improved from -0.0793 to -0.0117,
and emotion/genre AMI both improved slightly (0.1122→0.1149,
0.1099→0.1150) — but the held-out Pareto bar still clears in 0/4 seeds,
unchanged from the original buggy run. **The two bug fixes measurably
help but do not overturn this attempt's negative conclusion.** This is
consistent with the §4 diagnosis that the failure mode is the embedding
geometry / DEC self-sharpening dynamics, not the specific center-selection
or loss-scale artifacts. The other ~5 DEC-hybrid attempts (Euclidean,
vMF, decoupled un-detached/detached, PercepT-on-buddy-embedding) train no
reconstruction term at all or would require full InfoNCE retraining to
re-test the `prune_centers` bug in isolation (it only affects final center
selection, not training dynamics) — those remain untested against this
fix and are not claimed to be settled by this result, only the one
attempt actually re-run.

## 6c. Buddy's Stage 2 margin restored and widened — minimum-occupancy handling

**2026-09-27.** A deep Stage 1/2 analysis
([`src/test/20260927_deep_stage_analysis/deep_stage_analysis_report.md`](pilots/20260927_deep_stage_analysis/deep_stage_analysis_report.md))
found buddy's Stage 2 macro AUC is significantly correlated with per-topic
occupancy (Pearson r=0.531, p=0.019, n=19) — its three smallest held-out
topics (87/69/65 paintings) were dragging the average down, and buddy's
top-1 (exact-match) accuracy trailed PercepT's on 7 of 9 emotion
categories despite the AUC near-tie from §6b.

Two independent fixes were tested and 4-seed validated
([`src/test/20260927_deep_stage_analysis/candidate1_min_occupancy_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate1_min_occupancy_pilot_report.md),
[`candidate1_stress_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate1_stress_pilot_report.md),
[`candidate1_variant_c_stress_report.md`](pilots/20260927_deep_stage_analysis/candidate1_variant_c_stress_report.md)):

| configuration | K | macro AUC (4-seed mean ± std) | top-1 acc (4-seed mean) |
|---|---:|---:|---:|
| baseline (unmodified) | 19 | 0.5978 (single seed) | 0.173 (single seed) |
| A: class-balanced loss | 19 | 0.6116 ± 0.0053 | 0.198 |
| B: merge 3 smallest topics | 16 | 0.6218 ± 0.0027 | 0.167 |
| **C: merge + class-balanced (adopted)** | **16** | **0.6334 ± 0.0026** | **0.250** |

Variant C — the three smallest train communities merged into their
nearest larger neighbor by centroid cosine similarity (K: 19→16), *plus*
per-sample loss weighting inversely proportional to train-topic
frequency — combines both mechanisms and is the strongest and most
consistent across all four stress seeds (42/7/123/2024, std 0.0026).
It also closes most of the top-1-accuracy gap flagged in the deep
analysis (0.173→0.250, vs. PercepT's un-re-tested per-emotion accuracy
in that same report).

**Updated headline Stage 2 comparison (superseded again by §6d): buddy
0.6334 vs. PercepT's corrected 0.5925 — a +0.0409 margin**, restoring
(and exceeding) a clear, robust win over the §6b near-tie. This required
no change to buddy's Stage 1 embedding or graph — only Stage 2's target
vocabulary and loss weighting, using buddy's own already-frozen topic
structure.

## 6d. Buddy's Stage 2 mapper was badly under-tuned — LR/epoch sweep

**2026-09-27, same session.** Candidate 2 from the deep analysis's
priority list (mapper LR/capacity sweep) tested buddy's Stage 2 mapper's
learning rate and epoch count, on top of §6c's adopted merge+reweighting
configuration — never previously tuned; buddy had always used PercepT's
original `lr=1e-3, epochs=100` defaults unchanged
([`src/test/20260927_deep_stage_analysis/candidate2_mapper_sweep_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate2_mapper_sweep_pilot_report.md)):

| lr | epochs | macro AUC (seed 42) |
|---:|---:|---:|
| 3e-4 | 100 | 0.5441 |
| 1e-3 (previous default) | 100 | 0.6372 |
| 3e-3 | 100 | 0.7751 |
| **1e-2** | 100 | 0.8147 |
| 1e-2 | 200 | 0.8343 |
| **1e-2** | **400** | **0.8460** |

4-seed stress at the winning `lr=1e-2, epochs=400`: **mean 0.8461** (min
0.8460, max 0.8463, std 0.0001 — essentially seed-independent, consistent
with a small, low-capacity model converging to the same basin regardless
of init under full-batch training at this scale). Sanity-checked: zero
skipped topics at any sweep point, and the improvement is a smooth,
monotonic curve across the LR/epoch grid (not a discontinuous jump),
consistent with a genuinely under-optimized mapper being fixed, not an
evaluation artifact.

**This exceeds even PercepT's own best-ever tuned Stage 2 result (0.8290,
14-seed mean, from its own richer-multi-label-threshold LR sweep,
`percept_stage2_extended_seed_pilot_report.md`)** — though that PercepT
number uses a different (richer multi-label) target convention, so is not
directly comparable; the correct apples-to-apples comparator remains the
single-label-convention 0.5925.

**Final headline Stage 2 comparison: buddy 0.8461 vs. PercepT's corrected,
single-label 0.5925 — a +0.2536 margin.** Framing note, stated plainly:
per this investigation's explicit priority (improving PercepT was never
the goal, only ensuring its number is reliable), only buddy's mapper was
hyperparameter-tuned here; PercepT's corrected mapper still uses its
original, untuned `lr=1e-3, epochs=100`. This is an intentional asymmetry
matching the investigation's stated goals, not an oversight — but it
should be stated every time this margin is cited, since a symmetrically
tuned PercepT mapper was not tested and might also improve from its
0.5925.

## 6e. Topic-count (K) sweep — confirms the current configuration, no further gain

**2026-09-27, same session.** Candidate 3 tested whether a different
buddy topic count beats the adopted K=19 (merged to 16 for Stage 2).
Re-clustering the already-trained, frozen embedding at several Leiden
resolutions requires no InfoNCE retraining
([`src/test/20260927_deep_stage_analysis/candidate3_k_sweep_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate3_k_sweep_pilot_report.md)):

| merged K | emotion AMI | genre AMI | clears Pareto bar | Stage 2 macro AUC |
|---:|---:|---:|---|---:|
| 33 | 0.1282 | 0.1855 | no | 0.8568 |
| 17 | 0.1296 | 0.1997 | **yes** | 0.8406 |
| 9 | 0.1142 | 0.2555 | no | 0.8826 |
| 4 | 0.0834 | 0.1863 | no | 0.9078 |

The trade-off is clean and expected: coarser K (fewer, larger topics)
trivially raises Stage 2 macro AUC by making the classification task
easier, at the direct cost of Stage 1 topic quality — K=4 nearly matches
random-genre-level granularity and its AMIs collapse accordingly. The
only other bar-clearing K tested (17) scores lower on Stage 2 AUC (0.8406)
than the adopted K=16 (0.8461). **No tested K beats the current
configuration while also clearing the Pareto bar** — candidate 3 is a
clean negative result that confirms, rather than changes, the adopted
K=19/16 configuration.

## 6f. Richer multi-label targets — a real gain, found only after catching a flawed comparison

**2026-09-27, same session.** Candidate 4 tested richer multi-label
Stage 2 targets (k=20 cosine-vote fractions thresholded at three relative
cutoffs), on top of the candidate 1+2 configuration (0.8461 baseline).
The first attempt at this pilot
([`candidate4_rich_multilabel_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate4_rich_multilabel_pilot_report.md))
concluded negatively — all three cutoffs (0.50/0.30/0.15) scored below
0.8461 (0.8394/0.8322/0.8188) — **but this compared each cutoff's AUC
against that cutoff's own multi-label held-out targets, not against the
same single-label targets the 0.8461 baseline was scored against**. An
independent review (dispatched by Codex itself, not requested by this
session) caught this as a Critical finding: the exact same class of
apples-to-oranges comparison error already flagged in PercepT's own
multi-hot-threshold numbers (§6b). Codex declined to fix it, reasoning
the brief only asked for the stated comparison; that reasoning was
rejected here and the comparison was redone properly.

Re-scoring each cutoff's already-trained model against the baseline's own
single-label held-out targets
([`candidate4_fixed_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate4_fixed_pilot_report.md),
[`candidate4_fixed_stress_report.md`](pilots/20260927_deep_stage_analysis/candidate4_fixed_stress_report.md))
reverses the verdict entirely:

| cutoff | own-target macro AUC (flawed) | vs-baseline-target macro AUC (correct) |
|---:|---:|---:|
| 0.50 | 0.8394 | 0.8504 |
| 0.30 | 0.8322 | 0.8521 |
| 0.15 | 0.8188 | **0.8533** (seed 42) |

4-seed stress of the winning cutoff (0.15): **mean 0.8534** (min 0.8533,
max 0.8535, std 0.0001) — a robust **+0.0073** over the 0.8461 baseline,
consistent across all four seeds. A companion control (single-label
targets, unweighted loss, isolating the effect of dropping candidate 1's
class-balanced weighting alone) scored 0.8476, confirming most of the
+0.0073 gain is genuinely attributable to the richer multi-label targets,
not merely to the weighting change.

**Updated headline Stage 2 comparison: buddy 0.8534 vs. PercepT's
corrected, single-label 0.5925 — a +0.2609 margin** (same asymmetric-
tuning caveat as §6d applies: PercepT's mapper was not retuned to match).
**Candidate 4 is adopted**, superseding §6e's K=16-without-rich-targets
number as the new headline buddy Stage 2 configuration: merge the 3
smallest topics (K=19→16), threshold k=20 cosine-vote fractions at a 0.15
relative cutoff for multi-label targets, `lr=1e-2`, 400 epochs.

This episode is worth stating plainly: the flawed comparison was not
caught by this session's own first pass, nor by Codex's own dispatched
review-and-accept process — it was caught by a second, independent review
Codex itself requested, whose finding Codex then declined to act on. The
fix and the resulting real, validated gain came from treating that
declined finding as still worth investigating rather than as closed.

## 6g. Closing the asymmetric-tuning gap reverses the Stage 2 headline conclusion

**2026-09-28.** Every headline Stage 2 number from §6c onward carried the
same explicit caveat: buddy's mapper was hyperparameter-tuned (candidate
2, §6d) but PercepT's fixed mapper was left at its original, untuned
`lr=1e-3, epochs=100` — "an intentional asymmetry matching the
investigation's stated goals, not an oversight — but it should be stated
every time this margin is cited, since a symmetrically tuned PercepT
mapper was not tested and might also improve from its 0.5925" (§6d, and
repeated in §6f). This gap is now closed.

[`run_percept_mapper_symmetric_sweep_pilot.py`](../../src/test/20260927_deep_stage_analysis/run_percept_mapper_symmetric_sweep_pilot.py) /
[`percept_mapper_symmetric_sweep_pilot_report.md`](pilots/20260927_deep_stage_analysis/percept_mapper_symmetric_sweep_pilot_report.md)
mirrors candidate 2's own sweep exactly (`LR_GRID=(3e-4, 1e-3, 3e-3,
1e-2)`, `EPOCH_GRID=(100, 200, 400)`, same screen-then-4-seed-stress
discipline, same reused `AttentionPoolingMapper`, `evaluate_auc`,
`auc_summary` functions), applied to PercepT's fixed Stage 2 mapper
instead of buddy's:

| lr | epochs | macro AUC (seed 42) |
|---:|---:|---:|
| 3e-4 | 100 | 0.5094 |
| 1e-3 (previous default) | 100 | 0.5843* |
| 3e-3 | 100 | 0.7464 |
| **1e-2** | 100 | 0.8692 |
| 1e-2 | 200 | 0.9000 |
| **1e-2** | **400** | **0.9226** |

4-seed stress at `lr=1e-2, epochs=400`: **mean 0.9226** (min 0.9226, max
0.9227, std 0.0001 — as seed-independent as every other full-batch
mapper-tuning result in this investigation).

**Reproducibility note (\*):** this run's own fresh Stage-1 K=60/40 re-fit
scored 0.5843 at the untuned point, not the previously-published 0.5925.
Neither this script nor the original fixed pilot sets
`torch.use_deterministic_algorithms`/`cudnn.deterministic`, so a fresh
500-epoch DEC self-sharpening re-fit is not bit-reproducible run-to-run
at the same seed — confirmed by re-running the untuned sanity point
twice, both times landing on 0.5843, so this is a stable property of
*this* re-fit, not run-to-run jitter within it. This is itself a real,
previously-undocumented finding about the fixed pipeline's stability, not
a bug in this script. The +0.3383 screening margin and the 4-seed stress
are both measured against this run's own 0.5843 baseline internally, so
the tuning effect is measured consistently regardless of which absolute
number is "correct."

**Checked for the obvious artifact before trusting this result:** with 40
topics (vs. buddy's merged 16), a natural worry is that some PercepT
topics have so few held-out positives that per-topic AUC becomes
unstable/inflatable by rote memorization under more aggressive training.
Directly checked: held-out positive counts per topic range from 36 to
1,097 (median 151.5), zero topics under 20 — comparable in scale to
buddy's own three smallest topics (65-87). Targets are single-label
throughout (0.00% multi-labeled, matching the established 0.5925/0.8461/
0.8534 convention exactly — verified directly, not assumed). This is not
a repeat of the apples-to-oranges bug class found twice already in this
investigation (§6b, §6f).

**Updated headline Stage 2 comparison: buddy 0.8534 vs.
symmetrically-tuned PercepT 0.9226 — a -0.0692 margin. PercepT now wins.**
This reverses the conclusion of §6c/§6d/§6f, which all held while the
tuning asymmetry stood.

**What this does and does not mean:**
- It does **not** change §3's Stage 1 conclusions (AMI Pareto bar,
  silhouette, occupancy) — this is a Stage 2 (patch-image → frozen-topic
  classifier) result only, and Stage 1 was never re-fit or re-tuned here.
- It does **not** mean buddy's own Stage 2 work (candidates 1/2/4) was
  wrong or wasted — buddy's 0.5978→0.8534 progression is still real,
  validated, and adopted; PercepT's mapper simply had more headroom left
  when tuned the same way.
- It **does** mean the investigation's own repeated caveat turned out to
  be load-bearing: an acknowledged-but-unclosed asymmetry was hiding a
  larger effect than any of the four rounds of buddy-side improvement
  that preceded it. The honest reading is that this specific Stage 2
  mapper task (predicting a frozen topic assignment from CLIP patch
  tokens) rewards aggressive LR/epoch tuning heavily for both systems,
  and PercepT's fixed Stage 1 partition — despite its known occupancy
  collapse (§3, §6b) — turned out to be at least as learnable from patch
  features as buddy's, once its mapper was allowed to actually converge.
- The Stage 1 topic-formation comparison (the investigation's original
  and primary subject) is unaffected: buddy still clears the Pareto bar
  without PercepT's balance-hack or occupancy collapse. What changes is
  narrower but real: **"buddy's Stage 2 classifier beats PercepT's,
  best-effort vs. best-effort" is no longer a supportable claim.**

## 6h. Upweighting affect in buddy's graph construction — a clean negative

**2026-09-28.** Candidate 6 (lowest-priority item from the deep analysis's
updated list, §5 there) tested whether upweighting the affect InfoNCE loss
term (`total_loss = content_loss + LAMBDA_AFFECT * affect_loss`, swept
against the flat `LAMBDA_AFFECT=1.0` baseline) closes the "buddy's topics
are content/genre-driven, emotion rides along weakly" gap flagged as
Finding D in the deep analysis
([`run_candidate6_affect_upweight_pilot.py`](../../src/test/20260927_deep_stage_analysis/run_candidate6_affect_upweight_pilot.py) /
[`candidate6_affect_upweight_pilot_report.md`](pilots/20260927_deep_stage_analysis/candidate6_affect_upweight_pilot_report.md)).
Sanity check reproduced the established noise-schedule baseline exactly
(0.1306/0.1973/0.0488). Screened `LAMBDA_AFFECT ∈ {1, 2, 4, 8}` at seed
42: every upweighted value made held-out genre AMI collapse (0.1973→as low
as 0.0212) while emotion AMI did not meaningfully improve (best alternative
0.1180 at `LAMBDA_AFFECT=4`, below the flat baseline's 0.1306) — no point
cleared the Pareto bar or the +0.005 practical margin, so none was
stress-tested. **Clean negative, as this candidate's own priority ranking
anticipated.** This rules out the flat equal-weight loss as the bottleneck
behind Finding D; whatever limits buddy's emotion signal is more likely
the teacher graph construction, the representation, or the clustering
step itself, not simply how the two InfoNCE terms are weighted.

## 6i. Joint Stage 1 + Stage 2 hyperparameter sweep, and a 4-seed stress test of its finalists

**2026-09-30.** The hand-tuned candidate rounds above (§6c–§6h) were
replaced by one systematic search: a 20-parameter Bayesian W&B sweep
(earlier documents say 21 or 22: `teacher_graph_alpha` was dropped as
inert, and the sweep YAML has 20 parameters)
(`polysemic/CoSiR-buddy-percept-sweep/40i43gt5`) over a
**re-implemented** version of buddy's pipeline
(`scripts/buddy_percept_sweep/`). The re-implementation differs
structurally from the pilots behind §3–§6h; see "Comparability" below.
It covered Stage 1 (student architecture, learning rate, noise, affect
weight, teacher graph, Leiden resolution, topic merging) and Stage 2
(mapper size, learning rate, epochs, target construction). Each trial ran
both stages end to end at seed 42 and was scored as

```
objective = stage2_macro_auc if (emotion_ami > 0.1236 and genre_ami > 0.1954) else -1.0
```

That is, "clear this investigation's Stage 1 Pareto thresholds, then
maximize Stage 2 AUC." **How the AMIs are measured matters here.** In
this harness, both AMIs are *k-NN transfer* AMIs. Held-out paintings are
assigned to the merged train topics by a k-NN vote, with `transfer_k`
itself swept over {10, 20, 30, 40}, which is the method of §6e. §2/§3
instead re-clustered the held-out paintings independently with Leiden,
and that is where the thresholds and the attention-h1 baseline's "3/4
seeds" come from. On the pilots' attention-h1 snapshot (seed 42, 19
train topics), transfer at k=20 gives emotion AMI +0.0115 and genre AMI
+0.0126 above independent re-clustering, and emotion AMI rises with k:
0.1327 at k=10, 0.1364 at k=20, 0.1390 at k=50
([`heldout_label_transfer_pilot_report.md`](pilots/20260923_artelingo_buddy_analysis/heldout_label_transfer_pilot_report.md)).
This offset was measured on the pilot model, not on this harness, but
it indicates the gate was easier to pass here than in §3's sense of the
bar. The same caveat applies to the "clears Pareto bar" column in §6e,
whose AMIs are also transfer AMIs judged against thresholds set with
independent re-clustering. Genre AMI rests on only 159
genre-labelled held-out paintings.

The sweep ran on 9 GPUs across three DAS6 nodes for about 36 hours. It
was stopped on 2026-09-29 after about 2,267 trials, of which 335 had
passed the gate by then. Design:
[`2026-09-28-buddy-percept-sweep-design.md`](../../../superpowers/specs/2026-09-28-buddy-percept-sweep-design.md).

**Finalist filter.** Finalists were limited to gate-passing runs that had
topic merging on and ended with at most 45 topics. All 49 gate-passing
runs with more than 100 topics had merging off, and their graphs broke
into hundreds of tiny topics, where macro AUC is noisy and cannot be
compared with the other runs. This deliberately excludes the sweep's raw
#1 (0.9488 at K=455). The top 10 remaining runs, frozen in
[`finalists.json`](../../src/test/20260928_buddy_percept_sweep/finalists.json),
were each re-run at seeds 42/7/123/2024. Each seed re-fits Stage 1 as
well as Stage 2. The winner rule was fixed before the results came in:
most gate passes out of 4, then highest mean AUC. The stress test ran on
DAS6 at commit `0ca4f24`. Tooling:
[`run_top10_stress.py`](../../src/test/20260928_buddy_percept_sweep/run_top10_stress.py);
full table:
[`stress_summary.md`](pilots/20260928_buddy_percept_sweep/stress_summary.md);
raw logs: `src/test/20260928_buddy_percept_sweep/stress_logs/`.

| sweep rank | run | sweep objective (seed 42) | gate passes | Stage 2 AUC mean ± std | emotion AMI mean (per seed) | genre AMI mean | topics per seed |
|---:|---|---:|---:|---:|---|---:|---|
| 10 | **m8x7ifx4** | 0.9371 | **4/4** | **0.9355 ± 0.0046** | **0.1295** (0.1291/0.1302/0.1317/0.1269) | 0.3116 | 16/14/14/13 |
| 7 | bfrae3fr | 0.9395 | 3/4 | 0.9399 ± 0.0036 | 0.12359 (0.1239/0.1133/0.1252/0.1320) | 0.3294 | 12/10/14/13 |
| 5 | j4fp661m | 0.9442 | 3/4 | 0.9357 ± 0.0051 | 0.1273 (0.1309/0.1279/**0.12356**/0.1269) | 0.3233 | 17/19/22/21 |
| 3 | 1vavykgu | 0.9467 | 2/4 | 0.9482 ± 0.0035 | 0.1220 | 0.2599 | 30/27/26/27 |
| 8 | knp9y2wp | 0.9377 | 2/4 | 0.9360 ± 0.0054 | 0.1231 | 0.2779 | 23/21/22/20 |
| 9 | wokudgi4 | 0.9374 | 2/4 | 0.9309 ± 0.0076 | 0.1238 | 0.3214 | 14/14/15/13 |
| 1 | whysrv0g | 0.9514 | 1/4 | 0.9584 ± 0.0043 | 0.1111 | 0.3543 | 21/18/23/18 |
| 2 | i3zajuzr | 0.9477 | 1/4 | 0.9513 ± 0.0031 | 0.1189 | 0.3471 | 15/15/15/15 |
| 4 | woy8z3lw | 0.9443 | 1/4 | 0.9486 ± 0.0028 | 0.1195 | 0.3176 | 32/30/29/32 |
| 6 | 2lvvlkmy | 0.9414 | 1/4 | 0.9431 ± 0.0012 | 0.1198 | 0.3698 | 14/14/14/13 |

AUC mean includes seeds that failed the gate. j4fp661m's seed 123
(0.12356) fails the 0.1236 bar by less than 0.0001, and bfrae3fr's
emotion mean (0.12359) also sits just below it.

**What the stress test found:**

1. **Winner: `m8x7ifx4`**, the sweep's 10th-ranked finalist and the only
   one that clears the gate on all four seeds. Its transfer-AMI emotion
   is 0.1269–0.1317, clearing the 0.1236 threshold by 0.003 to 0.008 on
   every seed. That margin is *smaller* than the ~0.01 that transfer adds
   over §3's independent re-clustering, so this is **not** evidence that
   it clears §3's bar. Its genre AMI is 0.296–0.329, well above 0.1954.
   Its Stage 2 macro AUC is 0.9355 ± 0.0046 (min 0.9309) with 13–16
   topics. It uses the maximum swept `transfer_k=40`, and 8 of the 10
   finalists use k ≥ 30. The optimizer favoured large k, which by itself
   raises held-out emotion AMI (see above). Config:
   - **Stage 1:** attention-fusion student with **4 heads**
     (`heads=attn1`, `num_heads=4`; in this harness `heads` only selects
     attention vs. MLP fusion and `num_heads` sets the head count, so this
     is *not* the pilots' one-head "Attention-h1"), `d_shared=64`,
     `lr=4.58e-4`, `noise_std=0.1`, `lambda_affect=1.03`,
     `batch_size=2048`, `weight_decay=1e-5`, `content_pca_dim=80`,
     `teacher_graph_K=15`, `leiden_resolution=0.836`,
     `merge_small_threshold=0.02`, `transfer_k=40`.
   - **Stage 2:** mapper with `num_queries=8`, `mlp_head=one_hidden`,
     `mapper_lr=5.96e-3`, `mapper_epochs=400`,
     `weight_decay_stage2=1e-4`, `class_balanced_loss=false`,
     `target_cutoff=0.15`.
2. **The sweep's leaderboard suffered from the winner's curse.** Ranks 1,
   2, 4 and 6 of the sweep each pass the gate on only 1 of 4 seeds, and
   that one pass is always seed 42, the seed the sweep scored them on.
   The optimizer was maximizing AUC subject to an emotion AMI bar, so it
   drifted to the edge of that bar. The configs it ranked highest were
   the ones where seed 42 happened to land just above the bar (for
   example, `whysrv0g` scored 0.1242 at seed 42, but its other seeds
   averaged 0.107). **A single-seed sweep score for a gated objective is
   an optimistic estimate near the gate boundary.** This is why the stress
   test was needed, and why its winner comes from near the bottom of the
   finalist list. Re-seeding removes this seed-level luck. It does **not**
   remove split-level selection: the 2,267-trial search, the finalist
   filter, the winner choice and the reported numbers all use the same
   9,365-painting held-out split, and there is no untouched test split.
3. **Among these finalists, emotion AMI and Stage 2 AUC trade off.** The
   correlation of per-finalist means is r = −0.85; across all 40 per-seed
   runs it is r = −0.47. The highest-AUC configs (0.95–0.96) all sit
   below the emotion threshold on average. These are 10 configs picked
   near the gate, so this describes the frontier the search found, not a
   law of buddy's design space. Still, on that frontier, extra Stage 2
   AUC came with weaker emotion structure. Genre AMI never bound (lowest
   single run 0.2465). Emotion was the binding constraint, which fits
   Finding D and §6h: emotion is buddy's weak axis.
4. **The pipeline is deterministic at a fixed seed.** Every finalist's
   seed-42 re-run reproduced its sweep objective and topic count. The
   re-runs came from new stress jobs; two finalists ran back to back in
   one process, and the second still matched. Six matched bit for bit and four
   differed by 1e-16 (floating-point round-off). The spread in
   the table is seed variance, not run-to-run noise. This spread (AUC std
   0.001–0.008) is much larger than the std of 0.0001 in §6d/§6g. Those
   earlier stress tests held Stage 1 fixed and re-seeded only the mapper.
   Here each seed re-fits Stage 1, so the topic vocabulary itself changes
   (for example, 17–22 topics for `j4fp661m`). The two std figures should
   not be compared.

**Comparability. This does not overturn §6g.** It is tempting to set
0.9355 against buddy's earlier 0.8534 (§6f) and PercepT's
symmetrically tuned 0.9226 (§6g). Neither comparison is sound yet:
- **The harness's Stage 1 is structurally different from the pilots'
  buddy Stage 1**, not just re-coded. The design spec (§4, "Reused
  building blocks (exact files, no reimplementation)") asked for the
  pilots' own files. The implementation plan prescribed fresh
  re-implementations instead, a plan-level deviation that no task review
  caught. The differences:
  - **Content teacher graph.** The pilots build it with
    `pipeline.build_buddy_graphs(img, txt, K=20, alpha=0.5,
    connect_components=True)`: the union of an image-only and a
    text-only mutual-kNN graph, with minimum-degree and connectivity
    repair. The harness uses one mutual-kNN graph over PCA-reduced
    concatenated image+text features, with no union and no repair. Its
    affect teacher also skips that repair.
  - **InfoNCE.** The pilots use in-batch negatives
    (`anchors @ positives.T`). The harness scores each anchor against
    every training node, including the anchor itself, which is never
    masked out.
  - **Optimizer and stopping.** The pilots use Adam with plateau early
    stopping. The harness uses AdamW with a cosine schedule for a fixed
    200 epochs.

  The harness has no setting that reproduces the pilots' Stage 1, so
  "does it reproduce §6f's 0.8534?" cannot even be tested until a
  faithful mode is added. The local smoke test only checked that the
  numbers were finite and plausible (AUC 0.878–0.916, K=16).
- **The Stage 1 numbers use a different measurement.** The gate uses
  k-NN transfer AMI, which reads ~0.01 higher on emotion than §3's
  independent re-clustering at k=20 and rises further with k (see the
  top of this section).
- **Topic count and targets differ.** Here it is 13–16 topics with
  `transfer_k=40`; buddy in §6f had 16 topics with k=20, and PercepT in
  §6g had 40 single-label topics. Macro AUC is not comparable across
  topic vocabularies.
- **The tuning budgets are very unequal.** PercepT was not searched
  through this harness. Buddy received ~2,267 trials of joint Stage 1 +
  Stage 2 search; PercepT received only the small learning-rate/epoch
  grid on its mapper in §6g, with its Stage 1 fixed. Claiming a buddy
  win on this basis would repeat the one-sided tuning that §6g had to
  correct.

**Sweep limitations, for anyone reusing the harness:**
- The hyperband early-termination block was probably a no-op. It keys on
  `objective`, which is logged only once per run. The checkpoint metric
  was the training-batch loss, not the held-out recall that the spec
  asked for.
- `stage1_fused_silhouette` was not logged, so the sweep says nothing
  about the silhouette axis.
- `heads=attn1` and `heads=attn4` build identical models (only
  `num_heads` matters), so `heads` is effectively binary (MLP vs.
  attention), and W&B's parameter importance for it is misleading.
- Each trial ran as a fresh process (`wandb agent` with `program:`), so
  the in-memory input cache never carried over between trials.
  GoEmotions features were re-extracted every trial, which wasted
  roughly 40% of the compute.
- The number of topics skipped in the AUC average is not logged. So,
  unlike §6d/§6g, the winner is not confirmed to have zero skipped
  topics.

**What it does support:** only claims *internal to this harness*. On the
harness's transfer-AMI measurement, one configuration (`m8x7ifx4`)
clears the gate on 4/4 seeds and reaches Stage 2 macro AUC 0.93–0.94
with 13–16 topics. The sweep's single-seed leaderboard was unreliable
near the gate. This result is **not** comparable to §3's "attention-h1
clears the bar on 3/4 seeds", which used a stricter measurement. Two
cheap checks would firm it up before anyone cites it:
- Re-run `m8x7ifx4` on new seeds with `transfer_k=20` fixed.
- Score the §3 baseline with the transfer method at 4 seeds.

**The Stage 2 headline from §6g ("PercepT 0.9226 vs. buddy 0.8534")
stands until a matched head-to-head is run.** That needs a faithful
buddy Stage 1 mode in the harness, validated against §6f first, and then
both systems through that same harness with matched topic counts, the
same target construction, and a comparable tuning budget for PercepT.

## 6j. The two confirmation checks: the sweep winner's Stage 1 advantage does not hold up

**2026-09-30.** Both checks proposed at the end of §6i were run on DAS6
(commit `a906a89`). The winner `m8x7ifx4` was scored on its 4 original seeds
and on 4 new ones (11, 23, 57, 101), with `transfer_k` 40 (as tuned) and 20.
The pilots' unchanged attention-h1 baseline was re-fitted at seeds
42/7/123/2024. Every run was scored on both yardsticks:

- **independent:** re-cluster the held-out embedding on its own, with the
  pilots' repaired mutual-kNN graph and modularity Leiden (§2/§3's
  method, the one the Pareto bar was set with);
- **gate:** the sweep's measurement, where train topics are merged at 0.02 and
  held-out paintings are labelled by a k-NN vote.

Raw logs are in `src/test/20260930_harness_confirmation/logs/`.

| system | seeds | independent emotion AMI (mean, range) | independent genre AMI (mean) | clears bar, independent | gate emotion AMI, k=40 (mean) | clears gate, k=40 | Stage 2 AUC (mean) |
|---|---|---|---|---:|---:|---:|---:|
| attention-h1 baseline (pilot, untuned) | 42/7/123/2024 | **0.1230** (0.1179–0.1293) | 0.2403 | 2/4 | **0.1341** | 3/4 (seed 42 misses on genre, 0.1945) | — |
| sweep winner `m8x7ifx4` | original 4 | 0.1164 (0.1104–0.1234) | 0.3014 | 0/4 | 0.1295 | 4/4 | 0.9355 |
| sweep winner `m8x7ifx4` | new 4 | 0.1213 (0.1182–0.1251) | 0.3058 | 1/4 | 0.1314 | 4/4 | 0.9376 |

With `transfer_k` fixed at 20 on the new seeds, the winner's gate emotion
AMI is 0.1286 (still 4/4). Its independent AMI does not change. Its Stage 2
AUC barely changes: `transfer_k` also sets the k of the train-side
multi-label vote in `pipeline.run_trial`, so per-seed AUCs differ by up to
0.0005, and the 4-seed mean is 0.9376 at both values.

**What this shows:**

1. **The winner's "4/4 seeds" is an artefact of the transfer yardstick.**
   On the investigation's own independent measurement it clears the
   emotion bar on only 1 of 8 seeds (mean 0.1189).
2. **On emotion, the winner is no better than the untuned baseline on
   either yardstick.** The baseline scores higher on both: 0.1230 vs 0.1189
   independent, 0.1341 vs 0.1295–0.1314 gate. The sweep did not find a
   better emotion structure. It found configurations with much higher
   genre AMI (≈0.30 vs ≈0.24) and much higher Stage 2 AUC, at a small cost
   in emotion. This matches the emotion/AUC trade-off seen among the
   finalists in §6i.
3. **Its Stage 2 AUC is robust.** On 4 new seeds it scores 0.9376
   (0.924–0.944), in line with the stress test. This number is still
   harness-internal and not comparable to §6g, for the reasons listed in
   §6i.
4. **The pilot result depends on the GPU.** Re-fitting the "unchanged"
   baseline on DAS6 gives seed-42 independent AMIs of 0.1187 / 0.2639; the
   local GPU gave 0.1249 / 0.2404 (§3). Seed-level numbers therefore shift
   by up to ~0.006 between GPU types, which is about the width of the
   margins being argued over. From here on, every cross-system comparison
   runs on one GPU type (DAS6); the one local check (§6k's V1) compared
   against pilot numbers from the same local GPU.

§3's Stage 1 conclusion ("buddy clears the AMI bar without PercepT's
occupancy collapse") rests on the pilot baseline. It is weakened but not
reversed. On DAS6 the baseline clears the bar on 2/4 seeds (3/4 locally),
and its mean emotion AMI of 0.1230 sits just below the 0.1236 threshold.
The matched head-to-head (next section) scores both systems on both
yardsticks, on the same GPUs.

## 6k. Matched PercepT-vs-buddy head-to-head: plain-AUC winners and an emotion-constrained comparison

**2026-09-30.** Both systems ran through one harness:
- **Faithfulness to the pilots:** the buddy *pilot* Stage 1 port is
  bit-exact to the pilot; the PercepT port reproduces §6g at 0.9258 vs
  0.9226; the shared Stage 2 reproduces §6f at 0.85341 vs 0.8534. The buddy
  *harness* Stage 1 (§6i), which every buddy winner used, was not validated
  against the pilot.
- **Matched setup:** matched K (16 and 40), the same k = 20 held-out labels,
  a val/test split of the held-out set, and an equal 300-trial Bayesian
  search per system per K. The user approved the objective, K levels,
  split and budget; the rest were controller rulings.

Full report:
[`auto/percept/2026-09-30_matched_percept_buddy_h2h.md`](2026-09-30_matched_percept_buddy_h2h.md).

On the test half, 5 fresh seeds each. The test half was not used for
selection in this experiment, but it is part of the held-out set that §6a
to §6i tuned on, so the references (not shown here) carry a selection
advantage on it. PercepT is the baseline; Δ is buddy minus PercepT.

| selection | K | buddy AUC | PercepT AUC (baseline) | Δ test [95% CI] | Δ val stress | buddy / PercepT independent emotion AMI, test |
|---|---|---:|---:|---|---|---|
| plain AUC (primary) | 16 | 0.9931 | 0.9664 | +0.027 [+0.014, +0.040] | +0.025 | 0.060 / 0.079 |
| plain AUC (primary) | 40 | 0.9937 | 0.9604 | +0.033 [+0.025, +0.041] | +0.030 | 0.060 / 0.110 |
| emotion floor (secondary) | 16 | 0.9462 | 0.9435 | +0.003 [−0.013, +0.018] | +0.004 | 0.121 / 0.120 |
| emotion floor (secondary) | 40 | 0.9550 | 0.9598 | −0.005 [−0.013, +0.004] | +0.001 | 0.128 / 0.120 |

- **Primary (plain AUC, the approved protocol):** buddy's winners lead on
  Stage 2 AUC on both halves, but only by giving up affect. They set the
  affect-loss weight at the search's lower bound (0.25), and their topics
  keep 76% (K = 16, 0.060 / 0.079) and 55% (K = 40, 0.060 / 0.110) of
  PercepT's independent emotion AMI.
- **Secondary (emotion-constrained, designed after the interim leaderboard
  showed buddy's early lead came with emotion AMI ≈ 0.05; floor fixed on
  val before any test run):** no Stage 2 difference was detected at n = 5
  seeds on either half. The test CIs exclude only differences larger than
  about +0.018 / −0.013 (K = 16) and +0.004 / −0.013 (K = 40); with
  PercepT's native labels the K = 40 sign flips (+0.0007). At equal AUC
  buddy's constrained winners never showed less emotion than PercepT's:
  level on test, about 0.02 more on val (0.117 vs 0.096; 0.127 vs 0.108).
  PercepT's topics carried about 0.10 more genre AMI on test, but on val
  the difference was −0.005 and +0.030 (buddy's favour), so no genre
  difference is established.
- **Which comparison leads is open** (the user's choice).
- **§6g's headline ("PercepT 0.9226 vs buddy 0.8534") does not hold under
  matched conditions** under either selection.
- **Caveats:** buddy's winners all use the §6i harness Stage 1. The search
  drew the pilot Stage 1 in only 25 and 20 of 300 trials, yet those trials
  supplied 8 of the 10 constrained buddy finalists (52 to 55% of them
  cleared the floor, against 2 to 6% of harness trials), and at K = 16 the
  pilot runner-up showed no detectable difference from PercepT on val
  (0.9463 vs 0.9438; Δ +0.002, 95% CI [−0.023, +0.028], 4 seeds). The AUC-only
  search sampled buddy's high-emotion region thinly: `m8x7ifx4`, inside
  the buddy K = 16 search space, scores 0.9352 on test at 0.127 emotion
  AMI. A pilot-only buddy search remains open.

## 7. Artifact map

All new scripts, briefs, and reports referenced above live in
`src/test/20260923_artelingo_buddy_analysis/`. Every script was implemented
by Codex (via CCG's `codeagent-wrapper --backend codex`) against a written
brief, independently read and verified line-by-line before being run; every
GPU training run was executed directly (never left for Codex to hold, per
this project's own standing guidance that Codex sessions are unstable across
long-running GPU work — confirmed again tonight when a combined
implement-and-run dispatch for the baseline stress test crashed mid-run and
had to be re-run directly). Briefs (`*_BRIEF.md`) record the exact
specification given for each pilot and are kept alongside their scripts for
anyone auditing what was asked for versus what was measured.
