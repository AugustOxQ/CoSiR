# Brief: brainstorm memo — further method improvements, and extending buddy topic-formation to RedCaps (unsupervised)

Research and brainstorm task ONLY. Do not write, modify, or run any code.
Do not touch git. Produce a single Markdown memo and nothing else.

## Background you must absorb first

Read, in full:
- `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` —
  the complete investigation: what buddy's Stage 1 (Attention-h1 +
  Leiden) and Stage 2 (patch-attention mapper) achieve on ArtELingo, every
  variant tried, and why the DEC-hybrid line is closed. This is the
  method whose extension and improvement you are brainstorming.
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md` — the prior
  brainstorm memo and how its five candidates were resolved (all five were
  run; see the consolidated report's §3/§6 for outcomes). Do not repeat
  any of its five candidates or the closed DEC-hybrid line as a "new"
  idea — read carefully what has already been tried and closed.
- `run_learned_student_arch_sweep_pilot.py`, `run_buddy_stage2_pilot.py`,
  and `run_heldout_label_transfer_pilot.py` — the actual current-best
  method's implementation (architecture, Stage 2 mapper, label transfer).

## Part A: further method-improvement candidates (ArtELingo)

Propose concrete, well-reasoned next directions for improving buddy's
topic-formation method further, grounded in what has actually been tried
(cite the specific prior result each idea responds to). Consider at least:
graph-construction choices not yet swept for ArtELingo specifically (K,
alpha, edge-type composition -- note that Experiment 16's K(N) sweep was
done on RedCaps, not ArtELingo; is there reason to expect it transfers?),
Leiden resolution-parameter sensitivity (communities are currently
whatever Leiden's default finds -- has resolution ever been swept?),
ensembling or averaging across the multiple seeds/variants already
measured tonight rather than picking one winner, alternative or additional
teacher views beyond content+affect (does ArtELingo have any other
usable signal -- genre itself as a weak teacher, art-historical style/
period metadata, caption sentiment beyond GoEmotions), and Stage 2
architecture depth/capacity (the patch-attention mapper is a single linear
head on pooled patches -- has anything deeper ever been tried, and is
there evidence of underfitting from tonight's loss curves that would
justify it). For each: concrete mechanism, why plausible given cited
prior results, smallest test, and estimated risk/cost. Rank them.

## Part B: extending the method to RedCaps (unsupervised, no emotion/genre labels)

This is the harder, more novel half. RedCaps has no human-annotated
emotion or genre labels -- read, in full, before proposing anything:
- `src/conditional_buddy/buddy_graph.py` -- the existing, already-used
  buddy-graph construction (mutual-kNN over CLIP image/text embeddings)
  this whole project's RedCaps work is built on.
- `src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py`
  -- the existing, validated, training-free "subreddit lift" quality
  proxy for buddy graphs on RedCaps (already used across all three scales
  in Experiment 16). This is RedCaps' nearest equivalent to ArtELingo's
  emotion/genre AMI -- subreddit membership is the closest thing RedCaps
  has to a human-assigned topic-adjacent label, even though it is noisier
  and less directly comparable.
- `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
  §4 (Experiment 16) and `docs/reports/2026-09-01_buddy_k_scaling_stage_a.md`
  / `docs/reports/2026-09-02_buddy_k_ablation_stage_b.md` -- the existing
  RedCaps K(N) scaling results (150k/300k/500k_diverse feature stores
  already extracted and training-ready at
  `/data/SSD2/pre_extract/redcaps_{150k,300k_diverse,500k_diverse}`).
- Confirm by reading `run_learned_student_arch_sweep_pilot.py` and
  `run_pipeline.py` exactly what ArtELingo-specific assumptions are baked
  into tonight's method (GoEmotions-affect teacher view, genre_map,
  emotion majority-vote labels used only for AMI evaluation, not training)
  versus what is generic and already dataset-agnostic (buddy graph
  construction, symmetric InfoNCE, Leiden community detection,
  patch-attention Stage 2 mapper). State this distinction explicitly and
  precisely -- this determines exactly what needs to change for RedCaps
  versus what transfers unchanged.

Propose a concrete plan for a RedCaps version of this whole pipeline:
what plays the role of "content teacher" (the existing image+text mutual-
kNN buddy graph, unchanged) and what -- if anything -- plays the role of
"affect teacher" (RedCaps has no emotion signal; consider whether a
second, genuinely different teacher view is needed at all, or whether a
single-teacher variant of Attention-h1 is the right starting point --
reason explicitly about this, do not just assume the two-teacher
structure must be preserved). Propose evaluation without human labels:
subreddit-lift (already validated, reuse it exactly) for Stage 1 quality,
and a RedCaps-native Stage 2 analog -- an image-only mapper predicting
discovered topics, evaluated by whether it also predicts **subreddit**
better than a raw-feature control (mirroring tonight's ArtELingo
downstream-probe design exactly, substituting subreddit for emotion/
genre). Propose a staged, cheap-first pilot plan explicitly: what is the
smallest, cheapest experiment (state which RedCaps scale, what compute,
roughly how long) that would tell us whether this transfers at all,
before committing to anything at 300k/500k scale on DAS6. Flag any
genuine open design question you cannot resolve confidently, rather than
picking arbitrarily and hiding the uncertainty.

## Output

Rank all candidates from both parts by (expected value) / (implementation
+ verification cost), and give a clear top recommendation for what to
implement first tonight, with reasoning -- but present the full ranked
list, this is a brainstorm for a person (and a "brain" agent who will read
it next) to choose from, not a unilateral decision.

Write the memo to:
/tmp/claude-0/-project-CoSiR/a2df7b39-29a4-4d87-82ac-18432fa323d1/scratchpad/method_improvement_and_redcaps_brainstorm.md

Do not write anywhere else. Do not touch git. Confirm in your final
message only the file path and a one-line count of candidates evaluated
in each part.
