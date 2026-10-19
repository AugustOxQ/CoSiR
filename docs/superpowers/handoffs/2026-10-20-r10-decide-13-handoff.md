# Handoff: grill and spec the claim test (D2 on held data, 10-29 gate); then decision B for the cluster-loss track

## Brief
- Done: decisions C and A on the stage 2 sheet, the wrap-up, the literature check on cluster-level losses, the claim test facts brief.
- State: D2 0.096 / 0.370 / 0.454 on development data; PercepT core 0.077 / 0.229 / 0.436; above the instructed target on emotion only.
- Next: grill the claim test (D2 on held data, Leiden and PercepT readings), write its spec; then decision B for the cluster-loss track.
- Blocked: nothing.

> Written 2026-10-20 00:10 by "r10 decide 12" (cut at the context meter, 200k). Loop step reached: track 1 (claim
> test) after step 4, ready for the grill (step 5); track 2 (cluster-level losses) after step 3, before decision B.
> Direction log entry: [2026-10-20 · Stage 2 model sheet read](../direction_log.md).

## 1. Read in this order

1. The direction log entry of 2026-10-20: decision C, decision A and the claim test choices, with who decided.
2. The facts brief `.scratch/r10-claim-test/facts.md` (not in git; Codex, spot-checked): held split, feature coverage,
   D2 and PercepT core code paths and costs, the "paper settings" PercepT edit sites (§6), Leiden code (§7).
3. The literature note [2026-12-17_cluster_level_losses](../../reports/literature/2026-12-17_cluster_level_losses.md):
   Q5 (comparators for the claim test) now; Q1 to Q4 and the ranked ideas for track 2 later.
4. `.scratch/r10-cluster-losses/dec_comparison.md` (not in git; Codex, spot-checked): Table A lists every piece where
   PercepT core differs from the paper, which the "paper settings" PercepT must change.
5. The [stage 2 spec](../specs/2026-10-19-stage2-model-sheet-design.md) (D2's definition) and the
   [PercepT core spec](../specs/2026-10-18-percept-core-design.md) (A1 to A3).
6. `~/.claude/references/grill.md` and `~/.claude/rules/spec-templates.md` before the grill and the spec.

Skills: `grilling` and `domain-modeling` (the grill).

## 2. Decided by the user, not to be reopened

All in the direction log entry of 2026-10-20 unless noted.
- Decision C: the user's reading is recorded as written there.
- Decision A: two tracks. Track 1: the claim test spec next (type claim test, full loop). Track 2: step 3 for
  cluster-level losses (new idea, full loop): (1) cluster-level contrast per head, (2) teacher clusters as anchors,
  (3) DEC started from the good partition of our heads. Align-and-contrast across aspects is dropped; the short style
  and genre loop is not chosen (both parked).
- The claim test row is D2 (trunk + direct distillation 0.5).
- A Leiden reading in all major tests, beside k-means at k = 20.
- PercepT is scored both with its own assignment (top topic of its 40) and at matched k = 20; the gate is judged
  against the stronger reading per aspect.
- A "paper settings" PercepT (only the backbone differs from the paper) runs beside PercepT core, with a collapse rule.
- From earlier specs: gate 2026-10-29 = held-out emotion AMI at least +0.01 above PercepT core; the instructed bar is
  judged with retrieval on 2026-11-10; the default input carries the claim; time-box 2026-10-23 for the claim test
  spec. Run mode: day.

## 3. Where things stand

Image-side AMI, development data (5,000 evaluation paintings, k = 20, 3 seeds):

| Row | Emotion | Style | Genre |
|---|---:|---:|---:|
| D2 | 0.0962 | 0.3697 | 0.4540 |
| T1 (trunk, no distillation) | 0.0930 | 0.3591 | 0.4603 |
| PercepT core, matched k = 20 (P3 / P2 / P3) | 0.0772 | 0.2286 | 0.4359 |
| PercepT core, own assignment (best arm per aspect) | 0.082 | 0.220 | 0.401 |
| Target: instructed Qwen image | 0.0929 | 0.3952 | 0.4956 |
| Linear probe (reference ceiling) | 0.121 | 0.527 | 0.657 |

- With the stronger PercepT reading, the emotion bar is about 0.092; D2 clears it by about 0.004 on development data.
- Under random clusterings AMI and ARI stay at 0 for any cluster count; V and FMI move with the count (a simulation in
  r10 decide 12). Leiden rows need the count beside AMI, and V and FMI need a random baseline at the same count.
- PercepT's paper states no test split (one training set of about 1.2M pairs); our scores are on unseen paintings.
- Literature (12-17): no published support for DEC from a good partition; the closest per-aspect work is ESMC (AAAI
  2026). New comparators a reviewer may name: TURTLE, TEMI, CPP.

## 4. Open points for the next step

Track 1, the grill of the claim test (key questions answered by the user first, grill.md):
- **Key:** training data for D2 and PercepT in the claim test: the 13,400-painting fit sample as developed, or the full
  train side (42,969 paintings)? The code asserts the sample sizes; graphs scale about 10× (facts §4).
- **Key:** the Leiden protocol: which graph (September's mutual 20-NN, or a union kNN), which resolution (fixed 1.0,
  or chosen on development data, never held), and how held paintings get a community (Leiden on the held graph, or a
  kNN vote onto training communities as in September).
- **Key:** the gate rule with the new readings: which reading decides the 10-29 gate (k-means k = 20 as before, Leiden
  beside it?), seeds (3 or more), the margin on the seed mean, and the "paper settings" PercepT collapse rule.
- **Minor (one batch):** extraction on held rows (instructed Qwen images per aspect for the target bar; RoBERTa only if
  an arm needs it; facts §2); retrain with weights saved (no weights exist; facts §3, §5); V, ARI, FMI with random
  baselines; an in-sample reading beside the main one, to show the protocol effect (agent proposal); TURTLE or TEMI
  as extra comparators (probably 11-10, not 10-29); the run's end by 11:00 if overnight.

Track 2, after the claim test spec (or in its own decide chat if this one fills):
- **Key, decision B:** pick the idea from the literature note's ranked list (1 teacher-partition anchor, 2 TEMI-style
  head, 3 comparator probe), against the user's three forms. The note does not recommend plain DEC from a good
  partition; it suggests an anchor if the user still wants it.
- Cheap CPU checks proposed in r10 decide 12, not yet run: a re-score of the sheet's main rows at each label's class
  count and with Leiden; the joined-head check (three head outputs joined, one clustering).

## 5. Code and pitfalls

- D2 and PercepT core save outputs, not weights: the claim test must retrain (facts §3, §5). D2 takes about 0.6 min
  per seed for three aspects, PercepT core about 1.3 to 1.7 min per arm seed (facts §9).
- The instructed Qwen and RoBERTa caches cover no held rows; default Qwen, CLIP global and CLIP patches do (facts §2).
- `ms_train.py`, `ms_rows.py`, `ms_score.py` hashes are in every head's config; a new row type changes the trainer, so
  stage 0 runs again and F5 asserts one code version per row.
- Leiden: `leidenalg` 0.12.0 and `igraph` 1.0.0 are installed in the CoSiR env. September code: `detect_communities` in
  `src/conditional_buddy/prototype_seed.py` and the kNN vote in
  `src/test/20260923_artelingo_buddy_analysis/run_heldout_label_transfer_pilot.py`, both in the worktree
  `/project/CoSiR-buddy_prototype_conditioning`.
- Held data: the claim test is this line's first held read (`held-data.md`); no tuning, selection or early stopping
  reads it. `docs/superpowers/held_ledger.md` is a plain record.
- A new round folder needs a new module prefix (`sys.path` clash).

## 6. State at handoff

- Running: nothing (no GPU lock, background job, Codex job or subagent).
- Uncommitted: nothing. Outside git: `.scratch/r10-claim-test/facts.md`, `.scratch/r10-cluster-losses/dec_comparison.md`.
- On disk over 1 GB: `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger).
