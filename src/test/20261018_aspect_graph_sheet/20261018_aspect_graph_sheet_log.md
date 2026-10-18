# Aspect graph sheet (r10 sweep 1): run log

Spec: [2026-10-18-aspect-graph-sweep-design](../../../docs/superpowers/specs/2026-10-18-aspect-graph-sweep-design.md),
stages 0 and 1 (started before approval on the user's word, D10). Chat: `r10 sweep 1` (build and run in one chat,
night mode). Held data: none read; everything is on the frozen train-side sample.

## Timeline (Amsterdam time)

- 16:24 GPU check passed (driver 595.99.02, GPU free, no lock). `ags_trainimg.py` smoke (16 images, style_word):
  re-encode gate cosine 1.00000000.
- 16:25 GPU extraction started (`ags_extract_queue.sh`): style_word, then genre_word training images, 14,889 each.
- 16:30 Codex jobs for tickets 01 (graph rows) and 02 (cross-modal head), `gpt-6.1-sol` at high, each in its own
  worktree (`/project/CoSiR-wt-ags01`, `/project/CoSiR-wt-ags02`).
- 16:37 and 16:39 tickets 01 and 02 done (about 8 and 10 min); 16:41 both merged to main.
- 16:49 style_word training images sealed (1,368 s, 10.9 images/s).
- 16:58 early graph build, emotion and style (`ags_graphs_early.log`): `STAGE0 PASS 2.4204897561 / 2.3849313416 /
  2.5528809893`, the pilot's D, I, C, E, P, EC pair-identical on fit and dev. Style R0 style lift 7.07 (default image
  graph 5.37): the style instruction works on the training images, at the graph level (spec §5 assumption).
- 17:11 genre_word training images sealed (1,367 s); the CPU queue (`ags_queue.sh`) started by itself.
- 17:15 genre graphs built. Rows for the head (`graphs/selected.json`): emotion R0, R1, Psim50, Fres1, Gres; style R0,
  R1, Psim50, Fres05, Gres; genre R0, R1, Psim75, Fres025, Gres, Fside025. The best row by target lift was R0 for
  emotion and style (no sixth row); style's Fres1 was not eligible (16.8% of R0's pairs, under 20%).
- 17:20 heads train at about 35 s per epoch with three processes. Dev loss is lowest at epoch 1 to 4 and then rises;
  every run stops after 16 to 19 epochs (patience 15). The pilot's students did the same (best epoch 1 to 3).
- 18:00 seed 0 scored. Rows for seeds 1 and 2 (`head_seed_rows.json`, top two by seed-0 target AMI): emotion Fres1,
  Gres; style Fres05, Gres; genre Fside025, R0.
- 18:20 queue done: `AGS_DONE graphs=0 seed0_emotion=0 seed0_style=0 seed0_genre=0 score0=0 pick_seeds=0
  seeds12_emotion=0 seeds12_style=0 seeds12_genre=0 score012=0`. 39 head runs, no output missing. Sheets:
  [graphs/sheet.md](graphs/sheet.md) (63 graph rows), [ags_head_tables.md](ags_head_tables.md) (heads and the
  per-view references).
- 18:05 the user approved the storage summary; the unattended guard refused the deletes in this chat, so they are
  marked approved in `.scratch/pending_deletions.md` for the next chat.

## Agent defaults set in this chat (within the spec; none read held data)

The exact definitions: [ags_rows.md](ags_rows.md). Tickets: 01 `ags_graphs.py` (Codex), 02 `ags_head.py` and `ags_queue.sh` (Codex), both merged to main at 16:41 after the controller diff read and their smoke runs.

1. Genre's primary view (R0) is the genre_word caption graph; the image side enters through R0o, F-side, F-rank and
   G-agree. Why: the spec's genre "both sides" is a fusion, and the F-side family covers it; R0 stays one view, as for
   emotion and style.
2. Every aspect gets an extra reference row R0o (the other side alone) and R1c (full pruning with the CLIP content
   graph). 21 rows per aspect, 63 in all.
3. Similarities are divided by their sd over 200,000 random pairs before they are mixed (F-res, F-side, G-res,
   P-sim), so λ and w act in sd units. The content score is the mean of the default caption and default image scores.
4. F-orth removes the content direction of the same side (default captions for caption views, default images for
   image views). F-rank uses k = 60 (the usual reciprocal-rank constant). G-res uses λ_i = the painting's share of
   content pairs in R0.
5. Downstream eligibility: at least 200 dev pairs and at least 20% of R0's fit pairs, so a tiny graph cannot win on
   lift alone (spec §8 risk).
6. The head's loss: four buddy terms (image-image, image-caption both ways, caption-caption), weighted per pair, plus
   two own-caption terms, unweighted; the pilot's settings otherwise. Target AMI = mean of the image-side and
   caption-side AMI of the aspect's own label; it picks the two rows per aspect that get seeds 1 and 2.
