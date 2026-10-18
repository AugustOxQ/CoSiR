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
