# Model sheet (r10 stage 2): build and run log

Spec: [2026-10-19-stage2-model-sheet-design](../../../docs/superpowers/specs/2026-10-19-stage2-model-sheet-design.md)
(approved 2026-10-19 19:58). Build chat: `r10 build 7` (night mode, unattended). Tickets:
`.scratch/r10-model-sheet/issues/` (outside git; list in the run handoff). Held data: none read; everything is on the
frozen train-side sample (fit 13,400, dev 1,489, evaluation 5,000 paintings).

## Agent defaults set in the build (within the spec; none read held data)

1. **Epochs with three pair lists (D12).** Each head's epoch is one pass over its own fit list; its dev loss is
   checked at its own epoch end, and its best state (shared layer plus its own layer) is kept. A head that passes
   patience 15 leaves the step (its layer and its loss stop); the run ends when no head is left. Why: one head alone
   is then exactly stage 1's loop, so A1 and stage 0 run the same code as the trunk rows, and an overfitting head
   cannot keep moving the shared layer under the heads that still learn.
2. **A1** is three single-aspect runs, each seeded as stage 1. Why: with no shared layer, joint and single training
   differ only in the random streams; single runs keep stage 1's code path (the stage 0 check covers it).
3. **Shuffles in joint runs** use `default_rng([seed, aspect index])` per head (single runs: `default_rng(seed)`, as
   stage 1). Why: D1's three heads would otherwise see the same painting batches.
4. **Form b (D14):** one extra negative per pair endpoint per step, drawn from the node's buddies under the other two
   aspects' graph rows minus its own-aspect buddies; added to both directions of the four buddy terms only. Why: the
   cheapest form that puts each anchor's other-aspect neighbours against it in the head's own softmax.
5. **Form c (D15):** conditional similarity networks (Veit, Karaletsos and Belongie, CVPR 2017), at the defaults of
   the authors' code (github.com/andreasveit/conditional-similarity-networks: `--mask_loss 5e-4` on the L1 norm of
   the ReLU mask per sample; masks initialised N(0.9, 0.7); checked 2026-10-19; the paper text not checked). One mask
   per aspect, shared by the image and caption sides; masks get no weight decay; the code's embedding L2 term
   (`--embed_loss 5e-3`) is dropped because our outputs are L2-normalised after the mask.
6. **Distillation (D16):** direct = one-direction InfoNCE (head outputs as rows, instructed image embeddings of the
   same batch paintings as columns), same-painting columns masked; relational = KL(teacher || student) of the batch
   similarity rows, temperature 0.05 on both sides (the temperature of every other term), diagonal excluded.
7. **D1 batches:** 1,024 random fit paintings per head (the endpoint count of a pair batch); pair batches would bring
   the graph back into D1 through its negatives. D1's caption side is not trained.
8. **I1's caption input:** the aspect's instructed caption mean, as follow-up (c) of stage 1.
9. **Seed picks (D11):** A1, T1 and the top two rows per aspect by seed-0 image-side AMI of the head's own label.
   D17's "best D row": the highest mean over the three aspects of that AMI among D2 to D5.
10. **Bars:** the probe bar carries AMI only (its probabilities were not stored); V, ARI and FMI for the other bars
    come from their vectors.
11. **Queues:** one script per GPU stage, GPU work under `flock -o -w 7200 /tmp/gpu0.lock`, scoring on CPU after.

## Timeline (Amsterdam time)

- 20:10 tickets 01 (scorer) and 02 (trainer core) to Codex (`gpt-6.1-sol`, medium and high), worktrees
  `/project/CoSiR-wt-ms01`, `/project/CoSiR-wt-ms02`.
- 20:14 ticket 01 done (`659c286`): PercepT core check 45/45 equal at 6 decimals; back-fill 22 of 24 cells, the
  emotion and style network cells wait for stage 0's arrays. Controller diff read; merged (`a2f92fb`).
- 20:16 ticket 02 done (`f63f8da`): separate genre R0o seed 0, all fit pairs, 2 epochs on CPU: the four losses and
  both float16 arrays bitwise equal to stage 1's (its best epoch is 1). T1 smoke: three heads, distinct histories.
  Controller diff read; merged (`c7a4087`). The GPU path is not yet smoked.
- 20:20 cut to chat `r10 build 8` (context meter past 200k); tickets 03 and 04 go on there.
