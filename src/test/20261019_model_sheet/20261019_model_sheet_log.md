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

12. **D17 trigger tie (ticket 04):** if both T4 and T5 pass the D17 test, the form with the larger sum of margins
    beyond T1's sd wins; an exact tie goes to form b. Why: one reproducible rerun when the spec allows either form.
13. **D17 rerun and I1 train on seed 0 only** (ticket 04). Why: spec §3.4 gives three seeds only to A1, T1 and the
    top two rows per aspect.
14. **The §6 line (fix wave, review 1):** only default-input rows count, against the PercepT core bar's emotion AMI
    + 0.01 as the sheet computes it (0.077174 + 0.01 = 0.087174). Why: I1 uses instructed inputs (spec D5), and the
    gate is defined relative to PercepT core.
15. **Form c early stopping (fix wave, review 3):** dev loss for stopping and best-state choice excludes the mask L1
    term; the penalty is logged beside it. Why: outputs are normalised after the mask, so the term falls each epoch
    whatever the fit, and moved T5's best epoch later; CSN also selects on validation fit, not on the penalty.
16. **D17's best D row (fix wave, review 8):** chosen on seed 0 only, as default 9 says. Why: D rows have one or
    three seeds depending on the picks; one seed for all keeps the comparison fair.

## Timeline (Amsterdam time)

- 20:10 tickets 01 (scorer) and 02 (trainer core) to Codex (`gpt-6.1-sol`, medium and high), worktrees
  `/project/CoSiR-wt-ms01`, `/project/CoSiR-wt-ms02`.
- 20:14 ticket 01 done (`659c286`): PercepT core check 45/45 equal at 6 decimals; back-fill 22 of 24 cells, the
  emotion and style network cells wait for stage 0's arrays. Controller diff read; merged (`a2f92fb`).
- 20:16 ticket 02 done (`f63f8da`): separate genre R0o seed 0, all fit pairs, 2 epochs on CPU: the four losses and
  both float16 arrays bitwise equal to stage 1's (its best epoch is 1). T1 smoke: three heads, distinct histories.
  Controller diff read; merged (`c7a4087`). The GPU path is not yet smoked.
- 20:18 cut to chat `r10 build 8` (context meter past 200k); tickets 03 and 04 go on there.
- 20:21 tickets 03 (trainer extensions, `gpt-6.1-sol` high) and 04 (stage 0 runner, sheet, queues, medium) to
  Codex, worktrees `/project/CoSiR-wt-ms03`, `-ms04`. GPU smoke of ticket 02 (T1, A1 `--smoke --device cuda`, and
  genre A1 2 full epochs): valid outputs; CUDA losses close to CPU (genre epoch 1 dev 7.161 against 7.160); a full
  genre epoch takes about 1 s on the GPU against 17 s on CPU.
- 20:27 ticket 04 done (`5d4bc38`): sheet smoke on ticket 02 outputs, scoring 6.2 to 7.7 s per head (both sides);
  stage 1 AMI 6/6 equal; PercepT 45/45. Controller diff read; merged (`4b4ee36`).
- 20:32 ticket 03 done (`56b8c66`): six CPU smokes (T4, T5, D1, D2, D5, I1); stage 1 genre arrays and ticket 02's T1
  arrays still bitwise equal; D1 caption side bitwise at init. Controller diff read; merged (`92fa621`). CUDA smoke
  of the six rows: valid outputs; full-size joint rows take 1 to 3 s per epoch per head on the GPU.
- 20:36 to 20:47 stage 0 pre-check in this worktree (pre-fix code, CPU, three aspects in parallel): STAGE0 PASS;
  genre arrays bitwise equal, 36/36 AMI equal at 6 decimals, 7/7 best epochs equal, PercepT 45/45. The real stage 0
  reruns from main after the fix wave (the fix changes the trainer's code version).
- 20:51 run mode set to day (user).
- 20:55 final whole-branch review (`worker`, Opus `xhigh`): three must-fix findings (the §6 line counted I1; stage 0
  and score caches not tied to the code version; form c stopped on the mask penalty), six that can wait; scorer,
  bars, form b, distillation, joint loop and stage 0 re-derived exactly. Review in
  `.scratch/r10-model-sheet/notes/final_review.md` (outside git). Fix wave brief written (items F1 to F9, defaults
  14 to 16).
- 20:58 cut to chat `r10 build 9` (context meter past 200k); the fix wave goes on there.
- 21:00 fix wave to Codex (`gpt-6.1-sol` high), worktree `/project/CoSiR-wt-msfix`, branch `r10-ms-fix`.
- 21:09 fix wave done (`7814068`, note `ms_fix_build.md`): F1 to F3 and F5 to F9; 20 arrays still bitwise equal to
  stage 1 and tickets 02 and 03; form c dev loss plus the logged penalty equals the old formula within 4.2e-7; the
  pre-fix stage 0 copy fails the new `stale` gate (7 paths) while gates a, b, c still pass. Controller diff read;
  merged (`84dc2db`). CUDA smoke of T5 (4 epochs): penalty logged, best epoch = lowest dev loss, outputs valid.
- 21:13 scoped re-review (`worker`, Sonnet `high`): PASS, no must-fix item. Two notes for the run: (1) after F7,
  any config difference (max epochs, device, `--set`) replaces verified heads, so never rerun a row by hand into
  `outputs/` with other settings; (2) `SCORER_VERSION` hashes `ms_score.py` and `pav_score.py` only (as the brief
  asked), not `pc_score`, `apc_data` or `pc_data`.
- 21:15 `r10-model-sheet` merged into main (`e066c92`), pushed. 21:16 stage 0 from the main checkout (CPU).
