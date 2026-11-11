# Handoff: r21 build: tickets and build of the DEC arms on c*, from the approved spec, then open the run chat

## Brief
- Done: r21 decided and specced: a DEC loss on every head of c*, three forms, as ablation rows; spec approved 06:31.
- State: no code yet. One c* training takes 3.5 to 5.3 min per seed; its heads stop early (seed 0: epochs 4, 1, 18).
- Next: to-tickets from the spec, build, final review; merge to main, push, then open the r21 run chat.
- Blocked: nothing. The local GPU is held by r25 and r26 until about 10:00; smoke runs queue with --short.

> Written 2026-11-11 06:35 by the r21 decide chat. Loop step reached: step 6 done (spec approved), step 7 next.
> Direction log entry: [2026-11-11 · r21 decide](../direction_log.md).

## 1. Read in this order

1. The spec, from §1 onward: [2026-11-11-r21-dec-heads-design.md](../specs/2026-11-11-r21-dec-heads-design.md). It
   governs; this handoff only adds code facts.
2. `~/.claude/references/matt-chain.md` (the build chain: to-tickets, implement-spec, the final whole-branch review),
   and `~/.claude/references/loop-chats.md` ("Unattended chats": decision lines, nothing deleted).
3. The facts brief, with code paths and numbers:
   [r21_facts_brief.md](../../../src/test/20261111_r21_dec_heads/r21_facts_brief.md).
4. The constitution's "Decisions agents never make alone": `docs/superpowers/constitution.md` (its 2026-11-10
   amendment is in the working copy, uncommitted by another chat).

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

All on 2026-11-11, recorded in the spec's §4 (D1 to D7) and the direction log:
- Ablation rows beside the eight removals; c* stays the paper's model (user, 1a).
- DEC inside c*'s training after a warm-up, the target recomputed each epoch as in PercepT, c*'s losses kept (user, 2b).
- Three forms: plain, a pull toward the start clusters, a pull toward the teacher's clusters (user, after seeing the
  recommendation).
- Silhouette is measured; one joint row on all heads, with no arm per aspect; the ablation table's mAP; ArtELingo
  only; k = the class count; runs today or tonight (user, after seeing the recommendation).

## 3. Where things stand

- c*'s recipe is `k11=40` in the recipe sweep (`src/test/20261027_recipe_sweep/`, `rs_sheet.py`, `rs_knobs.py`,
  `rs_full.py`, `rs_ablation.py`). The eight removals are named configurations; a DEC arm can be added the same way, or
  as a new trainer that imports the frozen loop. The build chooses and writes a decision line.
- The trainer is `src/test/20261020_claim_test/ct2_d2.py` (each aspect keeps a deep copy of its head and the trunk at
  its best epoch); the model is in `src/test/20261019_model_sheet/ms_train.py:31` (2048 → 1024 trunk, 1024 → 2048
  heads, unit outputs).
- DEC parts: `src/test/20261018_percept_core/pc_sept.py:68` (Student-t soft assignments, the target, the KL loss;
  balance weight `LAMBDA_BALANCE = 1000`, stop at fewer than 0.1% changed).
- Saved c*: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/` (`weights/k11=40/`, `held/k11=40/`, and the
  validation exports under `train/k11=40/R6-5/`), seeds 0 to 4.
- Scoring: r17's `src/test/20261103_new_scores/ns_metrics.py` and `ns_clu_cells.py` (AMI, the class count);
  r20's `src/test/20261109_r20_figures/r20_ret.py` with r11's `rt_metrics.py` (mAP). Gate numbers: the spec's §3.5.
- c*'s per-seed test AMI at the class count is not saved; recompute it from the test exports. The 5-seed means must
  equal r17's file to 6 decimals (`/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage2/cells/clu__artelingo__qwen.json`).
- Nothing exists for r21 yet except `src/test/20261111_r21_dec_heads/r21_facts_brief.md`.

## 4. Open points for the next step

None for the user. The build chooses the code layout, the batch-level form of the DEC term on the image rows, and
the teacher-feature paths. Each result-shaping choice gets a decision line in `src/test/20261111_r21_dec_heads/decisions.md`.

## 5. Code and pitfalls

- CoSiR main has other chats' uncommitted edits (`src/test/20261111_paper_baselines/*`, the constitution, the
  handoff logs). Commit only r21's files. Build on its own branch per `matt-chain.md`; merge into main locally and push.
- Round folders go on `sys.path`: new modules use the prefix `r21_`. Earlier round folders are frozen: import them,
  never edit them.
- Gate 0b (spec §3.5): with DEC off, the new code must give the frozen loss to 1e-6 on CPU. CUDA reruns are not
  bit-exact (r19's rule review), hence the tolerances of gate 0c.
- Joins follow extraction IDs (C6 in the constitution; the project CLAUDE.md), never `range(n)`. This holds for the
  teacher's instructed features too.
- GPU through `res-queue gpu --short --label r21:<step>` (subagents add `--no-wait`); outputs in
  `/data/SSD2/cosir_r21/20261111_dec_heads/`; smoke outputs in the round folder or the scratchpad.
- Python: `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`; tests with `src/test` named.
- At the end: write the run handoff (`docs/superpowers/handoffs/2026-11-11-r21-run-handoff.md`), run the GPU check
  (`loop-chats.md`, "How to cut"), and open `loop-next <run handoff> --label "r21 run" --unattended`. The run chat
  checks the GPU power at gate 0c: under 200 W, the stages may run in the day; else from 19:00 (spec D17).
- The repo is public: no keys or personal data in files. The user has a meeting at 11:00: ask nothing about it.

## 6. State at handoff

- Running: nothing of this chat. r21's main chat opens beside this build chat.
- Uncommitted: nothing of r21 after this handoff's commit.
- On disk over 1 GB: nothing.
