# Experiment 13 — Symmetric conditioning (Option A): does removing the one-sided combiner explain the i2t/t2i freeze-ablation asymmetry?

> **Promoted docs-only from `experiment/two_side_conditioning`** (2026-08-31) per this project's two-branch reconciliation convention. **The code this report describes (`model.conditioning_mode="symmetric_shared"`, the SDD implementation ledger, the new unit tests) is NOT present on this branch (`experiment/condition_drift_retrieval_correlation`)** — it lives only on `experiment/two_side_conditioning` (worktree `/project/CoSiR-two_side_conditioning`). This copy exists so the finding is citable from either branch's report history; treat the "Code" line below as historical/foreign, not something to go looking for here.

**Date:** 2026-08-27
**Dataset:** RedCaps-150k
**Implementation branch/worktree:** `experiment/two_side_conditioning` (worktree `/project/CoSiR-two_side_conditioning`; originally implemented on `experiment/symmetric_conditioning_exp13`, branched from `experiment/condition_drift_retrieval_correlation` at `de4edd1`, then consolidated onto `experiment/two_side_conditioning`)
**Analysis branch:** `experiment/two_side_conditioning` (original); also promoted docs-only to `experiment/condition_drift_retrieval_correlation` (this copy)
**Code:** `src/model/cosirmodel.py`, `src/metrics/loss.py`, `src/hook/train_cosir.py`, `src/eval/pipeline.py`, `src/eval/metrics.py` (new opt-in `model.conditioning_mode="symmetric_shared"`); `.superpowers/sdd/2026-08-27-symmetric-conditioning-exp13/` (full SDD implementation ledger); `docs/superpowers/scratch/2026-08-27_codex_symmetric_combiner_brainstorm.md` (architecture design memo this implements) — **all only on `experiment/two_side_conditioning`**
**Motivated by:** `docs/reports/2026-08-27_combine_side_txt_replication.md` (C10) — flipping `combine_side` shrinks C9's i2t-vs-t2i freeze-ablation asymmetry by ~11× but does not eliminate it, leaving "why does image-side combination amplify this" unexplained
**Compute:** 1 smoke test (2 epochs × 2 arms) + 6 real runs (3 seeds × {trained, frozen}, 100 epochs each, ~2.5h wall time)

## TL;DR

Removed the architectural asymmetry itself (one combiner + one condition table applied to only one modality, `other_proj` on the other) rather than just flipping which side it favors. Implemented the smallest matched-capacity variant — **Option A**: one shared condition table, one shared (tied-weight) combiner applied to **both** image and text embeddings, every auxiliary loss term symmetrized. Result: **C9's headline i2t freeze-vs-trained effect (mean Δ=+4.67 R1, mean/SEM=+32.1 under the default image-side combiner) collapses to a noise-floor null** under symmetric conditioning (mean Δ=+0.53, mean/SEM=+0.8, not even sign-consistent across seeds), on both the primary (coupled-table oracle) and deployable (two-sided predictor) metrics. This is a stronger disconfirmation than the earlier `combine_side="txt"` replication (which still found a smaller-but-significant i2t effect) — **directly supports the architectural-amplification hypothesis**: the one-sided combiner is not just amplifying this effect, it looks like it may be substantially *causing* it.

## Method

**Implementation (see the SDD ledger on `experiment/two_side_conditioning` for full detail, not repeated here):** a new opt-in `model.conditioning_mode` config axis (`asymmetric` default = today's exact `combine_side` behavior, unchanged; `symmetric_shared` = Option A) was implemented end-to-end — model forward/combine API, symmetrized loss terms (preservation, delta/gate, laplacian, mixup, table regularizers, predictor distillation), training-loop integration, and a new three-tier evaluation (coupled-table oracle as primary, two-sided predictor retrieval as the deployable metric, independent two-sided oracle as a diagnostic-only upper bound — see the brainstorm memo's "Evaluation semantics" section for exact definitions). 24 new unit tests were added; the pre-existing test suite's 2 unrelated failures were confirmed unchanged. Implemented in an isolated git worktree/branch specifically so the default `asymmetric` path stays byte-for-byte unchanged regardless of this experiment's outcome.

**Training:** `scripts/run_condition_freeze_ablation.sh` (unmodified — the script's existing `EXTRA_OVERRIDES` env var was enough) with `EXTRA_OVERRIDES="model.conditioning_mode=symmetric_shared"`, matching every other 11.x/12.x/combine_side operating point exactly (RedCaps-150k, lr=1e-3, lr_label=1e-4, dim=16, alpha=0.5, buddy-init, same shared template as every prior run in this plan). 3 seeds × {trained, frozen} = 6 runs, wandb group `symmetric-shared-exp13`. A 2-epoch smoke test on the same config passed cleanly (both arms, no errors/NaNs) before committing to the full sweep.

**Analysis:** `scripts/analyze_condition_freeze_ablation.py` was extended with an additive `--metric-prefix` flag (default `test_oracle`, unchanged behavior) so the same paired frozen-vs-trained delta analysis could read the new `test_coupled_oracle/*` (primary) and `test_two_sided_predictor/*` (deployable) metric families the symmetric path logs instead of the old `test_oracle/*` keys, which this architecture no longer produces.

## Results

**Freeze-ablation deltas (`frozen − trained`, 3 seeds, paired within seed):**

| Metric tier | Direction | Option A (symmetric_shared) | txt-combine (12.5) | img-combine (C9, default) |
|---|---|---|---|---|
| Primary (`test_coupled_oracle`) | t2i | +0.10, mean/SEM=+0.5 (n.s.) | +0.13, mean/SEM=+0.3 (n.s.) | −0.27, mean/SEM=−2.0 (n.s., noise floor) |
| Primary (`test_coupled_oracle`) | i2t | **+0.53, mean/SEM=+0.8 (n.s., 2/3 wins)** | +0.40, mean/SEM=+4.0\* (sig, 3/3 wins) | **+4.67, mean/SEM=+32.1\* (sig, 3/3 wins)** |
| Deployable (`test_two_sided_predictor`) | t2i | +0.10, mean/SEM=+1.0 (n.s., 1/3 wins) | — | — |
| Deployable (`test_two_sided_predictor`) | i2t | **−0.40, mean/SEM=−1.6 (n.s., 0/3 wins)** | — | — |

`A = Δ_i2t − Δ_t2i` (the pre-registered asymmetry contrast): **img-combine +4.94 → txt-combine +0.27 → symmetric_shared +0.43** (primary metric).

Per-seed values (primary metric, `test_coupled_oracle`):
- t2i: seed 1 +0.40, seed 2 −0.30, seed 3 +0.20
- i2t: seed 1 +1.30, seed 2 −0.80, seed 3 +1.10

Freeze/drift sanity check passed: all 3 frozen-arm runs show `drift_from_init == 0` (the freeze mechanism works correctly under the new architecture too, same as every prior experiment in this plan).

## Interpretation

1. **Both the primary and deployable metrics agree, and both land inside the established noise floor (~0.1–0.7 R1) on both retrieval directions.** This is not merely "smaller than img-combine" (as the `combine_side="txt"` replication showed) — it is a null result by this project's own standard (`mean/SEM ≥ 2`), and the deployable-tier i2t delta is not even sign-consistent with the primary tier's (−0.40 vs. +0.53), the kind of sign instability expected from noise, not a real directional effect.
2. **Per Experiment 13's pre-registered success criteria** (originally `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 on `experiment/two_side_conditioning`; see this branch's copy for the promoted C11 summary): this is the **"Supports the architectural-amplification hypothesis"** outcome — `A` shrinks substantially relative to img-combine, and unlike the txt-combine replication, i2t itself is no longer a seed-replicated significant effect at all (2/3 and 0/3 win-rates on the two metric tiers, vs. 3/3 for both img-combine and txt-combine).
3. **This is a stronger result than the `combine_side="txt"` replication predicted.** That experiment showed the one-sided combiner *amplifies* the i2t effect by ~11×, implying a real i2t effect might survive symmetrization, just smaller. Instead, symmetrizing the combiner removes essentially all of it. The most direct reading: **C9's headline finding — "continued post-init training of the conditions regresses i2t retrieval" — is largely, possibly entirely, an artifact of conditioning only one modality**, not a property of buddy-graph training dynamics in general.
4. **What this does not (yet) establish**: why the one-sided combiner specifically produces this effect (a mechanistic account, e.g. via gradient asymmetry into the shared condition table, or the `other_proj` identity-init pathway never adapting) is still open — this experiment shows *that* removing the asymmetry removes the effect, not the causal mechanism by which the asymmetry produced it.

**Bottom line for the paper:** C9 (`docs/reports/2026-08-25_condition_freeze_ablation.md`) should no longer be read as "post-init condition training generically hurts i2t retrieval" — the evidence built across C9 → C10 (`combine_side="txt"`) → Experiment 13 (symmetric conditioning) now points to this being substantially an architectural artifact of the current one-sided combiner design, not an intrinsic property of training conditions on this buddy-graph signal.

## Caveats

- **`test_coupled_oracle` and `test_oracle` are different metric constructions, not the same metric under a different name** — `test_oracle` (img-combine/txt-combine) conditions only one side and projects the other through `other_proj`; `test_coupled_oracle` (symmetric_shared) conditions **both** sides with the same table representative and forms the similarity matrix from two conditioned outputs (see the brainstorm memo's "Evaluation semantics" section). The **deltas** (frozen − trained, computed identically within each family) are the valid unit of cross-architecture comparison here, per Experiment 13's own pre-registered design — but the raw R1 magnitudes are not directly comparable across families, only mentioned here as within-family context.
- **The bridge-node subgroup cross-reference (Experiment 13's secondary success criterion — does the `img_only_only`/`txt_only_only` concentration effect also symmetrize?) was not run.** `scripts/analyze_condition_retrieval_correlation.py` deeply assumes the asymmetric snapshot format (rebuilds `other_proj` from a saved state dict, reads a single `combine_side` to pick one gallery) and would need real adaptation, not a CLI flag, to support `symmetric_shared` checkpoints (no `other_proj`, two conditioned outputs). Deliberately deferred as a follow-up, not required to answer this experiment's primary question.
- **Only Option A was tested.** Separate-table/separate-combiner variants (B/C/D from the brainstorm) remain untested and are not motivated by this result — Option A already shows the effect disappears with the smallest, matched-capacity change, so there is no evidence yet that a more expressive symmetric variant would add anything.
- **n=3 seeds**, matching this plan's standard bar, but the effect sizes here are small in absolute terms (noise-floor range) — a null is harder to over-interpret than a large effect, but is still subject to the same seed-count caveat as every other result in this plan.

## Reproduction

> The commands below only work on `experiment/two_side_conditioning` (worktree `/project/CoSiR-two_side_conditioning`) — the `model.conditioning_mode` flag they reference does not exist on this branch.

```bash
# (from the worktree /project/CoSiR-two_side_conditioning, branch experiment/two_side_conditioning)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# smoke test (fast sanity gate)
SMOKE=1 WANDB_GROUP="symmetric-shared-exp13-smoke" WANDB_TAG="exp13-symmetric-shared-smoke" \
  EXTRA_OVERRIDES="model.conditioning_mode=symmetric_shared" bash scripts/run_condition_freeze_ablation.sh

# full 3-seed x 2-arm sweep
WANDB_GROUP="symmetric-shared-exp13" WANDB_TAG="exp13-symmetric-shared" \
  EXTRA_OVERRIDES="model.conditioning_mode=symmetric_shared" bash scripts/run_condition_freeze_ablation.sh

# analysis (primary + deployable metric tiers)
python scripts/analyze_condition_freeze_ablation.py --group "symmetric-shared-exp13" \
  --tag "exp13-symmetric-shared" --metric-prefix test_coupled_oracle
python scripts/analyze_condition_freeze_ablation.py --group "symmetric-shared-exp13" \
  --tag "exp13-symmetric-shared" --metric-prefix test_two_sided_predictor
```
