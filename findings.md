# Findings

- The existing Experiment 13 implementation is committed on branch `experiment/symmetric_conditioning_exp13`.
- `scripts/run_condition_freeze_ablation.sh` runs arms serially in order: trained then frozen. `SMOKE=1` sets two epochs, evaluation every epoch, and seed 1.
- The shared result target resolves to `/project/CoSiR/res/CoSiR_condition_freeze_ablation/redcaps_150k`.
