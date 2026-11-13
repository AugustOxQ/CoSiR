#!/usr/bin/env bash
# The λ = 0 control (run chat 3, 2026-11-13; decision 2026-11-12 20:33): one arm, dec-plain_l0_w5_S2, on seeds 0 to 4,
# its held exports (GPU, as stage 2), then the stage 2 table with the λ = 0 arm added (`stage2_lam0`) and the
# diagnostics for the same five arms. `diag` overwrites scores/diag.*; the 4-arm version is copied first.
source "$(dirname "${BASH_SOURCE[0]}")/r21_queue_lib.sh"
r21_args "$@"
LAM0=dec-plain_l0_w5_S2
ARMS=(dec-off dec-plain_l10_w5_S2 dec-start_l10_w5_S2 dec-teacher_l10_w5_S2 "$LAM0")
SCORES=/data/SSD2/cosir_r21/20261111_dec_heads/scores
for seed in 0 1 2 3 4; do r21_gpu lam0 15m "${R21_RUN[@]}" train --arm "$LAM0" --seeds "$seed" --device cuda; done
r21_gpu lam0 10m "${R21_RUN[@]}" held --arm "$LAM0" --seeds 0 1 2 3 4
r21_cpu lam0 2h "${R21_TABLE[@]}" table --arms "${ARMS[@]}" --name stage2_lam0
if [ "$R21_DRY" = 0 ] && [ ! -f "$SCORES/diag_before_lam0.json" ]; then
  cp "$SCORES/diag.json" "$SCORES/diag_before_lam0.json"; cp "$SCORES/diag.md" "$SCORES/diag_before_lam0.md"
fi
r21_cpu lam0 - "${R21_TABLE[@]}" diag --arms "${ARMS[@]}"
r21_done lam0
