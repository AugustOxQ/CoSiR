#!/bin/bash
# r29 stage 0 (regression check, spec §3.5): gates (i) to (vii) on K0 and the frozen records. Run from any checkout that
# holds the merged r29 code; the script works in its own folder. `--dry-run` prints the commands.
#   0a  gates (iii), (iv) store part, (i)(a), (i)(c), (ii), (v) image and text  -> modules on the integration branch
#   0b  gates of the stage 2 pieces that need no K*: (v) and (vi) MMArt, (vii) photo trainer, (v) and (vii) photo scorer,
#       (vii) ablation. A step whose module is not in this checkout is skipped and listed, not failed.
# Gate (i)(b) is a test (pytest), writes no record, and is not run here. Gate (iv)'s hash part needs trained K1 and K2:
# stage 1a runs it. Gates (viii) depend on K*'s form: stage 2 runs them. A gate record that exists is not run again
# (each gate has three tries); R29_FORCE=1 overrides.
source "$(dirname "${BASH_SOURCE[0]}")/r29_run_common.sh"
G=records/gates
S1=$OUT/cstar/siglip1

echo "== 0a =="
have $G/iii.json || q cpu 10m gate-iii -- "${PYC[@]}" r29_captions.py --write

# gate (iv) store part: the prefixed SigLIP 1 store and its two replays. Est: the plain store's 308,723 captions took
# txt_seconds 369.9 s (store progress.json), so the prefixed one is under 15 minutes.
PSTORE=/data/SSD2/pre_extract/artelingo_full/siglip_so400m_capprefix
if ! have $G/iv_store.json; then
  if [[ -f $PSTORE/index.json ]]; then
    q gpu short gate-iv-store -- "${PYC[@]}" r29_extract.py gate-iv --store "$PSTORE"
  else
    q gpu short extract-prefix -- "${PYC[@]}" r29_extract.py extract --gate-iv
  fi
fi

# gate (i)(a): r14's trainer and the new trainer (Qwen3 input), seed 0 each, about 6 minutes each (r14 run__s0 seconds
# 340.9). r14's own second run only when the first two differ (spec §3.5 (i)(a)).
if ! have $G/i_a.json; then
  [[ -d $OUT/gate_i_a/r14/weights ]] || q gpu short gate-i-a-r14 -- "${PYC[@]}" r29_trainer.py gate-i-a --run r14 --device cuda
  [[ -d $OUT/gate_i_a/new/weights ]] || q gpu short gate-i-a-new -- "${PYC[@]}" r29_trainer.py gate-i-a --run new --device cuda
  if ((DRY)) || ! "${PYC[@]}" -c 'import sys; import r29_trainer as T; r = T.gate_i_a_compare(write=False); sys.exit(0 if r["details"]["comparisons"]["new_vs_r14"]["equal"] else 1)'; then
    [[ -d $OUT/gate_i_a/r14_b/weights ]] || q gpu short gate-i-a-r14_b -- "${PYC[@]}" r29_trainer.py gate-i-a --run r14_b --device cuda
  fi
  have $G/i_a_weights.json || q cpu 10m gate-i-a-compare -- "${PYC[@]}" r29_trainer.py gate-i-a --compare
  q cpu 20m gate-i-a-verdict -- "${PYC[@]}" r29_decide.py gate-i-a
fi
# diagnostic (e) inputs: the new run's nine validation text cells and six image cells
have records/text_gate_i_a_new.json || q gpu short score-gate-i-a-new -- "${PYC[@]}" r29_text.py score --arm gate_i_a_new --set art_val --device cuda
have records/groups_gate_i_a_new.json || q cpu 20m groups-gate-i-a-new -- "${PYC[@]}" r29_decide.py groups --arm gate_i_a_new --seeds 0

# gate (i)(c): the export part on K0's weights, then the valpick part
have $G/i_c.json || q gpu short gate-i-c -- "${PYC[@]}" r29_trainer.py gate-i-c
have $G/i_c_valpick.json || q cpu 20m gate-i-c-valpick -- "${PYC[@]}" r29_decide.py gate-i-c-valpick

# gate (ii): K0's text scores replayed against r28 (stage0 log: 1.4 minutes)
have $OUT/text/qwen/scores || q gpu short gate-ii-score -- "${PYC[@]}" r29_text.py score --arm qwen --set art_val --device cuda
have $G/ii.json || q cpu 10m gate-ii -- "${PYC[@]}" r29_text.py gate-ii

# gate (v) image: replays of r17 SC1, SC3, SC7 and r14 Q1 on K0. Source of the estimate: the earlier run took 2183 s
# (logs/gate_v_image.log, 36.45 minutes), so --est 45m. Gate (v) text: records v_text.json seconds 25.
have $G/v_image.json || q cpu 45m gate-v-image -- "${PYC[@]}" r29_image_cells.py gate-v-image
have $G/v_text.json || q cpu 10m gate-v-text -- "${PYC[@]}" r29_text_cells.py gate-v-text

echo "== 0b (gates of the stage 2 pieces) =="
# MMArt, ticket 11 (gate (v) MMArt rebuilds; gate (vi) export part on K0's weights, then the scorer part on that try).
# Estimates are guesses (no timing source): r28's MMArt run left no log.
if [[ -f r29_mmart.py ]]; then
  have $G/v_mmart.json || q cpu 30m gate-v-mmart -- "${PYC[@]}" r29_mmart.py gate-v-mmart
  if ! have $G/vi.json; then
    q cpu 30m gate-vi-export -- "${PYC[@]}" r29_mmart.py gate-vi export --device cpu
    if ((DRY)); then TRY='<N>'; else TRY=$(ls -d "$OUT"/gate_vi/try* | sed 's/.*try//' | sort -n | tail -1); fi
    q cpu 45m gate-vi-score -- "${PYC[@]}" r29_mmart.py gate-vi score --try "$TRY"
  fi
else SKIPPED+=(r29_mmart.py); echo "skip: r29_mmart.py not in this checkout (ticket 11)"; fi
# photo trainer, ticket 12 (gate (vii) trainer part, fixed batch)
if [[ -f r29_photo.py ]]; then
  have $G/vii_trainer.json || q cpu 15m gate-vii-trainer -- "${PYC[@]}" r29_photo.py gate-vii
else SKIPPED+=(r29_photo.py); echo "skip: r29_photo.py not in this checkout (ticket 12)"; fi
# ---- TO CHECK AFTER TICKET 13 MERGES: r29_photo_cells.py CLI is taken from issues/13-photo-cells.md, not from the module ----
if [[ -f r29_photo_cells.py ]]; then
  have $G/v_photo.json || q cpu 30m gate-v-photo -- "${PYC[@]}" r29_photo_cells.py gate-v-photo
  have $G/vii_photo.json || q cpu 30m gate-vii-photo -- "${PYC[@]}" r29_photo_cells.py gate-vii-photo
else SKIPPED+=(r29_photo_cells.py); echo "skip: r29_photo_cells.py not in this checkout (ticket 13)"; fi
# ---- end of the ticket 13 block ----
# ablation, ticket 14 (gate (vii) ablation): its record exists at try 2; the controller re-runs it once at the build's end
# (decisions.md 01:53), so it is not run here.
if [[ -f r29_ablation.py ]]; then
  have $G/vii_ablation.json || q cpu 15m gate-vii-ablation -- "${PYC[@]}" r29_ablation.py gate-vii
else SKIPPED+=(r29_ablation.py); echo "skip: r29_ablation.py not in this checkout (ticket 14)"; fi

q cpu 5m gates-merge -- "${PYC[@]}" r29_decide.py gates
summary "stage 0"
