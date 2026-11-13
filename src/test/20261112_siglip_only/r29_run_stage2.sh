#!/bin/bash
# r29 stage 2 (spec §3.5, D3, D24, A1): usage  r29_run_stage2.sh {a|b} [--dry-run] [--amendment A1]
# Refuses unless records/pass_rule.json says "status": "pass" (D3), or says "fail" and the runner is called with --amendment A1 and
# the committed records/amendment_A1.json checks out (r29_paths.stage2_gate; spec §4 A1). Under A1 every Python step runs with
# R29_AMENDMENT=A1 in its environment (visible in every command line) and the records say "stage 2 ran on the fail under A1".
# Run the ladders and the sheet (`r29_ladder.py`) by hand with R29_AMENDMENT=A1 in the environment too. Order inside each part (D24): E2 and E3, MMArt, the photo
# model, the ablation. A piece that fails goes on the failure list and the others go on (§3.5).
#   a  2a sweep (development only): stores, training, validation exports, picks and ranges on validation.
#      The run chat commits the pick records before part b (the test commands of each piece refuse without them).
#   b  2b claim tests: held and test exports, test cells, the ablation table. ENDS BEFORE the ladders (Codex first, C13).
# D7: training jobs here are 33 to 40 minutes, so a run on a weekday between 10:00 and 19:00 (Amsterdam) is refused unless
# R29_ALLOW_DAYTIME=1 (a DAS6 run is the other way; this script is the local one).
# Estimates (--est) and their sources:
#   E2, E3 training, 5 seeds in one job: 40m. r14 train seconds, node402, k11=40: siglip2 369.3, 417.8, 421.9, 421.3, 409.4
#       (sum 2039.7 s = 34.0 min); clip_l14 398.1, 414.5, 409.5, 401.2, 380.6 (1994.0 s = 33.2 min); plus start-up.
#   F1 training, 5 seeds: 20m. Spec §9: r14 photo model 187 s per seed on node402 (5 x 187 s = 15.6 min).
#   Affection store, 526,749 explanations: 20m. SigLIP 1 text pass of the 308,723 ArtELingo captions took txt_seconds 369.9
#       (store progress.json); scaled by 526,749 / 308,723 = 631 s = 10.5 min.
#   Ablation, 5 seeds per removal: 40m. r28: instructed_graphs run seconds 498.3, 500.2, 422.9, 432.2, 440.4 (sum 2294 s =
#       38.2 min; r29_facts.md §8); the other three removals have no recorded time, spec §9 says about 7.5 minutes a run.
#   Ablation scoring (r29_ablation.py score): 20m. r28 ablation/score.log: 16.09 minutes (CPU queue).
#   SC9: two CPU image-group scorings, about 30 minutes each (spec §9); 45m as for the other CPU scorings.
#   Image picks and cells (CPU): 45m. No direct timing; gate (v) image, which replays picks and cells, took 2183 s
#       (logs/gate_v_image.log). MMArt export and picks and cells, photo picks and cells: no source, same 45m bound; 20m export.
source "$(dirname "${BASH_SOURCE[0]}")/r29_run_common.sh"
PART=${ARGS[0]:-}
[[ $PART == a || $PART == b ]] || { echo "usage: $0 {a|b} [--dry-run] [--amendment A1]"; exit 2; }
require_pass
require_committed records/prefix_pick.json
if [[ $PART == a ]]; then
  dow=$(TZ=Europe/Amsterdam date +%u); hr=$(TZ=Europe/Amsterdam date +%-H)
  if ((dow <= 5 && hr >= 10 && hr < 19)) && [[ -z ${R29_ALLOW_DAYTIME:-} ]]; then
    if ((DRY)); then echo "[dry-run] weekday daytime: a real run refuses (D7)"; else echo "REFUSE: weekday 10:00 to 19:00 (D7); set R29_ALLOW_DAYTIME=1 to override"; exit 2; fi
  fi
fi
K=$(kstar); FORM=$(kform)
SUF=''; [[ $FORM == prefixed ]] && SUF=_prefix
echo "K* = $K ($FORM)"
AB=$OUT/ablation/siglip1/$K

declare -A GROUP_PIDS

# ---------------------------------------------------------------- E2 and E3 (ticket 10)
for E in siglip2 clip_l14; do
  case $E in siglip2) SRC=siglip2$SUF; XENC=siglip2_so400m;; clip_l14) SRC=clipl14$SUF; XENC=clip_l14;; esac
  echo "== $E (caption source $SRC) =="
  if [[ $PART == a ]]; then
    # the prefixed store exists only when K2 won; the plain stores are the existing ones
    STORE=/data/SSD2/pre_extract/artelingo_full/$XENC$([[ $FORM == prefixed ]] && echo _capprefix)
    if [[ $FORM == prefixed && ! -f $STORE/index.json ]]; then
      q gpu short "extract-$SRC" -- "${PYC[@]}" r29_extract.py extract --encoder "$XENC" || continue
    fi
    # a rerun of 2a: `stores` raises on a gate (viii) that already passed (r29_encoders.py gate_viii), so a pass skips the step
    if [[ $(gate_status "viii_$SRC") == pass ]]; then echo "skip (gate viii_$SRC already pass)"; SKIPPED+=("records/gates/viii_$SRC.json")
    else q cpu 10m "gate-viii-$SRC" -- "${PYC[@]}" r29_encoders.py stores --encoder "$E" || continue; fi
    train_retry "$SRC" done_k "$OUT/cstar/$E/weights/$SRC" 40m "train-$E" "0 1 2 3 4" -- "${PYC[@]}" r29_encoders.py train \
      --encoder "$E" --seeds @SEEDS@ || continue
    q gpu short "export-val-$E" -- "${PYC[@]}" r29_encoders.py export --encoder "$E" --kind val || continue
    if ! have "records/image_groups_$SRC.json"; then
      q cpu 45m "image-groups-$E" -- "${PYC[@]}" r29_decide.py image-groups --encoder "$E" &
      GROUP_PIDS[$E]=$!
    fi
    q cpu 45m "picks-$E" -- "${PYC[@]}" r29_encoders.py picks --encoder "$E"
  else
    need_committed "cells-$E" "records/image_picks_$SRC.json" records/encoder_pick.json || continue
    q gpu short "export-held-$E" -- "${PYC[@]}" r29_encoders.py export --encoder "$E" --kind held || continue
    q cpu 45m "cells-$E" -- "${PYC[@]}" r29_encoders.py cells --encoder "$E"
  fi
done

# ---------------------------------------------------------------- MMArt (ticket 11)
echo "== MMArt =="
if [[ ! -f r29_mmart.py ]]; then SKIPPED+=(r29_mmart.py); echo "skip: r29_mmart.py not in this checkout"
elif [[ $PART == a ]]; then
  q cpu 20m "mmart-export-$K" -- "${PYC[@]}" r29_mmart.py export --arm "$K" && \
    q cpu 45m "mmart-picks-$K" -- "${PYC[@]}" r29_mmart.py picks --arm "$K"
else
  need_committed "mmart-cells-$K" "records/mmart_picks_$K.json" && \
    q cpu 45m "mmart-cells-$K" -- "${PYC[@]}" r29_mmart.py cells --arm "$K"
fi

# ---------------------------------------------------------------- photo model F1 (tickets 12 and 13)
echo "== photo =="
# F1 trains only on a passed gate (viii) affection; any other status goes on the failure list (no crash record: D21's are for real crashes)
photo_gate_ok() {
  local s; s=$(gate_status viii_affection)
  if [[ $s == pass || $s == missing ]]; then return 0; fi
  echo "F1 not trained: gate viii_affection is $s"; FAILS+=("train:f1 not run: gate viii_affection $s"); return 1
}
done_f1() { [[ -f $OUT/photo/siglip1/s$2/emotion.pt && -f $OUT/photo/siglip1/s$2/run.json ]]; }
if [[ ! -f r29_photo.py ]]; then SKIPPED+=(r29_photo.py); echo "skip: r29_photo.py not in this checkout"
elif [[ $PART == a ]]; then
  AFF=/data/SSD2/pre_extract/affection_local/siglip_so400m_txt
  {
    if [[ -f $AFF/index.json ]]; then have records/gates/viii_affection.json || q cpu 10m gate-viii-affection -- "${PYC[@]}" r29_photo.py gate-viii
    else q gpu 20m affection-store -- "${PYC[@]}" r29_photo.py store --device cuda; fi
  } && photo_gate_ok && train_retry f1 done_f1 x 20m train-f1 "0 1 2 3 4" -- "${PYC[@]}" r29_photo.py train --seeds @SEEDS@ --device cuda && {
    q gpu short f1-export-val-affection -- "${PYC[@]}" r29_photo.py export --dataset affection --splits val
    q gpu short f1-export-val-emoset -- "${PYC[@]}" r29_photo.py export --dataset emoset --splits val
    if [[ -f r29_photo_cells.py ]]; then q cpu 45m "photo-picks-$K" -- "${PYC[@]}" r29_photo_cells.py picks --arm "$K"
    else SKIPPED+=(r29_photo_cells.py); echo "skip: r29_photo_cells.py not in this checkout (ticket 13)"; fi
  }
else
  q gpu short f1-export-test-affection -- "${PYC[@]}" r29_photo.py export --dataset affection --splits test
  q gpu short f1-export-test-emoset -- "${PYC[@]}" r29_photo.py export --dataset emoset --splits test
  if [[ -f r29_photo_cells.py ]]; then
    need_committed "photo-cells-$K" "records/photo_picks_$K.json" && \
      q cpu 45m "photo-cells-$K" -- "${PYC[@]}" r29_photo_cells.py cells --arm "$K"
  else SKIPPED+=(r29_photo_cells.py); echo "skip: r29_photo_cells.py not in this checkout (ticket 13)"; fi
fi

# ---------------------------------------------------------------- ablation R1 to R4 (ticket 14)
echo "== ablation =="
if [[ ! -f r29_ablation.py ]]; then SKIPPED+=(r29_ablation.py); echo "skip: r29_ablation.py not in this checkout"
else
  for ROW in instructed_graphs content_correction buddy relational; do
    if [[ $PART == a ]]; then
      # D21's same-seed rerun when the process dies (for example an out-of-memory kill): a non-zero exit runs the train step a
      # second time. r29_ablation.py train caps every seed at two attempts in its journal attempts__s<seed>.json (an attempt
      # left `running` by a dead process counts as one) and refuses an `inconclusive` seed, so the second call never gives a
      # third try. The first failure goes on the failure list only when the second call fails too (qrun, then q).
      { qrun gpu 40m "ablation-train-$ROW" -- "${PYC[@]}" r29_ablation.py train --row "$ROW" --seeds 0 1 2 3 4 || \
          q gpu 40m "ablation-train-$ROW" -- "${PYC[@]}" r29_ablation.py train --row "$ROW" --seeds 0 1 2 3 4; } && \
        q gpu short "ablation-export-val-$ROW" -- "${PYC[@]}" r29_ablation.py export --row "$ROW" --seeds 0 1 2 3 4 --kind val
    else
      q gpu short "ablation-export-held-$ROW" -- "${PYC[@]}" r29_ablation.py export --row "$ROW" --seeds 0 1 2 3 4 --kind held
    fi
  done
  [[ $PART == b ]] && q cpu 20m ablation-score -- "${PYC[@]}" r29_ablation.py score
fi
if [[ $PART == a ]]; then
  for E in siglip2 clip_l14; do
    if [[ -n ${GROUP_PIDS[$E]:-} ]] && ! wait "${GROUP_PIDS[$E]}"; then FAILS+=("image-groups-$E"); fi
  done
  SC9_OK=1
  for E in siglip2 clip_l14; do
    case $E in siglip2) SRC=siglip2$SUF;; clip_l14) SRC=clipl14$SUF;; esac
    if [[ ! -f records/image_groups_$SRC.json ]] && (( ! DRY )); then
      FAILS+=("sc9: encoder-pick not run (records/image_groups_$SRC.json missing)")
      SC9_OK=0
    fi
    if [[ $(gate_status "sc9_replay_$E") == pass ]]; then
      echo "skip (gate sc9_replay_$E already pass)"; SKIPPED+=("records/gates/sc9_replay_$E.json")
    elif ! q cpu 45m "sc9-replay-$E" -- "${PYC[@]}" r29_decide.py sc9-replay --encoder "$E"; then
      FAILS+=("sc9: encoder-pick not run (sc9-replay-$E failed)")
      SC9_OK=0
    fi
  done
  if ((SC9_OK)); then have records/encoder_pick.json || q cpu 45m encoder-pick -- "${PYC[@]}" r29_decide.py encoder-pick; fi
fi
echo "NEXT (run chat, not this script): $([[ $PART == a ]] && echo 'commit the pick records (image_picks_*, mmart_picks_*, photo picks, encoder_pick.json); Codex re-derives encoder_pick.json (C13) before r29_run_stage2.sh b' || echo 'Codex re-derives every pick, range and cell; then the ladders (SC4 per encoder, SC5, SC6), the ablation sheet')"
summary "stage 2$PART"
