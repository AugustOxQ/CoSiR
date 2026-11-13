#!/bin/bash
# r29 stage 1 (spec §3.5): usage  r29_run_stage1.sh {a|b|c|t8} [--dry-run]
#   a  1a sweep: K1 and K2, seed 0, text group on validation; P1 comparators; gate (iv) hash part; merged gates.
#      ENDS BEFORE SC1: the run chat runs `r29_decide.py sc1`, Codex re-derives it, the chat commits records/prefix_pick.json
#      (and the 1a records) before part b.
#   b  1b incremental: K* seeds 1 to 4, validation exports and scores, picks and ranges (text and image), groups.
#      Refuses unless prefix_pick.json is committed. ENDS BEFORE SC2: the chat runs `r29_decide.py sc2`, Codex re-derives
#      it, the chat commits records/pass_rule.json with the pick records before part c.
#   t8 Table 8 (SC8, D8) alone: K*'s three caption towers (seed 0) against the SigLIP 1 text pass. Needs only its own inputs: the
#      committed prefix_pick.json (K*) and K*'s seed 0 weights from 1a. No pass rule: §6 "If SC1 or SC2 is parked ... Table 8
#      runs", so run it after 1b whatever SC2 said (a separate part, not a step of 1c, because 1c refuses without a pass_rule.json).
#      If SC1 left no K* (both arms out) it goes on the failure list with the reason. Writes records/time_text.json.
#   c  1c claim test: K* on ArtELingo test. Starts only from committed pass_rule.json, prefix_pick.json and
#      the pick records (checked with git; refuses). Runs on a pass and on a fail (D9). ENDS BEFORE the ladders: Codex
#      re-derives every pick, range and cell first (§3.4 "Order and C13"), then the chat runs the ladders.
# Every training job is one seed, 340 to 410 s in r14 (outputs/cstar/siglip1/train/k11=40/run__s*.json "seconds": 340.9,
# 407.3, 400.7, 410.0, 402.2 on node402), so --short and far under D7's hour. Text scoring: r28's logs, validation 3.28 min and
# held 6.90 min for all three aspects and five seeds.
source "$(dirname "${BASH_SOURCE[0]}")/r29_run_common.sh"
PART=${ARGS[0]:-}
[[ $PART == a || $PART == b || $PART == c || $PART == t8 ]] || { echo "usage: $0 {a|b|c|t8} [--dry-run]"; exit 2; }
W=$OUT/cstar/siglip1/weights
SEEDS5="0 1 2 3 4"

train_arm() {   # train_arm <arm> <seed>...
  local arm=$1; shift
  local s
  for s in "$@"; do
    train_retry "$arm" done_k "$W/$arm" short "train-$arm-s$s" "$s" -- "${PYC[@]}" r29_trainer.py train \
      --encoder siglip1 --cap "$arm" --seeds @SEEDS@ || return 1
  done
}

if [[ $PART == a ]]; then
  echo "== 1a =="
  trained=()
  for arm in siglip1 siglip1_prefix; do
    if [[ $arm == siglip1_prefix && -f records/gates/iv_store.json ]] && [[ $(jget records/gates/iv_store.json status) != pass ]]; then
      echo "K2 not trained: gate iv_store is not pass (sc1 reads the gate; §3.5 row 1a)"; SKIPPED+=("train:$arm"); continue
    fi
    train_arm "$arm" 0 || continue
    trained+=("$arm")
    have "$OUT/text/$arm/provenance/art_val/emotion__s0.json" || \
      q gpu short "score-val-$arm-s0" -- "${PYC[@]}" r29_text.py score --arm "$arm" --set art_val --seeds 0 --device cuda
  done
  # P1: plain SigLIP 1 captions in both forms (no training); their validation picks (D20 reads the seed 0 text group)
  for p1 in p1_siglip1 p1_siglip1_prefix; do
    q gpu short "score-val-$p1" -- "${PYC[@]}" r29_text.py score --arm "$p1" --set art_val --device cuda && \
      q cpu 10m "picks-$p1" -- "${PYC[@]}" r29_text.py picks --arm "$p1"
  done
  # gate (iv) hash part: the scorer's store hash against the trainer's records, for the arms that trained
  if ((${#trained[@]})); then
    q cpu 10m gate-iv-hash -- "${PYC[@]}" r29_text.py hash-check --arms "${trained[@]}"
  fi
  q cpu 5m gates-merge -- "${PYC[@]}" r29_decide.py gates
  echo "NEXT (run chat, not this script): python r29_decide.py sc1; Codex re-derivation (C13); commit records/prefix_pick.json and the 1a records; then r29_run_stage1.sh b"
  summary "stage 1a"
fi

if [[ $PART == b ]]; then
  echo "== 1b =="
  require_committed records/prefix_pick.json
  K=$(kstar)
  echo "K* = $K"
  # validation text picks are r23's rule on five seeds; image groups and picks follow the exports
  train_arm "$K" 1 2 3 4 || { summary "stage 1b (K* did not finish: no scoring)"; }
  q gpu short "export-val-$K" -- "${PYC[@]}" r29_trainer.py export --encoder siglip1 --cap "$K" --kind val --seeds $SEEDS5
  q gpu short "score-val-$K-s1-4" -- "${PYC[@]}" r29_text.py score --arm "$K" --set art_val --seeds 1 2 3 4 --device cuda
  q cpu 10m "picks-text-$K" -- "${PYC[@]}" r29_text.py picks --arm "$K"
  # No measured time for the two image jobs (r28's valpick and r29's picks left none); the nearest is gate (v) image,
  # 2183 s for picks and cells replays (logs/gate_v_image.log), so 45m is an upper bound; groups is a smaller piece: 30m.
  q cpu 30m "groups-$K" -- "${PYC[@]}" r29_decide.py groups --arm "$K" --seeds $SEEDS5
  q cpu 45m "picks-image-$K" -- "${PYC[@]}" r29_image_cells.py picks --arm "$K" --encoder siglip1
  echo "NEXT (run chat, not this script): python r29_decide.py sc2; Codex re-derivation (C13); commit records/pass_rule.json, text_$K.json, image_picks_$K.json, groups_$K.json, then r29_run_stage1.sh c"
  summary "stage 1b"
fi

if [[ $PART == c ]]; then
  echo "== 1c =="
  require_committed records/pass_rule.json records/prefix_pick.json
  K=$(kstar)
  require_committed "records/text_$K.json" "records/image_picks_$K.json"
  P1FORM=plain; [[ -f records/prefix_pick.json ]] && P1FORM=$(jget records/prefix_pick.json p1_form)
  [[ $P1FORM == prefixed ]] && require_committed records/text_p1_siglip1_prefix.json
  echo "K* = $K, P1 form = $P1FORM"
  q gpu short "export-held-$K" -- "${PYC[@]}" r29_trainer.py export --encoder siglip1 --cap "$K" --kind held --seeds $SEEDS5
  q gpu short "score-held-$K" -- "${PYC[@]}" r29_text.py score --arm "$K" --set art_held --device cuda
  # r23's plain P1 rows are r23's own files; the prefixed P1 rows are this round's and need their held scores
  [[ $P1FORM == prefixed ]] && q gpu short "score-held-p1-prefix" -- "${PYC[@]}" r29_text.py score --arm p1_siglip1_prefix --set art_held --device cuda
  q cpu 10m "text-held-$K" -- "${PYC[@]}" r29_text.py cells --arm "$K"
  # Estimates without a measured source: 10,000-draw cells; the image cells use gate (v) image's 2183 s as the bound
  q cpu 30m "text-cells-$K" -- "${PYC[@]}" r29_text_cells.py cells --arm "$K"
  q cpu 45m "image-cells-$K" -- "${PYC[@]}" r29_image_cells.py cells --arm "$K" --encoder siglip1
  echo "Table 8 is not a step of 1c: run r29_run_stage1.sh t8 (it needs no pass rule)"
  echo "NEXT (run chat, not this script): Codex re-derives every pick, range and cell (<= 1e-6); then r29_ladder.py ladder --claim SC3 --arm $K, diag-d, sheet, r29_decide.py diagnostics"
  summary "stage 1c"
fi

if [[ $PART == t8 ]]; then
  echo "== Table 8 (SC8) =="
  K=$(kstar)
  echo "K* = $K"
  # a non-exiting check: with no K* the SigLIP 1 half has nothing to sit beside, so the table is inconclusive, not a stop
  if need_committed table8-time records/prefix_pick.json; then
    if ((!DRY)) && ! { done_k "$W/$K" 0; }; then
      echo "SKIP Table 8: K* seed 0 weights are missing in $W/$K"; FAILS+=("table8-time: K* weights missing in $W/$K")
    else
      have records/time_text.json || q gpu short table8-time -- "${PYC[@]}" r29_time.py --weights-dir "$W/$K"
    fi
  fi
  summary "Table 8"
fi
