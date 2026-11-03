#!/usr/bin/env bash
# r19 node402 run order (spec §3.5, node order): the exact CLI commands for the run chat. Prints only; runs nothing.
# Use only CLUSTER=~/.claude/skills/cluster-run/cluster, from the worktree (branch) that holds the commit; always --node node402.
# Estimates come from r14's units.jsonl (A6000: X-c* 440-520 s per seed, X photo 150-260 s per seed, M5 about 4 min), plus loading.
cat <<'PLAN'
CLUSTER=/root/.claude/skills/cluster-run/cluster          # (spelled out: the `cluster` on PATH is another tool)
RUN="bash scripts/run_pp_node.sh"
0  once: the controller applies the DATA_MAP edit in the hand-back (16 entries, /local/wding/r19_in/*) to cluster.conf;
   commit with "cluster run" in the subject, then:   $CLUSTER sync --node node402
   Stores of the new three: ticket 02 must have written affection_local/{dinov2_large,clip_b32,clip_b16} first (photo, smoke).
1  probe (reads every input, imports pp_node and pp_matched in the node env; about 5 min with the data sync):
     $CLUSTER check  --node node402 --sync-data -- $RUN probe aux_art qwen_art aux_aff qwen_aff art_dinov2_large art_clip_b32 art_clip_b16 aff_dinov2_large aff_clip_b32 aff_clip_b16 art_clip_l14 aff_clip_l14 m5
     $CLUSTER launch --queue --node node402 --est 10m --label r19:probe -- $RUN probe aux_art qwen_art aux_aff qwen_aff art_dinov2_large art_clip_b32 art_clip_b16 aff_dinov2_large aff_clip_b32 aff_clip_b16 art_clip_l14 aff_clip_l14 m5
2  node smoke (clip_b32 seed 0, X-c* and X photo, 2 epochs, scratch output r19_pp_smoke; about 10 min):
     $CLUSTER check  --node node402 --sync-data -- $RUN smoke
     $CLUSTER launch --queue --node node402 --est 20m --label r19:smoke -- $RUN smoke
3  stage 0 (ii): CLIP L/14 c* seed 0 and photo seed 0 on node402 (training half; about 15 min):
     $CLUSTER check  --node node402 --sync-data -- $RUN regress-ii
     $CLUSTER launch --queue --node node402 --est 25m --label r19:ii -- $RUN regress-ii
   after the pull (see 6), the comparison on the local CPU (about 1 min), writes OUT/stage0/gate_ii.json:
     PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python src/test/20261103_past_picks/pp_node.py regress-ii --device cpu
   (it finds the pulled training, skips it, compares; exit 1 when the gate fails)
4  X-c*, one job (one GPU) per encoder, seeds 0 to 4 (about 45 min each; estimate 60m with loading):
     $CLUSTER check  --node node402 --sync-data -- $RUN part-a --encoder dinov2_large      (and clip_b32, clip_b16)
     $CLUSTER launch --queue --node node402 --est 60m --label r19:cstar_dinov2_large -- $RUN part-a --encoder dinov2_large
     $CLUSTER launch --queue --node node402 --est 60m --label r19:cstar_clip_b32     -- $RUN part-a --encoder clip_b32
     $CLUSTER launch --queue --node node402 --est 60m --label r19:cstar_clip_b16     -- $RUN part-a --encoder clip_b16
5  X photo, one job per encoder, each after that encoder's X-c* job ended with exit 0 (same node folder; about 20 min each; estimate 30m):
     $CLUSTER launch --queue --node node402 --est 30m --label r19:photo_dinov2_large -- $RUN photo --encoder dinov2_large
     $CLUSTER launch --queue --node node402 --est 30m --label r19:photo_clip_b32     -- $RUN photo --encoder clip_b32
     $CLUSTER launch --queue --node node402 --est 30m --label r19:photo_clip_b16     -- $RUN photo --encoder clip_b16
   M5 (DINOv2 matched arm, full scale, seeds 0 to 4; about 5 min; estimate 15m), any time after the probe:
     $CLUSTER check  --node node402 --sync-data -- $RUN m5
     $CLUSTER launch --queue --node node402 --est 15m --label r19:m5 -- $RUN m5
6  after each job:   $CLUSTER watch <tag> (background)    $CLUSTER pull --tag <tag>
   before the first pull, from the worktree you pull in:
     mkdir -p res && ln -sfn /data/SSD2/cosir_r19/20261103_past_picks/outputs res/r19_pp
   (smoke: res/r19_pp_smoke -> .../smoke_outputs; probe: res/r19_pp_probe -> .../probe_outputs; same pattern)
   Pulled units: OUT/node/units.jsonl, OUT/node/cstar_<enc>.json and photo_<enc>.json (status ok | failed; a seed that fails
   twice stops its kind, r14 D18). Gate (v) and the exports run locally after the pulls (ticket 05's and 06's commands).
7  smoke result on the node: list with `$CLUSTER clean --node node402 --jobs --tag <smoke tag>` (no --yes until the user agrees)
PLAN
