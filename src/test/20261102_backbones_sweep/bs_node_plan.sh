#!/usr/bin/env bash
# r14 node402 run order (spec 3.5): the exact CLI commands for the run chat. Prints only; runs nothing.
# Use only CLUSTER=~/.claude/skills/cluster-run/cluster, from the worktree (branch) that holds the commit; always --node node402.
# Run `launch` for the three part-a encoders and the three sweep workers one after another: each takes the lowest free GPU.
cat <<'PLAN'
CLUSTER=/root/.claude/skills/cluster-run/cluster          # (spelled out: the `cluster` on PATH is another tool)
0  commit with "cluster run" in the subject, then:   $CLUSTER sync --node node402
1  probe (about 1 min after the data is there):
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh probe aux_art aux_aff viii art_siglip2 aff_siglip2
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh probe aux_art aux_aff viii art_siglip2 aff_siglip2
2  smoke (4 units, scratch output r14_bs_smoke), then pull and clean it:
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh smoke      (the big Qwen stores sync here: run it in the background)
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh smoke
3  0(viii) and prep (each one GPU):
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh viii
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh viii
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh prep
4  part a, three launches (X-c* x5, then X photo x5 per encoder):
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh part-a --encoder siglip2     (and siglip1, clip_l14)
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh part-a --encoder siglip2
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh part-a --encoder siglip1
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh part-a --encoder clip_l14
5  sweeps, three workers on one claim queue (photo rows first, then knob rows):
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh sweep --kind photo knob
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh sweep --kind photo knob     (x3; add --reset-claims to the first only after a lost job)
6  after each job:   $CLUSTER watch <tag>    $CLUSTER pull --tag <tag>    then, for time left:
     python src/test/20261102_backbones_sweep/bs_node.py estimate
7  pulled results land in <worktree>/res/<name>/ (name = r14_bs, r14_bs_smoke, r14_bs_probe). Make res/r14_bs a symlink to
   /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs before the first pull, or move the pulled folder there (see the t10b hand-back).
8  smoke and probe results on the node: $CLUSTER clean --node node402 --path /local/wding/res/r14_bs_smoke   (list first; --yes only after review)
PLAN
