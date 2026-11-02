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
     FIRST wait for prep's done line (`$CLUSTER watch <prep tag>`): two first builds of the k80 graph race.
     $CLUSTER check --node node402 --sync-data -- bash scripts/run_bs_node.sh sweep --kind photo knob
     $CLUSTER launch --node node402 -- bash scripts/run_bs_node.sh sweep --kind photo knob     (x3; add --reset-claims to the first only after a lost job)
6  after each job:   $CLUSTER watch <tag>    $CLUSTER pull --tag <tag>    then, for time left:
     python src/test/20261102_backbones_sweep/bs_node.py estimate
   after each part a pull, the exports on the local CPU (the stage 1 and stage 3 inputs; stage 2 needs them too):
     bash src/test/20261102_backbones_sweep/bs_run_cpu.sh exports-part-a --enc <enc>
   (once, before stage 2: bash .../bs_run_cpu.sh exports-qwen). Run order after the node: exports-qwen, exports-part-a for all
   three encoders, stage1, stage2, stage3 (cells), C13 (Codex re-derives; commit RECORDS/c13.json), decide, stage4, stage5.
7  pulled results land in <worktree>/res/<name>/ (name = r14_bs, r14_bs_smoke, r14_bs_probe); the pull follows a symlink there
   (proved with the probe, verified 24 files). Before the first pull, from the worktree you pull in:
     mkdir -p res && ln -sfn /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs res/r14_bs     (smoke: res/r14_bs_smoke -> .../smoke_pull)
8  smoke result on the node: list with `$CLUSTER clean --node node402 --jobs --tag <smoke tag>` (no --yes until the user agrees)
PLAN
