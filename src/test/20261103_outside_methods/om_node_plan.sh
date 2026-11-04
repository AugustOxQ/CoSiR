#!/usr/bin/env bash
# r18 TPIPS on node402 (ticket 04): the exact commands for the controller, in order. Prints only; runs nothing.
# Use only CLUSTER=~/.claude/skills/cluster-run/cluster (the `cluster` on PATH is another tool), from the worktree
# (branch) that holds the commit, always --node node402. Launches use --queue, run in the background.
#
# Time per shard job, as variables (override from the environment; D20 batch mode is being measured, so both move):
#   OM_BATCH       batch size of every extract and of the gate probe (default 1; a larger batch only after gate (iv)
#                  passed for it: the probe at that batch must pass first, so gate-probe and the shards use the same one)
#   OM_EST_SHARD   expected wall time of ONE shard job (the --est of its launch). Estimate: 6.4 GPU h of work over 3 shards
#                  in parallel is about 2.1 h; with 2 sets x model load and chunk writes, about 2.5 h at batch 1.
#   OM_EST_PROBE   probe and gate-probe jobs (about 15 min each; --short)
OM_BATCH=${OM_BATCH:-1}
OM_EST_SHARD=${OM_EST_SHARD:-2h30m}
OM_EST_PROBE=${OM_EST_PROBE:-15m}
OM_NSHARD=3
R=res/r18_tpips                                         # where `pull` puts the node results, in the worktree
PROBE_R=res/r18_tpips_probe
STORE_ROOT=/data/SSD2/pre_extract                        # om_paths.TPIPS_STORE: artelingo_full/tpips_8b, mmart/tpips_8b
PY=/root/venvs/tpips/bin/python
TPIPS=src/test/20261103_outside_methods/om_tpips.py
SH=scripts/run_om_tpips_node.sh
L="label r18:tpips"
cat <<PLAN
CLUSTER=/root/.claude/skills/cluster-run/cluster
OM_BATCH=$OM_BATCH   OM_EST_SHARD=$OM_EST_SHARD   OM_EST_PROBE=$OM_EST_PROBE   shards=$OM_NSHARD

-- A. inputs the node needs (head node and cluster files; the CLI cannot do the env step, see 2)
0  Local, once: the manifests exist (ticket 03 'prepare') and the local batch-1 probe store is the reference:
     ls /data/SSD2/cosir_r18/20261103_outside_methods/outputs/tpips_inputs/{artelingo,mmart,probe}.json
     ls /data/SSD2/cosir_r18/20261103_outside_methods/outputs/stage0/probe_b1/{emotion,style,genre}/emb.npy
1  Weights, head node, detached and resumable (32 GB in all, over DATA_MAX_GB: not by sync); wait for both to say done:
     \$CLUSTER hf-fetch sywang/TPIPS-Embed-Qwen3VL-8B --revision 86a1493c60121e656a1c075bc7aed88f25fcd4d4 --wait      (background)
     \$CLUSTER hf-fetch Qwen/Qwen3-VL-Embedding-8B --revision 2c4565515e0f265c6511776e7193b22c0968ddc7 --wait         (background)
     \$CLUSTER hf-fetch sywang/TPIPS-Embed-Qwen3VL-8B --status ; \$CLUSTER hf-fetch Qwen/Qwen3-VL-Embedding-8B --status
2  Env: no new env. The job uses only the interpreter of the node's CoSiR env. Every numeric package comes from the input
   folder tpips_pylibs/full on PYTHONPATH, a copy of the local versions (7.5 GB; torch, torchvision, nvidia, transformers,
   tokenizers, safetensors, huggingface_hub, accelerate, PIL, numpy, with their dist-info, plus tpips and peft). The node's
   own transformers 5.16.1 and torch 2.14 failed gate (iv): round log, build 4. A second CLI profile cannot build an env:
   load_profiles refuses a shared REPO_URL. Gate (iv) compares node and local vectors.
3  DATA_MAP: these two pairs in cluster.conf (the images already map through wikiart_proj):
     /local/wding/r18_in/tpips_inputs:/data/SSD2/cosir_r18/20261103_outside_methods/outputs/tpips_inputs
     /local/wding/r18_in/tpips_pylibs:/data/SSD2/cosir_r18/20261103_outside_methods/node_inputs/tpips_pylibs
   and adopt the node dataset config:
     \$CLUSTER configs --adopt configs/dataset/om_node_cluster.yaml

-- B. sync and probe
4  commit with "cluster run" in the subject, then:   \$CLUSTER sync --node node402       ;   \$CLUSTER status --node node402
5  probe job (env, snapshots, 3 images through the mapped root, one image embedded; about 5 min with the model load):
     \$CLUSTER check --node node402 --sync-data -- bash $SH probe
     \$CLUSTER launch --node node402 --queue --short --$L -- bash $SH probe              (background; look for PROBE_ENV_OK and PROBE_EMBED_OK)
     \$CLUSTER watch <tag> ; \$CLUSTER pull --tag <tag>
   If a DATA_MAP path is wrong this job fails here, not in the 2.5 h jobs. Fix, new tag, repeat (twice at most).
6  gate probe (48 images x 3 factors at batch $OM_BATCH, one shard, merged on the node):
     \$CLUSTER launch --node node402 --queue --est $OM_EST_PROBE --$L -- bash $SH gate-probe --batch $OM_BATCH
     \$CLUSTER watch <tag> ; \$CLUSTER pull --tag <tag>     -> $R/tpips_store/probe_full/{emotion,style,genre}/

-- C. the extraction: 3 shard jobs, one GPU each (each takes the lowest free slot; each runs artelingo then mmart)
7  check once, then three launches of the same commit, each expected about $OM_EST_SHARD:
     \$CLUSTER check --node node402 --sync-data -- bash $SH extract --shard 0 --of $OM_NSHARD --batch $OM_BATCH
     \$CLUSTER launch --node node402 --queue --est $OM_EST_SHARD --$L -- bash $SH extract --shard 0 --of $OM_NSHARD --batch $OM_BATCH
     \$CLUSTER launch --node node402 --queue --est $OM_EST_SHARD --$L -- bash $SH extract --shard 1 --of $OM_NSHARD --batch $OM_BATCH
     \$CLUSTER launch --node node402 --queue --est $OM_EST_SHARD --$L -- bash $SH extract --shard 2 --of $OM_NSHARD --batch $OM_BATCH
   (background each. A lost job is relaunched with the same --shard and a new tag: its finished chunks on the node are skipped
    because the results folder is the same; do not clean the node's r18_tpips folder before all three are pulled.)
8  per job:   \$CLUSTER watch <tag>      \$CLUSTER pull --tag <tag>      (all three pulls land in the one folder $R/)

-- D. merge and gate (iv), local, tpips venv
9  put the pulled shards into the real stores (no chunks), then merge each set (keys, unit rows and sha256 are asserted):
     mkdir -p $STORE_ROOT/artelingo_full/tpips_8b $STORE_ROOT/mmart/tpips_8b
     rsync -a --exclude chunks $R/tpips_store/artelingo/ $STORE_ROOT/artelingo_full/tpips_8b/
     rsync -a --exclude chunks $R/tpips_store/mmart/ $STORE_ROOT/mmart/tpips_8b/
     PYTHONDONTWRITEBYTECODE=1 $PY $TPIPS merge --set artelingo --of $OM_NSHARD
     PYTHONDONTWRITEBYTECODE=1 $PY $TPIPS merge --set mmart --of $OM_NSHARD
10 gate (iv): the node's probe store against the local batch-1 reference, then the real stores (pass = exit 0, writes OUT/stage0/gate_iv.json):
     PYTHONDONTWRITEBYTECODE=1 $PY $TPIPS gate-iv --path das6=$R/tpips_store/probe_full
11 release the node: \$CLUSTER du --node node402 ; list what is left, show the user, \$CLUSTER clean only on a yes
   (the weights, 32 GB at /var/scratch/wding/cache/hub, and the env stay until the TPIPS cells are done).
PLAN
