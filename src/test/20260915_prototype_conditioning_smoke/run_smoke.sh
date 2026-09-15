#!/bin/bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
cd /project/CoSiR-buddy_prototype_conditioning

for P in 8 16 32; do
  echo "=== num_prototypes=$P ==="
  python main_cosir.py \
    dataset=redcaps_150k \
    model=clip_base \
    model.conditioning_mode=prototype_pooled \
    model.num_prototypes=$P \
    train.initialization_strategy=buddies \
    train.epochs=1 \
    train.max_train_samples=1500 \
    eval.evaluation_interval=1 \
    seed=42 \
    experiment.results_dir=/tmp/exp18_smoke_p${P} \
    wandb.mode=disabled
done
