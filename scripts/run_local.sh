#! /bin/bash
set -euo pipefail

# Reduce CUDA memory fragmentation (especially important for large-patch models like SigLIP)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 自动检测进程数（优先使用 CUDA_VISIBLE_DEVICES，其次 nvidia-smi，否则 CPU=1）
if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a __cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#__cvd_arr[@]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
else
  NUM_PROCS=1
fi
[ -z "${NUM_PROCS}" ] && NUM_PROCS=1
[ "${NUM_PROCS}" -lt 1 ] && NUM_PROCS=1

echo "Using ${NUM_PROCS} processes"

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.1 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 max high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e4 eval.oracle_aggregation="max" train.normalize=true

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.1 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 mean high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e4 eval.oracle_aggregation="mean" train.normalize=true # -> good

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.5 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 max high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e4 eval.oracle_aggregation="max" train.normalize=true

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.5 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 mean high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e4 eval.oracle_aggregation="mean" train.normalize=true


# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.1 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 max high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e3 eval.oracle_aggregation="max" train.normalize=true

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.1 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 mean high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e3 eval.oracle_aggregation="mean" train.normalize=true # Good

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.5 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 max high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e3 eval.oracle_aggregation="max" train.normalize=true

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=1000 loss.lambda_2=0.1 loss.lambda_3=0 model.dropout=0.5 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 1000 epochs normalizing text features in manifold loss knn10 mean high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e3 eval.oracle_aggregation="mean" train.normalize=true

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=500 loss.lambda_2=0.1 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="random normed imgtxt 1e-4 500 epochs normalizing text features in manifold loss knn10 max high knn loss try normalize initialization" optimizer.label_lr_multiplier=1.0e3 eval.oracle_aggregation="max" train.normalize=True

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="max" train.normalize=False train.representative_number=6

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="mean" train.normalize=False train.representative_number=1


# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=500 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="add clip base" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="max" train.normalize=True

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=500 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="add clip base" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="mean" train.normalize=True

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="max" train.normalize=True train.representative_number=12

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-6 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="mean" train.normalize=True train.representative_number=12

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-5 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="max" train.normalize=True train.representative_number=12

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-5 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="mean" train.normalize=True train.representative_number=12

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="max" train.normalize=True train.representative_number=12

# python main_cosir.py dataset=coco train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=200 loss.lambda_2=30 loss.lambda_3=0.0 model.dropout=0.1 model.num_layers=6 wandb.name="no embedding update sanity check" optimizer.label_lr_multiplier=5.0e5 eval.oracle_aggregation="mean" train.normalize=True train.representative_number=12