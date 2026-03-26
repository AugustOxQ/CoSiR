#! /bin/bash
set -euo pipefail

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

# # This section we do img x 96 outsideonly x [lr] x gumble x soft

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 outsideonly lr1e-4 gumble hard" #-> 1

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-5 gumble hard" #-> 2

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-6 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 outsideonly lr1e-6 gumble hard" #-> 3

# This section we do img x 96 grid x [lr] x gumble x soft

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 grid lr1e-4 gumble hard" #-> 1

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 grid lr1e-5 gumble hard" #-> 2

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-6 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 grid lr1e-6 gumble hard" #-> 3

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 grid lr1e-4 softmax hard" #-> x

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 grid lr1e-5 softmax hard" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-6 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img 96 grid lr1e-6 softmax hard" #-> x

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 grid lr1e-5 gunble argmax" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=15 loss.warm_up_epochs=10 loss.middle_epochs=15 train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 grid lr1e-4 gunble hard add predict img" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 model.dropout=0.25 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-4 gunble hard using img initialization 1e-5 dropout 0.25" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 model.dropout=0.5 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-4 gunble hard using img initialization 1e-5 dropout 0.5" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 model.dropout=0.75 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-4 gunble hard using img initialization 1e-5 dropout 0.75" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 model.dropout=0.25 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-5 gunble hard using img initialization 1e-5 dropout 0.25" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 model.dropout=0.5 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-5 gunble hard using img initialization 1e-5 dropout 0.5" #-> 4

# python main_cosir_phase2.py dataset=coco eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=30 model.dropout=0.75 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-5 gunble hard using img initialization 1e-5 dropout 0.75" #-> 4

# python main_cosir_phase2.py dataset=impressions eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=30 model.dropout=0.5 train.initialization_strategy="img" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="img96 outsideonly lr1e-4 gunble hard using img initialization 1e-5 dropout 0.5" #-> 4

python main_cosir_phase2.py dataset=impressions eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=500 model.dropout=0.1 train.initialization_strategy="imgtxt" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="imgtxt96 "