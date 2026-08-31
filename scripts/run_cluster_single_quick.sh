#! /bin/bash


input_lr=$1
min_radius=$2
python main_cosir.py dataset=coco_cluster featuremanager.chunk_size=4096 optimizer.lr=$input_lr eval.perform_evaluation=false model.min_radius=$min_radius