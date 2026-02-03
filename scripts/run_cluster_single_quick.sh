#! /bin/bash


input_lr=$1
python main_cosir.py dataset=cc3m_cluster featuremanager.chunk_size=4096 optimizer.lr=$input_lr