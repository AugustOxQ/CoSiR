#!/bin/bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LD_LIBRARY_PATH=/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
cd /project/CoSiR/src/test/20261025_backbone_check
for m in clip siglip2 pe qwen; do
  echo "== $m $(date +%T)"; /root/miniconda3/envs/CoSiR/bin/python run_backbone_check.py extract $m 2>&1 | grep -v -i warn
done
echo "== done $(date +%T)"
