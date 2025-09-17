#!/bin/bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

opts=(
    options/3task/3task_benchmark.yaml
    options/3task/3task_ratio0.3_benchmark.yaml
    options/5task/5task_benchmark.yaml
    options/7task/7task_benchmark.yaml
    options/7task/7task_ratio0.3_benchmark.yaml
)

for opt in "${opts[@]}"; do
    echo "==== Running $opt ===="
    torchrun \
      --nproc_per_node=1 \
      --master_port=4321 \
      ram/test.py \
      -opt "$opt" \
      --launcher pytorch
done
