#!/bin/bash

# export HOME=$PWD
# echo $HOME/

WORKDIR="$PWD"
echo "WORKDIR=$WORKDIR"
pwd

cd EddyFormer

nvidia-smi

ls

echo "<-------->"

DATA_PATH="data"
SAVE_PATH="."

LOGDIR="log_gs"
mkdir -p "$LOGDIR"

export PYTHONPATH="$PWD:$PYTHONPATH"

budgets=(1e3 2e3 4e3 8e3)
ndatas=(500 1000 2000 4000 8000 16000 20000)

for b in "${budgets[@]}"; do
    for ndata in "${ndatas[@]}"; do
        echo "Running with budget: $b seconds, data count: $ndata"
        nsm train --notqdm \
          --flow configs/flow/gray_scott_npy.py \
          --model configs/model/ef2d.py \
          --flow.config.base_dir data \
          --config.log.save_path "${SAVE_PATH}log_gs_N${ndata}_B${b}" \
          --config.log.wandb_project EddyFormer_GS \
          --config.train.batch_sharding False \
          --config.train.batch_size 64 \
          --config.train.vmap_batch 1 \
          --config.train.window 1 \
          --config.train.iteration 1000000 \
          --config.log.test_period 100 \
          --config.train.time_budget_s=${b} \
          --config.train.discard_budget_steps=5 \
          --config.train.datacnt ${ndata} \
          --config.train.data_gen_time=0.47509707514047617
    done
done