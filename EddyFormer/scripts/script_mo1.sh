#!/bin/bash

# export HOME=$PWD
# echo $HOME/

WORKDIR="$PWD"
echo "WORKDIR=$WORKDIR"
pwd

mkdir data
mkdir data/train
mkdir data/val

mv *_val.npy data/val
mv *.npy data/train

mv data EddyFormer

cd EddyFormer

nvidia-smi

ls

echo "<-------->"

DATA_PATH="data"
SAVE_PATH="."

BUDGET=100
LOGDIR="log_mo"
mkdir -p "$LOGDIR"

export PYTHONPATH="$PWD:$PYTHONPATH"

ndatas=(275 270 266 250 233 188)
b=5e4
for ndata in "${ndatas[@]}"; do
    echo "Running with budget: $b seconds, data count: $ndata"
    nsm train --notqdm \
        --flow configs/flow/mo_npy.py \
        --model configs/model/ef2d.py \
        --flow.config.base_dir data \
        --config.log.save_path "${SAVE_PATH}log_mo_N${ndata}_B${b}" \
        --config.log.wandb_project EddyFormer_MO \
        --config.train.batch_sharding False \
        --config.train.batch_size 64 \
        --config.train.vmap_batch 1 \
        --config.train.window 1 \
        --config.train.iteration 1000000 \
        --config.log.test_period 100 \
        --config.train.time_budget_s=${b} \
        --config.train.discard_budget_steps=5 \
        --config.train.datacnt ${ndata} \
        --config.train.data_gen_time=180
done

ndatas=(555 550 544 533 511 466)
BUDGET=1e5
for ndata in "${ndatas[@]}"; do
    echo "Running with budget: $b seconds, data count: $ndata"
    nsm train --notqdm \
        --flow configs/flow/mo_npy.py \
        --model configs/model/ef2d.py \
        --flow.config.base_dir data \
        --config.log.save_path "${SAVE_PATH}log_mo_N${ndata}_B${b}" \
        --config.log.wandb_project EddyFormer_MO \
        --config.train.batch_sharding False \
        --config.train.batch_size 64 \
        --config.train.vmap_batch 1 \
        --config.train.window 1 \
        --config.train.iteration 1000000 \
        --config.log.test_period 100 \
        --config.train.time_budget_s=${b} \
        --config.train.discard_budget_steps=5 \
        --config.train.datacnt ${ndata} \
        --config.train.data_gen_time=180
done

ndatas=(1110 1105 1100 1088 1066 1022)
b=2e5
for ndata in "${ndatas[@]}"; do
    echo "Running with budget: $b seconds, data count: $ndata"
    nsm train --notqdm \
        --flow configs/flow/mo_npy.py \
        --model configs/model/ef2d.py \
        --flow.config.base_dir data \
        --config.log.save_path "${SAVE_PATH}log_mo_N${ndata}_B${b}" \
        --config.log.wandb_project EddyFormer_MO \
        --config.train.batch_sharding False \
        --config.train.batch_size 64 \
        --config.train.vmap_batch 1 \
        --config.train.window 1 \
        --config.train.iteration 1000000 \
        --config.log.test_period 100 \
        --config.train.time_budget_s=${b} \
        --config.train.discard_budget_steps=5 \
        --config.train.datacnt ${ndata} \
        --config.train.data_gen_time=180
done


ndatas=(2220 2216 2211 2200 2177 2133)
b=4e5
for ndata in "${ndatas[@]}"; do
    echo "Running with budget: $b seconds, data count: $ndata"
    nsm train --notqdm \
        --flow configs/flow/mo_npy.py \
        --model configs/model/ef2d.py \
        --flow.config.base_dir data \
        --config.log.save_path "${SAVE_PATH}log_mo_N${ndata}_B${b}" \
        --config.log.wandb_project EddyFormer_MO \
        --config.train.batch_sharding False \
        --config.train.batch_size 64 \
        --config.train.vmap_batch 1 \
        --config.train.window 1 \
        --config.train.iteration 1000000 \
        --config.log.test_period 100 \
        --config.train.time_budget_s=${b} \
        --config.train.discard_budget_steps=5 \
        --config.train.datacnt ${ndata} \
        --config.train.data_gen_time=180
done