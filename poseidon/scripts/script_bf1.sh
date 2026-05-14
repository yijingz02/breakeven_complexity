#!/bin/bash

export HOME=$PWD
echo $HOME/

DATA_PATH="data"
SAVE_PATH="."

BUDGET=16e4
ndatas=(1000)
for ndata in "${ndatas[@]}"; do
    accelerate launch scOT/train.py \
        --config configs/multi_obstacle.yaml \
        --wandb_project_name poseidon_BF \
        --checkpoint_path $SAVE_PATH \
        --data_path $DATA_PATH \
        --ndata=$ndata \
        --time_budget=$BUDGET \
        --batch_size=16 \
        --lr=1e-3
done