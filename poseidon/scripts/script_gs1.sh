#!/bin/bash

export HOME=$PWD
echo $HOME/

DATA_PATH="data"
SAVE_PATH="."

BUDGET=1e3
ndatas=(500 1000 1400 2000 2800 4000 5600 8000 16000 20000)
for ndata in "${ndatas[@]}"; do
    accelerate launch scOT/train.py \
        --config configs/gray_scott.yaml \
        --wandb_project_name poseidon_GS \
        --checkpoint_path $SAVE_PATH \
        --data_path $DATA_PATH \
        --ndata=$ndata \
        --time_budget=$BUDGET \
        --batch_size=100 \
        --inference_batch_size=100 \
        --lr=1e-3
done
