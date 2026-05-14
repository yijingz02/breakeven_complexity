#!/bin/bash

cd DISCO

budgets=(16e4)
ndatas=(1000)
for b in "${budgets[@]}"; do
    for ndata in "${ndatas[@]}"; do
        python train.py \
            --yaml_config breakflow.yaml \
            --wandb_project disco_bf \
            --time_budget $b \
            --ntrain $ndata \
            --per_data_gen_cost 136.904521
    done
done