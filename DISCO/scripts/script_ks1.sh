#!/bin/bash

cd DISCO

budgets=(1000)
ndatas=(1000)
for b in "${budgets[@]}"; do
    for ndata in "${ndatas[@]}"; do
        python train.py \
            --yaml_config kuramotosivashinsky.yaml \
            --wandb_project disco_ks \
            --time_budget $b \
            --ntrain $ndata \
            --per_data_gen_cost 0.06673403584957123
    done
done