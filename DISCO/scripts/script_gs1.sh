#!/bin/bash

cd DISCO

budgets=(1000)
ndatas=(1000)
for b in "${budgets[@]}"; do
    for ndata in "${ndatas[@]}"; do
        python train.py \
            --yaml_config grayscott.yaml \
            --wandb_project disco_gs_fix \
            --time_budget $b \
            --ntrain $ndata \
            --per_data_gen_cost 0.47509707514047617
    done
done