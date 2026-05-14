#!/bin/bash

cd DISCO

budgets=(1000)
ndatas=(1000)
for b in "${budgets[@]}"; do
    for ndata in "${ndatas[@]}"; do
        python train.py \
            --yaml_config navierstokes.yaml \
            --wandb_project disco_ns \
            --time_budget $b \
            --ntrain $ndata \
            --per_data_gen_cost 0.04353754370212555
    done
done
