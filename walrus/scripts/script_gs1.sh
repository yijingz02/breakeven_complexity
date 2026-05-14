#!/bin/bash

cd walrus

DATA_PATH="data"
SAVE_PATH="."

ndatas=(1000)
BUDGET=1e3
for NDATA in "${ndatas[@]}"; do
  python train.py \
    distribution=local model=isotropic_model name=Walrus_ft_example trainer=globalnorm trainer.grad_acc_steps=1 server=gpuxl \
    optimizer=adam optimizer.lr=1e-3 logger.wandb_project_name="walrus_gs" \
    trainer.enable_amp=True model.gradient_checkpointing_freq=1 trainer.log_interval=1000 trainer.clip_gradient=1 \
    data.module_parameters.batch_size=100 \
    data.module_parameters.n_steps_input=10 \
    data.module_parameters.n_steps_output=1 \
    data.module_parameters.well_dataset_info.gs_exponax.path="${DATA_PATH}" \
    model/processor/space_mixing=full_spatial_attention \
    ++trainer.epsilon=1e-8 \
    model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=20000 \
    trainer.short_validation_length=20 \
    trainer.max_rollout_steps=188 \
    lr_scheduler=inv_sqrt_w_sqrt_ramps_longer trainer.val_frequency=100 trainer.rollout_val_frequency=100 \
    data.module_parameters.min_dt_stride=1 data.module_parameters.max_dt_stride=1 \
    trainer.prediction_type=delta data=gs_exponax trainer.max_epoch=10000 data_workers=10 model.override_dimensionality=0 auto_resume=True \
    checkpoint=none ++model.use_periodic_fixed_jitter=True ++model.input_field_drop=0 ++trainer.skip_spectral_metrics=True \
    finetuning_mods=all \
    ++trainer.video_validation=False \
    ++data.module_parameters.start_rollout_valid_output_at_t=11 \
    ++model.pretrained_path=../model_weights/halfwalrus_step160.pt \
    ++model.pretrained_strict=false \
    ++trainer.time_budget_s=$BUDGET \
    ++trainer.Ndata_generated=$NDATA \
    ++trainer.gen_time_s=0.47509707514047617 \
    ++trainer.save_path=$SAVE_PATH
done