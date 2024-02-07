#!/bin/bash
export EXPERIMENT_NUMBER=14
export EXPERIMENT_FROM_NUMBER=12
export RUN_NUMBER=99
export RUN_FROM_NUMBER=21
export SEED=1
export DATASET_NAME=Rail
export TASK="GEN"
export DEBUG=1
export SWEEPS=0
export DATAMODE=train
export WANDB=0
export DEVICE=0
export VITON_NAME=HR_VITON

./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number 12 --run_from_number 21 \
  --tocg_experiment_from_number 12 --tocg_run_from_number 21 \
  --tocg_discriminator_experiment_from_number 0 --tocg_discriminator_run_from_number 0 \
  --tocg_load_from_model Rail --tocg_discriminator_load_from_model Rail \
  --gen_experiment_from_number 0 --gen_run_from_number 0 \
  --gen_discriminator_experiment_from_number 0 --gen_discriminator_run_from_number 0 \
  --gen_load_from_model Original --gen_discriminator_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --load_last_step False \
  --gpu_ids 1 --device $DEVICE --res high_res --dataset_name $DATASET_NAME --run_wandb $WANDB \
  --niter 10000 --niter_decay 10000 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED