#!/bin/bash
export EXPERIMENT_NUMBER=9
export EXPERIMENT_FROM_NUMBER=0
export RUN_NUMBER=99
export RUN_FROM_NUMBER=0
export SEED=1
export DATASET_NAME=Rail
export TASK="GMM"
export DEBUG=1
export SWEEPS=0
export DATAMODE=train
export WANDB=0
export DEVICE=0
export VITON_NAME=CP_VTON_plus

./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number $EXPERIMENT_FROM_NUMBER --run_from_number $RUN_FROM_NUMBER \
  --gmm_experiment_from_number 99 --gmm_run_from_number 0 --gmm_load_from_model Original  \
  --tom_experiment_from_number 0 --tom_run_from_number 0 --tom_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --stage $TASK --load_last_step False  \
  --res low_res --dataset_name $DATASET_NAME  --run_wandb $WANDB \
  --low_res_dataset_name viton_plus \
  --niter 10000 --niter_decay 10000 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED