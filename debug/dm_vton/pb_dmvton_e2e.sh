#!/bin/bash
export EXPERIMENT_NUMBER=4
export EXPERIMENT_FROM_NUMBER=0
export RUN_NUMBER=99
export RUN_FROM_NUMBER=0
export SEED=1
export DATASET_NAME=Rail
export TASK="PB_Gen"
export DEBUG=1
export SWEEPS=0
export DATAMODE=train
export WANDB=0
export DEVICE=0
export VITON_NAME=DM_VTON


./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
    --experiment_from_number 2 --run_from_number 21 \
    --warp_experiment_from_number 2 --warp_run_from_number 21 --warp_load_from_model Rail \
    --gen_experiment_from_number 2 --gen_run_from_number 21 --gen_load_from_model Original \
    --dataset_name $DATASET_NAME --device 0 --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
    --viton_batch_size 32 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED

