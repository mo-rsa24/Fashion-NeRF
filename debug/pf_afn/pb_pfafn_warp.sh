#!/bin/bash
export EXPERIMENT_NUMBER=20
export EXPERIMENT_FROM_NUMBER=0
export RUN_NUMBER=99
export RUN_FROM_NUMBER=0
export SEED=1
export DATASET_NAME=Rail
export TASK="PB_Warp"
export DEBUG=1
export SWEEPS=0
export DATAMODE=train
export WANDB=0
export DEVICE=0
export VITON_NAME=PF_AFN

./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
    --experiment_from_number $EXPERIMENT_FROM_NUMBER \
    --run_from_number $RUN_FROM_NUMBER \
    --warp_load_from_model Original --dataset_name $DATASET_NAME --device $DEVICE --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
    --viton_batch_size 8 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED

