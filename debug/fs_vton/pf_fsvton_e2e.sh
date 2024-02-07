#!/bin/bash
export EXPERIMENT_NUMBER=36
export EXPERIMENT_FROM_NUMBER=0
export RUN_NUMBER=99
export RUN_FROM_NUMBER=0
export SEED=1
export DATASET_NAME=Rail
export TASK="PF_Gen"
export DEBUG=1
export SWEEPS=0
export DATAMODE=train
export WANDB=0
export DEVICE=0
export VITON_NAME=FS_VTON

./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number 26 --run_number 1 \
    --experiment_from_number 36 --run_from_number 21 \
    --parser_based_warp_experiment_from_number 30 --parser_based_warp_run_from_number 21 --warp_load_from_model Rail \
    --parser_based_gen_experiment_from_number 32 --parser_based_gen_run_from_number 21 --gen_load_from_model Rail \
    --parser_free_warp_experiment_from_number 34 --parser_free_warp_run_from_number 21 --parser_free_warp_load_from_model Rail \
    --parser_free_gen_experiment_from_number 0 --parser_free_gen_run_from_number 0 --parser_free_gen_load_from_model Original \
    --dataset_name $DATASET_NAME --validate False --device $DEVICE --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
    --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED