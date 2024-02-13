#!/bin/bash

experiment_number=22
run_number=30 # Compare all methods
experiment_from_number=0
run_from_number=0
seed_number=1
VITON_Type="Parser_Free" # [Parser_Based, Parser_Free, Share]
VITON_Name="PF_AFN" # [FS_VTON, PF_AFN,DM_VTON, ACGPN, CP_VTON, CP_VTON_plus, HR_VITON, Ladi_VITON,SD_VITON]
task="PB_Gen" # [PB_Gen, PB_Warp,PF_Warp, PF_Gen, EMASC, GMM, TOM, TOCG, GEN]
dataset_name="Rail" # [Original, Rail]

export EXPERIMENT_NUMBER=$experiment_number
export EXPERIMENT_FROM_NUMBER=$experiment_from_number
export RUN_NUMBER=$run_number
export RUN_FROM_NUMBER=$run_from_number
export DATASET_NAME=$dataset_name
export TASK=$task
export SEED=$seed_number
export DEBUG=0
export WANDB=1
export SWEEPS=0
export DATAMODE=train
export DEVICE=1
export VITON_NAME=$VITON_Name


/bin/hostname
nvidia-smi
source ~/.bashrc
conda activate NeRF
echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Experiment From Number: $EXPERIMENT_FROM_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Run From Number: $RUN_FROM_NUMBER"
echo "Dataset Name: $DATASET_NAME"
echo "Device: $DEVICE"

./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
    --experiment_from_number 20 --run_from_number 29 \
    --warp_experiment_from_number 20 --warp_run_from_number 29 --warp_load_from_model Rail \
    --gen_experiment_from_number 0 --gen_run_from_number 0 --gen_load_from_model Original \
    --dataset_name $DATASET_NAME --device $DEVICE --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 --val_count 10 \
    --viton_batch_size 32 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED 