#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster68,mscluster48,mscluster82
#SBATCH --ntasks=1
#SBATCH --partition=biggpu

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------

/bin/hostname
nvidia-smi
source ~/.bashrc
conda activate NeRF
export CUDA_VISIBLE_DEVICES=0
echo "Launching the script ${SLURM_JOB_NAME}"
echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Experiment From Number: $EXPERIMENT_FROM_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Run From Number: $RUN_FROM_NUMBER"
echo "Dataset Name: $DATASET_NAME"

./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
    --experiment_from_number 2 --run_from_number 29 \
    --warp_experiment_from_number 2 --warp_run_from_number 29 --warp_load_from_model Rail \
    --gen_experiment_from_number 0 --gen_run_from_number 0 --gen_load_from_model Original \
    --dataset_name $DATASET_NAME --device 0 --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 --val_count 10 \
    --viton_batch_size 32 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED