#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster82
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
echo "Launching the script ${SLURM_JOB_NAME}"




# export EXPERIMENT_NUMBER=14
# export EXPERIMENT_FROM_NUMBER=12
# export RUN_NUMBER=99
# export RUN_FROM_NUMBER=4
# export SEED=1
# export DATASET_NAME=Rail
# export TASK="GEN"
# export DEBUG=1
# export SWEEPS=0
# export DATAMODE=train
# export WANDB=1
# export DEVICE=0
# export VITON_NAME=HR_VITON

echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Experiment From Number: $EXPERIMENT_FROM_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Run From Number: $RUN_FROM_NUMBER"
echo "Dataset Name: $DATASET_NAME"
echo "Device: $DEVICE"
export CUDA_VISIBLE_DEVICES=$DEVICE

# tocg_experiment_from_number 11 - Original
# tocg_experiment_from_number 12 - Rail

# gen_experiment_from_number 13 - Original
# gen_experiment_from_number 14 - Rail


./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number 12 --run_from_number 15 \
  --tocg_experiment_from_number 12 --tocg_run_from_number 15 \
  --tocg_discriminator_experiment_from_number 0 --tocg_discriminator_run_from_number 0 \
  --tocg_load_from_model Rail --tocg_discriminator_load_from_model Rail \
  --gen_experiment_from_number 0 --gen_run_from_number 0 \
  --gen_discriminator_experiment_from_number 0 --gen_discriminator_run_from_number 0 \
  --gen_load_from_model Original --gen_discriminator_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --load_last_step False \
  --gpu_ids 1 --device $DEVICE --res high_res --dataset_name $DATASET_NAME --run_wandb $WANDB \
  --niter 10000 --niter_decay 10000 --display_count 100 --print_step 100 --save_period 100 --val_count 100 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED