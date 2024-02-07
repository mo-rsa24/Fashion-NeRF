#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster68,mscluster48,mscluster82,mscluster61,mscluster18
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

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
echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Experiment From Number: $EXPERIMENT_FROM_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Run From Number: $RUN_FROM_NUMBER"
echo "Dataset Name: $DATASET_NAME"
echo "Device: $DEVICE"
export CUDA_VISIBLE_DEVICES=$DEVICE


./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number 12 --run_from_number 4 \
  --tocg_experiment_from_number 12 --tocg_run_from_number 4 \
  --tocg_discriminator_experiment_from_number 12 --tocg_discriminator_run_from_number 4 \
  --tocg_load_from_model Rail --tocg_discriminator_load_from_model Rail \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --load_last_step False \
  --gpu_ids $DEVICE --device $DEVICE --res high_res --dataset_name $DATASET_NAME --run_wandb $WANDB \
  --niter 10000 --niter_decay 10000 --display_count 1000 --print_step 1000 --save_period 1000 --val_count 1000 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED

