#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=DM_PF
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster107,mscluster108
#SBATCH --ntasks=2
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
echo "Sharing Biggpu across DM_VTON and PF_AFN"

./scripts/viton/viton.sh --job_name DM_VTON --task PF_Warp --experiment_number 16 --run_number 31 \
    --experiment_from_number 2 --run_from_number 29 \
    --parser_based_warp_experiment_from_number 2 --parser_based_warp_run_from_number 29 --warp_load_from_model Rail \
    --parser_based_gen_experiment_from_number 4 --parser_based_gen_run_from_number 30 --gen_load_from_model Rail \
    --parser_free_warp_experiment_from_number 0 --parser_free_warp_run_from_number 0 --parser_free_warp_load_from_model Original \
    --dataset_name Rail --device 0 --load_last_step False --run_wandb 1 \
    --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 --val_count 10 \
    --viton_batch_size 32 --datamode train --debug 0 --sweeps 0 --seed 0 &

./scripts/viton/viton.sh --job_name PF_AFN --task PF_Warp --experiment_number 24 --run_number 31 \
    --experiment_from_number 0 --run_from_number 0 \
    --parser_based_warp_experiment_from_number 20 --parser_based_warp_run_from_number 29 --warp_load_from_model Rail \
    --parser_based_gen_experiment_from_number 22 --parser_based_gen_run_from_number 29 --gen_load_from_model Rail \
    --parser_free_warp_experiment_from_number 0 --parser_free_warp_run_from_number 0 --parser_free_warp_load_from_model Original \
    --dataset_name Rail --device 1 --load_last_step False --run_wandb 1 \
    --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 --val_count 10 \
    --viton_batch_size 32 --datamode train --debug 0 --sweeps 1 --seed 0 &


wait