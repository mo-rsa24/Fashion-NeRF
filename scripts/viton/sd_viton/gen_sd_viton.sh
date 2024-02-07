#!/bin/bash
declare -A args
while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            args[${1#--}]=$2
            shift
            ;;
    esac
    shift
done


python3  FashionNeRF.py --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --tocg_experiment_from_number ${args[tocg_experiment_from_number]} --tocg_run_from_number ${args[tocg_run_from_number]} \
  --tocg_discriminator_experiment_from_number ${args[tocg_discriminator_experiment_from_number]} --tocg_discriminator_run_from_number ${args[tocg_discriminator_run_from_number]} \
  --tocg_load_from_model ${args[tocg_load_from_model]} --tocg_discriminator_load_from_model ${args[tocg_discriminator_load_from_model]} \
  --gen_experiment_from_number ${args[gen_experiment_from_number]} --gen_run_from_number ${args[gen_run_from_number]} \
  --gen_discriminator_experiment_from_number ${args[gen_discriminator_experiment_from_number]} --gen_discriminator_run_from_number ${args[gen_discriminator_run_from_number]} \
  --gen_load_from_model ${args[gen_load_from_model]} --gen_discriminator_load_from_model ${args[gen_discriminator_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name SD_VITON --VITON_Model ${args[VITON_Model]} --load_last_step ${args[load_last_step]} \
  --gpu_ids 0 --device ${args[device]} --res high_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]} 

