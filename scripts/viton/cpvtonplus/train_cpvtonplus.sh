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

python3 FashionNeRF.py --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --gmm_experiment_from_number ${args[gmm_experiment_from_number]} --gmm_run_from_number ${args[gmm_run_from_number]} --gmm_load_from_model ${args[gmm_load_from_model]} \
  --tom_experiment_from_number ${args[tom_experiment_from_number]} --tom_run_from_number ${args[tom_run_from_number]} --tom_load_from_model ${args[tom_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name CP_VTON_plus --VITON_Model ${args[VITON_Model]} --stage ${args[stage]} --load_last_step ${args[load_last_step]} \
  --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --low_res_dataset_name viton_plus \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} \

