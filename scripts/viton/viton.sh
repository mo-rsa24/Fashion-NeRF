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
echo -e "\n" 
echo "${args[experiment_from_number]}"
echo "Starting ${args[job_name]} ..."
echo "Model ${args[task]} ..."
echo "Selected device ${args[device]} ..."
echo "Training for ${args[niter]} epochs ..."
echo "Debugging: ${args[debug]}"
echo "Sweep: ${args[sweeps]}"
echo "Run_Wandb: ${args[run_wandb]}"
echo "Seed: ${args[seed]}"
echo "Loading the last step ${args[load_last_step]}"
if [ "${args[job_name]}" == "DM_VTON" ]
then
    if [ "${args[task]}" == "PB_Warp" ]
    then     
    ./scripts/viton/dm_vton/pb_dmvton_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PB_Gen" ]
    then     
    
    ./scripts/viton/dm_vton/pb_dmvton_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_experiment_from_number ${args[warp_experiment_from_number]} --warp_run_from_number ${args[warp_run_from_number]} --gen_experiment_from_number ${args[gen_experiment_from_number]} --gen_run_from_number ${args[gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Warp" ]
    then     
    ./scripts/viton/dm_vton/pf_dmvton_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --debug ${args[debug]} --datamode ${args[datamode]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Gen" ]
    then     
    ./scripts/viton/dm_vton/pf_dmvton_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --parser_free_gen_experiment_from_number ${args[parser_free_gen_experiment_from_number]} --parser_free_gen_run_from_number ${args[parser_free_gen_run_from_number]} --parser_free_gen_load_from_model ${args[parser_free_gen_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --debug ${args[debug]} --datamode ${args[datamode]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

fi


if [ "${args[job_name]}" == "PF_AFN" ]
then
    if [ "${args[task]}" == "PB_Warp" ]
    then     
    ./scripts/viton/pf_afn/pb_pfafn_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PB_Gen" ]
    then     
    
    ./scripts/viton/pf_afn/pb_pfafn_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_experiment_from_number ${args[warp_experiment_from_number]} --warp_run_from_number ${args[warp_run_from_number]} --gen_experiment_from_number ${args[gen_experiment_from_number]} --gen_run_from_number ${args[gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Warp" ]
    then     
    ./scripts/viton/pf_afn/pf_pfafn_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Gen" ]
    then     
    ./scripts/viton/pf_afn/pf_pfafn_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --parser_free_gen_experiment_from_number ${args[parser_free_gen_experiment_from_number]} --parser_free_gen_run_from_number ${args[parser_free_gen_run_from_number]} --parser_free_gen_load_from_model ${args[parser_free_gen_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

fi


if [ "${args[job_name]}" == "FS_VTON" ]
then
    if [ "${args[task]}" == "PB_Warp" ]
    then     
    ./scripts/viton/fs_vton/pb_fsvton_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PB_Gen" ]
    then     
    
    ./scripts/viton/fs_vton/pb_fsvton_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --warp_experiment_from_number ${args[warp_experiment_from_number]} --warp_run_from_number ${args[warp_run_from_number]} --gen_experiment_from_number ${args[gen_experiment_from_number]} --gen_run_from_number ${args[gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --dataset_name ${args[dataset_name]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Warp" ]
    then     
    ./scripts/viton/fs_vton/pf_fsvton_warp.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

    if [ "${args[task]}" == "PF_Gen" ]
    then     
    ./scripts/viton/fs_vton/pf_fsvton_e2e.sh --task ${args[task]} --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} --parser_based_warp_experiment_from_number ${args[parser_based_warp_experiment_from_number]} --parser_based_warp_run_from_number ${args[parser_based_warp_run_from_number]} --parser_based_gen_experiment_from_number ${args[parser_based_gen_experiment_from_number]} --parser_based_gen_run_from_number ${args[parser_based_gen_run_from_number]} --warp_load_from_model ${args[warp_load_from_model]} --gen_load_from_model ${args[gen_load_from_model]} --parser_free_warp_experiment_from_number ${args[parser_free_warp_experiment_from_number]} --parser_free_warp_run_from_number ${args[parser_free_warp_run_from_number]} --parser_free_warp_load_from_model ${args[parser_free_warp_load_from_model]} --parser_free_gen_experiment_from_number ${args[parser_free_gen_experiment_from_number]} --parser_free_gen_run_from_number ${args[parser_free_gen_run_from_number]} --parser_free_gen_load_from_model ${args[parser_free_gen_load_from_model]} --dataset_name ${args[dataset_name]} --device ${args[device]} --load_last_step ${args[load_last_step]} --run_wandb ${args[run_wandb]} --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
    fi

fi


if [ "${args[job_name]}" == "ACGPN" ]
then
echo "Running ACGPN ..."
./scripts/viton/acgpn/acgpn.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
 --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
 --g_experiment_from_number ${args[g_experiment_from_number]} --g_run_from_number ${args[g_run_from_number]} --g_load_from_model ${args[g_load_from_model]} \
 --g1_experiment_from_number ${args[g1_experiment_from_number]} --g1_run_from_number ${args[g1_run_from_number]} --g1_load_from_model ${args[g1_load_from_model]} \
 --g2_experiment_from_number ${args[g2_experiment_from_number]} --g2_run_from_number ${args[g2_run_from_number]} --g2_load_from_model ${args[g2_load_from_model]}  \
 --d_experiment_from_number ${args[d_experiment_from_number]} --d_run_from_number ${args[d_run_from_number]} --d_load_from_model ${args[d_load_from_model]} \
 --d1_experiment_from_number ${args[d1_experiment_from_number]} --d1_run_from_number ${args[d1_run_from_number]} --d1_load_from_model ${args[d1_load_from_model]} \
 --d2_experiment_from_number ${args[d2_experiment_from_number]} --d2_run_from_number ${args[d2_run_from_number]} --d2_load_from_model ${args[d2_load_from_model]} \
 --d3_experiment_from_number ${args[d3_experiment_from_number]} --d3_run_from_number ${args[d3_run_from_number]} --d3_load_from_model ${args[d3_load_from_model]} \
 --unet_experiment_from_number ${args[unet_experiment_from_number]} --unet_run_from_number ${args[unet_run_from_number]} --unet_load_from_model ${args[unet_load_from_model]}  \
 --vgg_experiment_from_number ${args[vgg_experiment_from_number]} --vgg_run_from_number ${args[vgg_run_from_number]} --vgg_load_from_model ${args[vgg_load_from_model]} \
 --VITON_Type Parser_Based --VITON_Name ACGPN \
 --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
 --low_res_dataset_name ACGPN --load_last_step ${args[load_last_step]} \
 --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
 --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}

fi


if [ "${args[job_name]}" == "CP_VTON" ]
then
echo "Running CP_VTON ..."
./scripts/viton/cpvton/cpvton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --gmm_experiment_from_number ${args[gmm_experiment_from_number]} --gmm_run_from_number ${args[gmm_run_from_number]} --gmm_load_from_model ${args[gmm_load_from_model]} \
  --tom_experiment_from_number ${args[tom_experiment_from_number]} --tom_run_from_number ${args[tom_run_from_number]} --tom_load_from_model ${args[tom_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name CP_VTON --VITON_Model ${args[VITON_Model]} --stage ${args[stage]} --load_last_step ${args[load_last_step]} \
  --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --low_res_dataset_name viton_plus \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
fi

if [ "${args[job_name]}" == "HR_VITON" ]
then
echo "Running HR_VITON ..."
 if [ "${args[VITON_Model]}" == "TOCG" ] 
 then
  ./scripts/viton/hrviton/tocg_hrviton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
    --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
    --tocg_experiment_from_number ${args[tocg_experiment_from_number]} --tocg_run_from_number ${args[tocg_run_from_number]} \
    --tocg_discriminator_experiment_from_number ${args[tocg_discriminator_experiment_from_number]} --tocg_discriminator_run_from_number ${args[tocg_discriminator_run_from_number]} \
    --tocg_load_from_model ${args[tocg_load_from_model]} --tocg_discriminator_load_from_model ${args[tocg_discriminator_load_from_model]} \
    --VITON_Type Parser_Based --VITON_Name HR_VITON --VITON_Model ${args[VITON_Model]} --load_last_step ${args[load_last_step]} \
    --gpu_ids 0 --device ${args[device]} --res high_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
    --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
    --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
  fi 

  if [ "${args[VITON_Model]}" == "GEN" ] 
  then 
    ./scripts/viton/hrviton/gen_hrviton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
    --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
    --tocg_experiment_from_number ${args[tocg_experiment_from_number]} --tocg_run_from_number ${args[tocg_run_from_number]} \
    --tocg_discriminator_experiment_from_number ${args[tocg_discriminator_experiment_from_number]} --tocg_discriminator_run_from_number ${args[tocg_discriminator_run_from_number]} \
    --tocg_load_from_model ${args[tocg_load_from_model]} --tocg_discriminator_load_from_model ${args[tocg_discriminator_load_from_model]} \
    --gen_experiment_from_number ${args[gen_experiment_from_number]} --gen_run_from_number ${args[gen_run_from_number]} \
    --gen_discriminator_experiment_from_number ${args[gen_discriminator_experiment_from_number]} --gen_discriminator_run_from_number ${args[gen_discriminator_run_from_number]} \
    --gen_load_from_model ${args[gen_load_from_model]} --gen_discriminator_load_from_model ${args[gen_discriminator_load_from_model]} \
    --VITON_Type Parser_Based --VITON_Name HR_VITON --VITON_Model ${args[VITON_Model]} --load_last_step ${args[load_last_step]} \
    --gpu_ids 0 --device ${args[device]} --res high_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
    --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
    --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
  fi
fi



if [ "${args[job_name]}" == "SD_VITON" ]
then
echo "Running SD_VITON ..."
 if [ "${args[VITON_Model]}" == "TOCG" ] 
 then
  ./scripts/viton/sd_viton/tocg_sd_viton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
    --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
    --tocg_experiment_from_number ${args[tocg_experiment_from_number]} --tocg_run_from_number ${args[tocg_run_from_number]} \
    --tocg_discriminator_experiment_from_number ${args[tocg_discriminator_experiment_from_number]} --tocg_discriminator_run_from_number ${args[tocg_discriminator_run_from_number]} \
    --tocg_load_from_model ${args[tocg_load_from_model]} --tocg_discriminator_load_from_model ${args[tocg_discriminator_load_from_model]} \
    --VITON_Type Parser_Based --VITON_Name SD_VITON --VITON_Model ${args[VITON_Model]} --load_last_step ${args[load_last_step]} \
    --gpu_ids 0 --device ${args[device]} --res high_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
    --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
    --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}
  fi 

  if [ "${args[VITON_Model]}" == "GEN" ] 
  then 
    ./scripts/viton/sd_viton/gen_sd_viton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
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
  fi
fi



if [ "${args[job_name]}" == "Ladi_VITON" ]
then
echo "Running Ladi_VITON ..."

./scripts/viton/ladi_vton/tps_ladi_vton.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --tps_experiment_from_number ${args[tps_experiment_from_number]} --tps_run_from_number ${args[tps_run_from_number]} --tps_load_from_model ${args[tps_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name Ladi_VITON --VITON_Model ${args[VITON_Model]} --load_last_step ${args[load_last_step]} \
  --gpu_ids 0 --device ${args[device]} --res high_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}

fi

if [ "${args[job_name]}" == "CP_VTON_plus" ]
then

echo "Running CP_VTON_plus ..."
./scripts/viton/cpvtonplus/cpvtonplus.sh --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --gmm_experiment_from_number ${args[gmm_experiment_from_number]} --gmm_run_from_number ${args[gmm_run_from_number]} --gmm_load_from_model ${args[gmm_load_from_model]} \
  --tom_experiment_from_number ${args[tom_experiment_from_number]} --tom_run_from_number ${args[tom_run_from_number]} --tom_load_from_model ${args[tom_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name CP_VTON --VITON_Model ${args[VITON_Model]} --stage ${args[stage]} --load_last_step ${args[load_last_step]} \
  --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --low_res_dataset_name viton_plus \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]}

fi