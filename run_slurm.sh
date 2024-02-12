#!/bin/bash
# Remember for HR_VITON, 
#to change the fine_width & fine height 
#depending on TOCG or GEN

# run_number > 10 = Trained using new wandb
experiment_number=22
run_number=30 # Compare all methods
experiment_from_number=0
run_from_number=0
seed_number=1
VITON_Type="Parser_Free" # [Parser_Based, Parser_Free]
VITON_Name="PF_AFN" # [FS_VTON, PF_AFN,DM_VTON, ACGPN, CP_VTON, CP_VTON_plus, HR_VITON, Ladi_VITON,SD_VITON]
task="PB_Gen" # [PB_Gen, PB_Warp,PF_Warp, PF_Gen, EMASC, GMM, TOM, TOCG, GEN]
dataset_name="Rail" # [Original, Rail]
# log="viton.%N.%j"
log="viton.%N"
path="/home-mscluster/mmolefe/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/inference_pipeline/experiments/experiment_${experiment_number}/run_${run_number}/VITON/${VITON_Type}/${VITON_Name}/${dataset_name}"
mkdir -p "$path"
output="${path}/${log}.out"
error="${path}/${log}.err"
echo "Path: ${path}"
echo "output: $output"
echo "error: $error"
echo "$VITON_Name"

export EXPERIMENT_NUMBER=$experiment_number
export EXPERIMENT_FROM_NUMBER=$experiment_from_number
export RUN_NUMBER=$run_number
export RUN_FROM_NUMBER=$run_from_number
export DATASET_NAME=$dataset_name
export TASK=$task
export SEED=$seed_number
export DEBUG=0
export WANDB=0
export SWEEPS=1
export DATAMODE=train
export DEVICE=0
export VITON_NAME=$VITON_Name

if [ "$VITON_Name" == "DM_VTON" ]
then 
    if [ "$task" == "PB_Warp" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/dm_vton/pb_warp.slurm
    fi 

    if [ "$task" == "PB_Gen" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/dm_vton/pb_gen.slurm
    fi 

    if [ "$task" == "PF_Warp" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/dm_vton/pf_warp.slurm
    fi 

    if [ "$task" == "PF_Gen" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/dm_vton/pf_gen.slurm
    fi 
fi

if [ "$VITON_Name" == "PF_AFN" ]
then 
    if [ "$task" == "PB_Warp" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/pf_afn/pb_warp.slurm
    fi 

    if [ "$task" == "PB_Gen" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/pf_afn/pb_gen.slurm
    fi 

    if [ "$task" == "PF_Warp" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/pf_afn/pf_warp.slurm
    fi 

    if [ "$task" == "PF_Gen" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/pf_afn/pf_gen.slurm
    fi 
fi

if [ "$VITON_Name" == "FS_VTON" ]
then 
    if [ "$task" == "PB_Warp" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/fs_vton/pb_warp.slurm
    fi 

    if [ "$task" == "PB_Gen" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/fs_vton/pb_gen.slurm
    fi 

    if [ "$task" == "PF_Warp" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/fs_vton/pf_warp.slurm
    fi 

    if [ "$task" == "PF_Gen" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/fs_vton/pf_gen.slurm
    fi 
fi


if [ "$VITON_Name" == "HR_VITON" ]
then 
    if [ "$task" == "TOCG" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/hrviton/tocg_hrviton.slurm
    fi 

    if [ "$task" == "GEN" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/hrviton/gen_hrviton.slurm
    fi 
fi

if [ "$VITON_Name" == "SD_VITON" ]
then 
    if [ "$task" == "TOCG" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/sd_viton/tocg_sd_viton.slurm
    fi 

    if [ "$task" == "GEN" ]
    then 
       sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/sd_viton/gen_sd_viton.slurm
    fi 
fi

if [ "$VITON_Name" == "ACGPN" ]
then 
    sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/acgpn/acgpn.slurm
fi

if [ "$VITON_Name" == "CP_VTON" ]
then 
    if [ "$task" == "GMM" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/cpvton/cpvton_gmm.slurm
    fi 

    if [ "$task" == "TOM" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/cpvton/cpvton_tom.slurm
    fi 
fi

if [ "$VITON_Name" == "CP_VTON_plus" ]
then 
    
    if [ "$task" == "GMM" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/cpvtonplus/cpvtonplus_gmm.slurm
    fi 

    if [ "$task" == "TOM" ]
    then 
        sbatch -J "${VITON_Name}_${run_number}" -o "$output" -e "$error" slurms/viton/cpvtonplus/cpvtonplus_tom.slurm
    fi 
fi

