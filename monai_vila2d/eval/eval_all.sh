#!/bin/bash

# set common env vars
source set_env.sh

if [[ $# -ne 3 ]]; then
    print_usage
    exit 1
fi

export MODEL_PATH=$1
export OUTPUT_FOLDER_NAME=$2
export CONV_MODE=$3

# check if env vars are set
: ${CONTAINER:?"CONTAINER env var is not set!"}
: ${DATASETS:?"DATASETS env var is not set!"}
: ${CODE:?"CODE env var is not set!"}
 

sbatch eval_radvqa.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_slakevqa.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_pathvqa.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

sbatch eval_mimicvqa.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

sbatch eval_report_mimiccxr.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

sbatch eval_chestxray14_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chexpert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chestxray14_expert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chexpert_expert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

echo "Submitted all eval jobs"

squeue --me -l
