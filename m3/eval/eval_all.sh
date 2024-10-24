#!/bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
sbatch eval_report_mimiccxr_expert.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

sbatch eval_chestxray14_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chexpert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chestxray14_expert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 
sbatch eval_chexpert_expert_class.slurm $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE 

echo "Submitted all eval jobs"

squeue --me -l
