#!/bin/bash

#SBATCH --job-name=report_mimiccxr_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --dependency=singleton

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
: ${ACCOUNT:?"ACCOUNT env var is not set!"}

# common
export GENERATION_CONFIG='{"max_new_tokens":1024,"do_sample":false,"temperature":0}'


srun -o job_report_mimiccxr_${OUTPUT_FOLDER_NAME}_eval.log --container-image=$CONTAINER --acount=$ACCOUNT \
     --container-mounts=$CODE:/data/code,$DATASETS:/data/datasets,$MODEL_PATH:/data/model_path \
     --no-container-mount-home \
     --container-remap-root \
     --export=ALL \
     bash -c '

   set -e
   source /root/miniconda3/bin/activate
   source /root/.bashrc
   conda activate vila

   apt-get update
   apt-get install -y default-jre
   python -m pip show pycocoevalcap || python -m pip install pycocoevalcap

   cd /data/code/VILA_github  
   export PYTHONPATH=/data/code/VILA_github
   echo "PYTHONPATH is $PYTHONPATH"


   IMAGE_DIR="/data/datasets/vlm/mimic-cxr-512_v2/images/"
   OUTPUT_PATH="/data/code/eval/$OUTPUT_FOLDER_NAME/report_mimiccxr"
   mkdir -p $OUTPUT_PATH


   IMAGE_LIST_FILE=/data/code/monai_vlm/m3/eval/scripts/report_updated/images_test_front.txt
   TEXT_GT_DIR=/data/datasets/vlm/vila/text_gt/test_1
   TEXT_GT_CLEAN_DIR=/data/datasets/vlm/vila/text_gt/test_1_update7_clean_front

   RESULT_FILE="$OUTPUT_PATH/results.json"
   RESULT_CLEAN_FILE="$OUTPUT_PATH/results_clean.json"


   OUTPUT_DIR=$OUTPUT_PATH/output_checkpoint
   OUTPUT_CLEAN_DIR=${OUTPUT_DIR}_clean
   mkdir -p ${OUTPUT_DIR}


   NUM_GPUS=8

   # Step 1: Run inference for each GPU in parallel
   echo "Running inference with multi-GPU..."

   for (( gpu=0; gpu<$NUM_GPUS; gpu++ ))
   do

      CUDA_VISIBLE_DEVICES=${gpu} python /data/code/monai_vlm/m3/eval/scripts/report_updated/inference.py \
         --model-path /data/model_path \
         --image-list-file ${IMAGE_LIST_FILE} \
         --images-folder ${IMAGE_DIR} \
         --output-folder ${OUTPUT_DIR} \
         --conv-mode ${CONV_MODE} \
         --num_gpus ${NUM_GPUS} \
         --gpu_id ${gpu} &

   done
   wait


   # Step 2: Clean output text
   echo "Cleaning output texts..."
   mkdir -p ${OUTPUT_CLEAN_DIR}
   python /data/code/monai_vlm/m3/eval/scripts/report_updated/cleaning.py ${OUTPUT_DIR} ${OUTPUT_CLEAN_DIR}


   echo "Computing metrics..."
   python /data/code/monai_vlm/m3/eval/scripts/report_updated/metrics.py \
      --gt_dir $TEXT_GT_DIR \
      --pred_dir $OUTPUT_CLEAN_DIR \
      --output $RESULT_FILE 


   # Step 3: Computing metrics
   echo "Computing metrics..."

   python /data/code/monai_vlm/m3/eval/scripts/report_updated/metrics.py \
      --gt_dir $TEXT_GT_CLEAN_DIR \
      --pred_dir $OUTPUT_CLEAN_DIR \
      --output $RESULT_CLEAN_FILE 


   # Step 4: Computing GREEN score

   pip install --ignore-requires-python git+https://github.com/Stanford-AIMI/GREEN.git@b1f2f8caf1c10ecba46ef2134ef1c0e89e517dfc
   apt-get install -y bc


   GPU_IDS=0,1,2,3,4,5,6,7
   RESULT_GREEN_FILE="$OUTPUT_PATH/result_green.json"
   RESULT_CLEAN_GREEN_FILE="$OUTPUT_PATH/result_clean_green.json"

   echo "Computing GREEN score..."
   bash /data/code/monai_vlm/m3/eval/scripts/report_updated/eval_green_score.sh ${TEXT_GT_DIR} ${OUTPUT_CLEAN_DIR} "$GPU_IDS" ${RESULT_GREEN_FILE}

   echo "Computing GREEN score clean ..."
   bash /data/code/monai_vlm/m3/eval/scripts/report_updated/eval_green_score.sh ${TEXT_GT_CLEAN_DIR} ${OUTPUT_CLEAN_DIR} "$GPU_IDS" ${RESULT_CLEAN_GREEN_FILE}
      
   echo "All steps completed successfully."

   '
