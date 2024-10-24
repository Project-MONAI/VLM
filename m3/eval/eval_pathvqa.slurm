#!/bin/bash

#SBATCH --job-name=pathvqa_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH -A monai

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

# common
export GENERATION_CONFIG='{"max_new_tokens":1024,"do_sample":false,"temperature":0}'


srun -o job_pathvqa_${OUTPUT_FOLDER_NAME}_eval.log --container-image=$CONTAINER \
     --container-mounts=$CODE:/data/code,$DATASETS:/data/datasets,$MODEL_PATH:/data/model_path \
     --no-container-mount-home \
     --container-remap-root \
     --export=ALL \
     bash -c '
	source /root/miniconda3/bin/activate
	source /root/.bashrc
	conda activate vila

	cd /data/code/VILA_github  
	export PYTHONPATH=/data/code/VILA_github
	echo "PYTHONPATH is $PYTHONPATH"

	DATA_PATH="/data/datasets/vnath/datasets/pathvqa/jsons/pathvqa_instruct_test.json"
	IMAGE_DIR="/data/datasets/vnath/datasets/pathvqa/images"
    OUTPUT_ROOT="/data/code/eval/$OUTPUT_FOLDER_NAME/pathvqa"
	OUTPUT_PATH="$OUTPUT_ROOT/outputs.jsonl"
	RESULT_PATH="$OUTPUT_ROOT/results.json"


    python -m llava.eval.model_vqa_science \
            --model-path /data/model_path \
            --question-file $DATA_PATH \
            --image-folder $IMAGE_DIR \
            --answers-file $OUTPUT_PATH \
            --num-chunks 1 \
            --chunk-idx 0 \
            --single-pred-prompt \
            --generation-config "$GENERATION_CONFIG" \
            --conv-mode $CONV_MODE 

	python /data/code/monai_vlm/m3/eval/scripts/metric_pathvqa.py \
		--input "/data/datasets/vnath/datasets/pathvqa/qas/test_vqa.pkl" \
		--answers $OUTPUT_PATH \
		--output $RESULT_PATH \

    '

