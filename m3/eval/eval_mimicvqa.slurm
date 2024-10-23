#!/bin/bash

#SBATCH --job-name=mimicvqa_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH -A healthcareeng_monai

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

# TODO standardize gt data location


srun -o job_mimicvqa_${OUTPUT_FOLDER_NAME}_eval.log \
    --container-image=$CONTAINER \
    --container-mounts=$CODE:/data/code,$DATASETS:/data/datasets,$MODEL_PATH:/data/model_path \
    --no-container-mount-home \
    --container-remap-root \
    --export=ALL \
	bash -c '
	set -e
	source /root/miniconda3/bin/activate
	source /root/.bashrc
	conda activate vila

	cd /data/code/VILA_github  
	export PYTHONPATH=/data/code/VILA_github
	echo "PYTHONPATH is $PYTHONPATH"

	IMAGE_DIR="/data/datasets/MIMIC_VQA/images_small"
	GT_ROOT="/data/datasets/MIMIC_VQA/drax_evaluation/groundtruth_vqa_json_basenames"

	OUTPUT_ROOT="/data/code/eval/$OUTPUT_FOLDER_NAME/mimicvqa"
	RESULT_PATH="$OUTPUT_ROOT/results.json"

	mkdir -p $OUTPUT_ROOT

    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/abnormality/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/abnormality_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --conv-mode $CONV_MODE &

    CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/presence/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/presence_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --conv-mode $CONV_MODE &

    CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/view/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/view_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --conv-mode $CONV_MODE &

    CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/location/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/location_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --conv-mode $CONV_MODE &

    CUDA_VISIBLE_DEVICES=4 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/level/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/level_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --conv-mode $CONV_MODE &

    CUDA_VISIBLE_DEVICES=5 python -m llava.eval.model_vqa_science \
        --model-path /data/model_path \
        --question-file $GT_ROOT/type/llava_med_instruct_mimicvqa_test_type.json \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_ROOT/type_llava_med_instruct_mimicvqa_test_type_answers_images_small.jsonl \
        --generation-config "$GENERATION_CONFIG" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --conv-mode $CONV_MODE &

    wait


	python /data/code/monai_vlm/m3/eval/scripts/mimic_vqa/metric_mimicvqa.py \
	--input $GT_ROOT \
	--answers $OUTPUT_ROOT \
	--output $RESULT_PATH \
    '

