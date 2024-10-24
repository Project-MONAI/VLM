#!/bin/bash

#SBATCH --job-name=chestxray14_class_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
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

srun -o job_chestxray14_class_${OUTPUT_FOLDER_NAME}_eval.log --container-image=$CONTAINER \
     --container-mounts=$CODE:/data/code,${DATASETS}:/data/datasets,$MODEL_PATH:/data/model_path \
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

	OUTPUT_PATH="/data/code/eval/$OUTPUT_FOLDER_NAME/chestxray14_class"
	RESULT_PATH="/data/code/eval/$OUTPUT_FOLDER_NAME/chestxray14_class/results.json"

	mkdir -p $OUTPUT_PATH

	CUDA_VISIBLE_DEVICES=0,1 python /data/code/monai_vlm/m3/eval/scripts/classification/infer_chestxray14.py --idx 0 --mpath /data/model_path --conv $CONV_MODE --output $OUTPUT_PATH &
	CUDA_VISIBLE_DEVICES=2,3 python /data/code/monai_vlm/m3/eval/scripts/classification/infer_chestxray14.py --idx 1 --mpath /data/model_path --conv $CONV_MODE --output $OUTPUT_PATH &
	CUDA_VISIBLE_DEVICES=4,5 python /data/code/monai_vlm/m3/eval/scripts/classification/infer_chestxray14.py --idx 2 --mpath /data/model_path --conv $CONV_MODE --output $OUTPUT_PATH &
	CUDA_VISIBLE_DEVICES=6,7 python /data/code/monai_vlm/m3/eval/scripts/classification/infer_chestxray14.py --idx 3 --mpath /data/model_path --conv $CONV_MODE --output $OUTPUT_PATH &

    wait

	python /data/code/monai_vlm/m3/eval/scripts/classification/metric_chestxray14.py --answers $OUTPUT_PATH --output $RESULT_PATH

    '

