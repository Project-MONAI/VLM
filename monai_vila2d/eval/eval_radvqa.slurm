#!/bin/bash

#SBATCH --job-name=radvqa_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH -A healthcareeng_monai


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


srun -o job_radvqa_${OUTPUT_FOLDER_NAME}_eval.log --container-image=$CONTAINER \
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

	DATA_PATH="/data/datasets/vnath/datasets/radvqa/radvqa_test_instruct.json"
	IMAGE_DIR="/data/datasets/vnath/datasets/radvqa/VQA_RAD_Image_Folder"
    OUTPUT_ROOT="/data/code/eval/$OUTPUT_FOLDER_NAME/radvqa"
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


	python /data/code/monai_vlm/monai_vila2d/eval/scripts/metric_radvqa.py \
		--input $DATA_PATH \
		--answers $OUTPUT_PATH \
		--output $RESULT_PATH \

    '

