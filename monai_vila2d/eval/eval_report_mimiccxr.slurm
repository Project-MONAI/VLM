#!/bin/bash

#SBATCH --job-name=report_mimiccxr_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2:00:00
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


srun -o job_report_mimiccxr_${OUTPUT_FOLDER_NAME}_eval.log --container-image=$CONTAINER \
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
    bash /data/code/monai_vlm/monai_vila2d/eval/scripts/report/replace_script.sh pycocoevalcap eval.py  /data/code/monai_vlm/monai_vila2d/eval/scripts/report/eval.py


    cd /data/code/VILA_github  
    export PYTHONPATH=/data/code/VILA_github
    echo "PYTHONPATH is $PYTHONPATH"

    IMAGE_DIR="/data/datasets/vlm/mimic-cxr-512_v2/images/"
    OUTPUT_PATH="/data/code/eval/$OUTPUT_FOLDER_NAME/report_mimiccxr"
    RESULT_PATH="$OUTPUT_PATH/results.json"

    mkdir -p $OUTPUT_PATH

    CUDA_VISIBLE_DEVICES=0 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_0 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE &

     CUDA_VISIBLE_DEVICES=1 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_1 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE &

     CUDA_VISIBLE_DEVICES=2 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_2 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE &            

     CUDA_VISIBLE_DEVICES=3 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_3 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE & 

     CUDA_VISIBLE_DEVICES=4 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_4 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE & 

     CUDA_VISIBLE_DEVICES=5 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_5 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE & 

     CUDA_VISIBLE_DEVICES=6 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_6 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE & 

     CUDA_VISIBLE_DEVICES=7 python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/report_mimiccxr_infer.py \
        --model-path /data/model_path \
        --image-list-file /data/datasets/vlm/lists/part_7 \
        --images-folder $IMAGE_DIR \
        --output-folder $OUTPUT_PATH \
        --conv-mode $CONV_MODE &

    wait

    python /data/code/monai_vlm/monai_vila2d/eval/scripts/report/compute_metrics_caption.py \
        --gt_dir /data/datasets/vlm/vila/text_gt/test_1 \
        --pred_dir $OUTPUT_PATH \
        --output $RESULT_PATH 

    '

