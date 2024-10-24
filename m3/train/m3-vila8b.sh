#!/bin/bash

# Set master address and worker list
master_addr=$(hostname)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export worker_list=$(hostname)

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

VILA_CODE_PATH="<path_to_vila_code>"

# User-defined variables
NODES=4
GPUS_PER_NODE=8
MASTER_PORT=25001
STAGE2_PATH="/path/to/your/model"
OUTPUT_DIR="/path/to/output/checkpoints"
CONTAINER_IMAGE="/path/to/your/container/image.sqsh"
CONTAINER_MOUNTS="/path/to/mounts"
WANDB_API_KEY="your_wandb_api_key"

# Removed srun and directly execute the commands
source /root/miniconda3/bin/activate
source /root/.bashrc
cd $VILA_CODE_PATH
conda activate vila
echo `which python`
wandb login <wandb_api_key>

export PYTHONPATH=$VILA_CODE_PATH
echo "PYTHONPATH is $PYTHONPATH"

echo "MASTER_ADDR="$MASTER_ADDR
CURRENT_RANK=${SLURM_PROCID:-"0"}
n_node=${SLURM_JOB_NUM_NODES:-1}
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list | rank $CURRENT_RANK of $n_node"

bs=16
STAGE2_PATH=$STAGE2_PATH
OUTPUT=provide_output_dir_name

HEALTHCARE_DS=$(for i in {1..10}; do echo -n usmle+; done)
HEALTHCARE_DS+=$(for i in {1..4}; do echo -n radvqa+; done)
HEALTHCARE_DS+=$(for i in {1..4}; do echo -n slake+; done)
HEALTHCARE_DS+=$(for i in {1..4}; do echo -n pathvqa+; done)
HEALTHCARE_DS+=$(for i in {1..2}; do echo -n expert_mimic_vqa+; done)
HEALTHCARE_DS+=$(for i in {1..8}; do echo -n expert_vista3d+; done)
HEALTHCARE_DS+=$(for i in {1..16}; do echo -n expert_brats+; done)
HEALTHCARE_DS+=$(for i in {1..1}; do echo -n mimic_report_dcl_train_update0_clean+; done)
HEALTHCARE_DS+=$(for i in {1..1}; do echo -n mimic_report_expert_dcl_train_update0_clean+; done)
HEALTHCARE_DS=${HEALTHCARE_DS%+}

torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $STAGE2_PATH \
    --version llama_3 \
    --data_mixture ${HEALTHCARE_DS} \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
