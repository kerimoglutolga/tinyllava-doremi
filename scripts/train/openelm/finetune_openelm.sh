#!/bin/bash
if [ $# -ne 10 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
CONV_VERSION="$7"
VERSION="$8"
TRAIN_RECIPE="$9"
MODEL_MAX_LENGTH="${10}"

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed --include localhost:0,1,2,3 --master_port 29503 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen\
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-pretrain \
    --output_dir /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune-doremi \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune \
    --use_doremi True \
    --doremi_reference_model_path /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune \
