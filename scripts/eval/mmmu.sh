#!/bin/bash

MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-pretrain/checkpoint-1090"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-pretrain"
EVAL_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava/data/eval"

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/MMMU/eval

python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json
