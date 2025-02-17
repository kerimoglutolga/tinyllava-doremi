#!/bin/bash

MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune/checkpoint-5195"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune"
EVAL_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava/data/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
