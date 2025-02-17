DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k_cleaned.json
IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data

LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048

export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/.cache
export HF_HOME=/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/.cache

export HF_TOKEN=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
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

#bash scripts/train/openelm/pretrain_openelm.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/openelm/finetune_openelm.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
