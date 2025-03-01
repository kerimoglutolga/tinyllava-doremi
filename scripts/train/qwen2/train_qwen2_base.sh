DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data

LLM_VERSION=Qwen/Qwen2-0.5B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-0_5b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/qwen2/pretrain_qwen2.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/qwen2/finetune_qwen2.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
