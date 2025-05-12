#!/bin/bash
DATASET_PATH="YOUR_DATASET_PATH"
EXP_NAME="YOUR_EXP_NAME"
DATASET_NAME="YOUR_DATASET_NAME"
IMAGE_ENCODER="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
UNET_BACKBONE="models/sdxl/dreamshaper.safetensors"

LOGGING_DIR="log/${EXP_NAME}"
mkdir -p "${LOGGING_DIR}"
cp "$0" "${LOGGING_DIR}/$(basename "$0")_$(date '+%Y%m%d%H%M%S').sh"

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 29500 \
    --num_processes 8 \
    --multi_gpu \
    train.py \
    --unet_backbone ${UNET_BACKBONE} \
    --pretrained_model_name_or_path ${PRETRAINED_MODEL_PATH} \
    --pretrained_ip_adapter_path models/ipa_weights/ip-adapter-plus-face_sdxl_vit-h.bin \
    --num_tokens 16 \
    --image_encoder_path models/image_encoder \
    --output_dir output_models/${EXP_NAME} \
    --logging_dir log/${EXP_NAME} \
    --train_height 1344\
    --train_width 768 \
    --train_batch_size 2 \
    --mixed_precision bf16 \
    --report_to tensorboard \
    --num_image_tokens 16 \
    --dataset_name ${DATASET_PATH} \
    --multi \
    --max_num_objects 2 \
    --use_controlnet \
    --control_cond skeleton \
    --save_steps 5000 \
    --learning_rate 1e-6 \
    --max_train_steps 50000 \
    --train_controlnet
    
