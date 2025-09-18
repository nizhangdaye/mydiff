# export WANDB_API_KEY="API KEY"
# export WANDB_PROJ="diffusionsat_satclip_finetune_controlnet"
# export WANDB_ENTITY="YOUR USER OR TEAM NAME"

export HF_HOME="/mnt/data/zwh/cache/huggingface"
export MODEL_NAME="/mnt/data/zwh/log/instruct-pix2pix/experiment_0"

export OUT_DIR="/mnt/data/zwh/log/instruct-pix2pix-CRS-UniControlNet/experiment_2"
# 离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch --config_file="$1" \
  --mixed_precision="fp16" \
  --main_process_port=43736 \
  --gpu_ids="$CUDA_VISIBLE_DEVICES" \
  train_instruct_pix2pix_CRS_Uni_ControlNet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 \
  --unet_path="/mnt/data/zwh/model/DiffusionSat/checkpoint-100000" \
  --train_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --checkpointing_steps=1 \
  --checkpoints_total_limit=5 \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --output_dir="${OUT_DIR}" \
  --resume_from_checkpoint="latest" \
  --dataloader_num_workers=4 \
  --validation_epochs=1 \
  --num_train_epochs=10 \
  --enable_xformers_memory_efficient_attention \
  --use_ema
# --lr_warmup_steps=1000 \
# --use_fixed_edit_text \
# num_train_epochs=2000 \
# --seed \
# --wandb="${WANDB_PROJ}" \


# CUDA_VISIBLE_DEVICES=1 ./launch_scripts/launch_512_xbd_instruct_pix2pix.sh config/launch_accelerate_configs/single_gpu.yaml