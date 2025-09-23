# export WANDB_API_KEY="API KEY"
# export WANDB_PROJ="diffusionsat_satclip_finetune_controlnet"
# export WANDB_ENTITY="YOUR USER OR TEAM NAME"

export HF_HOME="/mnt/s1/zwh/cache/huggingface"

# 离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch --config_file="$1" train_instruct_pix2pix_T2I.py

# CUDA_VISIBLE_DEVICES=1 ./launch_scripts/launch_512_instruct_pix2pix_T2I.sh config/launch_accelerate_configs/single_gpu.yaml