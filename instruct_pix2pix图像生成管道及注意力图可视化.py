from io import BytesIO

import PIL
import requests
import safetensors
import torch
from attention_map_diffusers import attn_maps, init_pipeline, save_attention_maps

from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)

iamge_patch = r"/mnt/data/zwh/data/maxar/brazil-flood-2024-5/disaster_image_pairs/before/105001003ACFED00_213131133010_211_from_10300100F6524600_213131133010.png"
image = PIL.Image.open(iamge_patch)

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "/mnt/data/zwh/log/instruct-pix2pix/experiment_fixed-flooding-prompt_0",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")

##### 1. Replace modules and Register hook #####
pipeline = init_pipeline(pipeline)

# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# 打印调度器类型
print(pipeline.scheduler.__class__.__name__)

# # 启用xformers
# if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
#     pipeline.enable_xformers_memory_efficient_attention()

prompt = "add flooding to the image"
image = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=20,
    image_guidance_scale=1.5,
    guidance_scale=7,
).images[0]


image_name = r"/home/zhangwh/Projects/mydiff/image/pix2pix.png"
image.save(image_name)

##### 2. Process and Save attention map #####
save_attention_maps(
    attn_maps,
    pipeline.tokenizer,
    prompt,
    base_dir="/home/zhangwh/Projects/mydiff/image",
    unconditional=False,
)


# state_dict = safetensors.torch.load_file(
#     "/mnt/data/zwh/log/instruct-pix2pix/unet/diffusion_pytorch_model.safetensors"
# )
# pipeline.unet.load_state_dict(state_dict)

# image = pipeline(prompt=prompt, image=image).images[0]
# image.save(image_name.replace(".png", "_pix2pix_finetune.png"))
