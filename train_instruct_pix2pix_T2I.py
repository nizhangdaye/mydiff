import argparse
import logging
import math
import os
import shutil
import time
from typing import Any, Dict, List

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    T2IAdapter,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from src.data.instruct_pix2pix_dataset import ImageEditDataset, collate_fn
from src.pipeline.instructP2P_T2I_Pipeline import InstructPix2PixT2IAdapterPipeline

logger = get_logger(__name__, log_level="INFO")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model(model_cfg: dict):
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_cfg["unet_path"],
        subfolder="unet",
        use_safetensors=False,
    )
    if model_cfg.get("t2i_adapter_path"):
        logger.info(f"Loading T2IAdapter from {model_cfg.get('t2i_adapter_path')}")
        adapter = T2IAdapter.from_pretrained(model_cfg.get("t2i_adapter_path"))
    else:
        logger.info("Initializing new T2IAdapter (random weights)")
        adapter = T2IAdapter(
            in_channels=model_cfg.get("t2i_adapter_in_channels"),
            channels=model_cfg.get("t2i_adapter_channels"),
            num_res_blocks=model_cfg.get("t2i_adapter_num_res_blocks"),
            downscale_factor=model_cfg.get("t2i_adapter_downscale_factor"),
            adapter_type=model_cfg.get("t2i_adapter_type"),
        )
    return noise_scheduler, tokenizer, text_encoder, vae, unet, adapter


def init_instructpix2pix_unet(unet):
    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels,
            out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    return unet


def run_epoch(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    noise_scheduler: DDPMScheduler,
    train_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    weight_dtype: torch.dtype,
    generator: torch.Generator | None,
    global_step: int,
    progress_bar: tqdm,
    ema_unet: EMAModel | None,
    adapter: Any | None = None,
    ema_adapter: EMAModel | None = None,
) -> tuple[float, int]:
    """运行单个训练 epoch。

    返回: (epoch_average_loss, 更新后的 global_step)
    统计的平均损失为该 epoch 内所有优化(完成一次梯度累计)步骤 loss 的算术平均。
    """
    # 用于累计日志用的 loss（按 step 打印）
    accum_loss_for_log = 0.0
    # 用于计算 epoch 平均 loss（只在完成一次优化步骤时统计）
    total_optim_loss = 0.0
    optim_steps_in_epoch = 0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            with accelerator.autocast():
                # 1. 将编辑后图像编码为 latent
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. 采样噪声与时间步
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # 3. 前向扩散：向 latent 加噪
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4. 文本条件
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 5. 原始图像 latent（mode）
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # 6. 条件 dropout (classifier-free guidance 支持)
                if data_cfg.get("conditioning_dropout_prob") is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * data_cfg.get("conditioning_dropout_prob")
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    null_inputs = tokenizer(
                        [""],
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    null_conditioning = text_encoder(null_inputs.input_ids.to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= data_cfg.get("conditioning_dropout_prob")).to(image_mask_dtype)
                        * (random_p < 3 * data_cfg.get("conditioning_dropout_prob")).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds

                # 7. 拼接 Unet 输入 image latent 与 noisy latent
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # 8. 拼接 T2I Adapter 输入
                controlnet_image_before_depth = (
                    batch["before_depth_pixel_values"].to(dtype=weight_dtype).repeat(1, 3, 1, 1)
                )  # [bs, 3, h, w]
                controlnet_image_before_seg = batch["before_seg_pixel_values"].to(dtype=weight_dtype)  # [bs, 3, h, w]

                # 将两种输入进行拼接
                t2iadapter_image = torch.cat(
                    [controlnet_image_before_depth, controlnet_image_before_seg], dim=1
                )  # [bs, 6, h, w]

                # 9. 确定监督目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # 10. 前向 + loss
                # TODO: 目前 T2Iadapter 只支持单种条件，暂时用深度条件
                down_block_additional_residuals = adapter(controlnet_image_before_depth)
                # NOTE:
                #   diffusers >=1.2.0 开始弃用在非 ControlNet 场景把 T2IAdapter 输出通过 down_block_additional_residuals 传入；
                #   T2IAdapter 的 residual 应该走 intrablock 形式: down_intrablock_additional_residuals。
                #   如果用户的 diffusers 版本仍旧只接受旧参数，会抛出 TypeError；为兼容性做一次回退。
                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_intrablock_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                    ],
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # 11. 分布式收集用于日志（保持与原实现一致）
            avg_loss = accelerator.gather(loss.repeat(train_cfg.get("train_batch_size"))).mean()
            accum_loss_for_log += avg_loss.item() / train_cfg.get("gradient_accumulation_steps")

            # 12. 反向与优化
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), opt_cfg.get("max_grad_norm"))
            optimizer.step()
            if lr_scheduler is not None:  # 非 plateau scheduler
                lr_scheduler.step()
            optimizer.zero_grad()

        # 13. 若完成一次优化（梯度已同步）
        if accelerator.sync_gradients:
            if model_cfg.get("use_ema"):
                if ema_unet is not None:
                    ema_unet.step(unet.parameters())
                if ema_adapter is not None and adapter is not None:
                    ema_adapter.step(adapter.parameters())
            progress_bar.update(1)
            global_step += 1
            optim_steps_in_epoch += 1
            total_optim_loss += accum_loss_for_log

            # 记录 step 级日志
            logs = {"step_loss": accum_loss_for_log}
            if lr_scheduler is not None:
                logs["lr"] = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            accum_loss_for_log = 0.0

    epoch_avg_loss = (total_optim_loss / optim_steps_in_epoch) if optim_steps_in_epoch > 0 else float("nan")
    return epoch_avg_loss, global_step


def log_validation(
    pipeline: InstructPix2PixT2IAdapterPipeline,
    cfg: Dict[str, Any],
    accelerator: Accelerator,
    generator: torch.Generator | None,
    epoch: int,
    val_dataset,
):
    """运行验证：对验证集每个样本生成编辑结果并保存/记录。

    保存路径: output_dir/validation/epoch-{epoch}/sample_{i}.png
    记录: 可选 wandb 表、tensorboard 图像+文本。
    """
    # Scheduler 切 UniPC（更快推理）
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if cfg["precision"].get("enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()

    resolution = cfg["data"].get("resolution")
    inference_steps = cfg["validation"].get("num_inference_steps")

    logger.info(f"Running validation epoch={epoch} (steps={inference_steps}) on {len(val_dataset)} samples...")

    autocast_ctx = torch.autocast(accelerator.device.type)
    validation_results = []

    with autocast_ctx:
        for idx in range(len(val_dataset)):
            raw_sample = val_dataset.dataset[idx]
            before_image = raw_sample["before"]
            after_image = raw_sample["after"]
            before_depth = raw_sample["before_depth"]
            before_seg = raw_sample["before_seg"]

            # prompt 选择
            if getattr(val_dataset, "use_fixed_edit_text") and getattr(val_dataset, "fixed_edit_mapping"):
                edit_instruction = val_dataset.fixed_edit_mapping[raw_sample["disaster_type"]]
            else:
                edit_instruction = raw_sample["edit"]

            # 转 PIL
            def to_pil(x):
                if isinstance(x, Image.Image):
                    return x
                return Image.fromarray(x)

            before_image = to_pil(before_image).resize((resolution, resolution))
            after_image = to_pil(after_image).resize((resolution, resolution))
            before_depth_img = to_pil(before_depth).resize((resolution, resolution)).convert("RGB")
            before_seg_img = to_pil(before_seg).resize((resolution, resolution))

            # 条件图构建（深度 + seg -> 6 通道），若缺失则用黑图
            before_depth_tensor = transforms.ToTensor()(before_depth_img)
            before_seg_tensor = transforms.ToTensor()(before_seg_img)
            conditioning_image = (
                torch.cat([before_depth_tensor, before_seg_tensor], dim=0).unsqueeze(0).to(accelerator.device)
            )

            # TODO: T2I 目前不支持多种条件输入，暂时使用 before_depth_tensor
            edited_image = pipeline(
                prompt=edit_instruction,
                image=before_image,
                control_image=before_depth_tensor,
                num_inference_steps=inference_steps,
                generator=generator,
                guidance_scale=7.0,
                image_guidance_scale=1.5,
                adapter_scale=1.0,
            ).images[0]

            # 可视化网格
            vis_w, vis_h = resolution, resolution
            grid = Image.new("RGB", (vis_w * 3, vis_h * 2), (255, 255, 255))
            grid.paste(before_image, (0, 0))
            grid.paste(after_image, (vis_w, 0))
            grid.paste(before_depth_img, (vis_w * 2, 0))
            grid.paste(before_seg_img.convert("RGB"), (0, vis_h))
            grid.paste(edited_image, (vis_w, vis_h))

            validation_results.append(
                {
                    "sample_idx": idx,
                    "vis_image": grid,
                    "prompt": edit_instruction,
                }
            )

    # Trackers logging
    for tracker in getattr(accelerator, "trackers", []):
        for r in validation_results:
            img_np = np.array(r["vis_image"]).transpose(2, 0, 1)
            tracker.writer.add_image(f"validation/sample_{r['sample_idx']}", img_np, epoch)
            tracker.writer.add_text(f"validation/prompt_{r['sample_idx']}", r["prompt"], epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_instruct_p2p_T2I.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 解包配置子字典，便于引用
    model_cfg = cfg.get("model")
    train_cfg = cfg.get("training")
    opt_cfg = cfg.get("optimizer")
    sched_cfg = cfg.get("lr_scheduler")
    data_cfg = cfg.get("data")
    log_cfg = cfg.get("logging")
    precision_cfg = cfg.get("precision")
    ckpt_cfg = cfg.get("checkpointing")

    output_dir = os.path.join(cfg["output_root"], f"{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    cfg["output_dir"] = output_dir

    logging_dir = os.path.join(output_dir, log_cfg.get("logging_dir"))
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps"),
        mixed_precision=precision_cfg.get("mixed_precision"),
        log_with=log_cfg.get("report_to"),
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    seed = cfg.get("seed")
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        print(f"配置文件：")
        print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet, adapter = load_model(model_cfg)

    unet = init_instructpix2pix_unet(unet)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    adapter.train()

    # Create EMA for the unet.
    if model_cfg.get("use_ema"):
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema_adapter = EMAModel(adapter.parameters(), model_cls=T2IAdapter, model_config=adapter.config)

    if precision_cfg.get("enable_xformers_memory_efficient_attention"):
        unet.enable_xformers_memory_efficient_attention()
        adapter.enable_xformers_memory_efficient_attention()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if model_cfg.get("use_ema"):
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    ema_adapter.save_pretrained(os.path.join(output_dir, "t2iadapter_ema"))

                # models 里顺序：unet, adapter （accelerator.prepare 顺序）
                for i, model in enumerate(models):
                    sub_dir = "unet" if isinstance(model, UNet2DConditionModel) else "t2iadapter"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if model_cfg.get("use_ema"):
                load_unet_ema = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_unet_ema.state_dict())
                ema_unet.to(accelerator.device)
                del load_unet_ema
                if os.path.isdir(os.path.join(input_dir, "t2iadapter_ema")):
                    load_adapter_ema = EMAModel.from_pretrained(os.path.join(input_dir, "t2iadapter_ema"))
                    ema_adapter.load_state_dict(load_adapter_ema.state_dict())
                    ema_adapter.to(accelerator.device)
                    del load_adapter_ema

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, UNet2DConditionModel):
                    load_unet_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_unet_model.config)
                    model.load_state_dict(load_unet_model.state_dict())
                    del load_unet_model
                else:  # adapter
                    load_adapter_model = T2IAdapter.from_pretrained(os.path.join(input_dir, "t2iadapter"))
                    model.load_state_dict(load_adapter_model.state_dict())
                    del load_adapter_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if train_cfg.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if precision_cfg.get("allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    learning_rate = opt_cfg.get("learning_rate")
    if opt_cfg.get("scale_lr"):
        learning_rate = (
            learning_rate
            * train_cfg.get("gradient_accumulation_steps")
            * train_cfg.get("train_batch_size")
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if opt_cfg.get("use_8bit_adam"):
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # 需要训练的参数集合：UNet + Adapter
    params_to_optimize = list(unet.parameters()) + list(adapter.parameters())

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=learning_rate,
        betas=(opt_cfg.get("adam_beta1"), opt_cfg.get("adam_beta2")),
        weight_decay=opt_cfg.get("adam_weight_decay"),
        eps=opt_cfg.get("adam_epsilon"),
    )

    # Create dataset using the custom ImageEditDataset class

    with accelerator.main_process_first():
        # Create training dataset
        train_dataset = ImageEditDataset(
            dataset_path=data_cfg.get("dataset_path"),
            tokenizer=tokenizer,
            resolution=data_cfg.get("resolution"),
            split="train",
            center_crop=data_cfg.get("center_crop"),
            random_flip=data_cfg.get("random_flip"),
            max_samples=train_cfg.get("max_train_samples"),
            seed=seed,
            use_fixed_edit_text=data_cfg.get("use_fixed_edit_text"),
        )

        # Create validation dataset for visualization
        val_dataset = ImageEditDataset(
            dataset_path=data_cfg.get("dataset_path"),
            tokenizer=tokenizer,
            resolution=data_cfg.get("resolution"),
            split="test",
            center_crop=False,
            random_flip=False,
            max_samples=10,  # 限制验证样本数量
            seed=seed,
            use_fixed_edit_text=data_cfg.get("use_fixed_edit_text", False),
        )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_cfg.get("train_batch_size"),
        num_workers=data_cfg.get("dataloader_num_workers"),
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = sched_cfg.get("lr_warmup_steps") * accelerator.num_processes
    max_train_steps = train_cfg.get("max_train_steps")
    if max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / train_cfg.get("gradient_accumulation_steps")
        )
        num_training_steps_for_scheduler = (
            train_cfg.get("num_train_epochs") * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes

    if sched_cfg.get("lr_scheduler") == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("plateau_factor"),
            patience=sched_cfg.get("plateau_patience"),
            min_lr=sched_cfg.get("min_lr"),
        )
    else:
        lr_scheduler = get_scheduler(
            sched_cfg.get("lr_scheduler"),
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

    # Prepare everything with our `accelerator`.
    unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, adapter, optimizer, train_dataloader, lr_scheduler
    )

    if model_cfg.get("use_ema"):
        ema_unet.to(accelerator.device)
        ema_adapter.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_cfg.get("gradient_accumulation_steps"))
    if max_train_steps is None:
        max_train_steps = train_cfg.get("num_train_epochs") * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix")

    # Train!
    total_batch_size = (
        train_cfg.get("train_batch_size") * accelerator.num_processes * train_cfg.get("gradient_accumulation_steps")
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_cfg.get('train_batch_size')}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_cfg.get('gradient_accumulation_steps')}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Epoch-based resume: look for checkpoint-epoch-* only
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-epoch-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-epoch-")[1]))
            latest = dirs[-1] if dirs else None
            if latest is None:
                accelerator.print("No epoch checkpoint found, starting fresh.")
            else:
                epoch_num = int(latest.split("-epoch-")[1])
                accelerator.print(f"Resuming from epoch checkpoint {latest}")
                accelerator.load_state(os.path.join(output_dir, latest))
                first_epoch = epoch_num  # 继续该 epoch（已完成的 epoch 数=epoch_num）
                global_step = epoch_num * num_update_steps_per_epoch
        else:
            # Provided path could be absolute or relative; extract epoch number
            base = os.path.basename(resume_from_checkpoint.rstrip("/"))
            if base.startswith("checkpoint-epoch-"):
                epoch_num = int(base.split("-epoch-")[1])
                if os.path.isdir(resume_from_checkpoint):
                    load_dir = resume_from_checkpoint
                else:
                    load_dir = os.path.join(output_dir, base)
                if not os.path.isdir(load_dir):
                    accelerator.print(f"Checkpoint directory {load_dir} not found; starting fresh.")
                else:
                    accelerator.print(f"Resuming from epoch checkpoint {load_dir}")
                    accelerator.load_state(load_dir)
                    first_epoch = epoch_num
                    global_step = epoch_num * num_update_steps_per_epoch
            else:
                accelerator.print(f"Provided resume path {resume_from_checkpoint} is not an epoch checkpoint; ignored.")

    for epoch in range(first_epoch, num_train_epochs):
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs}")

        epoch_avg_loss, global_step = run_epoch(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler if not sched_cfg.get("lr_scheduler") == "plateau" else None,
            noise_scheduler=noise_scheduler,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            opt_cfg=opt_cfg,
            model_cfg=model_cfg,
            weight_dtype=weight_dtype,
            generator=generator,
            global_step=global_step,
            progress_bar=progress_bar,
            ema_unet=ema_unet if model_cfg.get("use_ema") else None,
            adapter=adapter,
            ema_adapter=ema_adapter if model_cfg.get("use_ema") else None,
        )

        # 当前 epoch 的进度条完成后关闭（避免多 epoch 时 tqdm 行数累积）
        progress_bar.close()

        # 按 epoch 保存（若启用）
        if accelerator.is_main_process and ckpt_cfg.get("checkpoint_epochs"):
            if (epoch + 1) % ckpt_cfg.get("checkpoint_epochs") == 0:
                # checkpoint 轮换
                if train_cfg.get("checkpoints_total_limit") is not None:
                    checkpoints = os.listdir(output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint-epoch")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-epoch-")[1]))
                    if len(checkpoints) >= train_cfg.get("checkpoints_total_limit"):
                        num_to_remove = len(checkpoints) - train_cfg.get("checkpoints_total_limit") + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        logger.info(
                            f"{len(checkpoints)} epoch checkpoints exist, removing {len(removing_checkpoints)}: {', '.join(removing_checkpoints)}"
                        )
                        for rc in removing_checkpoints:
                            shutil.rmtree(os.path.join(output_dir, rc))
                save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
                accelerator.save_state(save_path)
                logger.info(f"[Epoch Save] Saved state to {save_path}")

        # epoch 平均损失日志
        # if accelerator.is_main_process:  TODO 其他进程的优化器学习率不会更新
        accelerator.print(f"Epoch {epoch} average loss: {epoch_avg_loss:.6f}")
        log_payload = {"epoch_avg_loss": epoch_avg_loss}
        if sched_cfg.get("lr_scheduler") == "plateau":
            # ReduceLROnPlateau 在 epoch 结束后根据 loss 调整
            lr_scheduler.step(epoch_avg_loss)
            log_payload.update({"lr": lr_scheduler.get_last_lr()[0]})
        accelerator.log(log_payload, step=epoch)

        # ========== 最优模型保存逻辑 (基于 epoch 平均 loss) ==========
        # 在主进程比较并保存最优 checkpoint (含状态)。
        if accelerator.is_main_process and not math.isnan(epoch_avg_loss):
            # 使用属性缓存 best
            if not hasattr(main, "_best_epoch_loss"):
                main._best_epoch_loss = float("inf")  # type: ignore
            if epoch_avg_loss < main._best_epoch_loss:  # type: ignore
                prev = getattr(main, "_best_epoch_loss")
                main._best_epoch_loss = float(epoch_avg_loss)  # type: ignore
                best_dir = os.path.join(output_dir, "checkpoint-best")
                # 若存在旧 best，可先删除（保持单一目录）
                if os.path.isdir(best_dir):
                    try:
                        shutil.rmtree(best_dir)
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Failed removing old best checkpoint dir: {e}")
                accelerator.save_state(best_dir)
                logger.info(
                    f"[Best Model] Epoch {epoch} new best loss {epoch_avg_loss:.6f} (prev {prev:.6f}). Saved to {best_dir}"
                )

        if accelerator.is_main_process:
            if epoch % train_cfg.get("validation_epochs") == 0:
                if model_cfg.get("use_ema"):
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                    ema_adapter.store(adapter.parameters())
                    ema_adapter.copy_to(adapter.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = InstructPix2PixT2IAdapterPipeline(
                    vae=unwrap_model(vae),
                    text_encoder=unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=unwrap_model(unet),
                    scheduler=noise_scheduler,  # 或推理时替换为 UniPCMultistepScheduler
                    adapter=unwrap_model(adapter),
                    safety_checker=None,
                    requires_safety_checker=False,
                )

                log_validation(pipeline, cfg, accelerator, generator, epoch, val_dataset)

                if model_cfg.get("use_ema"):
                    ema_unet.restore(unet.parameters())
                    ema_adapter.restore(adapter.parameters())

                del pipeline
                torch.cuda.empty_cache()

        if global_step >= max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if model_cfg.get("use_ema"):
            ema_unet.copy_to(unet.parameters())
            ema_adapter.copy_to(adapter.parameters())

        pipeline = InstructPix2PixT2IAdapterPipeline(
            vae=unwrap_model(vae),
            text_encoder=unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unwrap_model(unet),
            scheduler=noise_scheduler,  # 或推理时替换为 UniPCMultistepScheduler
            adapter=unwrap_model(adapter),
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(output_dir)

    if accelerator.is_main_process:
        log_validation(pipeline, cfg, accelerator, generator, epoch, val_dataset)
    accelerator.end_training()


if __name__ == "__main__":
    main()
