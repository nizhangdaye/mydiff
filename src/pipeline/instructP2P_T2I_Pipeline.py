import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipelineOutput,
)

try:  # 兼容不同 diffusers 版本 (adapter 所在模块路径)
    from diffusers import T2IAdapter
except Exception:  # pragma: no cover
    T2IAdapter = Any  # type: ignore


class InstructPix2PixT2IAdapterPipeline(StableDiffusionInstructPix2PixPipeline):
    """融合 InstructPix2Pix 与 T2I-Adapter 的推理管道。

    相比原始 `StableDiffusionInstructPix2PixPipeline`：
    1. 仍然通过在 UNet 输入处拼接 `image_latents` (额外 4 通道) 达成图像编辑条件；
    2. 额外使用 T2IAdapter 对原始输入 RGB 图像编码，得到多尺度 residual list，注入到 UNet 对应 down blocks；
    3. 支持 classifier-free guidance + image guidance（与官方 InstructPix2Pix 相同的 3-way split 策略）。

    推理输入：
        prompt: 文本或批量文本
        image: 待编辑图像 (PIL / numpy / tensor)
        adapter_image: 送入 T2IAdapter 的图像（通常与 image 相同，可允许外部传入以支持其他预处理）

    说明：
        diffusers 中 StableDiffusionAdapterPipeline 的 residual 注入方式是将 adapter forward 产生的特征逐层相加。
        这里我们仿照 ControlNet 版本的写法，在调用 unet 时通过 down_block_additional_residuals 传入；
        然而 InstructPix2Pix 的 UNet forward 已支持 `down_block_additional_residuals` 与 `mid_block_additional_residual`。
        T2IAdapter 仅提供多个 down residual，不提供 mid residual，因此 mid 传 None。
    """

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        adapter: Optional[Any] = None,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.adapter = adapter
        self.register_modules(adapter=adapter)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput,  # 编辑前的图片
        control_image: Optional[PipelineImageInput] = None,  # 深度，语义等
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        adapter_scale: float = 1.0,
        **kwargs,
    ):
        """执行图像编辑推理。

        adapter_scale: 对 adapter residual 进行整体缩放。
        """
        device = self._execution_device

        # 1. Encode prompt (支持 classifier-free guidance + image guidance)
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. 预处理图像 (InstructPix2Pix 输入 & Adapter 输入)
        image_tensor = self.image_processor.preprocess(image)
        adapter_tensor = self.image_processor.preprocess(control_image)

        # 3. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. 准备 image_latents (原始图像 latents，用于与噪声 latents concat)
        image_latents = self.prepare_image_latents(
            image_tensor,
            1 if isinstance(prompt, str) else len(prompt),
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 5. 采样初始 latents
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            (1 if isinstance(prompt, str) else len(prompt)) * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Adapter forward -> residuals 列表 (按照 StableDiffusionAdapterPipeline 约定顺序)
        # adapter 期望原始 RGB (归一化后) 输入，image_processor 输出范围在 [-1,1]
        adapter_residuals = self.adapter(adapter_tensor.to(device=device, dtype=prompt_embeds.dtype))
        # 允许整体缩放
        if adapter_scale != 1.0:
            adapter_residuals = [r * adapter_scale for r in adapter_residuals]

        # 7. 去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # 拼接图像条件 (与官方 InstructPix2Pix 一致)
                unet_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # UNet 前向 (将 adapter residuals 填入 down_block_additional_residuals)
                noise_pred = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_intrablock_additional_residuals=adapter_residuals,
                    mid_block_additional_residual=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # Guidance (text / image / unconditional) 三分策略
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # 调度器 step
                extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # 进度条更新
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. 解码并后处理
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        self.adapter.save_pretrained(os.path.join(save_directory, "adapter"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(os.path.join(save_directory, "feature_extractor"))
        self.save_config(save_directory)
