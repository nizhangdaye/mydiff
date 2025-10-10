import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipelineOutput,
)
from torch import nn

try:  # 兼容不同 diffusers 版本 (IPAdapter mixin 中的方法调用)
    from diffusers import T2IAdapter  # noqa: F401  # 仅为类型提示/潜在复用
except Exception:  # pragma: no cover
    T2IAdapter = Any  # type: ignore

from src.models.ip_adapter.ip_adapter import IPAdapterUNet2DConditionModel
from utils.depth_processing import simple_multi_threshold


class InstructPix2PixIpAdapterPipeline(StableDiffusionInstructPix2PixPipeline):
    """将 InstructPix2Pix 与自定义 IPAdapterUNet2DConditionModel 结合的推理管道。

    训练阶段对 UNet 的使用方式：将 (noisy_latents, original_image_latents) 在 channel 维拼接输入；
    同时通过 `reference_image_embeds` (来源于 depth -> 多阈值伪 RGB -> CLIP vision encoder 倒数第二层 hidden states)
    注入到修改后的注意力层 (IPAdapterAttnProcessor2_0) 中。

    本推理管道复现上述流程：
    1. 对 "原始待编辑图" 编码得到 image_latents (与官方 InstructPix2Pix 一样)；
    2. 采样初始噪声 latents；
    3. 若提供 depth/condition 图 (control_image)，执行 simple_multi_threshold -> 244 维度分类伪图；
       然后走 image_encoder 得到 hidden_states[-2] 作为 reference_image_embeds；
       也允许直接传入 ip_adapter_image_embeds (优先级更高)；
    4. 去噪循环中，每一步将 (latents, image_latents) concat 后送入 ip_adapter.unet；
       做 classifier-free guidance (text + image guidance 三分)；
    5. 解码输出。

    备注：与训练一致，这里仅使用单一 reference 条件；若未来扩展可支持列表并在 IPAdapterUNet2DConditionModel 内部对 batch 维或 concat 处理。
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        执行图像生成的主推理流程。

        支持：
        - 文本到图像（text-to-image）或图像到图像（image-to-image）生成
        - 分类器自由引导（Classifier-Free Guidance）
        - IP-Adapter 图像条件嵌入
        - 自定义回调函数
        - 多种输出格式（如 PIL 图像、NumPy 数组、PyTorch 张量等）
        """

        # 兼容旧版参数：从 kwargs 中提取已弃用的 callback 和 callback_steps
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 如果传入的是 PipelineCallback 或 MultiPipelineCallbacks 对象，
        # 则自动使用其定义的 tensor_inputs 作为回调需要的张量变量名
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 保存当前使用的引导尺度（用于后续可能的属性访问）
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        # 获取模型运行设备（如 'cuda' 或 'cpu'）
        device = self._execution_device

        # ==============================
        # 1. 定义调用参数（batch size）
        # ==============================
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # 若未提供 prompt，则使用已有的 prompt_embeds 推断 batch size
            batch_size = prompt_embeds.shape[0]

        # 再次获取设备（冗余，但安全）
        device = self._execution_device

        # ==============================
        # 2. 编码文本提示（prompt）
        # ==============================
        # 将文本 prompt 转换为文本嵌入（text embeddings）
        # 同时处理 negative prompt（用于 CFG）
        # 如果已提供 prompt_embeds，则跳过编码
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,  # 是否启用分类器自由引导
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # ==============================
        # 3. 处理 IP-Adapter 图像嵌入（可选）
        # ==============================
        # IP-Adapter 是一种通过额外图像提供风格/内容引导的机制
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # ==============================
        # 4. 预处理输入图像（用于 img2img 或 pix2pix 类任务）
        # ==============================
        # 将输入图像（如 PIL、np.ndarray、torch.Tensor）统一预处理为标准张量
        image = self.image_processor.preprocess(image)

        # ==============================
        # 5. 设置推理时间步（timesteps）
        # ==============================
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # 例如 [999, 998, ..., 0]

        # ==============================
        # 6. 准备图像的潜在表示（image latents）
        # ==============================
        # 将输入图像编码为 VAE 潜在空间中的张量
        # 若启用 CFG，则会复制一份用于无条件分支
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        # 计算最终输出图像的高宽（以像素为单位）
        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor  # VAE 下采样因子（通常为 8）
        width = width * self.vae_scale_factor

        # ==============================
        # 7. 初始化随机潜在变量（latents）
        # ==============================
        # 这些是扩散过程的起点（纯噪声）
        num_channels_latents = self.vae.config.latent_channels  # 通常为 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,  # 若提供，则直接使用；否则生成新噪声
        )

        # ==============================
        # 8. 验证 UNet 输入通道数是否匹配
        # ==============================
        # UNet 输入 = 噪声潜在变量 + 图像潜在变量
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"UNet 配置期望 {self.unet.config.in_channels} 个输入通道，"
                f"但接收到 {num_channels_latents} (latents) + {num_channels_image} (image) = "
                f"{num_channels_latents + num_channels_image}。请检查 UNet 配置或输入图像。"
            )

        # ==============================
        # 9. 准备调度器额外参数（如 DDIM 的 eta）
        # ==============================
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ==============================
        # 11. 主去噪循环（Denoising Loop）
        # ==============================
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ----------------------------------------
                # a. 构建模型输入：拼接 latents（用于 CFG）
                # ----------------------------------------
                # 对于 InstructPix2Pix 等模型，CFG 需要 3 个分支：
                #   - 无条件（unconditional）
                #   - 文本条件（text-conditional）
                #   - 图像条件（image-conditional）
                if self.do_classifier_free_guidance:
                    # 复制 latents 三次：[uncond, text_cond, image_cond]
                    latent_model_input = torch.cat([latents] * 3)
                else:
                    latent_model_input = latents

                # 调度器可能对输入进行缩放（如 DDIM 的 sqrt(1 - alpha) 归一化）
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 将图像潜在变量（image_latents）沿通道维度拼接到噪声潜在变量后
                # 形成 UNet 的完整输入：[B, C_latent + C_image, H, W]
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # ----------------------------------------
                # b. 调用 UNet 预测噪声
                # ----------------------------------------
                noise_pred = self.unet(
                    scaled_latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,  # 文本嵌入
                    ip_adapter_image_embeds=ip_adapter_image_embeds,  # IP-Adapter 嵌入
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]  # 取第一个输出（噪声预测）

                # ----------------------------------------
                # c. 应用分类器自由引导（CFG）
                # ----------------------------------------
                if self.do_classifier_free_guidance:
                    # 将预测的噪声按 batch 分成三份
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)

                    # InstructPix2Pix 的特殊 CFG 公式：
                    #   输出 = 无条件 + guidance_scale * (文本条件 - 图像条件)
                    #         + image_guidance_scale * (图像条件 - 无条件)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # ----------------------------------------
                # d. 调度器更新 latents（x_t -> x_{t-1}）
                # ----------------------------------------
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # ----------------------------------------
                # e. 执行 step-end 回调（允许修改中间变量）
                # ----------------------------------------
                if callback_on_step_end is not None:
                    # 构建回调所需的局部变量字典
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]  # 从当前局部作用域获取变量

                    # 调用回调函数，可能返回修改后的变量
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # 更新可能被回调修改的变量
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # ----------------------------------------
                # f. 旧版回调 & 进度条更新
                # ----------------------------------------
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # ==============================
        # 12. 解码潜在变量为最终图像
        # ==============================
        if not output_type == "latent":
            # 使用 VAE 解码器将 latents 转换为像素空间图像
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            # 若 output_type 为 "latent"，直接返回 latents
            image = latents
        # ==============================
        # 13. 后处理图像（归一化、格式转换等）
        # ==============================
        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # ==============================
        # 14. 释放 GPU 内存（如果启用了模型卸载）
        # ==============================
        self.maybe_free_model_hooks()

        # ==============================
        # 15. 返回结果
        # ==============================
        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.ip_adapter_image_proj_model.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.ip_adapter_image_proj_model.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.ip_adapter_image_proj_model.image_projection_layers
            ):
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, True
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    (
                        single_image_embeds,
                        single_negative_image_embeds,
                        single_negative_image_embeds,
                    ) = single_image_embeds.chunk(3)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        # 保存基础组件 (保持与其他 pipeline 一致)
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        if self.image_encoder is not None:
            self.image_encoder.save_pretrained(os.path.join(save_directory, "image_encoder"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(os.path.join(save_directory, "feature_extractor"))
        self.save_config(save_directory)
