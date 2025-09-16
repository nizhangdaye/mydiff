import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipelineOutput,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

from ..models.crs_diff_uni_controlnet.uni_controlnet import CRSDifUniControlNet


class InstructPix2PixControlNetPipeline(StableDiffusionInstructPix2PixPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        controlnet=None,
        requires_safety_checker: bool = True,
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
        self.controlnet = controlnet
        self.register_modules(controlnet=controlnet)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        image,
        control_image,
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
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        device = self._execution_device

        # 1. Encode prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale > 1.0 and image_guidance_scale >= 1.0,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Preprocess image
        image = self.image_processor.preprocess(image)
        control_image = self.image_processor.preprocess(control_image)

        # 3. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare image latents
        image_latents = self.prepare_image_latents(
            image,
            1 if isinstance(prompt, str) else len(prompt),
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            guidance_scale > 1.0 and image_guidance_scale >= 1.0,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 5. Prepare latents
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

        # 6. Prepare controlnet condition
        controlnet_cond = control_image.to(device=device, dtype=prompt_embeds.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 3)
                    if guidance_scale > 1.0 and image_guidance_scale >= 1.0
                    else latents
                )
                scaled_latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                # 拼接 image_latents 仅用于 UNet
                unet_input = torch.cat(
                    [scaled_latent_model_input, image_latents], dim=1
                )

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                # UNet forward
                noise_pred = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # Guidance
                if guidance_scale > 1.0 and image_guidance_scale >= 1.0:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = (
                        noise_pred.chunk(3)
                    )
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Step
                extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 8. Decode latents
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=[True] * image.shape[0]
        )

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # 保存基础组件
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        self.controlnet.save_pretrained(os.path.join(save_directory, "controlnet"))
        # 保存 feature_extractor（如果有）
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(
                os.path.join(save_directory, "feature_extractor")
            )
        # 保存 model_index.json
        self._save_config(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 加载各组件
        vae = AutoencoderKL.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "vae")
        )
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "text_encoder")
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "tokenizer")
        )
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "unet")
        )
        scheduler = DDPMScheduler.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "scheduler")
        )
        if os.path.exists(os.path.join(pretrained_model_name_or_path, "controlnet")):
            controlnet = CRSDifUniControlNet.from_pretrained(
                os.path.join(pretrained_model_name_or_path, "controlnet")
            )
        # 加载 feature_extractor（如果有）
        feature_extractor = None
        fe_dir = os.path.join(pretrained_model_name_or_path, "feature_extractor")
        if os.path.exists(fe_dir):
            feature_extractor = CLIPFeatureExtractor.from_pretrained(fe_dir)
        # 其他参数
        kwargs.setdefault("safety_checker", None)
        kwargs.setdefault("feature_extractor", feature_extractor)
        kwargs.setdefault("image_encoder", None)
        kwargs.setdefault("requires_safety_checker", False)
        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
            **kwargs,
        )
