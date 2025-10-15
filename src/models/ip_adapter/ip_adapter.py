"""
ref to https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
"""

import os
import os.path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import PIL
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.models.embeddings import MultiIPAdapterImageProjection
from torch import nn
from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers

# from anysd.src.adapter import IPAdapterAttnProAnySD
# from anysd.src.pipe import AnySDInstructPix2PixPipeline
# from anysd.src.unet import UNet2DConditionAnySD
from .attention_processor import AttnProcessor2_0
from .image_proj_model import Resampler, SemanticMapTokenizer


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


# TODO: 使用继承 UNet2DConditionModel
class IPAdapterUNet2DConditionModel(UNet2DConditionModel):
    def __init__(
        self,
        # ========== 原生 UNet2DConditionModel 参数 ==========
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        # ========== IP-Adapter 新增参数 ==========
        num_ip_adapter_image_tokens: Optional[int] = 16,  # 必须提供，
        hidden_size: Optional[int] = 1664,  # image encoder hidden size, 必须提供
        # ========== 语义 tokenizer 参数（可选） ==========
        num_semantic_classes: int = 3,
        tokenizer_patch_size: int = 16,
        use_sincos_pos_emb: bool = True,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )

        self.register_to_config(num_ip_adapter_image_tokens=num_ip_adapter_image_tokens)
        self.register_to_config(hidden_size=hidden_size)
        self.register_to_config(num_semantic_classes=num_semantic_classes)
        self.register_to_config(tokenizer_patch_size=tokenizer_patch_size)
        self.register_to_config(use_sincos_pos_emb=use_sincos_pos_emb)

        self.ip_adapter_image_proj_model = self._initialize_image_proj_model(hidden_size, num_ip_adapter_image_tokens)

        # 语义图 tokenizer：将离散语义/结构图转为 [B,N,D] tokens，其中 D 对齐为 hidden_size（与 Resampler.embedding_dim 一致）
        self.semantic_tokenizer = SemanticMapTokenizer(
            num_classes=num_semantic_classes,
            embed_dim=hidden_size,
            patch_size=tokenizer_patch_size,
            use_sincos_pos_emb=use_sincos_pos_emb,
        )

        # 若用户希望模型一创建就具有 IP-Adapter 能力，则在此初始化 attention processors
        # 注意: 这会把当前 to_k/ to_v 权重复制到 to_k_ip / to_v_ip 中，适用于直接 `from_pretrained` 场景。
        # 在从现有基础 UNet 迁移 (`from_unet`) 时我们会先复制外部权重，再调用初始化，以获得更合理的拷贝。
        self._initialize_adapter_modules(num_tokens=num_ip_adapter_image_tokens)

    def _initialize_image_proj_model(self, hidden_size, num_tokens):
        """初始化图像特征投影模块并把必要字段写入 config。

        之前只注册了 encoder_hid_dim_type, 未把真正需要在二次 from_pretrained 时
        被 diffusers 基类读取的 encoder_hid_dim / num_ip_adapter_image_tokens 写入 config，
        导致 save_pretrained -> from_pretrained 后缺失字段而报错:
        ValueError: `encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to ip_image_proj.

        参数:
            image_encoder: CLIPVisionModelWithProjection (需含 hidden_size)。
            num_tokens: 生成的图像提示 token 数 (Resampler 查询数)。
        """

        image_projection_layer = Resampler(
            dim=self.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=num_tokens,
            embedding_dim=hidden_size,
            output_dim=self.config.cross_attention_dim,
            ff_mult=4,
        )
        image_projection_layers = [image_projection_layer]

        # TODO: 支持多种图像映射器，目前只有一种
        return MultiIPAdapterImageProjection(image_projection_layers)

    def _initialize_adapter_modules(self, num_tokens):
        """
        训练时调用，初始化 adapter 模块
        推理时不调用
        """
        attn_procs = {}
        unet_sd = self.state_dict()
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.0.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.0.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                # attn_procs[name] = IPAdapterAttnProAnySD(
                #     hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens
                # )
                attn_procs[name] = IPAdapterAttnProcessor2_0(  # IPAdapterXFormersAttnProcessor
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens
                )
                attn_procs[name].load_state_dict(weights)
        self.set_attn_processor(
            attn_procs
        )  # set in unet , https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/unet.py#L871

        adapter_modules = torch.nn.ModuleList(self.attn_processors.values())
        return adapter_modules

    def _verify_cross_attn_text_branch_weights_equal(self, src_unet: UNet2DConditionModel) -> bool:
        """验证 cross-attn 文本分支(attn2)的权重是否与原始 UNet 完全一致。

        对比以下参数（且排除新增的 *_ip 参数）：
        - attn2.to_q.(weight|bias)
        - attn2.to_k.(weight|bias)
        - attn2.to_v.(weight|bias)
        - attn2.to_out.0.(weight|bias)

        返回 True 表示完全一致；否则打印每个不一致参数的最大绝对误差并返回 False。
        """
        src_sd = src_unet.state_dict()
        dst_sd = self.state_dict()

        def is_cross_text_param(k: str) -> bool:
            if "_ip" in k:
                return False
            return (".attn2." in k) and (
                k.endswith("to_q.weight")
                or k.endswith("to_q.bias")
                or k.endswith("to_k.weight")
                or k.endswith("to_k.bias")
                or k.endswith("to_v.weight")
                or k.endswith("to_v.bias")
                or k.endswith("to_out.0.weight")
                or k.endswith("to_out.0.bias")
            )

        mismatches: List[Tuple[str, float]] = []
        checked_count = 0
        for k, v_dst in dst_sd.items():
            if not is_cross_text_param(k):
                continue
            if k not in src_sd:
                # should not happen for non *_ip keys, but be defensive
                mismatches.append((k, float("inf")))
                continue
            v_src = src_sd[k]
            # move to cpu and align dtype to float32 for stable comparison
            a = v_dst.detach().to(torch.float32).cpu()
            b = v_src.detach().to(torch.float32).cpu()
            checked_count += 1
            if a.shape != b.shape:
                mismatches.append((k, float("inf")))
                continue
            if not torch.allclose(a, b, atol=1e-6, rtol=0):
                diff = (a - b).abs().max().item()
                mismatches.append((k, diff))

        if len(mismatches) == 0:
            print(
                f"[IPAdapterUNet.verify] Cross-attn(text) weights identical to source UNet. Checked params: {checked_count}."
            )
            return True
        else:
            print(
                f"[IPAdapterUNet.verify] Cross-attn(text) weights differ from source UNet on {len(mismatches)}/{checked_count} params:"
            )
            for k, d in mismatches:
                print(f"  - {k}: max_abs_diff={d:.6e}")
            return False

    def _verify_ip_kv_copied_from_unet_kv(self, src_unet: UNet2DConditionModel) -> bool:
        """验证 IP-Adapter 的 K/V 投影(to_k_ip/to_v_ip)是否与原始 UNet 的 to_k/to_v 完全一致。

        遍历所有 cross-attn(attn2) 的处理器参数：
        - *.attn2.processor.to_k_ip.0.weight vs *.attn2.to_k.weight
        - *.attn2.processor.to_v_ip.0.weight vs *.attn2.to_v.weight

        返回 True 表示全部一致；否则打印每个不一致参数的最大绝对误差并返回 False。
        """
        src_sd = src_unet.state_dict()
        dst_sd = self.state_dict()

        def paired_base_key(ip_key: str) -> Optional[str]:
            if ip_key.endswith(".attn2.processor.to_k_ip.0.weight"):
                return ip_key.replace(".processor.to_k_ip.0.weight", ".to_k.weight").replace(
                    ".attn2.processor", ".attn2"
                )
            if ip_key.endswith(".attn2.processor.to_v_ip.0.weight"):
                return ip_key.replace(".processor.to_v_ip.0.weight", ".to_v.weight").replace(
                    ".attn2.processor", ".attn2"
                )
            return None

        mismatches: List[Tuple[str, float]] = []
        checked_count = 0
        for k_ip, v_ip in dst_sd.items():
            base_k = paired_base_key(k_ip)
            if base_k is None:
                continue
            if base_k not in src_sd:
                mismatches.append((k_ip + " (missing base)", float("inf")))
                continue
            a = v_ip.detach().to(torch.float32).cpu()
            b = src_sd[base_k].detach().to(torch.float32).cpu()
            checked_count += 1
            if a.shape != b.shape:
                mismatches.append((k_ip, float("inf")))
                continue
            if not torch.allclose(a, b, atol=1e-6, rtol=0):
                diff = (a - b).abs().max().item()
                mismatches.append((k_ip, diff))

        if len(mismatches) == 0:
            print(
                f"[IPAdapterUNet.verify] IP to_k_ip/to_v_ip identical to source UNet's to_k/to_v. Checked params: {checked_count}."
            )
            return True
        else:
            print(
                f"[IPAdapterUNet.verify] IP to_k_ip/to_v_ip differ from source UNet on {len(mismatches)}/{checked_count} params:"
            )
            for k, d in mismatches:
                print(f"  - {k}: max_abs_diff={d:.6e}")
            return False

    def init_instructpix2pix_unet(self):
        """
        训练时调用
        """

        # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
        # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
        # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
        # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
        # initialized to zero.
        in_channels = 8
        out_channels = self.conv_in.out_channels
        self.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,
                out_channels,
                self.conv_in.kernel_size,
                self.conv_in.stride,
                self.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.conv_in.weight)
            self.conv_in = new_conv_in

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # =========== 自定义参数 ===========
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        ip_adapter_semantic_map: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # =========== 自定义： IP-Adapter image projection ===========
        if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
            encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

        # 期望输入形状 [B, C, H, W]（C=1/3），内部会进行取第1通道、round->clamp->long
        tokens = self.semantic_tokenizer(ip_adapter_semantic_map)  # [B, N, D]
        ip_adapter_image_embeds = [tokens.unsqueeze(1)]  # 适配 MultiIPAdapterImageProjection 的 [B,1,N,D]

        ip_adapter_image_embeds = self.ip_adapter_image_proj_model(ip_adapter_image_embeds)
        encoder_hidden_states = (encoder_hidden_states, ip_adapter_image_embeds)
        # ===========================================================================

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        num_ip_adapter_image_tokens: int = 16,
        hidden_size: int = 1664,  # TODO: 1664 是不是太大了？
        copy_weights: bool = True,
        # ------ 语义 tokenizer 参数 ------
        num_semantic_classes: int = 3,
        tokenizer_patch_size: int = 16,
        use_sincos_pos_emb: bool = True,
    ) -> "IPAdapterUNet2DConditionModel":
        """参考 `CRSDifUniControlNet.from_unet` 的构建方式, 直接从一个已经加载好的基础 UNet
        (通常是 *原生* Stable Diffusion 的 `UNet2DConditionModel`) 构造 IPAdapter 版本, 避免
        旧 checkpoint + 新模块 通过 `from_pretrained` 触发的 meta tensor 问题。

        参数:
            unet: 已经 `from_pretrained` 过的基础 UNet 实例。
            num_ip_adapter_image_tokens: 图像提示 token 数量。
            hidden_size: 图像编码器输出 hidden size (CLIP vision hidden_size)。
            copy_weights: 是否把可兼容的权重复制到新模型 (建议 True)。
            keep_attn_processors: 是否沿用原 unet 的 attention processors (若已替换 LoRA 等)。
            init_instructpix2pix: 是否同时调用 `init_instructpix2pix_unet()` 扩展输入通道为 8。

        返回:
            一个新的 `IPAdapterUNet2DConditionModel` 实例。
        """
        cfg = unet.config
        # 构造新的 IPAdapterUNet (此时所有参数是正常初始化, 不会是 meta)
        ip_unet = cls(
            sample_size=getattr(cfg, "sample_size", None),
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            center_input_sample=cfg.center_input_sample,
            flip_sin_to_cos=cfg.flip_sin_to_cos,
            freq_shift=cfg.freq_shift,
            down_block_types=cfg.down_block_types,
            mid_block_type=cfg.mid_block_type,
            up_block_types=cfg.up_block_types,
            only_cross_attention=cfg.only_cross_attention,
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            downsample_padding=cfg.downsample_padding,
            mid_block_scale_factor=cfg.mid_block_scale_factor,
            dropout=cfg.dropout,
            act_fn=cfg.act_fn,
            norm_num_groups=cfg.norm_num_groups,
            norm_eps=cfg.norm_eps,
            cross_attention_dim=cfg.cross_attention_dim,
            transformer_layers_per_block=getattr(cfg, "transformer_layers_per_block", 1),
            reverse_transformer_layers_per_block=getattr(cfg, "reverse_transformer_layers_per_block", None),
            encoder_hid_dim=getattr(cfg, "encoder_hid_dim", None),
            encoder_hid_dim_type=getattr(cfg, "encoder_hid_dim_type", None),
            attention_head_dim=getattr(cfg, "attention_head_dim", 8),
            num_attention_heads=getattr(cfg, "num_attention_heads", None),
            dual_cross_attention=getattr(cfg, "dual_cross_attention", False),
            use_linear_projection=getattr(cfg, "use_linear_projection", False),
            class_embed_type=getattr(cfg, "class_embed_type", None),
            addition_embed_type=getattr(cfg, "addition_embed_type", None),
            addition_time_embed_dim=getattr(cfg, "addition_time_embed_dim", None),
            num_class_embeds=getattr(cfg, "num_class_embeds", None),
            upcast_attention=getattr(cfg, "upcast_attention", False),
            resnet_time_scale_shift=getattr(cfg, "resnet_time_scale_shift", "default"),
            resnet_skip_time_act=getattr(cfg, "resnet_skip_time_act", False),
            resnet_out_scale_factor=getattr(cfg, "resnet_out_scale_factor", 1.0),
            time_embedding_type=getattr(cfg, "time_embedding_type", "positional"),
            time_embedding_dim=getattr(cfg, "time_embedding_dim", None),
            time_embedding_act_fn=getattr(cfg, "time_embedding_act_fn", None),
            timestep_post_act=getattr(cfg, "timestep_post_act", None),
            time_cond_proj_dim=getattr(cfg, "time_cond_proj_dim", None),
            conv_in_kernel=getattr(cfg, "conv_in_kernel", 3),
            conv_out_kernel=getattr(cfg, "conv_out_kernel", 3),
            projection_class_embeddings_input_dim=getattr(cfg, "projection_class_embeddings_input_dim", None),
            attention_type=getattr(cfg, "attention_type", "default"),
            class_embeddings_concat=getattr(cfg, "class_embeddings_concat", False),
            mid_block_only_cross_attention=getattr(cfg, "mid_block_only_cross_attention", None),
            cross_attention_norm=getattr(cfg, "cross_attention_norm", None),
            addition_embed_type_num_heads=getattr(cfg, "addition_embed_type_num_heads", 64),
            # ----- 新增参数 -----
            num_ip_adapter_image_tokens=num_ip_adapter_image_tokens,
            hidden_size=hidden_size,
            num_semantic_classes=num_semantic_classes,
            tokenizer_patch_size=tokenizer_patch_size,
            use_sincos_pos_emb=use_sincos_pos_emb,
        )

        # 先保存原 unet 权重
        unet_sd = unet.state_dict()
        if copy_weights:
            # 1) 先把基础 UNet 兼容权重复制到新模型
            result = ip_unet.load_state_dict(unet_sd, strict=False)
            print("[IPAdapterUNet.from_unet] Copied base UNet parameters. Missing new keys (expected for IP modules):")
            for key in result.missing_keys:
                print(f"  - {key}")
            if result.unexpected_keys:
                print("[IPAdapterUNet.from_unet] Unexpected keys (in source but not used):")
                for key in result.unexpected_keys:
                    print(f"  - {key}")

            # 2) 强制重新初始化 IPAdapter attn processors，使 to_k_ip / to_v_ip 从已复制好的 to_k / to_v 再拷贝
            ip_unet._initialize_adapter_modules(num_tokens=num_ip_adapter_image_tokens)
            print("[IPAdapterUNet.from_unet] Synchronized to_k/to_v -> to_k_ip/to_v_ip for all cross-attn processors.")

            # 3) 验证 cross-attn 文本分支权重（不含 *_ip）是否与原始 UNet 完全一致
            ip_unet._verify_cross_attn_text_branch_weights_equal(unet)

            # 4) 验证 IP-Adapter 的 K/V 投影是否从原始 UNet 的 to_k/to_v 正确拷贝
            ip_unet._verify_ip_kv_copied_from_unet_kv(unet)

        return ip_unet


# 测试
if __name__ == "__main__":
    from diffusers import UNet2DConditionModel

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="image_encoder"
    )
    with torch.no_grad():
        img = torch.randn(1, 3, 224, 224)  # 或实际图像预处理后
        vision_out = image_encoder(img, output_hidden_states=True)  # return the hidden states of all layers
        ip_adapter_image_embeds = vision_out.last_hidden_state  # (B, 257, hidden)
    ip_adapter_image_embeds = ip_adapter_image_embeds.unsqueeze(1)  # (B, 1, 257, hidden)
    print(f"ip_adapter_image_embeds shape: {ip_adapter_image_embeds.shape}")

    print(f"======================================训练测试=====================================================")
    unet = UNet2DConditionModel.from_pretrained("/mnt/data/zwh/model/DiffusionSat/checkpoint-100000", subfolder="unet")
    ipadapter_unet = IPAdapterUNet2DConditionModel.from_unet(unet)
    ipadapter_unet.init_instructpix2pix_unet()

    # 打印模型结构
    print(ipadapter_unet)
    # 进行一次前向传播测试
    noise_pred = ipadapter_unet(
        sample=torch.randn(1, 8, 64, 64),
        timestep=10,
        encoder_hidden_states=torch.randn(1, 77, 1024),
        ip_adapter_image_embeds=[ip_adapter_image_embeds],
        return_dict=False,
    )[0]

    print(noise_pred.shape)
    ipadapter_unet.save_pretrained("./ipadapter_unet")

    # print(f"====================================推理测试=====================================================")
    # ipadapter_unet = IPAdapterUNet2DConditionModel.from_pretrained("./ipadapter_unet")
    # noise_pred = ipadapter_unet(
    #     sample=torch.randn(1, 8, 64, 64),
    #     timestep=10,
    #     encoder_hidden_states=torch.randn(1, 77, 1024),
    #     ip_adapter_image_embeds=[ip_adapter_image_embeds],
    #     return_dict=False,
    # )[0]
    # print(ipadapter_unet)
    # print(noise_pred.shape)
