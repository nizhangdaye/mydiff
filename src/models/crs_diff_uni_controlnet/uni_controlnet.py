from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.controlnets.controlnet import ControlNetModel, ControlNetOutput
from diffusers.models.embeddings import TimestepEmbedding
from torch.nn import functional as F

from .uni_control_local_adapter import (
    LocalAdapterInjectionHelper,
    LocalResBlock,
    Uni_ControlNetLocalAdapterFeatureExtractor,
)


@dataclass
class SatelliteControlNetOutput(ControlNetOutput):
    """
    扩展的 ControlNet 输出，包含遥感图像特定信息
    """

    metadata_embeddings: Optional[torch.Tensor] = None


class CRSDifUniControlNet(ControlNetModel):
    """
    针对遥感图像定制的 ControlNet
    支持元数据处理和定制注意力机制
    """

    @register_to_config
    def __init__(
        self,
        # ====== 原 ControlNetModel 配置参数 ======
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (
            16,
            32,
            96,
            256,
        ),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        # ====== 自定义扩展参数 ======
        use_metadata: bool = False,
        num_metadata: int = 7,
        use_local_adapter_injection: bool = True,
        replace_zero_conv: bool = False,
        local_adapter_inject_channels: Tuple[int, int, int, int] = (192, 256, 384, 512),
        **kwargs,
    ):
        # 调用父类初始化
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            only_cross_attention=only_cross_attention,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            global_pool_conditions=global_pool_conditions,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            **kwargs,
        )

        # 立即删除不使用的 controlnet_cond_embedding 以节省显存（需与 EMA 重建保持一致）
        if hasattr(self, "controlnet_cond_embedding"):
            del self.controlnet_cond_embedding

        self.use_metadata = use_metadata
        self.use_local_adapter_injection = use_local_adapter_injection

        # 元数据嵌入
        if use_metadata:
            timestep_input_dim = self.time_embedding.linear_1.in_features
            time_embed_dim = self.time_embedding.linear_1.out_features
            self.metadata_embedding = nn.ModuleList(
                [TimestepEmbedding(timestep_input_dim, time_embed_dim) for _ in range(num_metadata)]
            )
            self.num_metadata = num_metadata
        else:
            self.metadata_embedding = None

        # 局部适配器注入
        if use_local_adapter_injection:
            self.uni_control_local_adapter_feature_extractor = Uni_ControlNetLocalAdapterFeatureExtractor(
                self.config.conditioning_channels, local_adapter_inject_channels
            )
            self._replace_with_local_resblocks(local_adapter_inject_channels)

        if replace_zero_conv:
            self._replace_with_zero_conv()

    def _replace_with_zero_conv(self):
        """将 controlnet 的零卷积进行替换"""
        for i in range(len(self.controlnet_down_blocks)):
            # print(f"替换第 {i} 个 controlnet_down_block 的零卷积")
            old_block = self.controlnet_down_blocks[i]
            self.controlnet_down_blocks[i] = nn.Conv2d(old_block.in_channels, old_block.out_channels, kernel_size=1)
        # print("替换最后一个 controlnet_mid_block 的零卷积")
        old_mid = self.controlnet_mid_block
        self.controlnet_mid_block = nn.Conv2d(old_mid.in_channels, old_mid.out_channels, kernel_size=1)

    def _replace_with_local_resblocks(self, inject_channels):
        """替换每个 down_block 中第一个 ResBlock 为 LocalResBlock"""
        for block_idx, down_block in enumerate(self.down_blocks):
            LocalAdapterInjectionHelper.replace_resnet_with_local_resblock(
                down_block=down_block,
                inject_channels=inject_channels[block_idx],
                resnet_eps=self.config.norm_eps,
                resnet_groups=self.config.norm_num_groups,
                resnet_act_fn=self.config.act_fn,
            )

    def forward(
        self,
        sample: torch.Tensor,  # x noise
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,  # text embeds
        # original_image_embeds: torch.Tensor,  # image embeds
        controlnet_cond: torch.Tensor,  # local adapter cond
        conditioning_scale: float = 1.0,
        metadata: Optional[torch.Tensor] = None,
        cond_metadata: Optional[torch.Tensor] = None,
        satclip_loc=None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[SatelliteControlNetOutput, Tuple]:
        """
        前向传播，支持元数据输入

        Args:
            metadata: 遥感图像元数据字典，包含：
                - 'lat': 纬度信息
                - 'lon': 经度信息
                - 'time': 时间信息
                - 'sensor_params': 传感器参数
        """

        # # check channel order
        # channel_order = self.config.controlnet_conditioning_channel_order
        # if channel_order == "rgb":
        #     pass
        # elif channel_order == "bgr":
        #     controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        # else:
        #     raise ValueError(
        #         f"unknown `controlnet_conditioning_channel_order`: {channel_order}"
        #     )

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # CUSTOM, 处理元数据
        if self.metadata_embedding is not None:
            assert metadata is not None
            assert len(metadata.shape) == 2 and metadata.shape[1] == self.num_metadata, (
                f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"
            )

            md_bsz = metadata.shape[0]
            metadata = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
            metadata = metadata.to(dtype=self.dtype)
            for i, md_embed in enumerate(self.metadata_embedding):
                md_emb = md_embed(metadata[:, i, :])  # (N, D)
                emb = emb + md_emb  # (N, D)

            if cond_metadata is not None:
                assert cond_metadata.shape[1] == self.num_metadata, (
                    f"Invalid cond metadata shape: {cond_metadata.shape}. Need batch x num_metadata x (optional) num_cond"
                )
                if len(cond_metadata.shape) == 3:
                    md_bsz = cond_metadata.shape[0]
                    num_cond = cond_metadata.shape[2]
                    cond_metadata = self.time_proj(cond_metadata.view(-1)).view(
                        md_bsz, self.num_metadata, num_cond, -1
                    )  # (N, num_md, D)
                    cond_metadata = cond_metadata.to(dtype=self.dtype)
                    for i, md_embed in enumerate(self.metadata_embedding):
                        md_emb = md_embed(cond_metadata[:, i, :, :]).sum(dim=1)  # sum across time
                        emb = emb + md_emb
                else:
                    assert len(cond_metadata.shape) == 2
                    md_bsz = cond_metadata.shape[0]
                    cond_metadata = self.time_proj(cond_metadata.view(-1)).view(
                        md_bsz, self.num_metadata, -1
                    )  # (N, num_md, D)
                    cond_metadata = cond_metadata.to(dtype=self.dtype)
                    for i, md_embed in enumerate(self.metadata_embedding):
                        md_emb = md_embed(cond_metadata[:, i, :])  # (N, D)
                        emb = emb + md_emb  # (N, D)

        aug_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError("text_embeds required for text_time addition_embed_type")
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError("time_ids required for text_time addition_embed_type")
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        # sample = torch.cat([sample, original_image_embeds], dim=1)
        sample = self.conv_in(sample)

        # CUSTOM, 提取局部特征
        local_features = None
        if self.use_local_adapter_injection:
            local_features = self.uni_control_local_adapter_feature_extractor(controlnet_cond)

        # 3. down - 修改此部分以传递局部条件
        down_block_res_samples = (sample,)
        for block_idx, downsample_block in enumerate(self.down_blocks):
            current_local_feature = local_features[block_idx] if local_features else None

            # 对于每个下采样块，需要特殊处理第一个 ResNet
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                if hasattr(downsample_block, "resnets") and isinstance(downsample_block.resnets[0], LocalResBlock):
                    # 使用辅助类处理包含 LocalResBlock 的下采样块
                    sample, res_samples = LocalAdapterInjectionHelper.forward_down_block_with_local_resnet(
                        down_block=downsample_block,
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        local_condition=current_local_feature,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
            else:
                if hasattr(downsample_block, "resnets") and isinstance(downsample_block.resnets[0], LocalResBlock):
                    sample, res_samples = LocalAdapterInjectionHelper.forward_down_block_with_local_resnet(
                        downsample_block,
                        sample,
                        emb,
                        None,
                        None,
                        None,
                        current_local_feature,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return SatelliteControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    @classmethod
    def from_unet(
        cls,
        unet,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_in_channels: int = 3,  # 条件图像通道总数
        replace_zero_conv: bool = False,
    ):
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_in_channels,
            replace_zero_conv=replace_zero_conv,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            result = controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
            print("Parameters initialized directly:")
            for key in result.missing_keys:
                print(f"  - {key}")

            print("Unexpected keys:")
            for key in result.unexpected_keys:
                print(f"  - {key}")

            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        # # 修改 conv_in 层以适应 instruct_pix2pix 的输入
        # in_channels = 8
        # out_channels = controlnet.conv_in.out_channels
        # controlnet.register_to_config(in_channels=in_channels)

        # with torch.no_grad():
        #     new_conv_in = nn.Conv2d(
        #         in_channels,
        #         out_channels,
        #         controlnet.conv_in.kernel_size,
        #         controlnet.conv_in.stride,
        #         controlnet.conv_in.padding,
        #     )
        #     new_conv_in.weight.zero_()
        #     new_conv_in.weight[:, :4, :, :].copy_(controlnet.conv_in.weight)
        #     controlnet.conv_in = new_conv_in

        return controlnet


# 测试
if __name__ == "__main__":
    # 初始化 uni_controlnet
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        "/mnt/data/zwh/model/DiffusionSat/checkpoint-100000",
        subfolder="unet",
    )
    uni_controlnet = CRSDifUniControlNet.from_unet(
        unet,
        conditioning_in_channels=6,
        replace_zero_conv=True,
    )
    # 打印模型结构
    print(uni_controlnet)
    # 进行一次前向传播测试
    down_block_res_samples, mid_block_res_sample = uni_controlnet(
        sample=torch.randn(1, 4, 64, 64),
        timestep=10,
        encoder_hidden_states=torch.randn(1, 77, 1024),
        controlnet_cond=torch.randn(1, 6, 512, 512),
        return_dict=False,
    )
    print(down_block_res_samples[0].shape)
    print(mid_block_res_sample.shape)
