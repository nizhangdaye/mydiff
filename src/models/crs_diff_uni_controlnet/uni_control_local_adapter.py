"""
局部适配器模块
包含用于遥感图像条件注入的局部适配器组件
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diffusers.models.activations import get_activation
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D

from ...utils.models_utils import zero_module


class SelfAttention(nn.Module):
    """自注意力模块，用于增强特征表示"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Query, Key, Value transformations
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        # Softmax attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, width, height = x.size()
        query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, width * height)
        value = self.value_conv(x).view(batch, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)

        return out + x  # Skip connection


class FDN(nn.Module):
    """Feature-wise Dependent Normalization 特征依赖归一化"""

    def __init__(self, norm_nc, label_nc, eps=1e-5):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=True, eps=eps)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out


# TODO: 这个地方可以有其他的改进，比如使用交叉注意力替代
class EnhancedFDN(nn.Module):
    """增强的特征依赖归一化，集成自注意力机制"""

    def __init__(self, norm_nc, label_nc, eps=1e-5):
        super(EnhancedFDN, self).__init__()
        self.fdn = FDN(norm_nc, label_nc, eps=eps)
        self.attention = SelfAttention(norm_nc)

    def forward(self, x, local_features):
        # 先应用自注意力增强特征表示
        x = self.attention(x)
        # 再应用特征依赖归一化
        out = self.fdn(x, local_features)
        return out


class LocalResBlock(nn.Module):
    """
    A Resnet block with local condition injection.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        inject_channels (`int`): the number of channels in local condition features.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"silu"`): the activation function to use.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        temb_channels: int = 512,
        inject_channels: int,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Local condition normalization layers
        self.norm1 = EnhancedFDN(in_channels, inject_channels, eps=eps)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Time embedding projection
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = EnhancedFDN(out_channels, inject_channels, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2_zero = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.nonlinearity = get_activation(non_linearity)

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        local_conditions: torch.Tensor,
    ) -> torch.Tensor:
        # First normalization with local conditions
        hidden_states = self.norm1(x, local_conditions)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Time embedding
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)
        while len(temb.shape) < len(hidden_states.shape):
            temb = temb[..., None]
        hidden_states = hidden_states + temb

        # Second normalization with local conditions
        hidden_states = self.norm2(hidden_states, local_conditions)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2_zero(hidden_states)

        output_tensor = self.skip_connection(x) + hidden_states
        return output_tensor


class Uni_ControlNetLocalAdapterFeatureExtractor(nn.Module):
    """
    统一控制局部适配器特征提取器
    将输入的条件图像转换为多尺度的局部特征，用于注入到不同的下采样块中
    """

    def __init__(self, local_channels, inject_channels):
        """
        Args:
            local_channels: 输入条件图像的通道数
            inject_channels: 各个下采样层的注入通道数 tuple
        """
        super().__init__()

        # 预处理特征提取器
        self.pre_extractor = nn.Sequential(
            nn.Conv2d(local_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 下采样 2x
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 下采样 4x
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        # 多尺度特征提取器
        self.extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        128, inject_channels[0], 3, padding=1, stride=2
                    ),  # 下采样 8x
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        inject_channels[0],
                        inject_channels[1],
                        3,
                        padding=1,
                        stride=2,
                    ),  # 下采样 16x
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        inject_channels[1],
                        inject_channels[2],
                        3,
                        padding=1,
                        stride=2,
                    ),  # 下采样 32x
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        inject_channels[2],
                        inject_channels[3],
                        3,
                        padding=1,
                        stride=2,
                    ),  # 下采样 64x
                    nn.SiLU(),
                ),
            ]
        )

        # 零初始化卷积层，用于稳定训练
        self.zero_convs = nn.ModuleList(
            [
                zero_module(
                    nn.Conv2d(inject_channels[0], inject_channels[0], 3, padding=1)
                ),
                zero_module(
                    nn.Conv2d(inject_channels[1], inject_channels[1], 3, padding=1)
                ),
                zero_module(
                    nn.Conv2d(inject_channels[2], inject_channels[2], 3, padding=1)
                ),
                zero_module(
                    nn.Conv2d(inject_channels[3], inject_channels[3], 3, padding=1)
                ),
            ]
        )

    def forward(self, local_conditions):
        """
        前向传播，提取多尺度局部特征

        Args:
            local_conditions: 输入条件图像 [B, local_channels, H, W]

        Returns:
            output_features: 多尺度特征列表，对应4个下采样层
        """
        # 预处理
        local_features = self.pre_extractor(local_conditions)
        assert len(self.extractors) == len(self.zero_convs)

        output_features = []

        # 逐层提取特征
        for idx in range(len(self.extractors)):
            local_features = self.extractors[idx](local_features)
            # 应用零初始化卷积，确保训练稳定性
            output_features.append(self.zero_convs[idx](local_features))

        return output_features


class LocalAdapterInjectionHelper:
    """
    局部适配器注入辅助类
    提供替换和前向传播的辅助方法
    """

    @staticmethod
    def replace_resnet_with_local_resblock(
        down_block,
        inject_channels,
        resnet_eps,
        resnet_groups,
        resnet_act_fn,
    ):
        """
        将下采样块中的第一个ResNet替换为LocalResBlock

        Args:
            down_block: 下采样块
            inject_channels: 注入通道数
        """
        if hasattr(down_block, "resnets") and len(down_block.resnets) > 0:
            first_resnet = down_block.resnets[0]

            # 创建 LocalResBlock 替换第一个 ResNet
            local_resblock = LocalResBlock(
                in_channels=first_resnet.in_channels,
                out_channels=first_resnet.out_channels,
                temb_channels=first_resnet.time_emb_proj.in_features,
                inject_channels=inject_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                non_linearity=resnet_act_fn,
            )

            # 执行替换
            down_block.resnets[0] = local_resblock

    @staticmethod
    def forward_down_block_with_local_resnet(
        down_block,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        local_condition: torch.Tensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        additional_residuals: Optional[torch.Tensor] = None,
    ):
        """
        处理包含LocalResBlock的下采样块的前向传播

        Args:
            down_block: 下采样块
            hidden_states: 隐藏状态
            temb: 时间嵌入
            encoder_hidden_states: 编码器隐藏状态
            attention_mask: 注意力掩码
            cross_attention_kwargs: 交叉注意力参数
            local_condition: 局部条件特征

        Returns:
            hidden_states: 处理后的隐藏状态
            output_states: 所有中间状态
        """
        # from .uni_control_local_adapter import LocalResBlock

        output_states = ()

        if isinstance(down_block, CrossAttnDownBlock2D):
            blocks = list(zip(down_block.resnets, down_block.attentions))
            for i, (resnet, attn) in enumerate(blocks):
                if isinstance(resnet, LocalResBlock):
                    # 第一个 ResNet 是 LocalResBlock，传递局部条件
                    hidden_states = resnet(hidden_states, temb, local_condition)
                else:
                    # 其他 ResNet 正常处理
                    hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # apply additional residuals to the output of the last pair of resnet and attention blocks
                if i == len(blocks) - 1 and additional_residuals is not None:
                    hidden_states = hidden_states + additional_residuals

                output_states = output_states + (hidden_states,)

        elif isinstance(down_block, DownBlock2D):
            for resnet in down_block.resnets:
                if isinstance(resnet, LocalResBlock):
                    # 第一个 ResNet 是 LocalResBlock，传递局部条件
                    hidden_states = resnet(hidden_states, temb, local_condition)
                else:
                    hidden_states = resnet(hidden_states, temb)
                output_states = output_states + (hidden_states,)
        else:
            raise ValueError(
                f"Unsupported down_block type: {type(down_block)}. Expected CrossAttnDownBlock2D or DownBlock2D."
            )

        # 处理下采样层
        if down_block.downsamplers is not None:
            for downsampler in down_block.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


# 导出主要类和函数
__all__ = [
    "SelfAttention",
    "FDN",
    "EnhancedFDN",
    "LocalResBlock",
    "UniControlLocalAdapterFeatureExtractor",
    "LocalAdapterInjectionHelper",
]
