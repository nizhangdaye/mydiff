from typing import Optional

import torch
import torch.nn.functional as F


class FloodAwareAttnProcessor2_0:
    def __init__(self, depth_bias_weight: float = 1.0):
        """
        Args:
            depth_bias_weight (float): 空间偏置强度，默认为 1.0。
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FloodAwareAttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        depth_mask=None,  # ← 低洼区域掩码  1 为低洼，0 为非低洼
        depth_bias_weight=1,  # ← 可选：运行时覆盖权重
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ================================================
        # 应用空间偏置（仅当 depth_mask 提供时）
        # ================================================
        if depth_mask is not None:
            # 无论 depth_mask 是三通道还是单通道，每个通道都一样，取第一个通道
            low_mask = depth_mask[:, :1, :, :]  # shape: [B, 1, H, W]

            # 插值到当前特征图尺寸
            current_size = int(sequence_length**0.5)
            if low_mask.shape[-2:] != (current_size, current_size):
                low_mask = F.interpolate(low_mask.float(), size=(current_size, current_size), mode="nearest")
            # 展平为 [B, L]
            low_mask_flat = low_mask.view(batch_size, 1, sequence_length)  # (B, 1, L)

            # 构建偏置：query 位置为低洼时，所有 key 都加分
            spatial_bias = low_mask_flat.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1) → broadcast to (B, heads, L, L)

            # 手动计算 attention
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)
            attn_scores = attn_scores + depth_bias_weight * spatial_bias

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            hidden_states = torch.matmul(attn_weights, value)
        else:
            # 默认行为
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        # ==================================================

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
