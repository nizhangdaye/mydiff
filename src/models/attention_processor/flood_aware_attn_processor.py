from typing import Optional

import torch
import torch.nn.functional as F


class FloodAwareAttnProcessor2_0:
    def __init__(self):
        """
        Args:
            depth_bias_weight (float): 空间偏置强度，默认为 1.0。
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FloodAwareAttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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
        # 改为通过 SDPA 的 attn_mask 传入加性偏置，避免手写 matmul/softmax
        # ================================================
        if depth_mask is not None:
            # 无论 depth_mask 是三通道还是单通道，每个通道都一样，取第一个通道
            low_mask = depth_mask[:, :1, :, :]  # shape: [B, 1, H, W]

            # 插值到当前特征图尺寸（与 query 的 token 数一致）
            q_len = query.shape[-2]
            # 非 4D 情况下假设方形网格（常见于 UNet flatten 后）；否则用户应提供匹配尺寸
            current_size = int(q_len**0.5)
            if current_size * current_size != q_len or low_mask.shape[-2:] != (current_size, current_size):
                low_mask = F.interpolate(low_mask.float(), size=(current_size, current_size), mode="nearest")

            # 展平为 [B, Lq]，构造 (B, 1, Lq, 1) 的加性偏置，沿 heads 与 keys 维度广播
            low_mask_flat = low_mask.view(batch_size, -1).to(dtype=query.dtype)

            spatial_bias = low_mask_flat.view(batch_size, 1, q_len, 1)
            spatial_bias = depth_bias_weight * spatial_bias  # (B, 1, Lq, 1)
            k_len = key.shape[-2]

            # 合并已有 attention_mask：支持 bool 掩码或加性掩码
            if attention_mask is None:
                # 显式扩展到 (B, H, Lq, Sk) 并确保连续
                attn_mask_total = spatial_bias.expand(batch_size, attn.heads, q_len, k_len).contiguous()
            else:
                if attention_mask.dtype == torch.bool:
                    # 将布尔掩码转换为加性掩码：被屏蔽位置为 -inf
                    float_mask = torch.zeros_like(attention_mask, dtype=query.dtype)
                    float_mask = float_mask.masked_fill(~attention_mask, float("-inf"))
                else:
                    float_mask = attention_mask.to(dtype=query.dtype)
                # 将 spatial_bias 显式扩展到 float_mask 的形状，避免广播视图的非连续最后一维
                sb_expanded = spatial_bias.expand_as(float_mask).contiguous()
                attn_mask_total = (float_mask + sb_expanded).contiguous()

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
            )
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
