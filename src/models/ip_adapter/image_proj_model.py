import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)  # 可学习的基础查询 token

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class SemanticMapTokenizer(nn.Module):
    """
    将离散语义/结构图 (类别索引) 转为 token 序列 [B, N, D]，供 Resampler 消化。

    设计要点（最小改动、与现有 IP-Adapter 对齐）：
    - 输入: 语义图张量，形状 [B, C, H, W]，其中 C 可为 1 或 3；取第 1 个通道作为类别索引源。
    - 类别索引应为整数 [0, num_classes-1]；若输入为浮点（如由阈值分类产生），在前向中 round->clamp->long。
    - nn.Embedding 将每个像素映射到维度 D（必须与 UNet/IP-Adapter 配置的 `hidden_size` 一致）。
    - 使用平均池化进行 patchify，将像素级特征聚合到 patch 级别，降低序列长度；
      patch_size 默认为 16，若 H/W 不是 patch_size 的整数倍，将自动向下取整。
    - 添加无参 2D 正余弦位置编码（可开关），提升几何感知；保持零开销参数。

    输出: [B, N, D]，其中 N = (H//patch_size) * (W//patch_size)。
    注：外部若需传给 MultiIPAdapterImageProjection，可先在 batch 维做 `unsqueeze(1)` 得到 [B, 1, N, D]，再包成 list。
    """

    def __init__(
        self,
        num_classes: int = 256,
        embed_dim: int = 1024,
        patch_size: int = 16,
        use_sincos_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.use_sincos_pos_emb = bool(use_sincos_pos_emb)

        self.class_embed = nn.Embedding(self.num_classes, self.embed_dim)
        # 使用平均池化进行 patch 聚合（在通道维是 embedding 维）
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size, ceil_mode=False)

        # ✅ LayerNorm 归一化，稳定输出
        self.norm = nn.LayerNorm(self.embed_dim)

        # ✅ 缓存位置编码（在 forward 中懒加载）
        self.register_buffer("_pos_cache", None, persistent=False)

    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype):
        """生成 2D 正余弦位置编码，形状 [1, H, W, D]。
        参考 ViT 位置编码风格，按 (H, W) 网格生成，再拼接为 dim 维。
        """

        def get_1d_pos_embed(length: int, d_half: int):
            pos = torch.arange(length, device=device, dtype=dtype)
            omega = torch.arange(d_half, device=device, dtype=dtype)
            omega = 1.0 / (10000 ** (omega / d_half))
            out = torch.einsum("n,d->nd", pos, omega)
            return torch.cat([out.sin(), out.cos()], dim=1)  # [length, 2*d_half]

        d_h = dim // 2
        d_w = dim - d_h
        # 保证偶数分配到 sin/cos
        d_h_even = (d_h // 2) * 2
        d_w_even = (d_w // 2) * 2
        emb_h = get_1d_pos_embed(h, d_h_even // 2)  # [H, d_h_even]
        emb_w = get_1d_pos_embed(w, d_w_even // 2)  # [W, d_w_even]

        emb_h = emb_h.unsqueeze(1).expand(h, w, d_h_even)
        emb_w = emb_w.unsqueeze(0).expand(h, w, d_w_even)

        # 若 dim 为奇数或不能整除，截断/补零到 dim
        pos = torch.cat([emb_h, emb_w], dim=-1)  # [H, W, d_h_even + d_w_even]
        if pos.shape[-1] < dim:
            pad = torch.zeros(h, w, dim - pos.shape[-1], device=device, dtype=dtype)
            pos = torch.cat([pos, pad], dim=-1)
        elif pos.shape[-1] > dim:
            pos = pos[:, :, :dim]

        return pos.unsqueeze(0)  # [1, H, W, D]

    def forward(self, semantic_map: torch.Tensor) -> torch.Tensor:
        """
        semantic_map: [B, C, H, W], C 可以为 1 或 3；仅使用第 1 个通道作为类别索引。
        返回: tokens [B, N, D]，N = (H//p)*(W//p)。
        """
        if semantic_map.dim() != 4:
            raise ValueError(f"semantic_map must be 4D [B,C,H,W], got {semantic_map.shape}")

        b, c, h, w = semantic_map.shape
        # 取第一个通道，四舍五入到最近的类索引
        idx = semantic_map[:, 0].round().clamp(0, self.num_classes - 1).long()  # [B, H, W]

        # 像素级嵌入
        pix_emb = self.class_embed(idx)  # [B, H, W, D]
        pix_emb = pix_emb.permute(0, 3, 1, 2)  # [B, D, H, W]

        # patch 平均池化聚合
        pooled = self.pool(pix_emb)  # [B, D, H', W']
        _, d, hp, wp = pooled.shape

        if self.use_sincos_pos_emb:
            # ✅ 缓存并复用位置编码
            if (
                self._pos_cache is None
                or self._pos_cache.shape[1:] != (d, hp, wp)
                or self._pos_cache.device != pooled.device
                or self._pos_cache.dtype != pooled.dtype
            ):
                pos = self._build_2d_sincos_pos_embed(hp, wp, d, pooled.device, pooled.dtype)
                pos = pos.permute(0, 3, 1, 2)  # [1, D, H', W']
                self._pos_cache = pos
            pooled = pooled + self._pos_cache

        tokens = pooled.permute(0, 2, 3, 1).reshape(b, hp * wp, d)  # [B, N, D]
        tokens = self.norm(tokens)  # ✅ LayerNorm

        return tokens
