from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class customGPTConfig:
    in_channels: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1


def build_rope_cache(T: int, head_dim: int, device: torch.device):
    """
    Returns cos, sin with shape (1, 1, T, head_dim//2)
    for broadcasting over (B, H, T, head_dim//2)
    """
    assert head_dim % 2 == 0
    half = head_dim // 2

    inv_freq = 1.0 / (10000 ** (torch.arange(half, device=device) / half))  # (half,)
    pos = torch.arange(T, device=device)  # (T,)
    angle = pos[:, None] * inv_freq[None, :]  # (T, half)

    cos = angle.cos()[None, None, :, :]  # (1,1,T,half)
    sin = angle.sin()[None, None, :, :]  # (1,1,T,half)
    return cos, sin


def apply_rope_heads(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: (B, H, T, head_dim)
    cos/sin: (1,1,T,head_dim//2)
    """
    head_dim = x.size(-1)
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # RoPE cache (built lazily per max T seen)
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        self._rope_T = 0

    def _ensure_rope_cache(self, T: int, device: torch.device):
        if self._rope_T >= T and self._rope_cos.device == device:
            return
        cos, sin = build_rope_cache(T, self.head_dim, device)
        self._rope_cos = cos
        self._rope_sin = sin
        self._rope_T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape

        h = self.ln1(x)
        qkv = self.qkv(h)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE (cache)
        self._ensure_rope_cache(T, x.device)
        cos = self._rope_cos[:, :, :T, :]
        sin = self._rope_sin[:, :, :T, :]
        q = apply_rope_heads(q, cos, sin)
        k = apply_rope_heads(k, cos, sin)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B,H,T,head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        y = self.out(y)

        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class CustomGPT(nn.Module):
    def __init__(self, cfg: customGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            GPTBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x
        for blk in self.blocks:
            h = blk(h)

        h = self.ln_f(h)          # (B, T, D)
        z = self.out_proj(h)
        return z
    

class InputProjection(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        # TS2Vec: per-timestamp fully connected layer
        self.proj = nn.Linear(input_dim, proj_dim, bias=True)

    def forward(self, x):  # x: (B, T, F)
        return self.proj(x)  # (B, T, proj_dim)


class TS2VecMaxPooling(nn.Module):
    def __init__(self, n_scales: int = 6):
        super().__init__()
        self.n_scales = n_scales

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h -> (B, T, D)
        x = h.transpose(1, 2)  # (B, D, T)
        outs = []
        for _ in range(self.n_scales):
            outs.append(x)
            if x.size(-1) < 2:  break
            x = F.max_pool1d(x, kernel_size=2, stride=2)

        return outs
    
