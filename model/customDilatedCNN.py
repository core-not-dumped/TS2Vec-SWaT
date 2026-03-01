import torch
import torch.nn as nn
import torch.nn.functional as F

class _DilatedConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float, causal: bool):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Conv1d expects (B, C, T)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self):
        # Conv1d: Kaiming init (for GELU)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.conv.weight.data *= 0.1          # 🔥 residual 안정화 (중요)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # LayerNorm
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def _pad(self, x_ct: torch.Tensor) -> torch.Tensor:
        # x_ct: (B, C, T)
        # padding needed to preserve length
        pad = (self.kernel_size - 1) * self.dilation
        if self.causal:
            # pad only on the left: (pad_left, pad_right)
            return F.pad(x_ct, (pad, 0))
        else:
            # symmetric-ish padding to keep length
            left = pad // 2
            right = pad - left
            return F.pad(x_ct, (left, right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x_ct = x.transpose(1, 2)          # (B, C, T)
        x_ct = self._pad(x_ct)
        y_ct = self.conv(x_ct)            # (B, C, T)
        y = y_ct.transpose(1, 2)          # (B, T, C)

        y = F.gelu(y)
        y = self.dropout(y)
        y = y + residual                  # residual
        y = self.ln(y)                    # LN over C
        return y


class CustomDilatedCNN(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
        dilation_base: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _DilatedConvBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dilation=(dilation_base ** i),
                dropout=dropout,
                causal=causal,
            )
            for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C=d_model)
        h = x
        for blk in self.blocks:
            h = blk(h)               # (B, T, d_model)
        h = self.ln_f(h)
        z = self.out_proj(h)         # (B, T, d_model)
        return z