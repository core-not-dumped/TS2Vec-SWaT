import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        # LSTM output dim = hidden_size * num_dirs
        # to keep output (B,T,d_model), set hidden_size accordingly
        assert d_model % self.num_dirs == 0, "d_model must be divisible by num_dirs for bidirectional LSTM."
        hidden_size = d_model // self.num_dirs

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C=d_model)
        h, _ = self.lstm(x)        # (B, T, d_model)
        h = self.ln_f(h)           # (B, T, d_model)
        z = self.out_proj(h)       # (B, T, d_model)
        return z

