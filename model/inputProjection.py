import torch.nn as nn

class InputProjection(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        # TS2Vec: per-timestamp fully connected layer
        self.proj = nn.Linear(input_dim, proj_dim, bias=True)

    def forward(self, x):  # x: (B, T, F)
        return self.proj(x)  # (B, T, proj_dim)