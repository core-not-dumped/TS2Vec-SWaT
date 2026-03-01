import torch.nn as nn

class InputProjection(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        # TS2Vec: per-timestamp fully connected layer
        self.proj = nn.Linear(input_dim, proj_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init (TS2Vec에 가장 안정적)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data *= 0.5
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):  # x: (B, T, F)
        return self.proj(x)  # (B, T, proj_dim)