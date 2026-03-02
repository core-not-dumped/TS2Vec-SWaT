import torch
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
    

class InputProjection_W_TimestampMasking(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64, mask_prob: float = 0.3):
        super().__init__()
        self.mask_prob = mask_prob
        self.proj_u = nn.Linear(input_dim, proj_dim, bias=False)  # unmask용
        self.proj_m = nn.Linear(1, proj_dim, bias=False)  # mask용
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.proj_u, self.proj_m):
            nn.init.xavier_uniform_(layer.weight)
            layer.weight.data *= 0.5

    def forward(self, x: torch.Tensor, return_mask: bool = False, no_mask: bool = False):
        B, T, F = x.shape
        if no_mask:
            mask = torch.zeros(B, T, 1, device=x.device)
        else:
            mask = (torch.rand(B, T, 1, device=x.device) < self.mask_prob).to(x.dtype)
        mask_inv = 1.0 - mask

        xu = self.proj_u(x * mask_inv)  # (B, T, D)
        xm = self.proj_m(mask)  # (B, T, D)

        out = xu + xm  # (B, T, D)

        if return_mask:
            return out, mask
        return out