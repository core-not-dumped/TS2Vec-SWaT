import torch
import torch.nn as nn
import torch.nn.functional as F

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
    