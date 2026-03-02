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
    

# time stamp, sensor masking 둘다 적용
class InputProjection_W_TimeSensor_Masking(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64, mask_prob: float = 0.3):
        super().__init__()
        self.mask_prob = mask_prob
        self.W = nn.Parameter(torch.empty(input_dim, proj_dim)) # (C, D)
        self.Wm = nn.Parameter(torch.empty(input_dim, proj_dim)) # (C, D)
        self.ln = nn.LayerNorm(proj_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.Wm)
        self.W.data *= 0.5
        self.Wm.data *= 0.5

    def forward(self, x: torch.Tensor, return_mask: bool = False, no_mask: bool = False):
        B, T, C = x.shape
        x = x.unsqueeze(-1) * self.W.unsqueeze(0).unsqueeze(0)  # (B, T, C, D) 

        if no_mask:
            mask = torch.zeros(B, T, 1, 1, device=x.device)
        else:
            time_stamp_mask = (torch.rand(B, T, 1, 1, device=x.device) < self.mask_prob)
            sensor_mask = (torch.rand(B, 1, C, 1, device=x.device) < self.mask_prob)
            mask = time_stamp_mask | sensor_mask
            mask = mask.to(x.dtype)  # (B, T, C, 1)
        mask_inv = 1.0 - mask

        xu = (x * mask_inv).sum(dim=2) # (B, T, D)
        xm = (self.Wm.unsqueeze(0).unsqueeze(0) * mask).sum(dim=2) # (B, T, D)

        out = xu + xm  # (B, T, D)
        out = self.ln(out)

        if return_mask:
            return out, mask
        return out
    
    def masking_forward(self, x: torch.Tensor, time: int, sensor_idx: int):
        B, T, C = x.shape
        x = x.unsqueeze(-1) * self.W.unsqueeze(0).unsqueeze(0)  # (B, T, C, D)
        mask = torch.zeros(B, T, C, 1, device=x.device)
        mask[:, time, sensor_idx] = 1.0
        mask_inv = 1.0 - mask

        xu = (x * mask_inv).sum(dim=2) # (B, T, D)
        xm = (self.Wm.unsqueeze(0).unsqueeze(0) * mask).sum(dim=2) # (B, T, D)

        out = xu + xm  # (B, T, D)
        out = self.ln(out)

        return out