from typing import Dict, Dict, Optional, Tuple

import torch

def augment_view_return2(x: torch.Tensor, data_len: int):
    x = x.transpose(1, 2)  # (B, C, L)
    B, C, L = x.shape

    # 시작 위치 샘플링 (각 배치마다 다르게)
    max_start = L - data_len
    s1 = torch.randint(0, max_start + 1, (B,), device=x.device)
    s2 = torch.randint(0, max_start + 1, (B,), device=x.device)

    # 벡터화 slicing: (B, data_len) 인덱스 만들기
    idx_base = torch.arange(data_len, device=x.device).unsqueeze(0)  # (1, data_len)
    idx1 = (s1.unsqueeze(1) + idx_base)  # (B, data_len)
    idx2 = (s2.unsqueeze(1) + idx_base)  # (B, data_len)

    # gather를 위해 (B, C, data_len) 형태로 인덱스 확장
    idx1 = idx1.unsqueeze(1).expand(B, C, data_len)
    idx2 = idx2.unsqueeze(1).expand(B, C, data_len)

    x1 = x.gather(dim=2, index=idx1)
    x2 = x.gather(dim=2, index=idx2)

    x1 = x1.transpose(1, 2)  # (B, data_len, C)
    x2 = x2.transpose(1, 2)  # (B, data_len, C)

    return x1, x2


def augment_view_return1(x: torch.Tensor, data_len: int):
    x = x.transpose(1, 2)  # (B, C, L)
    B, C, L = x.shape

    # 시작 위치 샘플링 (각 배치마다 다르게)
    max_start = L - data_len
    s1 = torch.randint(0, max_start + 1, (B,), device=x.device)

    # 벡터화 slicing: (B, data_len) 인덱스 만들기
    idx_base = torch.arange(data_len, device=x.device).unsqueeze(0)  # (1, data_len)
    idx1 = (s1.unsqueeze(1) + idx_base)  # (B, data_len)

    # gather를 위해 (B, C, data_len) 형태로 인덱스 확장
    idx1 = idx1.unsqueeze(1).expand(B, C, data_len)
    x1 = x.gather(dim=2, index=idx1)
    x1 = x1.transpose(1, 2)  # (B, data_len, C)

    return x1

def augment_view_return_slide(x: torch.Tensor, data_len: int, slide_len: int):
    x = x.transpose(1, 2)  # (B, C, L)
    B, C, L = x.shape

    # 시작 위치 샘플링 (각 배치마다 다르게)
    max_start = L - data_len - slide_len + 1
    s = torch.randint(0, max_start + 1, (B,), device=x.device)

    base = torch.arange(data_len, device=x.device)  # (data_len,)
    offsets = torch.arange(slide_len, device=x.device)

    # idx_all: (B, slide_len, data_len)
    idx_all = s[:, None, None] + offsets[None, :, None] + base[None, None, :]

    # index를 (B, C, slide_len, data_len)로 확장
    idx_exp = idx_all[:, None, :, :].expand(B, C, slide_len, data_len)
    x_exp = x[:, :, None, :].expand(B, C, slide_len, L)           # (B, C, slide_len, T)

    out = x_exp.gather(dim=3, index=idx_exp)                      # (B, C, slide_len, data_len)
    out = out.permute(0, 2, 3, 1)                                 # (B, slide_len, data_len, C)
    out = out.reshape(B * slide_len, data_len, C)  # (B*slide_len, data_len, C)
    return out

def augment_view_return_masking(x: torch.Tensor, data_len: int, masking_len: int):
    x = x.transpose(1, 2)  # (B, C, L)
    B, C, L = x.shape

    # 시작 위치 샘플링 (각 배치마다 다르게)
    max_start = L - data_len
    s = torch.randint(0, max_start + 1, (B,), device=x.device)  # (B,)
    base = torch.arange(data_len, device=x.device)              # (data_len,)

    idx = s[:, None] + base[None, :]                            # (B, data_len)
    idx_exp = idx[:, None, :].expand(B, C, data_len)            # (B, C, data_len)

    crop = x.gather(dim=2, index=idx_exp)                       # (B, C, data_len)
    crop = crop.permute(0, 2, 1)                                # (B, data_len, C)

    # masking_len개 버전 만들기: (B, K, data_len, C)
    out = crop.unsqueeze(1).repeat(1, masking_len, 1, 1)                  # (B, K, data_len, C)

    # 각 k에 대해 서로 다른 위치를 0으로 마스킹 (끝쪽부터 1개씩)
    # k=0 -> -1, k=1 -> -2, ...
    k_idx = torch.arange(masking_len, device=x.device)          # (K,)
    t_idx = (data_len - 1 - k_idx)                              # (K,)

    out[:, k_idx, t_idx, :] = 0
    out = out.reshape(B * masking_len, data_len, C)  # (B * K, data_len, C)

    return out

def augment_view_return_masking_random(x: torch.Tensor, masking_len: int):
    # x: (B, L, C)
    B, L, C = x.shape
    x = x.unsqueeze(1).repeat(1, masking_len, 1, 1)
    mask = (torch.rand(B, masking_len, L, device=x.device) > 0.3).float()
    x = x * mask.unsqueeze(-1) # (B, data_len, D)
    x = x.reshape(B * masking_len, L, C)
    return x


@torch.no_grad()
def inject_spike_anomaly(
    x: torch.Tensor,
    p: float = 0.3,          # 샘플당 anomaly 넣을 확률
    amp_start: float = 1.0,
    amp_end: float = 2.0,
    dur: int = 1,            # 연속 길이(1이면 진짜 spike)
    seed: int | None = None
):
    B, T, C = x.shape
    dur = max(1, min(dur, T))

    g = torch.Generator(device=x.device)
    if seed is not None:
        g.manual_seed(seed)

    x2 = x.clone()
    mask = torch.zeros((B, T, C), device=x.device, dtype=torch.bool)

    # 어떤 배치에 anomaly 넣을지
    do = torch.rand((B,), generator=g, device=x.device) < p

    for b in range(B):
        if not do[b]:
            continue
        t0 = int(torch.randint(0, T - dur + 1, (1,), generator=g, device=x.device).item())
        c0 = int(torch.randint(0, C, (1,), generator=g, device=x.device).item())
        sign = -1.0 if bool(torch.rand((), generator=g, device=x.device) < 0.5) else 1.0
        amp = torch.rand((), generator=g, device=x.device) * (amp_end - amp_start) + amp_start

        x2[b, t0:t0+dur, c0] = x2[b, t0:t0+dur, c0] + sign * amp
        mask[b, t0:t0+dur, c0] = True

    return x2, mask, amp