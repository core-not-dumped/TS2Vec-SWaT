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


def timestamp_masking(x: torch.Tensor, masking_ratio: float):
    # x: (B, data_len, D)
    B, T, D = x.shape
    mask = (torch.rand(B, T, device=x.device) > masking_ratio).float()
    x = x * mask.unsqueeze(-1) # (B, data_len, D)
    return x


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
def augment_anomaly(
    x: torch.Tensor,                       # (B,T,C)
    p_sample: float = 0.5,                 # 배치에서 몇 % 샘플에 이상 넣을지
    t_frac_range: Tuple[float, float] = (0.05, 0.2),  # 이상 구간 길이 비율
    k_sensor_range: Tuple[int, int] = (1, 5),         # 이상 센서 개수 범위
    mode_probs: Dict[str, float] = None,   # 이상 타입 확률
    severity: float = 3.0,                 # 이상 강도(대략 z-score 스케일로 쓰기 좋음)
    baseline: str = "per_sample",          # "per_sample" | "global"
    return_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      x_corrupt: (B,T,C)
      y_anom:    (B,T)   (샘플별 시간 이상 라벨; 옵션)
      mask_anom: (B,T,C) (어느 (t,c)에 주입했는지; 옵션)
    """
    if mode_probs is None:
        mode_probs = {
            "spike": 0.25,
            "step":  0.20,
            "drift": 0.20,
            "noise": 0.20,
            "drop":  0.15,
        }

    device = x.device
    B, T, C = x.shape
    x0 = x.clone()

    # robust scale estimate (배치별/샘플별 스케일)
    if baseline == "per_sample":
        mu = x0.mean(dim=1, keepdim=True)                 # (B,1,C)
        sigma = x0.std(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        mu = x0.mean(dim=(0, 1), keepdim=True)            # (1,1,C)
        sigma = x0.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    # 어떤 샘플에 이상 넣을지
    do = (torch.rand(B, device=device) < p_sample)        # (B,)
    if not do.any():
        if return_mask:
            return x0, torch.zeros(B, T, device=device), torch.zeros(B, T, C, device=device)
        return x0, None, None

    # 이상 타입 샘플링 준비
    modes = list(mode_probs.keys())
    probs = torch.tensor([mode_probs[m] for m in modes], device=device)
    probs = probs / probs.sum()

    mask_anom = torch.zeros(B, T, C, device=device)
    y_anom = torch.zeros(B, T, device=device)

    for b in range(B):
        if not do[b]:
            continue

        # time window 선택
        frac = torch.empty(1, device=device).uniform_(*t_frac_range).item()
        L = max(1, int(round(T * frac)))
        t0 = int(torch.randint(0, max(1, T - L + 1), (1,), device=device).item())
        t1 = t0 + L

        # sensor subset 선택
        k0, k1 = k_sensor_range
        k = int(torch.randint(k0, min(C, k1) + 1, (1,), device=device).item())
        sensors = torch.randperm(C, device=device)[:k]

        # 모드 선택
        m = modes[int(torch.multinomial(probs, 1).item())]

        # 표준화된 단위로 이상 만들기: (L,k)
        # amp는 센서별 스케일 반영해서 severity*z 로 들어감
        amp = (severity * sigma[b, 0, sensors])            # (k,)
        amp = amp.view(1, -1)                              # (1,k)

        if m == "spike":
            # 구간 내 랜덤 몇 포인트만 큰 스파이크
            num_spikes = max(1, L // 10)
            idx = torch.randint(0, L, (num_spikes,), device=device)
            x0[b, t0:t1, sensors] += 0.0  # no-op
            x0[b, t0 + idx, sensors] += amp * torch.sign(torch.randn(num_spikes, k, device=device))

        elif m == "step":
            # 구간 전체에 일정 오프셋
            direction = torch.sign(torch.randn(1, k, device=device))
            x0[b, t0:t1, sensors] += amp * direction

        elif m == "drift":
            # 선형 드리프트
            ramp = torch.linspace(0, 1, L, device=device).view(L, 1)
            direction = torch.sign(torch.randn(1, k, device=device))
            x0[b, t0:t1, sensors] += ramp * (amp * direction)

        elif m == "noise":
            # 분산 폭증
            x0[b, t0:t1, sensors] += torch.randn(L, k, device=device) * (0.5 * amp)

        elif m == "drop":
            # flatline / dropout: 평균으로 고정
            x0[b, t0:t1, sensors] = mu[b, 0, sensors].view(1, -1).expand(L, k)

        else:
            raise ValueError(f"Unknown mode: {m}")

        # 마스크/라벨 기록
        mask_anom[b, t0:t1, sensors] = 1.0
        y_anom[b, t0:t1] = 1.0

    if return_mask:
        return x0, y_anom, mask_anom
    return x0, None, None