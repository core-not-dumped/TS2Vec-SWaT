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