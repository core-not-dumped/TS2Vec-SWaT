import torch
from tqdm import tqdm
import numpy as np

from data.augmentation import *
from src.hyperparam import *


def last_repr_from_model(model, pooling_layer, x):
    out = model(x)
    outs = pooling_layer(out)
    h = outs[-1][:, :, -1]
    return h


# 마지막만 마스킹하고, hierarchical 하게 L1 계산
@torch.no_grad()
def score_by_masking_last(model, proj_layer, pooling_layer, loader, device, progress: float = 1.0,):
    """
    Score = mean_s ( L1( r_s(last) , r_s_mask(last) ) )
      where r_s(last) is the last-timestep representation at scale s.

    Masking: suffix masking on the *last masking_len timesteps* of projected input x
      x_mask[:, -masking_len:, :] = mask_value
    """
    model.eval()
    proj_layer.eval()

    scores, labels = [], []
    max_steps = int(len(loader) * progress)

    for step, (x, y, ts) in enumerate(tqdm(loader, total=max_steps)):
        if step >= max_steps:   break
        x = x.to(device)  # (B, C, T)
        x = augment_view_return1(x, data_len)  # (B, data_len, C)
        y_np = y.detach().cpu().numpy()

        x = proj_layer(x, no_mask=True)  # (B, data_len, d_model)

        # suffix masking (last part only)
        x_mask = x.clone()
        x_mask[:, -1, :] = 0

        # forward + multi-scale pooling
        out = model(x)              # expected: (B, T, D) or whatever your pooling_layer expects
        out_m = model(x_mask)

        rs = pooling_layer(out)     # list of (B, D, L_s)  (based on your TS2VecMaxPooling)
        rs_m = pooling_layer(out_m) # list of (B, D, L_s)

        # hierarchical L1 on last timestep per scale
        per_scale = []
        S = min(len(rs), len(rs_m))
        for s in range(S):
            a = rs[s][:, :, -1]     # (B, D)
            b = rs_m[s][:, :, -1]   # (B, D)
            d = (a - b).abs().sum(dim=-1)  # (B,)
            per_scale.append(d)

        s_batch = torch.stack(per_scale, dim=0).mean(dim=0)  # (B,)
        scores.append(s_batch.detach().cpu().numpy())
        labels.append(y_np)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels


# masking 위치를 랜던하게 만들고 score 계산
@torch.no_grad()
def score_by_learnable_masking_random(model, proj_layer, pooling_layer, loader, device, masking_len, progress=1.0):
    model.eval(); proj_layer.eval()
    scores, labels = [], []
    max_steps = int(len(loader) * progress)

    for step, (x, y, ts) in enumerate(tqdm(loader, total=max_steps)):
        if step >= max_steps:   break
        x = x.to(device)  # (B, C, T)
        x = augment_view_return1(x, data_len) # (B, data_len, C)
        y_np = y.detach().cpu().numpy()

        # input projection
        x_origin = proj_layer(x, no_mask=True) # (B, data_len, d_model)
        x_mask = proj_layer(x.repeat_interleave(masking_len, dim=0), no_mask=False) # (B * masking_len, data_len, d_model)

        # unmasked
        r_origin = last_repr_from_model(model, pooling_layer, x_origin) # (B, d_model)
        r_mask = last_repr_from_model(model, pooling_layer, x_mask) # (B * masking_len, d_model)
        r_mask = r_mask.reshape(-1, masking_len, r_mask.size(-1))  # (B, masking_len, d_model)

        # compute similarity between r and each masked version of r
        s = (r_origin.unsqueeze(1) - r_mask).abs().sum(dim=-1)  # (B, masking_len)
        s = s.mean(dim=-1) # (B,)

        scores.append(s.detach().cpu().numpy())
        labels.append(y_np)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels


# masking 위치를 sequential하게 만들고 score 계산
@torch.no_grad()
def score_by_learnable_masking_sequential(model, proj_layer, pooling_layer, loader, device, masking_len, progress=1.0):
    model.eval(); proj_layer.eval()
    scores, labels = [], []
    max_steps = int(len(loader) * progress)

    for step, (x, y, ts) in enumerate(tqdm(loader, total=max_steps)):
        if step >= max_steps:   break
        x = x.to(device)  # (B, C, T)
        x = augment_view_return1(x, data_len) # (B, data_len, C)
        y_np = y.detach().cpu().numpy()

        out = proj_layer(x, no_mask=True) # (B, data_len, d_model)
        r_origin = last_repr_from_model(model, pooling_layer, out) # (B, d_model)

        masked_x = x.unsqueeze(1).repeat(1, masking_len, 1, 1)
        mask = torch.ones(data_len, data_len, device=x.device).bool()
        mask.fill_diagonal_(0)
        masked_x = (masked_x * mask[None, :, :, None]).reshape(-1, data_len, x.size(-1))
        masked_out = proj_layer(masked_x, no_mask=True)
        r_mask = last_repr_from_model(model, pooling_layer, masked_out) # (B * masking_len, d_model)
        r_mask = r_mask.reshape(-1, masking_len, r_mask.size(-1))  # (B, masking_len, d_model)

        # compute similarity between r and each masked version of r
        s = (r_origin.unsqueeze(1) - r_mask).abs().sum(dim=-1)  # (B, masking_len)
        s = s.mean(dim=-1) # (B,)

        scores.append(s.detach().cpu().numpy())
        labels.append(y_np)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels


# 각 batch의 masking 위치를 랜덤하게 만들고 score 계산 (모든 dataloader 사용 x)
@torch.no_grad()
def get_anomaly_score_one(x, proj_layer, model, pooling_layer, report_masking_len):
    out = proj_layer(x, no_mask=True) # (B, data_len, d_model)
    out = last_repr_from_model(model, pooling_layer, out) # (B, d_model)
    changed_out = proj_layer(x.repeat_interleave(report_masking_len, dim=0), no_mask=False) # (B * report_masking_len, data_len, d_model)
    changed_out = last_repr_from_model(model, pooling_layer, changed_out) # (B * report_masking_len, d_model)
    changed_out = changed_out.view(-1, report_masking_len, changed_out.size(-1)) # (B, report_masking_len, d_model)
    anomaly_score = (changed_out - out.unsqueeze(1)).abs().sum(dim=-1) # (B, report_masking_len)
    anomaly_score = anomaly_score.mean(dim=-1) # (B,)

    return anomaly_score


@torch.no_grad()
def get_timewise_anomaly_score_one(x, proj_layer, model, pooling_layer, report_masking_len, data_len):
    B, T, C = x.shape
    fold_x = x.unfold(1, data_len, 1)
    fold_x = fold_x.permute(0, 1, 3, 2) # (B, ~, data_len, C)
    fold_x = fold_x.contiguous().view(-1, data_len, C) # (B * ~, data_len, C)
    fold_out = proj_layer(fold_x, no_mask=True) # (B * ~, data_len, d_model)
    fold_out = last_repr_from_model(model, pooling_layer, fold_out) # (B * ~, d_model)
    fold_out = fold_out.view(B, -1, fold_out.size(-1)) # (B, ~, d_model)

    masked_fold_x = fold_x.repeat_interleave(report_masking_len, dim=0) # (B * ~ * report_masking_len, data_len, C)
    masked_fold_out = proj_layer(masked_fold_x, no_mask=False) # (B * ~ * report_masking_len, data_len, d_model)
    masked_fold_out = last_repr_from_model(model, pooling_layer, masked_fold_out) # (B * ~ * report_masking_len, d_model)
    masked_fold_out = masked_fold_out.view(B, -1, report_masking_len, masked_fold_out.size(-1)) # (B, ~, report_masking_len, d_model)

    timestamp_anomaly_score = (masked_fold_out - fold_out.unsqueeze(2)).abs().sum(dim=-1).mean(dim=-1) # (B, ~)
    torch.set_printoptions(threshold=float('inf'))
    print(timestamp_anomaly_score)
    return timestamp_anomaly_score


# sensor를 가렸을 때 anomaly score가 얼마나 변하는지 계산
@torch.no_grad()
def get_sensorwise_anomaly_score_one(x: torch.Tensor, proj_layer, model, pooling_layer, time_anomaly_sus):
    B, T, C = x.shape
    if time_anomaly_sus.sum() == 0: time_anomaly_sus = ~time_anomaly_sus
    x_rep = x.repeat_interleave(C, dim=0)
    sensor_out = proj_layer(x_rep, no_mask=True) # (B * C, data_len, d_model)
    sensor_out = last_repr_from_model(model, pooling_layer, sensor_out) # (B * C, d_model)
    sensor_masked_out = proj_layer.sensor_mask_forward(x, time_anomaly_sus, no_mask=True) # (B * C, data_len, d_model)
    sensor_masked_out = last_repr_from_model(model, pooling_layer, sensor_masked_out) # (B * C, d_model)
    sensor_score = (sensor_masked_out - sensor_out).abs().sum(dim=-1) # (B * C,)
    sensor_score = sensor_score.view(B, C) # (B, C)

    return sensor_score
