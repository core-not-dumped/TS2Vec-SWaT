import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from data.augmentation import *
from src.hyperparam import *

def ts2vec_dual_loss_vec(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B, D, L = z1.shape
    device = z1.device

    a = z1.permute(0, 2, 1).contiguous()  # (B,L,D)  = r_{i,t}
    b = z2.permute(0, 2, 1).contiguous()  # (B,L,D)  = r'_{i,t}

    # -------------------------
    # Instance-wise (batch 끼리 비교)
    # logits_ab_inst: (L,B,B) = a[i,t]·b[j,t] / tau
    # logits_aa_inst: (L,B,B) = a[i,t]·a[j,t] / tau
    # -------------------------
    logits_ab_inst = torch.einsum("itd,jtd->tij", a, b) / tau
    logits_aa_inst = torch.einsum("itd,jtd->tij", a, a) / tau

    # numerator log: diag over (i==j)
    log_num_inst = logits_ab_inst.diagonal(dim1=1, dim2=2)  # (L,B)

    # denominator log: log( sum_j exp(ab) + sum_{j!=i} exp(aa) )
    # compute logsumexp for ab over j
    logsum_ab = torch.logsumexp(logits_ab_inst, dim=2)  # (L,B)
    # mask out j==i for aa term then logsumexp
    eyeB = torch.eye(B, device=device, dtype=torch.bool)[None, :, :]  # (1,B,B)
    logits_aa_masked = logits_aa_inst.masked_fill(eyeB, float("-inf"))
    logsum_aa = torch.logsumexp(logits_aa_masked, dim=2)  # (L,B); could be -inf if B==1

    # combine two sums in log-space: log( exp(logsum_ab) + exp(logsum_aa) )
    # use logaddexp for stability; handle -inf safely
    log_den_inst = torch.logaddexp(logsum_ab, logsum_aa)

    loss_inst = -(log_num_inst - log_den_inst).mean()

    # -------------------------
    # Temporal (time step 끼리 비교)
    # logits_ab_temp: (B,L,L) = a[i,t]·b[i,t'] / tau
    # logits_aa_temp: (B,L,L) = a[i,t]·a[i,t'] / tau
    # -------------------------
    logits_ab_temp = torch.einsum("itd,isd->its", a, b) / tau  # (B,L,L)
    logits_aa_temp = torch.einsum("itd,isd->its", a, a) / tau  # (B,L,L)

    log_num_temp = logits_ab_temp.diagonal(dim1=1, dim2=2)  # (B,L)

    logsum_ab_t = torch.logsumexp(logits_ab_temp, dim=2)    # (B,L)

    eyeL = torch.eye(L, device=device, dtype=torch.bool)[None, :, :]
    logits_aa_t_masked = logits_aa_temp.masked_fill(eyeL, float("-inf"))
    logsum_aa_t = torch.logsumexp(logits_aa_t_masked, dim=2) # (B,L)

    log_den_temp = torch.logaddexp(logsum_ab_t, logsum_aa_t)

    loss_temp = -(log_num_temp - log_den_temp).mean()

    return loss_temp + loss_inst


def hier_loss_ts2vec_dual(outs1, outs2, tau=0.2):
    return sum(
        ts2vec_dual_loss_vec(o1, o2, tau)
        for o1, o2 in zip(outs1, outs2)
    )


def to_vec(outs):
    """
    outs: list of (B,D,Lk) or tensor (B,D,L)
    return: (B, D_total) vector
    """
    if isinstance(outs, (list, tuple)):
        vecs = [o.mean(dim=-1) for o in outs]     # (B,D)
        z = torch.cat(vecs, dim=1)                # (B, sumD)
    else:
        z = outs.mean(dim=-1)                     # (B,D)
    return z


def compute_centroid(model, pooling_layer, loader, device):
    model.eval()
    zs = []
    with torch.no_grad():
        for x, y, ts in tqdm(loader):
            x = x.to(device)
            x = augment_view_return1(x, data_len)
            out = model(x)
            outs = pooling_layer(out)
            z = to_vec(outs)                  # (B, D_total)
            z = F.normalize(z, dim=-1)        # cosine 기반으로 안정
            zs.append(z.cpu())
    Z = torch.cat(zs, dim=0)                  # (N, D_total)
    mu = Z.mean(dim=0, keepdim=True)          # (1, D_total)
    mu = F.normalize(mu, dim=-1)
    return mu


def score_by_centroid(model, pooling_layer, loader, mu, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y, ts in tqdm(loader):
            x = x.to(device)
            x = augment_view_return1(x, data_len)
            out = model(x)
            outs = pooling_layer(out)
            z = to_vec(outs)
            z = F.normalize(z, dim=-1)
            sim = (z.cpu() @ mu.t()).squeeze(1)     # (B,)
            s = (1.0 - sim)                         # 높을수록 이상
            scores.append(s)
            labels.append(y.cpu().float())          # 0 normal, 1 attack
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def last_repr_from_model(model, pooling_layer, x):
    out = model(x)
    outs = pooling_layer(out)
    h = outs[-1][:, :, -1]
    return h


@torch.no_grad()
def score_by_masking(model, proj_layer, pooling_layer, loader, device, masking_len, progress=1.0):
    model.eval()
    proj_layer.eval()
    scores = []
    labels = []
    max_steps = int(len(loader) * progress)

    for step, (x, y, ts) in enumerate(tqdm(loader, total=max_steps)):
        if step >= max_steps:   break
        x = x.to(device)  # (B, C, T)
        x_mask = augment_view_return1(x, data_len) # (B, data_len, C)
        y_np = y.detach().cpu().numpy()

        # input projection
        x = proj_layer(x) # (B, data_len, d_model)
        x_mask = augment_view_return_masking(x, data_len, masking_len) # (B * masking_len, data_len, d_model)

        # unmasked
        r = last_repr_from_model(model, pooling_layer, x) # (B, D)
        r_mask = last_repr_from_model(model, pooling_layer, x_mask) # (B * masking_len, D)
        r_mask = r_mask.reshape(-1, masking_len, r_mask.size(-1))  # (B, masking_len, D)

        # compute similarity between r and each masked version of r
        s = (r.unsqueeze(1) - r_mask).abs().sum(dim=-1)  # (B, masking_len)
        s = s.mean(dim=-1) # (B,)

        scores.append(s.detach().cpu().numpy())
        labels.append(y_np)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels