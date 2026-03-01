import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from data.augmentation import *
from src.hyperparam import *

def infonce_ts2vec_vec(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B, D, L = z1.shape

    # (B,D,L) -> (B,L,D) then normalize
    a = F.normalize(z1.permute(0, 2, 1), dim=-1)  # (B,L,D)
    b = F.normalize(z2.permute(0, 2, 1), dim=-1)  # (B,L,D)

    # logits per time: (L,B,B)
    logits = torch.einsum("bld,cld->lbc", a, b) / tau
    labels = torch.arange(B, device=z1.device)

    # CE expects (N, C): flatten L times
    loss_ab = F.cross_entropy(logits.reshape(L * B, B), labels.repeat(L))
    loss_ba = F.cross_entropy(logits.transpose(1, 2).reshape(L * B, B), labels.repeat(L))

    return 0.5 * (loss_ab + loss_ba)


def hier_loss_ts2vec(outs1, outs2, tau=0.2):
    return sum(
        infonce_ts2vec_vec(o1, o2, tau)
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
def score_by_masking(model, proj_layer, pooling_layer, loader, device, masking_len):
    model.eval()
    scores = []
    labels = []

    for x, y, ts in tqdm(loader):
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

        r = F.normalize(r, dim=-1)           # (B, D)
        r_mask = F.normalize(r_mask, dim=-1) # (B, masking_len, D)

        # compute similarity between r and each masked version of r
        s = (r.unsqueeze(1) * r_mask).sum(dim=-1)  # (B, masking_len)
        s = s.mean(dim=-1)  # (B,)
        s = 1.0 - s

        scores.append(s.detach().cpu().numpy())
        labels.append(y_np)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels