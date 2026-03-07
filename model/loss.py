import torch
import torch.nn.functional as F
import numpy as np


def topk_percentage(score_n, score_a):
    scores = np.concatenate([score_n, score_a])
    labels = np.concatenate([
        np.zeros(len(score_n)),  # normal = 0
        np.ones(len(score_a))    # attack = 1
    ])

    # score 기준 내림차순 정렬 후 top-k
    idx = np.argsort(scores)[::-1][:len(score_a)]

    attack_count = np.sum(labels[idx] == 1)
    normal_count = np.sum(labels[idx] == 0)

    attack_ratio = attack_count / len(score_a) * 100
    normal_ratio = normal_count / len(score_a) * 100

    return attack_ratio, normal_ratio


def ts2vec_dual_loss_vec(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    B, D, L = z1.shape
    device = z1.device

    a = z1.permute(0, 2, 1).contiguous()  # (B,L,D)  = r_{i,t}
    b = z2.permute(0, 2, 1).contiguous()  # (B,L,D)  = r'_{i,t}

    # -------------------------
    # Instance-wise (batch 끼리 비교)
    # logits_ab_inst: (L,B,B) = a[i,t]·b[j,t]
    # logits_aa_inst: (L,B,B) = a[i,t]·a[j,t]
    # -------------------------
    logits_ab_inst = torch.einsum("itd,jtd->tij", a, b)
    logits_aa_inst = torch.einsum("itd,jtd->tij", a, a)

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
    # logits_ab_temp: (B,L,L) = a[i,t]·b[i,t']
    # logits_aa_temp: (B,L,L) = a[i,t]·a[i,t']
    # -------------------------
    logits_ab_temp = torch.einsum("itd,isd->its", a, b)  # (B,L,L)
    logits_aa_temp = torch.einsum("itd,isd->its", a, a)  # (B,L,L)

    log_num_temp = logits_ab_temp.diagonal(dim1=1, dim2=2)  # (B,L)

    logsum_ab_t = torch.logsumexp(logits_ab_temp, dim=2)    # (B,L)

    eyeL = torch.eye(L, device=device, dtype=torch.bool)[None, :, :]
    logits_aa_t_masked = logits_aa_temp.masked_fill(eyeL, float("-inf"))
    logsum_aa_t = torch.logsumexp(logits_aa_t_masked, dim=2) # (B,L)

    log_den_temp = torch.logaddexp(logsum_ab_t, logsum_aa_t)

    loss_temp = -(log_num_temp - log_den_temp).mean()

    return loss_temp + loss_inst


def hier_loss_ts2vec_dual(outs1, outs2):
    return sum(
        ts2vec_dual_loss_vec(o1, o2)
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

