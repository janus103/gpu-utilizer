import torch
from util import feat_to_bcn, batch_channel_mean, batch_channel_var, batch_standardize, batch_corr

@torch.no_grad()
def corr_from_stem_feat(F: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    F: (B,C,H,W) -> R: (B,C,C)
    """
    X = feat_to_bcn(F)
    mu = batch_channel_mean(X)
    var = batch_channel_var(X, mu)
    Z = batch_standardize(X, mu, var, eps=eps)
    return batch_corr(Z)

@torch.no_grad()
def update_running_V(running_V: torch.Tensor, R0: torch.Tensor, R_list: list[torch.Tensor],
                     count: int, B: int):
    """
    running_V: (C,C) running mean of sensitivity
    R0: (B,C,C) correlation of original
    R_list: list of Rk, each is (B,C,C) for an augmentation family/view
    """
    # mean over augment views of squared diff
    diffs = []
    for Rk in R_list:
        diffs.append((Rk - R0) ** 2)
    V_batch = torch.stack(diffs, dim=0).mean(dim=0)  # (B,C,C)
    V_b = V_batch.mean(dim=0)  # (C,C)
    new_count = count + B
    running_V = (running_V * count + V_b * B) / max(new_count, 1)
    return running_V, new_count

@torch.no_grad()
def make_prior_P_from_V(V: torch.Tensor, mode: str = "soft", topk: int = 256, thr: float = None):
    """
    V: (C,C) sensitivity matrix
    Returns P: (C,C) in {0,1} (hard) or [0,1] (soft)
    """
    C = V.size(0)
    V2 = V.clone()
    V2.fill_diagonal_(0.0)  # usually ignore diagonal for "style correlation" selection

    if mode == "soft":
        # normalize to [0,1]
        P = V2 / (V2.max() + 1e-12)
        return P

    if mode == "thr":
        assert thr is not None
        return (V2 > thr).float()

    if mode == "topk":
        # select topk edges in upper triangle and symmetrize
        iu, ju = torch.triu_indices(C, C, offset=1, device=V.device)
        vals = V2[iu, ju]
        k = min(topk, vals.numel())
        _, idx = torch.topk(vals, k=k, largest=True)
        P = torch.zeros_like(V2)
        P[iu[idx], ju[idx]] = 1.0
        P[ju[idx], iu[idx]] = 1.0
        return P

    raise ValueError(f"Unknown mode: {mode}")