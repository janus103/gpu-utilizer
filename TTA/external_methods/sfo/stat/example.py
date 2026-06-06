import torch
from util import feat_to_bcn, batch_channel_mean, batch_channel_var, batch_standardize, batch_corr
from util_mask import corr_from_stem_feat, update_running_V
from util import update_running_mean, update_running_var, update_running_R

@torch.no_grad()
def calib_update(
    F0: torch.Tensor,              # stem(x): (B,C,H,W)
    F_aug_list: list[torch.Tensor],# [stem(tau_k(x)) ...], each (B,C,H,W)  (offline only)
    state: dict,
    eps: float = 1e-5,
):
    """
    state holds running stats:
      - mu_s: (1,C,1)
      - var_s: (1,C,1)
      - R_s: (C,C)
      - V_s: (C,C)  -> later convert to P
      - n: int (num samples accumulated)
    """
    X0 = feat_to_bcn(F0)                 # (B,C,N)
    mu0 = batch_channel_mean(X0)         # (B,C,1)
    var0 = batch_channel_var(X0, mu0)    # (B,C,1)
    Z0 = batch_standardize(X0, mu0, var0, eps=eps)
    R0 = batch_corr(Z0)                  # (B,C,C)

    B = X0.size(0)
    # update mu_s, var_s, R_s
    state["mu_s"], state["n"] = update_running_mean(state["mu_s"], mu0, state["n"], B)
    state["var_s"], _         = update_running_var(state["var_s"], var0, state["n"]-B, B)  # careful: reuse previous count
    state["R_s"], _           = update_running_R(state["R_s"], R0, state["n"]-B, B)

    # update V_s (sensitivity)
    R_aug_list = [corr_from_stem_feat(Fk, eps=eps) for Fk in F_aug_list]  # list of (B,C,C)
    state["V_s"], _ = update_running_V(state["V_s"], R0, R_aug_list, state["n"]-B, B)
    return state
