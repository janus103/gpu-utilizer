import torch

# 공통 변환: Feature map to BCN vector
def feat_to_bcn(F: torch.Tensor) -> torch.Tensor:
    # F: (B,C,H,W) -> X: (B,C,N)
    B, C, H, W = F.shape
    return F.view(B, C, H * W)

# Mean 계산 1: 지금 들어온 배치에서 평균을 계산 (즉시 통계)
@torch.no_grad()
def batch_channel_mean(X: torch.Tensor) -> torch.Tensor:
    # X: (B,C,N) -> mu: (B,C,1)
    return X.mean(dim=-1, keepdim=True)

# Mean 계산 2 그 평균들을 계속 누적해서 최종 mu_s 만들기 위한 업데이트 (누적 통계)
@torch.no_grad()
def update_running_mean(running_mu: torch.Tensor, mu_batch: torch.Tensor, count: int, B: int):
    """
    running_mu: (1,C,1) dataset running mean
    mu_batch: (B,C,1) batch per-sample mean
    count: number of samples accumulated so far
    B: current batch size
    """
    mu_b = mu_batch.mean(dim=0, keepdim=True)  # (1,C,1)
    new_count = count + B
    running_mu = (running_mu * count + mu_b * B) / max(new_count, 1)
    return running_mu, new_count

# Variance 계산 1
@torch.no_grad()
def batch_channel_var(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    # X: (B,C,N), mu: (B,C,1) -> var: (B,C,1)
    return ((X - mu) ** 2).mean(dim=-1, keepdim=True)

# Variance 계산 2
@torch.no_grad()
def update_running_var(running_var: torch.Tensor, var_batch: torch.Tensor, count: int, B: int):
    # running_var: (1,C,1), var_batch: (B,C,1)
    var_b = var_batch.mean(dim=0, keepdim=True)  # (1,C,1)
    new_count = count + B
    running_var = (running_var * count + var_b * B) / max(new_count, 1)
    return running_var, new_count

# Standardization
@torch.no_grad()
def batch_standardize(X: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # X,mu,var: (B,C,N)/(B,C,1)/(B,C,1)
    return (X - mu) / torch.sqrt(var + eps)

# Correlation 계산 1
@torch.no_grad()
def batch_corr(Z: torch.Tensor) -> torch.Tensor:
    # Z: (B,C,N) -> R: (B,C,C)
    B, C, N = Z.shape
    return torch.bmm(Z, Z.transpose(1, 2)) / float(N)

# Correlation 계산 2
@torch.no_grad()
def update_running_R(running_R: torch.Tensor, R_batch: torch.Tensor, count: int, B: int):
    """
    running_R: (C,C) running mean of correlation
    R_batch: (B,C,C) per-sample correlation
    """
    R_b = R_batch.mean(dim=0)  # (C,C)
    new_count = count + B
    running_R = (running_R * count + R_b * B) / max(new_count, 1)
    return running_R, new_count