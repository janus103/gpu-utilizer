#!/usr/bin/env python3
"""
Train a simple fully-connected autoencoder on the stem Conv2d weights of a
pretrained model (e.g., resnet50 from timm).

Input features:
    For each (out_channel, in_channel) kernel of the stem Conv2d (7x7), compute
    mean and variance. Flatten all (mean, var) pairs to a vector of length
    2 * input_chan_size * output_chan_size. This is the encoder input size.

Outputs:
    - Train an autoencoder with a latent vector size specified by CLI.
    - Report MSE, RMSE, MAE for the best model (priority: lowest MSE, then
      lowest RMSE, then lowest MAE).
    - Save only the best model to AE_checkpoints/{model}_ae_{latent}.pth in the
      current working directory.
"""
import argparse
import math
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from timm.models import create_model


class StemStatsDataset(Dataset):
    """Dataset holding one sample: flattened mean/var stats over stem conv kernels."""

    def __init__(self, weight: torch.Tensor) -> None:
        # weight shape: [out_channels, in_channels, kH, kW]
        if weight.ndim != 4:
            raise ValueError(f"Expected 4D conv weight, got shape {weight.shape}")
        means = weight.mean(dim=(2, 3))  # [out, in]
        vars_ = weight.var(dim=(2, 3), unbiased=False)  # [out, in]
        flat = torch.cat([means.flatten(), vars_.flatten()], dim=0)  # [2*out*in]
        self.features = flat.unsqueeze(0).float()  # single sample

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]


class StemAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def load_stem_weight(model_name: str, device: torch.device) -> torch.Tensor:
    """Load pretrained model and return stem conv weight on CPU."""
    model = create_model(model_name, pretrained=True).to(device)
    model.eval()
    # Try common stem attributes
    if hasattr(model, "conv1"):
        stem = model.conv1
    elif hasattr(model, "stem") and hasattr(model.stem, "conv1"):
        stem = model.stem.conv1
    else:
        raise AttributeError("Could not find stem conv (conv1) in the model.")
    weight = stem.weight.detach().cpu()
    return weight


def evaluate(model: StemAutoEncoder, loader: DataLoader, device: torch.device):
    model.eval()
    mse_loss = nn.MSELoss(reduction="mean")
    total_se = 0.0
    total_abs = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            diff = recon - batch
            total_se += torch.sum(diff.pow(2)).item()
            total_abs += torch.sum(diff.abs()).item()
            total_count += diff.numel()
    mse = total_se / total_count
    rmse = math.sqrt(mse)
    mae = total_abs / total_count
    return mse, rmse, mae


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weight = load_stem_weight(args.model, device)
    in_ch = weight.shape[1]
    out_ch = weight.shape[0]
    input_dim = 2 * in_ch * out_ch

    dataset = StemStatsDataset(weight)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)

    model = StemAutoEncoder(input_dim=input_dim, latent_dim=args.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_mse = float("inf")
    best_rmse = float("inf")
    best_mae = float("inf")
    best_path = None
    save_dir = os.path.join(os.getcwd(), "AE_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = f"{args.model}_ae_{args.latent_size}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        mse, rmse, mae = evaluate(model, loader, device)

        improved = (
            (mse < best_mse)
            or (math.isclose(mse, best_mse) and rmse < best_rmse)
            or (math.isclose(mse, best_mse) and math.isclose(rmse, best_rmse) and mae < best_mae)
        )
        if improved:
            best_mse, best_rmse, best_mae = mse, rmse, mae
            torch.save(
                {
                    "model": args.model,
                    "latent_size": args.latent_size,
                    "state_dict": model.state_dict(),
                    "metrics": {"mse": best_mse, "rmse": best_rmse, "mae": best_mae},
                    "input_dim": input_dim,
                },
                ckpt_path,
            )
            best_path = ckpt_path

        print(
            f"Epoch {epoch:03d} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} "
            f"{'(best)' if improved else ''}"
        )

    print("\nBest metrics:")
    print(f"  MSE : {best_mse:.6f}")
    print(f"  RMSE: {best_rmse:.6f}")
    print(f"  MAE : {best_mae:.6f}")
    if best_path:
        print(f"Saved best model to: {best_path}")
    else:
        print("No improvement recorded; model not saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an autoencoder on stem Conv2d stats (mean/var) of a pretrained model."
    )
    parser.add_argument("--model", type=str, default="resnet50", help="Model name for timm.create_model")
    parser.add_argument("--latent-size", type=int, required=True, help="Latent vector size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for AE training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
