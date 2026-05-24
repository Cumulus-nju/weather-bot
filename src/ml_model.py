"""Multi-variate ML refinement model.

Architecture: residual dilated CNN (~78K params)
  - Input:  5-channel IDW first-guess (t2m, d2m, u10, v10, msl)
  - Output: 5-channel refined field
  - Learns the correction: output = IDW + CNN(IDW)

Training: MSE(target, output) + λ * gradient_smoothness(output)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("weather-bot.ml")

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "data" / "training"
MODEL_PATH = MODEL_DIR / "refiner.pt"

# Channel order: 5 meteorological variables + land-sea mask
CHANNELS = ["t2m", "d2m", "u10", "v10", "msl", "lsm"]
N_IN_CHANNELS = 6
N_OUT_CHANNELS = 5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class Refiner(nn.Module):
    """Residual refinement CNN — multi-channel in, multi-channel out."""

    def __init__(self, in_channels: int = N_IN_CHANNELS, out_channels: int = N_OUT_CHANNELS):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 32, 1)
        self.block1 = DilatedResBlock(32, dilation=1)
        self.block2 = DilatedResBlock(32, dilation=2)
        self.block3 = DilatedResBlock(32, dilation=4)
        self.block4 = DilatedResBlock(32, dilation=8)
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        correction = self.head(h)
        return x[:, :N_OUT_CHANNELS] + correction


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def gradient_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference of spatial gradients — encourages smoothness."""
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(2, 3))
    dy_true, dx_true = torch.gradient(y_true, dim=(2, 3))
    return (F.l1_loss(dy_pred, dy_true) + F.l1_loss(dx_pred, dx_true)) / 2


def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """MSE + gradient smoothness penalty, averaged over all channels."""
    mse = F.mse_loss(y_pred, y_true)
    grad = gradient_loss(y_pred, y_true)
    return mse + alpha * grad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(epochs: int = 30, batch_size: int = 8, lr: float = 1e-3, alpha: float = 0.3):
    """Train the multi-channel refiner CNN on pre-built .npy data."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data — shape (N, 5, H, W)
    X_train = torch.from_numpy(np.load(MODEL_DIR / "X_train.npy"))
    Y_train = torch.from_numpy(np.load(MODEL_DIR / "Y_train.npy"))
    X_val = torch.from_numpy(np.load(MODEL_DIR / "X_val.npy"))
    Y_val = torch.from_numpy(np.load(MODEL_DIR / "Y_val.npy"))

    logger.info(f"Train: X={tuple(X_train.shape)} Y={tuple(Y_train.shape)}")
    logger.info(f"Val:   X={tuple(X_val.shape)} Y={tuple(Y_val.shape)}")

    # Per-channel normalization: mean/std computed across (N, H, W) for each channel
    x_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    x_std = X_train.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std

    y_mean = Y_train.mean(dim=(0, 2, 3), keepdim=True)
    y_std = Y_train.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Y_train = (Y_train - y_mean) / y_std
    Y_val = (Y_val - y_mean) / y_std

    # Save normalization stats + channel names
    torch.save({
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "channels": CHANNELS,
    }, MODEL_DIR / "norm_stats.pt")

    model = Refiner()
    logger.info(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(X_train)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = combined_loss(y_pred, yb, alpha=alpha)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            n_val = len(X_val)
            val_preds = []
            for start in range(0, n_val, batch_size * 2):
                xb = X_val[start : start + batch_size * 2]
                val_preds.append(model(xb))
            y_val_pred = torch.cat(val_preds, dim=0)
            val_loss = combined_loss(y_val_pred, Y_val, alpha=alpha).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            improved = "*"
        else:
            improved = ""

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs}  "
            f"train: {train_loss / n_batches:.4f}  val: {val_loss:.4f}  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}{improved}"
        )

    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {MODEL_PATH}")

    # --- Per-channel baseline comparison ---
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        x_sample = X_val[:50]
        y_sample = Y_val[:50]
        refined = model(x_sample)
        logger.info("Per-channel IDW vs refined MSE (normalized):")
        for i, name in enumerate(CHANNELS[:5]):  # only the 5 met variables
            idw_mse = F.mse_loss(x_sample[:, i], y_sample[:, i]).item()
            ref_mse = F.mse_loss(refined[:, i], y_sample[:, i]).item()
            imp = (1 - ref_mse / idw_mse) * 100 if idw_mse > 0 else 0
            logger.info(f"  {name:6s}  IDW={idw_mse:.4f}  CNN={ref_mse:.4f}  improvement={imp:.1f}%")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_refiner() -> Refiner | None:
    """Load trained multi-channel model. Returns None if model not found."""
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}")
        return None

    model = Refiner()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    norm_stats = torch.load(MODEL_DIR / "norm_stats.pt", weights_only=True)
    model._norm = norm_stats
    model._channels = norm_stats.get("channels", CHANNELS)
    logger.info(f"Loaded refiner ({count_params(model):,} params, "
                f"{len(model._channels)} channels: {model._channels})")
    return model


def refine(model: Refiner, idw_fields: dict[str, np.ndarray],
           lsm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run ML refinement on a dict of IDW fields. Returns refined fields.

    idw_fields: {"t2m": (H,W), "d2m": (H,W), ...} — 5 meteorological variables
    lsm: (H,W) land-sea mask — if None, loaded from data/training/land_sea_mask.npy
    returns: dict of 5 refined fields
    """
    norm = model._norm
    channels = model._channels  # e.g. ["t2m", "d2m", "u10", "v10", "msl", "lsm"]

    if lsm is None:
        lsm = np.load(MODEL_DIR / "land_sea_mask.npy").astype(np.float32)

    # Stack 5 IDW vars + mask: (6, H, W)
    var_channels = [name for name in channels if name != "lsm"]
    x = np.stack([idw_fields[name] for name in var_channels] + [lsm], axis=0)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, 6, H, W)
    x = (x - norm["x_mean"]) / norm["x_std"]

    with torch.no_grad():
        y = model(x)

    y = y * norm["y_std"] + norm["y_mean"]
    y = y.squeeze(0).numpy()  # (5, H, W)

    return {name: y[i] for i, name in enumerate(var_channels)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
