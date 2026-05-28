"""Multi-variate ML refinement model — seasonal edition.

Architecture: residual dilated CNN (~86K params)
  - Input:  7-channel (6 IDW fields + land-sea mask)
  - Output: 6-channel refined field (t2m, d2m, u10, v10, msl, tp)
  - Learns correction: output = IDW + CNN(IDW)

Training: ocean-masked MSE + gradient smoothness, per-season models.
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

CHANNELS = ["t2m", "d2m", "u10", "v10", "msl", "tp", "lsm"]
N_IN_CHANNELS = 7
N_OUT_CHANNELS = 6
SEASONS = ["spring", "summer", "autumn", "winter"]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.dropout(h)
        h = self.conv2(h)
        return F.relu(x + h)


class Refiner(nn.Module):
    def __init__(self, in_channels: int = N_IN_CHANNELS, out_channels: int = N_OUT_CHANNELS,
                 dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 32, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.block1 = DilatedResBlock(32, dilation=1, dropout=dropout)
        self.block2 = DilatedResBlock(32, dilation=2, dropout=dropout)
        self.block3 = DilatedResBlock(32, dilation=4, dropout=dropout)
        self.block4 = DilatedResBlock(32, dilation=8, dropout=dropout)
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        h = self.dropout(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        correction = self.head(h)
        return x[:, :N_OUT_CHANNELS] + correction


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Ocean-masked loss
# ---------------------------------------------------------------------------


def masked_mse(y_pred: torch.Tensor, y_true: torch.Tensor,
               lsm: torch.Tensor) -> torch.Tensor:
    """MSE computed only on land grid points (lsm > 0.5)."""
    land = (lsm > 0.5).float()
    n_land = land.sum() + 1e-8
    diff = (y_pred - y_true) * land
    return (diff ** 2).sum() / n_land


def masked_gradient_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         lsm: torch.Tensor) -> torch.Tensor:
    """Gradient smoothness loss on land grid points only."""
    land = (lsm > 0.5).float()

    def _masked_grad(t):
        dy, dx = torch.gradient(t, dim=(2, 3))
        return dy * land, dx * land

    dy_pred, dx_pred = _masked_grad(y_pred)
    dy_true, dx_true = _masked_grad(y_true)

    n_land = land.sum() + 1e-8
    return (F.l1_loss(dy_pred, dy_true, reduction="sum") +
            F.l1_loss(dx_pred, dx_true, reduction="sum")) / (2 * n_land)


def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                  lsm: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Ocean-masked MSE + gradient smoothness, averaged over all channels."""
    mse = masked_mse(y_pred, y_true, lsm)
    grad = masked_gradient_loss(y_pred, y_true, lsm)
    return mse + alpha * grad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_season(season: str, epochs: int = 50, batch_size: int = 8,
                 lr: float = 1e-3, alpha: float = 0.3,
                 patience: int = 8, dropout: float = 0.15):
    """Train a single-season refiner CNN."""
    data_dir = MODEL_DIR / season
    if not (data_dir / "X_train.npy").exists():
        logger.error(f"Training data not found for {season} at {data_dir}")
        return

    model_path = MODEL_DIR / f"refiner_{season}.pt"
    stats_path = MODEL_DIR / f"norm_stats_{season}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = torch.from_numpy(np.load(data_dir / "X_train.npy"))
    Y_train = torch.from_numpy(np.load(data_dir / "Y_train.npy"))
    X_val = torch.from_numpy(np.load(data_dir / "X_val.npy"))
    Y_val = torch.from_numpy(np.load(data_dir / "Y_val.npy"))

    logger.info(f"[{season}] Train: X={tuple(X_train.shape)} Y={tuple(Y_train.shape)}")
    logger.info(f"[{season}] Val:   X={tuple(X_val.shape)} Y={tuple(Y_val.shape)}")

    # Per-channel normalization (computed on land only for target, but full grid for input)
    x_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    x_std = X_train.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std

    y_mean = Y_train.mean(dim=(0, 2, 3), keepdim=True)
    y_std = Y_train.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Y_train = (Y_train - y_mean) / y_std
    Y_val = (Y_val - y_mean) / y_std

    torch.save({
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "channels": CHANNELS, "season": season,
    }, stats_path)

    model = Refiner(dropout=dropout)
    logger.info(f"[{season}] Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(X_train)
    n_val = len(X_val)
    best_val_loss = float("inf")
    best_epoch = 0
    stall = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            optimizer.zero_grad()
            y_pred = model(xb)
            lsm = xb[:, -1:]  # land-sea mask is always the last channel
            # Broadcast lsm to all output channels for loss
            lsm_bc = lsm.expand(-1, N_OUT_CHANNELS, -1, -1)
            loss = combined_loss(y_pred, yb, lsm_bc, alpha=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_preds = []
            for start in range(0, n_val, batch_size * 2):
                xb = X_val[start: start + batch_size * 2]
                val_preds.append(model(xb))
            y_val_pred = torch.cat(val_preds, dim=0)
            lsm_val = X_val[:, -1:].expand(-1, N_OUT_CHANNELS, -1, -1)
            val_loss = combined_loss(y_val_pred, Y_val, lsm_val, alpha=alpha).item()

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            stall = 0
            improved = "*"
        else:
            stall += 1

        logger.info(
            f"[{season}] Epoch {epoch + 1:3d}/{epochs}  "
            f"train: {train_loss / n_batches:.4f}  val: {val_loss:.4f}  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}{improved}"
        )

        if stall >= patience:
            logger.info(f"[{season}] Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"[{season}] Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # --- Per-channel evaluation ---
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        x_sample = X_val[:min(50, n_val)]
        y_sample = Y_val[:min(50, n_val)]
        refined = model(x_sample)
        logger.info(f"[{season}] Per-channel IDW vs refined MSE (normalized, land-only):")
        lsm_sample = x_sample[:, -1]
        for i, name in enumerate(CHANNELS[:6]):
            land = lsm_sample > 0.5
            idw_mse = F.mse_loss(
                x_sample[:, i][land], y_sample[:, i][land]
            ).item()
            ref_mse = F.mse_loss(
                refined[:, i][land], y_sample[:, i][land]
            ).item()
            imp = (1 - ref_mse / idw_mse) * 100 if idw_mse > 0 else 0
            logger.info(f"  {name:6s}  IDW={idw_mse:.4f}  CNN={ref_mse:.4f}  improvement={imp:.1f}%")


def train_all_seasons(epochs: int = 50, **kwargs):
    """Train models for all 4 seasons sequentially."""
    for season in SEASONS:
        logger.info(f"{'='*50}")
        logger.info(f"Training {season} model")
        logger.info(f"{'='*50}")
        train_season(season, epochs=epochs, **kwargs)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_refiner(season: str | None = None) -> Refiner | None:
    """Load trained seasonal model. Uses current season if not specified."""
    if season is None:
        from config import get_season
        season = get_season()

    model_path = MODEL_DIR / f"refiner_{season}.pt"
    stats_path = MODEL_DIR / f"norm_stats_{season}.pt"

    if not model_path.exists():
        logger.warning(f"Model not found for {season} at {model_path}")
        return None

    model = Refiner()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    norm_stats = torch.load(stats_path, weights_only=True)
    model._norm = norm_stats
    model._channels = norm_stats.get("channels", CHANNELS)
    model._season = season
    logger.info(f"Loaded refiner [{season}] ({count_params(model):,} params)")
    return model


def refine(model: Refiner, idw_fields: dict[str, np.ndarray],
           lsm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run ML refinement on a dict of IDW fields. Returns refined fields.

    idw_fields: {"t2m": (H,W), "d2m": (H,W), ...} — 6 meteorological variables
    lsm: (H,W) land-sea mask — if None, loaded from data/training/land_sea_mask.npy
    returns: dict of 6 refined fields (ocean masking is handled by plotter)
    """
    norm = model._norm
    channels = model._channels

    if lsm is None:
        lsm = np.load(MODEL_DIR / "land_sea_mask.npy").astype(np.float32)

    var_channels = [name for name in channels if name != "lsm"]
    x = np.stack([idw_fields[name] for name in var_channels] + [lsm], axis=0)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    x = (x - norm["x_mean"]) / norm["x_std"]

    with torch.no_grad():
        y = model(x)

    y = y * norm["y_std"] + norm["y_mean"]
    y = y.squeeze(0).numpy()

    return {name: y[i] for i, name in enumerate(var_channels)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_all_seasons()
