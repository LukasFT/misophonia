"""
The main training script for training on synthetic data
"""

import sys
from pathlib import Path

# Add parent directory of misophonia-dataset to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds  # noqa: F401

# from torch.utils.tensorboard import SummaryWriter
# from torchmetrics.functional import (
#     scale_invariant_signal_distortion_ratio as si_sdr,
# )
from torchmetrics.functional import (
    scale_invariant_signal_noise_ratio as si_snr,
)

# from torchmetrics.functional import (
#     signal_distortion_ratio as sdr,
# )
from torchmetrics.functional import (
    signal_noise_ratio as snr,
)

# from .model import MisophoniaANCNet


def loss_fn(_output: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pred = _output["x"]
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


def train_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch: int = 0,
) -> float:
    model = model.train()

    losses = []

    for inputs, gt in train_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(inputs)

        loss = loss_fn(output, gt)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch + 1}: Loss = {np.mean(losses)}")
    return np.mean(losses)


def train_model(
    model: nn.Module,
    train_loader: wds.WebLoader,
    *,
    n_epochs: int,
    device: torch.device,
) -> None:

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0005, weight_decay=0)

    for epoch in range(n_epochs):
        losses = train_epoch(model, device, optimizer, train_loader, epoch)
        print(f"Epoch {epoch + 1}: Loss = {losses}")
