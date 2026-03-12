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
import torch.nn.functional as F  # noqa: N812
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


def custom_collate_fn(
    batch: list[list[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # Pad the audio to all be the same length (the length of the longest audio in the batch)
    max_len = max([mix.shape[-1] for mix, _, _ in batch])

    mixes = []
    gts = []
    labels = []
    masks = []
    for mix, label, gt in batch:
        pad_len = max_len - mix.shape[-1]
        assert pad_len >= 0, "Error calculating batch padding"

        mix = F.pad(torch.from_numpy(mix).to(torch.float32), (0, pad_len))  # Convert and pad mix
        gt = F.pad(torch.from_numpy(gt).to(torch.float32), (0, pad_len))  # Convert and pad gt

        mask = torch.zeros_like(mix)
        mask[:, -pad_len:] = 1.0

        mixes.append(mix)
        gts.append(gt)
        labels.append(torch.from_numpy(label).to(torch.float32))  # Convert label
        masks.append(mask)

    inputs = {
        "mix": torch.stack(mixes),
        "label_vector": torch.stack(labels),
    }
    gt = torch.stack(gts)
    masks = torch.stack(masks)

    return inputs, gt, masks


def loss_fn(_output: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pred = _output["x"]
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


def train_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch: int = 0,
    # writer: SummaryWriter = None,
) -> float:
    model = model.train()

    losses = []

    i = 0
    for inputs, gt, mask in train_loader:
        # in loader return mask that is [B, C, N]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gt = gt.to(device)
        mask = mask.to(device)

        # TODO: Remove debug
        print(f"{inputs['mix'].shape=}, {gt.shape=}, {mask.shape=}")

        optimizer.zero_grad()

        output = model(inputs)
        output = output * mask  # only calculate loss on actual audio (force model output to be 0 on padded parts)

        loss = loss_fn(output, gt)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {loss.item()}")  # TODO: Remove debug
        i += 1
        if i > 5:
            # TODO: Remove debug
            raise NotImplementedError(
                "Stopping after 5 batches for testing purposes. Remove this condition to train on the full dataset."
            )

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
