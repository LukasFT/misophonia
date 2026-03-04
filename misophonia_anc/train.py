"""
The main training script for training on synthetic data
"""

import sys
from pathlib import Path

# Add parent directory of misophonia-dataset to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import argparse
import logging
import multiprocessing
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds  # noqa: F401

# from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
from torchmetrics.functional import (
    signal_distortion_ratio as sdr,
)
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr, signal_noise_ratio as snr
from tqdm import tqdm  # pylint: disable=unused-import

from misophonia_dataset.misophonia_dataset import PremadeMisophoniaDataset

from .model import MisophoniaANCNet


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
    device: torch.device,
    train_loader: wds.WebLoader,
    batch_size: int,
    n_epochs: int,
    num_workers: int,
    log_dir: Path,
) -> None:

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0005, weight_decay=0)
    # writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(n_epochs):
        losses = train_epoch(model, device, optimizer, train_loader, epoch)
        print(f"Epoch {epoch + 1}: Loss = {losses}")


if __name__ == "__main__":
    # Load data
    # data = PremadeMisophoniaDataset(name="demo-v1", base_save_dir=Path("../data"))
    # train_data = data.get_split(split="train")

    shard_glob = "data/demo-v1/train/shards/data-*.tar"
    batch_size = 32

    # # Load model
    model = MisophoniaANCNet(label_len=10, pretrained_path=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_workers = 0  # multiprocessing.cpu_count()
    # train(
    #     model,
    #     device,
    #     optimizer,
    #     train_data,
    #     n_epochs=3,
    #     n_items=len(train_data),
    #     num_workers=num_workers,
    #     log_dir=Path("../logs"),
    # )
