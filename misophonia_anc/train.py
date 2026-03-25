"""
The main training script for training on synthetic data
"""

import sys
from pathlib import Path

# Add parent directory of misophonia-dataset to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import sys
from pathlib import Path

import eliot
import matplotlib.pyplot as plt
import mlflow  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds  # noqa: F401
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import signal_noise_ratio as snr

from ._utils import print_mem

# from .model import MisophoniaANCNet


def loss_fn(_output: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pred = _output["x"]
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


def si_snr_improvement(mix: torch.tensor, pred: torch.tensor, gt: torch.tensor) -> torch.Tensor:
    return si_snr(pred, gt) - si_snr(mix, gt)


def train_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch: int = 0,
) -> float:
    model = model.train()

    batch_train_losses = []

    for batch_idx, (inputs, gt, audio_lens) in enumerate(train_loader):
        # in loader return mask that is [B, C, N]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gt = gt.to(device)
        audio_lens = audio_lens.to(device)
        _, _, T = gt.shape  # noqa: N806

        # print_mem("after inputs")
        optimizer.zero_grad()

        # Mask output
        output = model(inputs)
        pred = output["x"]
        time_idx = torch.arange(T, device=pred.device).unsqueeze(0)
        mask = (time_idx < audio_lens.unsqueeze(1)).unsqueeze(1)
        output["x"] = pred * mask

        loss = loss_fn(output, gt)
        loss.backward()
        # print_mem("after backward")
        optimizer.step()

        loss_value = loss.item()
        batch_train_losses.append(loss_value)
        if mlflow.active_run() is not None:
            mlflow.log_metric(
                "train/loss_batch", loss_value, step=epoch * len(train_loader) + batch_idx, dataset="train"
            )

    epoch_train_loss = float(np.mean(batch_train_losses))

    if mlflow.active_run() is not None:
        mlflow.log_metric("train/loss_epoch", epoch_train_loss, step=epoch, dataset="train")

    return epoch_train_loss


def val_epoch(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    epoch: int = 0,
) -> float:
    model = model.eval()

    batch_val_losses = []
    val_si_snrs = []

    with torch.no_grad():
        for batch_idx, (inputs, gt, audio_lens) in enumerate(val_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}  # [B, 2, N]
            _, _, T = gt.shape  # noqa: N806

            gt = gt.to(device)
            audio_lens = audio_lens.to(device)

            # Mask output
            output = model(inputs)
            pred = output["x"]
            time_idx = torch.arange(T, device=pred.device).unsqueeze(0)
            mask = (time_idx < audio_lens.unsqueeze(1)).unsqueeze(1)
            output["x"] = pred * mask

            loss = loss_fn(output, gt)

            loss_value = loss.item()
            val_si_snr_improvement = si_snr_improvement(inputs["mixes"], output["x"], gt).mean().item()
            batch_val_losses.append(loss_value)
            val_si_snrs.append(val_si_snr_improvement)

            if mlflow.active_run() is not None:
                global_step = epoch * len(val_loader) + batch_idx
                mlflow.log_metric("val/loss_batch", loss_value, step=global_step, dataset="val")
                mlflow.log_metric("val/si_snr_batch", val_si_snr_improvement, step=global_step, dataset="val")

    epoch_val_loss = float(np.mean(batch_val_losses))
    epoch_val_si_snr = float(np.mean(val_si_snrs))

    if mlflow.active_run() is not None:
        mlflow.log_metric("val/loss_epoch", epoch_val_loss, step=epoch, dataset="val")
        mlflow.log_metric("val/si_snr_epoch", epoch_val_si_snr, step=epoch, dataset="val")

    return epoch_val_loss, epoch_val_si_snr


def train_model(
    model: nn.Module,
    *,
    train_loader: wds.WebLoader,
    val_loader: wds.WebLoader,
    n_epochs: int,
    device: torch.device,
    save_dir: Path,
) -> None:

    model = model.to(device)
    print_mem("after model")
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0005, weight_decay=0)

    train_losses = []
    val_losses = []
    val_si_snrs = []
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, device, optimizer, train_loader, epoch)
        train_losses.append(train_loss)

        val_loss, val_si_snr = val_epoch(model, device, val_loader, epoch)
        val_losses.append(val_loss)
        val_si_snrs.append(val_si_snr)

        eliot.log_message(
            f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val SI-SNR = {val_si_snr}",
            level="debug",
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_si_snr": val_si_snr,
                },
                step=epoch,
            )

    if save_dir is not None:
        plot_dir = save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _make_plots(plot_dir, train_losses, val_losses, val_si_snrs)


def _make_plots(plot_dir: Path, train_losses: list, val_losses: list, val_si_snrs: list) -> None:
    # Loss plot
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(plot_dir / "loss_plot.png")
    plt.close()

    # Si-SNR plot
    plt.figure()
    plt.plot(val_si_snrs, label="Validation Si-SNR")

    plt.xlabel("Epoch")
    plt.ylabel("SNR")
    plt.title("Validation Si-SNR")
    plt.legend()

    plt.savefig(plot_dir / "si_snr_plot.png")
    plt.close()
