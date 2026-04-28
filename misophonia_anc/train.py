"""
The main training script for training on synthetic data
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

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
from tqdm import tqdm

try:
    from .confidential_losses import mrccmse_loss  # noqa: F401
except ImportError:
    eliot.log_message(
        "Could not import MultiResolutionCCMSE loss. Make sure you have access to the private repository containing confidential losses and that it is properly installed.",
        level="warning",
    )


if TYPE_CHECKING:
    from .model import MisophoniaANCNet


def loss_fn(
    _output: dict[str, torch.Tensor], tgt: torch.Tensor, audio_lens: torch.Tensor, loss_option: str = "time"
) -> torch.Tensor:
    pred = _output["x"]

    def _time_loss(
        pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor, gamma: float = 0.25
    ) -> torch.Tensor:
        """
        Computes loss with .7 weight on snr and .3 weight on si-snr. Applies double weighting to the right channel
        """
        B = pred.shape[0]
        batch_loss = []
        for i in range(B):
            left_pred = pred[i, 0, : audio_lens[i]]
            right_pred = pred[i, 1, : audio_lens[i]]
            left_term = -0.9 * snr(left_pred, tgt[i, 0, : audio_lens[i]])
            right_term = -0.9 * snr(right_pred, tgt[i, 1, : audio_lens[i]])

            avg_term = 0.5 * left_term + 0.5 * right_term
            max_term = max(left_term, right_term)
            batch_loss.append(avg_term + gamma * max_term)
        return sum(batch_loss) / len(batch_loss)

    def _freq_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
        """
        Computes multiresolution CCMSE on
        """
        B = pred.shape[0]
        batch_loss = []
        for i in range(B):
            item_loss = mrccmse_loss(pred[i, :, : audio_lens[i]], tgt[i, :, : audio_lens[i]])  # type: ignore
            batch_loss.append(item_loss)
        return sum(batch_loss) / len(batch_loss)

    if loss_option == "time":
        return _time_loss(pred, tgt, audio_lens)
    elif loss_option == "freq":
        return _freq_loss(pred, tgt, audio_lens)
    elif loss_option == "combined":
        return 0.5 * _freq_loss(pred, tgt, audio_lens) + 0.5 * _time_loss(pred, tgt, audio_lens)
    else:
        raise ValueError(f"Invalid loss option: {loss_option}")


def si_snr_improvement(
    mix: torch.tensor, pred: torch.tensor, gt: torch.tensor, audio_lens: torch.tensor
) -> torch.Tensor:
    B = pred.shape[0]
    si_snr_improvements = []
    for i in range(B):
        improvement = si_snr(pred[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]) - si_snr(
            mix[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]
        )
        si_snr_improvements.append(improvement.mean())
    return sum(si_snr_improvements) / len(si_snr_improvements)


def truncated_si_snr(pred: torch.tensor, gt: torch.tensor, audio_lens: torch.tensor) -> torch.Tensor:
    B = pred.shape[0]
    si_snrs = []
    for i in range(B):
        si_snrs.append(si_snr(pred[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]).mean())
    return sum(si_snrs) / len(si_snrs)


def train_epoch(
    model: nn.Module,
    *,
    device: torch.device,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    start_global_step: int = 0,
    epoch: int = 0,
    loss_option: str = "time",
) -> tuple[float, int]:
    model = model.train()

    batch_train_losses = []

    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Training (epoch {epoch})", unit="batch"):
        inputs = batch["inputs"]
        gt = batch[model.ground_truth_target]
        audio_lens = batch["audio_lens"]

        # in loader return mask that is [B, C, N]
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["mix"] = inputs["mix"].to(device)
        inputs["label_vector"] = inputs["label_vector"].to(device)
        inputs["is_control"] = inputs["is_control"].to(device)

        gt = gt.to(device)
        audio_lens = audio_lens.to(device)
        _, _, T = gt.shape  # noqa: N806

        optimizer.zero_grad()

        # Mask output
        output = model(inputs)

        loss = loss_fn(output, gt, audio_lens, loss_option=loss_option)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        batch_train_losses.append(loss_value)
        if mlflow.active_run() is not None:
            mlflow.log_metric("train/loss_batch", loss_value, step=start_global_step + batch_idx)

    epoch_train_loss = float(np.mean(batch_train_losses))
    return epoch_train_loss, start_global_step + batch_idx + 1


def val_epoch(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    *,
    start_global_step: int = 0,
    epoch: int = 0,
    loss_option: str = "time",
) -> tuple[float, float, int]:
    """
    Function to evaluate model on validation set each epoch.

    Args:
        See train_model() for arg description

    Returns:
        epoch_val_loss (float): epoch loss on val set
        epoch_val_si_snr_improvement (float): si_snri on val set
        global_step (int): number of val batches the on which the model has been ran

    """
    model = model.eval()

    batch_val_losses = []
    val_si_snr_improvements = []
    val_si_snrs = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation (epoch {epoch})", unit="batch"):
            inputs = batch["inputs"]
            gt = batch[model.ground_truth_target]
            audio_lens = batch["audio_lens"]

            # inputs = {k: v.to(device) for k, v in inputs.items()}  # [B, 2, N]
            inputs["mix"] = inputs["mix"].to(device)
            inputs["label_vector"] = inputs["label_vector"].to(device)
            inputs["is_control"] = inputs["is_control"].to(device)

            _, _, T = gt.shape  # noqa: N806

            gt = gt.to(device)
            audio_lens = audio_lens.to(device)

            # Mask output
            output = model(inputs)
            pred = output["x"]

            loss = loss_fn(output, gt, audio_lens, loss_option=loss_option)

            loss_value = loss.item()
            val_si_snr_improvement = si_snr_improvement(inputs["mix"], output["x"], gt, audio_lens).item()
            val_si_snr = truncated_si_snr(output["x"], gt, audio_lens).item()

            batch_val_losses.append(loss_value)
            val_si_snr_improvements.append(val_si_snr_improvement)
            val_si_snrs.append(val_si_snr)

            if mlflow.active_run() is not None:
                mlflow.log_metric("val/loss_batch", loss_value, step=start_global_step + batch_idx)
                mlflow.log_metric(
                    "val/si_snr_improvement_batch", val_si_snr_improvement, step=start_global_step + batch_idx
                )

    epoch_val_loss = float(np.mean(batch_val_losses))
    epoch_val_si_snr_improvement = float(np.mean(val_si_snr_improvements))
    epoch_val_si_snr = float(np.mean(val_si_snrs))
    global_step = start_global_step + batch_idx

    return epoch_val_loss, epoch_val_si_snr_improvement, epoch_val_si_snr, global_step + 1


def train_model(
    model: "MisophoniaANCNet",
    *,
    train_loader: wds.WebLoader,
    val_loader: wds.WebLoader,
    n_epochs: int,
    checkpoint_epoch: int = -1,
    device: torch.device,
    loss_option: str,
    save_dir: Path,
    lr: float = 0.0005,
    weight_decay: float = 0.0,
    global_step_train: int = 0,
    global_step_val: int = 0,
) -> None:
    """
    Main function to run training loop on Misophonia ANC model. Checkpoints model weights after each epoch. Logs batch and epoch losses for both
    train and val set to mlflow project as well as si_snri on val set. Optionally plots and saves metrics to a local directory.

    Args:
        model (MisophoniaANCNet): Model to train
        train_loader (wds.WebLoader): Train dataset in the form of a WebLoader
        val_loader (wds.WebLoader): Val dataset in the form of a WebLoader
        n_epochs (int): number of epochs for training
        checkpoint_epoch (int): epoch to start checkpointing from. Set to -1 if the model is randomly initialized and set to the epoch number of the loaded checkpoint if resuming training from a checkpoint.
        lr (float): learning rate during trainer
        weight_decay (float): weight decay to apply to optimizer
        device (torch.device): cuda or cpu
        save_dir: Path to save model weights and metric plots
        global_step_train (int): Metadata for MLflow to report total number of training batches already logged.
        global_step_val (int): Metadata for MLflow to report total number of validation batches already logged.
    """

    model = model.to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    # Tracking metrics
    train_losses = []
    val_losses = []
    val_si_snr_improvements = []
    val_si_snrs = []

    # Checkpoint trackers
    best_epoch = -1
    best_val_si_snr_improvement = -np.inf
    for epoch in range(checkpoint_epoch + 1, n_epochs):
        train_loss, global_step_train = train_epoch(
            model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_option=loss_option,
            start_global_step=global_step_train,
            epoch=epoch,
        )
        train_losses.append(train_loss)

        val_loss, val_si_snr_improvement, val_si_snr, global_step_val = val_epoch(
            model,
            device,
            val_loader,
            loss_option=loss_option,
            start_global_step=global_step_val,
            epoch=epoch,
        )
        val_losses.append(val_loss)
        val_si_snrs.append(val_si_snr)
        val_si_snr_improvements.append(val_si_snr_improvement)

        eliot.log_message(
            f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val SI-SNRi = {val_si_snr_improvement}",
            level="debug",
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_si_snr_improvement": val_si_snr_improvement,
                    "epoch/val_si_snr": val_si_snr,
                    "epoch/global_step_train": global_step_train,
                    "epoch/global_step_val": global_step_val,
                },
                step=epoch,
            )

        # Checkpointing
        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"weights_epoch_{epoch}.pt"
        model.save_checkpoint(
            ckpt_path,
            epoch=epoch,
            global_step_train=global_step_train,
            global_step_val=global_step_val,
            val_si_snr_improvement=val_si_snr_improvement,
            val_si_snr=val_si_snr,
            val_loss=val_loss,
            train_loss=train_loss,
        )

        if val_si_snr_improvement > best_val_si_snr_improvement:
            best_epoch = epoch
            best_val_si_snr_improvement = val_si_snr_improvement

    # Plotting
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _make_plots(plot_dir, train_losses, val_losses, val_si_snrs, val_si_snr_improvements)

    # Rename best model weights
    best_ckpt = ckpt_dir / f"weights_epoch_{best_epoch}.pt"
    final_path = ckpt_dir / "best_weights.pt"

    if best_epoch >= 0:
        shutil.copy(best_ckpt, final_path)  # safer than rename


def _make_plots(
    plot_dir: Path, train_losses: list, val_losses: list, val_si_snrs: list, val_si_snr_improvements: list
) -> None:
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

    # Si-SNRi plot
    plt.figure()
    plt.plot(val_si_snr_improvements, label="Val Si-SNRi")

    plt.xlabel("Epoch")
    plt.ylabel("Si-SNRi")
    plt.title("Validation Si-SNRi")
    plt.legend()

    plt.savefig(plot_dir / "si_snr_improvement_plot.png")
    plt.close()

    # Si-SNR plot
    plt.figure()
    plt.plot(val_si_snrs, label="Val Si-SNR")

    plt.xlabel("Epoch")
    plt.ylabel("Si-SNR")
    plt.title("Validation Si-SNR")
    plt.legend()

    plt.savefig(plot_dir / "si_snr_plot.png")
    plt.close()
