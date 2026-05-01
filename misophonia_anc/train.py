"""
The main training script for training on synthetic data
"""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import eliot
import mlflow  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds  # noqa: F401
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from tqdm import tqdm

from ._utils import CustomMlFlowLogger, SimpleCounter, perform_eval, prepare_dir_or_file

try:
    from .confidential_losses import mrccmse_loss  # noqa: F401
except ImportError:
    eliot.log_message(
        "Could not import MultiResolutionCCMSE loss. Make sure you have access to the private repository containing confidential losses and that it is properly installed.",
        level="warning",
    )


if TYPE_CHECKING:
    from .model import MisophoniaANCNet


def get_loss_fn_from_name(loss_option: str) -> Callable:
    if loss_option == "time":
        return _time_loss
    elif loss_option == "time_with_si_snr":
        return _time_snr_and_si_snr_loss
    elif loss_option == "freq":
        return _freq_loss
    elif loss_option == "combined":
        return _combined_loss
    else:
        raise ValueError(f"Invalid loss option: {loss_option}")


def _time_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    """
    Pure SNR, 50/50 from each channel
    """
    batch_size = pred.shape[0]
    batch_loss = []
    for i in range(batch_size):
        left_pred = pred[i, 0, : audio_lens[i]]
        right_pred = pred[i, 1, : audio_lens[i]]
        left_term = -snr(left_pred, tgt[i, 0, : audio_lens[i]])
        right_term = -snr(right_pred, tgt[i, 1, : audio_lens[i]])

        avg_term = 0.5 * left_term + 0.5 * right_term
        batch_loss.append(avg_term)
    return sum(batch_loss) / len(batch_loss)


def _time_snr_and_si_snr_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    """
    Pure SNR, 50/50 from each channel
    """
    batch_size = pred.shape[0]
    batch_loss = []
    for i in range(batch_size):
        left_pred = pred[i, 0, : audio_lens[i]]
        right_pred = pred[i, 1, : audio_lens[i]]

        left_term_snr = -snr(left_pred, tgt[i, 0, : audio_lens[i]])
        right_term_snr = -snr(right_pred, tgt[i, 1, : audio_lens[i]])

        avg_term_snr = 0.5 * left_term_snr + 0.5 * right_term_snr

        left_term_si = -si_snr(left_pred, tgt[i, 0, : audio_lens[i]])
        right_term_si = -si_snr(right_pred, tgt[i, 1, : audio_lens[i]])

        avg_term_si = 0.5 * left_term_si + 0.5 * right_term_si

        term = 0.9 * avg_term_snr + 0.1 * avg_term_si

        batch_loss.append(term)
    return sum(batch_loss) / len(batch_loss)


def _freq_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    """
    Computes multiresolution CCMSE on
    """
    batch_size = pred.shape[0]
    batch_loss = []
    for i in range(batch_size):
        item_loss = mrccmse_loss(pred[i, :, : audio_lens[i]], tgt[i, :, : audio_lens[i]])  # type: ignore
        batch_loss.append(item_loss)
    return sum(batch_loss) / len(batch_loss)


def _combined_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    time_loss = _time_loss(pred, tgt, audio_lens)
    freq_loss = _freq_loss(pred, tgt, audio_lens)
    return 0.5 * time_loss + 0.5 * freq_loss


def si_snr_improvement(
    mix: torch.tensor, pred: torch.tensor, gt: torch.tensor, audio_lens: torch.tensor
) -> torch.Tensor:
    batch_size = pred.shape[0]
    si_snr_improvements = []
    for i in range(batch_size):
        improvement = si_snr(pred[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]) - si_snr(
            mix[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]
        )
        si_snr_improvements.append(improvement.mean())
    return sum(si_snr_improvements) / len(si_snr_improvements)


def truncated_si_snr(pred: torch.tensor, gt: torch.tensor, audio_lens: torch.tensor) -> torch.Tensor:
    batch_size = pred.shape[0]
    si_snrs = []
    for i in range(batch_size):
        si_snrs.append(si_snr(pred[i, :, : audio_lens[i]], gt[i, :, : audio_lens[i]]).mean())
    return sum(si_snrs) / len(si_snrs)


def train_epoch(
    model: nn.Module,
    *,
    device: torch.device,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    step_counter: SimpleCounter,
    epoch: int = 0,
    loss_fn: Callable = _time_loss,
) -> tuple[float, float]:
    model = model.train()

    batch_train_losses = []
    log_to_mlflow = mlflow.active_run() is not None
    mlflow_logger = CustomMlFlowLogger()

    with mlflow_logger:
        for batch_num, batch in tqdm(enumerate(train_loader), desc=f"Training (epoch {epoch})", unit="batch"):
            if (
                log_to_mlflow and (batch_num % 100 == 0 or batch_num % 100 == 1) and device == torch.device("cuda")
            ):  # Log VRAM every 10 batches two times in a row
                mlflow_logger.log_metrics(
                    {
                        "debug/train/batch_vram_allocated_gb": (torch.cuda.memory_allocated(device) / (1024**3)),
                        "debug/train/batch_vram_reserved_gb": (torch.cuda.memory_reserved(device) / (1024**3)),
                        "debug/train/batch_vram_free_gb": (
                            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                        )
                        / (1024**3),
                        "debug/train/batch_vram_total_gb": (
                            torch.cuda.get_device_properties(device).total_memory / (1024**3)
                        ),
                    },
                    step=step_counter.current,
                    synchronous=False,
                )

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

            output = model(inputs)

            loss = loss_fn(output["x"], gt, audio_lens)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            batch_train_losses.append(loss_value)
            step_counter.increment()
            if log_to_mlflow:
                mlflow_logger.log_metrics(
                    {
                        "train/batch/loss": loss_value,
                        "debug/train/batch_vram_reserved_gb": (torch.cuda.memory_reserved(device) / (1024**3)),
                    },
                    step=step_counter.current,
                    synchronous=False,
                )

    batch_train_losses = np.array(batch_train_losses)
    epoch_train_loss = float(np.mean(batch_train_losses))
    epoch_train_loss_std = float(np.std(batch_train_losses))
    return epoch_train_loss, epoch_train_loss_std


def train_model(
    model: "MisophoniaANCNet",
    *,
    train_loader: wds.WebLoader,
    val_loader: wds.WebLoader,
    n_epochs: int,
    checkpoint_epoch: int = 0,
    device: torch.device,
    loss_option: str,
    save_dir: Path,
    skip_subtraction: bool = True,
    lr: float = 0.0005,
    weight_decay: float = 0.0,
    global_step_train_start: int = 0,
    global_step_val_start: int = 0,
) -> None:
    """
    Main function to run training loop on Misophonia ANC model. Checkpoints model weights after each epoch. Logs batch and epoch losses for both
    train and val set to mlflow project as well as si_snri on val set. Optionally plots and saves metrics to a local directory.

    Args:
        model (MisophoniaANCNet): Model to train
        train_loader (wds.WebLoader): Train dataset in the form of a WebLoader
        val_loader (wds.WebLoader): Val dataset in the form of a WebLoader
        n_epochs (int): number of epochs for training
        checkpoint_epoch (int): epoch to start checkpointing from. Set to 0 if the model is randomly initialized and set to the epoch number of the loaded checkpoint if resuming training from a checkpoint.
        lr (float): learning rate during trainer
        weight_decay (float): weight decay to apply to optimizer
        device (torch.device): cuda or cpu
        save_dir: Path to save model weights and metric plots
        skip_subtraction: Skip subtraction methods when calculating metrics during val evaluation.
        global_step_train (int): Metadata for MLflow to report total number of training batches already logged.
        global_step_val (int): Metadata for MLflow to report total number of validation batches already logged.
    """

    model = model.to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    loss_fn = get_loss_fn_from_name(loss_option)

    # Checkpoint trackers
    best_epoch = -1
    best_val_si_snr_improvement = -np.inf

    # FIXME: Counting global steps is done using two different implementations for val and train
    global_step_train_counter = SimpleCounter(global_step_train_start)
    global_step_val_counter = SimpleCounter(global_step_val_start)

    for epoch in range(checkpoint_epoch + 1, n_epochs + 1):
        # Perform train epcoh
        train_loss, train_loss_std = train_epoch(
            model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=loss_fn,
            step_counter=global_step_train_counter,
            epoch=epoch,
        )

        # Perform val epoch
        results_file = save_dir / "eval_results" / f"original_run_{epoch}_val_results.json"
        aggregated_results_file = save_dir / "eval_results" / f"original_run_{epoch}_val_aggregated_results.json"
        samples_dir = save_dir / "samples" / f"original_run_{epoch}" / "val"

        prepare_dir_or_file(results_file, overwrite=True, is_dir=False)
        prepare_dir_or_file(aggregated_results_file, overwrite=True, is_dir=False)
        prepare_dir_or_file(samples_dir, overwrite=True, is_dir=True)

        _, eval_results_agg = perform_eval(
            model,
            val_loader,
            device=device,
            save_results_to=results_file,
            save_aggregated_results_to=aggregated_results_file,
            save_samples_to=samples_dir,
            save_num_samples=20,
            mlflow_global_step=global_step_val_counter,
            loss_fn=loss_fn,
            skip_subtraction=skip_subtraction,
            split_name="val",
        )

        val_si_snr = eval_results_agg["x"]["si_snr_mean"]
        val_si_snr_improvement = eval_results_agg["x"]["si_snr_improvement_mean"]
        val_loss = eval_results_agg["x"]["loss_mean"]

        epoch_metrics = {
            "train/epoch/loss": train_loss,
            "train/epoch/loss_std": train_loss_std,
            "val/epoch/loss": val_loss,
            "val/epoch/loss_std": eval_results_agg["x"]["loss_std"],
            "val/epoch/si_snr_improvement": val_si_snr_improvement,
            "val/epoch/si_snr_improvement_std": eval_results_agg["x"]["si_snr_improvement_std"],
            "val/epoch/si_snr_std": eval_results_agg["x"]["si_snr_std"],
            "val/epoch/si_snr": val_si_snr,
            "val/epoch/snr_improvement": eval_results_agg["x"]["snr_improvement_mean"],
            "val/epoch/snr_improvement_std": eval_results_agg["x"]["snr_improvement_std"],
            "val/epoch/snr": eval_results_agg["x"]["snr_mean"],
            "val/epoch/snr_std": eval_results_agg["x"]["snr_std"],
            "train/epoch/global_step": global_step_train_counter.current,
            "val/epoch/global_step": global_step_val_counter.current,
        }
        eliot.log_message(
            f"Epoch {epoch}:\n{json.dumps(epoch_metrics, indent=4)}",
            level="debug",
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(epoch_metrics, step=epoch)

        # Checkpointing
        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"weights_epoch_{epoch}.pt"
        model.save_checkpoint(
            ckpt_path,
            epoch=epoch,
            global_step_train=global_step_train_counter.current,
            global_step_val=global_step_val_counter.current,
            val_si_snr_improvement=val_si_snr_improvement,
            val_si_snr=val_si_snr,
            val_loss=val_loss,
            train_loss=train_loss,
        )

        if val_si_snr_improvement > best_val_si_snr_improvement:
            best_epoch = epoch
            best_val_si_snr_improvement = val_si_snr_improvement

    # Rename best model weights
    best_ckpt = ckpt_dir / f"weights_epoch_{best_epoch}.pt"
    final_path = ckpt_dir / "best_weights.pt"

    if best_epoch >= 0:
        shutil.copy(best_ckpt, final_path)  # safer than rename
