"""
The main training script for training on synthetic data
"""

import json
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

from ._utils import CustomMlFlowLogger, SimpleCounter, _debug_to_mlflow, perform_eval, prepare_dir_or_file

try:
    from .confidential_losses import mrccmse_loss  # noqa: F401
except ImportError:
    eliot.log_message(
        "Could not import MultiResolutionCCMSE loss. Make sure you have access to the private repository containing confidential losses and that it is properly installed.",
        level="warning",
    )


if TYPE_CHECKING:
    from .model import MisophoniaANCNet, ModelEMA


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
    for i in range(batch_size):  # Calculate per-item in batch
        for channel in range(pred.shape[1]):  # Calculate loss per-channel of item (average over channels)
            pred_channel = pred[i, channel, : audio_lens[i]]
            tgt_channel = tgt[i, channel, : audio_lens[i]]
            term = -snr(pred_channel, tgt_channel)
            batch_loss.append(term)
    return sum(batch_loss) / len(batch_loss)


def _time_snr_and_si_snr_loss(pred: torch.Tensor, tgt: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    """
    Pure SNR, 50/50 from each channel
    """
    batch_size = pred.shape[0]
    assert pred.shape[1] == 2, "Time with SI-SNR loss currently only supports stereo audio."
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
    gradient_clip_max_norm: float | None = None,
    ema: "ModelEMA | None" = None,
) -> tuple[float, float]:
    model = model.train()

    batch_train_losses = []
    log_to_mlflow_every = 100 if mlflow.active_run() is not None else None  # Only log batch metrics for every x batches
    mlflow_logger = CustomMlFlowLogger()

    with mlflow_logger:
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Training (epoch {epoch})", unit="batch"):
            # Debug even less often than log_to_mlflow_every:
            if log_to_mlflow_every and (batch_idx % 1000 == 0 or batch_idx % 1000 == 1):
                _debug_to_mlflow(mlflow_logger, step_counter, device, prefix="train_")

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

            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)

            optimizer.step()

            if ema is not None:
                ema.update(model)

            loss_value = loss.item()
            batch_train_losses.append(loss_value)
            step_counter.increment()
            if log_to_mlflow_every is not None:
                if batch_idx % log_to_mlflow_every == 0:
                    mlflow_logger.log_metrics(
                        {"train/batch/loss": loss_value},
                        step=step_counter.current,
                        synchronous=False,
                    )
                mlflow_logger.flush_if_needed(synchronous=False)

    batch_train_losses = np.array(batch_train_losses)
    epoch_train_loss = float(np.mean(batch_train_losses))
    epoch_train_loss_std = float(np.std(batch_train_losses))
    return epoch_train_loss, epoch_train_loss_std


def train_model(
    model: "MisophoniaANCNet",
    *,
    train_loader_factory: Callable[[int], wds.WebLoader],
    val_loader: wds.WebLoader,
    n_epochs: int,
    checkpoint_epoch: int = 0,
    device: torch.device,
    loss_option: str,
    save_dir: Path,
    skip_subtraction: bool = True,
    eval_mono_to_stereo: bool = False,
    lr: float = 0.0005,
    lr_schedule_config: dict | None = None,
    weight_decay: float = 0.0,
    gradient_clip_max_norm: float | None = None,
    global_step_train_start: int = 0,
    global_step_val_start: int = 0,
    ema: "ModelEMA | None" = None,
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
        lr_schedule_config: Configuration for learning rate scheduler. If None, no scheduler is used.
        weight_decay (float): weight decay to apply to optimizer
        gradient_clip_max_norm (float | None): Maximum norm for gradient clipping. If None, no gradient clipping is applied.
        device (torch.device): cuda or cpu
        save_dir: Path to save model weights and metric plots
        skip_subtraction: Skip subtraction methods when calculating metrics during val evaluation.
        eval_mono_to_stereo: If True, combine every other pair of predictions into a stereo signal.
        global_step_train (int): Metadata for MLflow to report total number of training batches already logged.
        global_step_val (int): Metadata for MLflow to report total number of validation batches already logged.
    """

    model = model.to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    loss_fn = get_loss_fn_from_name(loss_option)

    if ema is not None:
        ema.to(device)

    scheduler = None
    scheduler_metric = None
    last_learning_rate = lr
    if lr_schedule_config is not None and len(lr_schedule_config) > 0:
        activate_scheduler_after_epoch = lr_schedule_config.get("activate_after_epoch", 0)
        """ First apply the scheduler from this epoch onwards  """
        if checkpoint_epoch > activate_scheduler_after_epoch:
            raise NotImplementedError("Checkpointing after LR scheduler is activated is currently not supported.")
        if lr_schedule_config.get("type") == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_schedule_config.get("mode", "max"),
                factor=lr_schedule_config.get("factor", 0.5),
                patience=lr_schedule_config.get("patience", 5),
                min_lr=lr_schedule_config.get("min_lr", 0),
                cooldown=lr_schedule_config.get("cooldown", 0),
            )
            activate_scheduler_after_epoch = lr_schedule_config.get("activate_after_epoch", 40)
            scheduler_metric = lr_schedule_config.get("scheduler_metric", "val/epoch/snr")
        else:
            raise ValueError(f"Unsupported lr_schedule type {lr_schedule_config}")

    global_step_train_counter = SimpleCounter(global_step_train_start)
    global_step_val_counter = SimpleCounter(global_step_val_start)

    for epoch in range(checkpoint_epoch + 1, n_epochs + 1):
        # Perform train epoch
        train_loader = train_loader_factory(epoch)
        train_loss, train_loss_std = train_epoch(
            model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=loss_fn,
            step_counter=global_step_train_counter,
            epoch=epoch,
            gradient_clip_max_norm=gradient_clip_max_norm,
            ema=ema,
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
            mono_to_stereo=eval_mono_to_stereo,
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

        if ema is not None:
            # Same as results_file etc. but with ema_ prefix
            ema_results_file = results_file.with_name(f"ema_{results_file.name}")
            ema_aggregated_results_file = aggregated_results_file.with_name(f"ema_{aggregated_results_file.name}")
            ema_samples_dir = samples_dir.with_name(f"ema_{samples_dir.name}")

            prepare_dir_or_file(ema_results_file, overwrite=True, is_dir=False)
            prepare_dir_or_file(ema_aggregated_results_file, overwrite=True, is_dir=False)
            prepare_dir_or_file(ema_samples_dir, overwrite=True, is_dir=True)

            _, ema_eval_results_agg = perform_eval(
                ema.model,
                val_loader,
                device=device,
                save_results_to=ema_results_file,
                save_aggregated_results_to=ema_aggregated_results_file,
                save_samples_to=ema_samples_dir,
                save_num_samples=20,
                mlflow_global_step=None,  # No not log EMA results on batch level
                loss_fn=loss_fn,
                skip_subtraction=skip_subtraction,
                split_name="val",
                mono_to_stereo=eval_mono_to_stereo,
            )
            epoch_metrics.update(
                {
                    "ema/val/epoch/loss": ema_eval_results_agg["x"]["loss_mean"],
                    "ema/val/epoch/loss_std": ema_eval_results_agg["x"]["loss_std"],
                    "ema/val/epoch/si_snr_improvement": ema_eval_results_agg["x"]["si_snr_improvement_mean"],
                    "ema/val/epoch/si_snr_improvement_std": ema_eval_results_agg["x"]["si_snr_improvement_std"],
                    "ema/val/epoch/si_snr": ema_eval_results_agg["x"]["si_snr_mean"],
                    "ema/val/epoch/si_snr_std": ema_eval_results_agg["x"]["si_snr_std"],
                    "ema/val/epoch/snr_improvement": ema_eval_results_agg["x"]["snr_improvement_mean"],
                    "ema/val/epoch/snr_improvement_std": ema_eval_results_agg["x"]["snr_improvement_std"],
                    "ema/val/epoch/snr": ema_eval_results_agg["x"]["snr_mean"],
                    "ema/val/epoch/snr_std": ema_eval_results_agg["x"]["snr_std"],
                }
            )

        eliot.log_message(
            f"Epoch {epoch}:\n{json.dumps(epoch_metrics, indent=4)}",
            level="debug",
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(epoch_metrics, step=epoch)

        if scheduler is not None and epoch >= activate_scheduler_after_epoch:
            scheduler.step(epoch_metrics[scheduler_metric])

            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != last_learning_rate:
                eliot.log_message(f"Learning rate changed from {last_learning_rate} to {new_lr}", level="debug")
                last_learning_rate = new_lr

        # Checkpointing
        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"weights_epoch_{epoch}.pt"
        ### Debug
        print(f"Saving checkpoint to {ckpt_path}")
        print(f"EMA: {ema}")
        ###
        model.save_checkpoint(
            ckpt_path,
            epoch=epoch,
            global_step_train=global_step_train_counter.current,
            global_step_val=global_step_val_counter.current,
            val_si_snr_improvement=val_si_snr_improvement,
            val_si_snr=val_si_snr,
            val_loss=val_loss,
            train_loss=train_loss,
            ema_model=ema,
        )
