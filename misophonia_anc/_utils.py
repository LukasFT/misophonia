"""A collection of useful helper functions"""

# ruff: noqa: ANN001 # FIXME: Improve quality

import json
import math
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Sequence, Sized
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import eliot
import matplotlib.pyplot as plt
import mlflow
import mlflow.entities
import mlflow.utils.time
import numpy as np
import pandas as pd
import pydantic
import soundfile as sf
import torch
import torch.nn.functional as F  # noqa: N812  # noqa: N812
import torchaudio
import webdataset as wds
import yaml
from torch.profiler import ProfilerActivity, profile, record_function
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from tqdm import tqdm

from misophonia_dataset.interface import DEFAULT_LABEL_ORDER, BaseModel, MisophoniaItem, SplitT
from misophonia_dataset.main import get_default_datasets_names
from misophonia_dataset.misophonia_dataset import MisophoniaDatasetSplit

if TYPE_CHECKING:
    from .model import MisophoniaANCNet

# Initialize random generator for reproducibility
rng = np.random.default_rng()

SAMPLE_RATE = 44100

######################################
# Preprocess and Data Loading Utils #
######################################


def preprocess_to_webdataset_pt(
    shards_dir: str | Path,
    dataset_split: MisophoniaDatasetSplit,
    *,
    samples_per_shard: int,
    num_workers: int = None,
    show_progress: bool = True,
    metadata: dict | None = None,
) -> str:
    """
    Preprocess GeneratedMisophoniaDataset into WebDataset .tar shards using multithreading. Save tensors.

    Args:
        shards_dir: Directory to save the .tar shards. Assumes shards_dir already contains split in name
        dataset_split: The dataset split to preprocess.
        samples_per_shard: Number of samples per .tar shard
        num_workers: Number of threads to use for parallel processing. If None, defaults to number of CPU cores.
        show_progress: Whether to show a progress bar during preprocessing.
        metadata: Optional dictionary of metadata to save alongside the dataset. Will be saved as metadata.json in the shards_dir.

    Returns:
        A glob pattern for the generated .tar shards. Used for loading wds.WebDataset.
    """

    def _audio_to_flac_bytes(audio: np.ndarray | torch.Tensor, sample_rate: int) -> bytes:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio of shape [C, T], got {audio.shape}")

        with BytesIO() as buffer:
            sf.write(buffer, audio.T, samplerate=sample_rate, format="FLAC", subtype="PCM_24")
            return buffer.getvalue()

    def process_item(idx) -> tuple[dict, dict]:
        item: MisophoniaItem = dataset_split[idx]
        # This function should call your actual preprocessing
        # preprocess_item_to_arrays -> returns (X, y, label_vec)
        mix_array = item.get_mix_audio()
        isolated_trigger_array = item.get_isolated_trigger_audio()
        clean_mix_array = item.get_clean_mix_audio()
        label_array = item.get_label_vector(label_order=DEFAULT_LABEL_ORDER)
        sample_rate = item.global_mixing_params.sample_rate

        mix_torch = torch.from_numpy(mix_array)
        mix_vs_isolated_trigger_metrics = calculate_default_metrics(
            mix_torch,
            torch.from_numpy(isolated_trigger_array),
            sample_rate=sample_rate,
        )
        mix_vs_clean_mix_metrics = calculate_default_metrics(
            mix_torch, torch.from_numpy(clean_mix_array), sample_rate=sample_rate
        )

        metadata = {
            "uuid": item.uuid,
            "sample_rate": sample_rate,
            "fg_categories": item.foreground_categories,
            "bg_categories": item.background_categories,
            "is_trigger": item.is_trigger,
            "mix_vs_isolated_trigger_metrics": mix_vs_isolated_trigger_metrics,
            "mix_vs_clean_mix_metrics": mix_vs_clean_mix_metrics,
            "fg_freesound_ids": tuple(fg.source_item.freesound_id for fg in item.foregrounds),
            "bg_freesound_ids": tuple(bg.source_item.freesound_id for bg in item.backgrounds),
        }
        metadata_str = json.dumps(metadata)
        mix_flac = _audio_to_flac_bytes(mix_array, sample_rate)
        isolated_trigger_flac = _audio_to_flac_bytes(isolated_trigger_array, sample_rate)
        clean_mix_flac = _audio_to_flac_bytes(clean_mix_array, sample_rate)

        sample = {
            "__key__": f"{idx:09d}",
            "mix.flac": mix_flac,
            "label.npy": label_array,
            "isolated_trigger.flac": isolated_trigger_flac,
            "clean_mix.flac": clean_mix_flac,
            "metadata.json": metadata_str,
        }
        return sample, metadata

    num_workers = num_workers or os.cpu_count() or 1

    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(shards_dir / "data-%06d.tar")
    metrics = []
    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            size = len(dataset_split)
            results = executor.map(process_item, range(size))
            if show_progress:
                results = tqdm(results, total=size, desc=f"Saving {dataset_split.split} items")

            for item, item_metadata in results:
                sink.write(item)
                metrics.append(
                    {
                        "pred_name": "mix_vs_isolated_trigger",
                        "metrics": item_metadata["mix_vs_isolated_trigger_metrics"],
                    }
                )
                metrics.append(
                    {
                        "pred_name": "mix_vs_clean_mix",
                        "metrics": item_metadata["mix_vs_clean_mix_metrics"],
                    }
                )

    shard_glob = str(shards_dir / "data-*.tar")

    # Save metadata
    metadata_file = shards_dir / "metadata.json"
    metadata_with_metrics = dict(metadata or {})
    metadata_with_metrics["aggregated_metrics"] = aggregate_results(metrics)

    with metadata_file.open("w") as f:
        json.dump(metadata_with_metrics, f, indent=4)

    return shard_glob


class MisophoniaDatasetPreprocessedConfig(BaseModel):
    from_premade: str | None = pydantic.Field(
        None,
        description="Whether to use a premade dataset with the given name. If a non-empty string, use that name of the premade dataset.",
    )
    generated_source_data: list[str] | tuple[str, ...] = pydantic.Field(
        get_default_datasets_names(),
        description="If from_premade is False, will generate dataset using the given source datasets. See GeneratedMisophoniaDataset for options.",
    )
    generated_config: dict | None = pydantic.Field(
        None,
        description="If premade_config not given, will generate dataset using the given config. See GeneratedMisophoniaDataset.get_split for options.",
    )


GtTargets = Literal["isolated_trigger", "clean_mix"]


class MisophoniaANCConfig(BaseModel):
    dataset_splits: dict[SplitT, MisophoniaDatasetPreprocessedConfig] = pydantic.Field(
        ..., description="For each split, the config for the preprocessed dataset to use for training/eval."
    )

    num_epochs: int = pydantic.Field(10, description="Number of epochs to train for.")
    batch_size: int = pydantic.Field(1, description="Batch size for training.")
    loss_option: str = pydantic.Field(
        "time", description="Domain in which to apply loss. See train.get_loss_fn_from_name for options."
    )

    model_params: dict = pydantic.Field(
        {}, description="Dictionary of parameters to initialize the model. See the MisophoniaANCNet class for options."
    )

    model_hyperparams: dict = pydantic.Field(
        {}, description="Dictionary of parameters to initialize the model. See the train_model() for options."
    )

    subtraction_methods: list[str] | tuple[str, ...] | None = pydantic.Field(
        None,
        description="Whether to perform post-hoc subtraction using the original mix and the model's prediction, and if so, which method to use for subtraction. "
        "See MisophoniaANCNet.register_subtraction_method for details.",
    )

    stereo_to_mono: bool = pydantic.Field(
        False,  # noqa: FBT003
        description="Wheater to split each sample into two mono samples, effectively doubling the batch size, i.e. (B, T, 2) -> (2B, T, 1).",
    )

    mlflow_experiment: str | None = pydantic.Field(
        None, description="MLflow experiment name to log training metrics to."
    )

    # validate the if stereo_to_mono = True, then model_params.audio_channels must be 1
    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_stereo_to_mono(cls, values: dict[str, Any]) -> dict[str, Any]:
        stereo_to_mono = values.get("stereo_to_mono", False)
        audio_channels = (values.get("model_params") or {}).get("audio_channels", 2)
        if stereo_to_mono and audio_channels != 1:
            raise ValueError("If stereo_to_mono is True, model_params.audio_channels must be 1.")
        return values

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, *, defaults={}) -> "MisophoniaANCConfig":
        conf = dict(defaults)
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Cannot load config since file does not exist: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        conf.update(data)
        return cls(**conf)


def to_mono_batch(batch: dict) -> dict:
    """Make the batch twice the size by converting stereo audio to mono (two samples per sample)."""
    # (B, N, T) -> (N*B, 1, T)
    original_batch_size, original_channels, length = batch["inputs"]["mix"].shape
    new_batch_size = original_batch_size * original_channels
    new_channels = 1

    batch["inputs"]["mix"] = batch["inputs"]["mix"].reshape(new_batch_size, new_channels, length)

    if "isolated_trigger" in batch:
        batch["isolated_trigger"] = batch["isolated_trigger"].reshape(new_batch_size, new_channels, length)

    if "clean_mix" in batch:
        batch["clean_mix"] = batch["clean_mix"].reshape(new_batch_size, new_channels, length)

    batch["inputs"]["label_vector"] = batch["inputs"]["label_vector"].repeat_interleave(original_channels, dim=0)
    batch["inputs"]["is_control"] = batch["inputs"]["is_control"].repeat_interleave(original_channels, dim=0)
    batch["audio_lens"] = batch["audio_lens"].repeat_interleave(original_channels, dim=0)

    batch["idxs"] = [f"{idx}_{ch}" for idx in batch["idxs"] for ch in range(original_channels)]

    if "metadata" in batch:
        batch["metadata"] = [batch["metadata"][i // original_channels] for i in range(new_batch_size)]
        # remove pre-computed metrics
        for metadata in batch["metadata"]:
            if "mix_vs_isolated_trigger_metrics" in metadata:
                del metadata["mix_vs_isolated_trigger_metrics"]
            if "mix_vs_clean_mix_metrics" in metadata:
                del metadata["mix_vs_clean_mix_metrics"]

    return batch


def to_stereo_batch(batch: dict) -> dict:
    """Convert a batch of mono audio back to stereo by duplicating the mono channel."""
    # (B, 1, T) -> (B/2, 2, T)
    original_batch_size, original_channels, length = batch["inputs"]["mix"].shape
    assert original_channels == 1
    assert original_batch_size % 2 == 0

    new_batch_size = original_batch_size // 2
    new_channels = 2
    batch["inputs"]["mix"] = batch["inputs"]["mix"].reshape(new_batch_size, new_channels, length)
    if "isolated_trigger" in batch:
        batch["isolated_trigger"] = batch["isolated_trigger"].reshape(new_batch_size, new_channels, length)
    if "clean_mix" in batch:
        batch["clean_mix"] = batch["clean_mix"].reshape(new_batch_size, new_channels, length)

    # Remove every other label, len etc. since they are duplicated
    batch["inputs"]["label_vector"] = batch["inputs"]["label_vector"][::2]
    batch["inputs"]["is_control"] = batch["inputs"]["is_control"][::2]
    batch["audio_lens"] = batch["audio_lens"][::2]
    batch["idxs"] = [idx.rsplit("_", 1)[0] for idx in batch["idxs"][::2]]

    if "metadata" in batch:
        batch["metadata"] = batch["metadata"][::2]

    return batch


def to_stereo_output(output: dict) -> dict:
    """Convert a batch of mono outputs back to stereo by duplicating the mono channel."""
    # (B, 1, T) -> (B/2, 2, T)

    res = {}
    for pred_name, pred in output.items():
        original_batch_size, original_channels, length = pred.shape
        assert original_channels == 1
        assert original_batch_size % 2 == 0

        new_batch_size = original_batch_size // 2
        new_channels = 2

        res[pred_name] = pred.reshape(new_batch_size, new_channels, length)

    return res


def make_custom_collate_fn(
    *,
    include_metadata: bool,
    include_isolated_trigger: bool,
    include_clean_mix: bool,
    max_length: int | None = None,
    stereo_to_mono: bool = False,
) -> Callable[[dict], dict]:
    max_length = torch.inf if max_length is None else max_length

    def custom_collate_fn(
        batch: dict,
    ) -> dict:
        """
        Pads mixes and gt so that they are equal length. Passes length of each audio to properly mask on loss function.

        If max_length is given, it will only use the first max_length samples of the audio.

        Also randomly assign control sounds a class in the label vector during training, since they don't have a specific class.
        This is done by randomly assigning a 1 to one of the trigger classes in the label vector.
        This is done because the purpose of the control sounds is to teach the model that even if a category is queried,
            it might need to predict silence if there is no trigger sound of that category in the mix.
        """

        def _to_audio_tensor(audio: np.ndarray | torch.Tensor | bytes) -> torch.Tensor:
            if isinstance(audio, torch.Tensor):
                return audio.float()
            if isinstance(audio, np.ndarray):
                return torch.from_numpy(audio).float()
            if isinstance(audio, (bytes, bytearray, memoryview)):
                decoded_audio, _ = sf.read(BytesIO(audio), dtype="float32", always_2d=True)
                return torch.from_numpy(decoded_audio.T).float()
            raise TypeError(f"Unsupported audio type: {type(audio)!r}")

        mixes = []
        if include_isolated_trigger:
            isolated_triggers = []
        if include_clean_mix:
            clean_mixes = []
        labels = []
        audio_lens = []
        is_controls = []
        metadatas = []
        idxs = []

        # Convert to torch
        for sample in batch:
            for key in sample:
                if key.endswith(".flac"):
                    sample[key] = _to_audio_tensor(sample[key])
                elif key.endswith(".npy"):
                    sample[key] = torch.from_numpy(sample[key]).float()

        # Longest sample in the batch determines the length to pad to
        chunk_size = max(sample["mix.flac"].shape[-1] for sample in batch)
        chunk_size = min(chunk_size, max_length)

        for sample in batch:
            idxs.append(sample["__key__"])
            label = sample["label.npy"]
            mix = sample["mix.flac"]

            if mix.shape[-1] > max_length:
                mix = mix[:, :max_length]

            if include_metadata:
                metadatas.append(sample["metadata.json"])

            is_control = label.sum() == 0  # Check if the label vector is all zeros (indicating a control sound)
            is_controls.append(1 if is_control else 0)
            if is_control:
                # Randomly assign a class to the control sound in the label vector
                # See note in docstring for motivation
                random_class = rng.integers(0, len(label))
                label[random_class] = 1

            labels.append(label)

            L = mix.shape[-1]  # noqa: N806
            audio_lens.append(L)
            # audio is shorter than chunk_size → pad

            mixes.append(F.pad(mix, (0, chunk_size - L)))

            if include_isolated_trigger:
                isolated_trigger = F.pad(sample["isolated_trigger.flac"], (0, chunk_size - L))
                if isolated_trigger.shape[-1] > max_length:
                    isolated_trigger = isolated_trigger[:, :max_length]
                isolated_triggers.append(isolated_trigger)
            if include_clean_mix:
                clean_mix = F.pad(sample["clean_mix.flac"], (0, chunk_size - L))
                if clean_mix.shape[-1] > max_length:
                    clean_mix = clean_mix[:, :max_length]
                clean_mixes.append(clean_mix)

        res = {
            "idxs": idxs,
            "inputs": {
                "mix": torch.stack(mixes),
                "label_vector": torch.stack(labels),
                "is_control": torch.tensor(is_controls),
            },
            "audio_lens": torch.tensor(audio_lens),  # Mask to indicate padded parts of the audio
        }
        if include_metadata:
            res["metadata"] = metadatas
        if include_isolated_trigger:
            res["isolated_trigger"] = torch.stack(isolated_triggers)
        if include_clean_mix:
            res["clean_mix"] = torch.stack(clean_mixes)

        if stereo_to_mono:
            res = to_mono_batch(res)

        return res

    return custom_collate_fn


def make_dataloader(
    files: Iterable[str | Path],
    *,
    batch_size: int,
    num_workers: int,
    include_isolated_trigger: bool = True,
    include_clean_mix: bool = True,
    include_metadata: bool = False,
    max_length: int | None = None,
    stereo_to_mono: bool = False,
) -> wds.WebLoader:
    """
    Make a WebLoader from the given .tar files.

    Args:
        files: An iterable of paths to .tar shard files.
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker threads for loading data.

    Returns:
        An iterable WebLoader that yields batches of data from the given .tar files.

    """
    files = tuple(str(file) for file in files)
    assert len(files) > 0, "No files provided to make_dataloader."
    eliot.log_message(f"Loading data from `{files[0]}` etc...", level="debug")
    eliot.log_message(
        f"Using {num_workers} workers loading WebDataset (total CPU count = {os.cpu_count()}, allocated = {get_allocated_cpus()}).",
        level="debug",
    )

    included_filenames = {"mix.flac", "label.npy", "metadata.json"}
    if include_isolated_trigger:
        included_filenames.add("isolated_trigger.flac")
    if include_clean_mix:
        included_filenames.add("clean_mix.flac")

    def _include_file(fname: str) -> bool:
        """
        Only include files that are in the included_filenames set.

        Example:
            _include_file("000000991.mix.flac") -> True
            _include_file("000000991.isolated_trigger.flac") -> True if include_isolated_trigger is True, False otherwise

        """
        return any(fname.endswith(included_fname) for included_fname in included_filenames)

    data = (
        wds.WebDataset(
            files,
            empty_check=False,
            shardshuffle=1,  # Number of shards to keep in memory at the time (as I understand it)
            select_files=_include_file,
        )
        .shuffle(batch_size)  # Number of samples to shuffle in memory at the time (as I understand it)
        .decode("torch")  # converts the saved numpy arrays to tensors
    )

    data = data.batched(
        batch_size,
        # Make batches of the same size, and randomly assign control sounds a class
        collation_fn=make_custom_collate_fn(
            include_metadata=include_metadata,
            include_isolated_trigger=include_isolated_trigger,
            include_clean_mix=include_clean_mix,
            max_length=max_length,
            stereo_to_mono=stereo_to_mono,
        ),
    )

    return wds.WebLoader(
        data,
        batch_size=None,  # We set batch size in the WebDataset pipeline, so we set it to None here
        num_workers=num_workers,
    )


def calculate_default_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    mix: torch.Tensor | None = None,
    mix_metrics: dict | None = None,
    sample_rate: int = SAMPLE_RATE,
    loss_fn: Callable | None = None,
    calculate_ild_itd: bool = False,
) -> dict[str, float]:
    si_snr_both = si_snr(preds, target)
    snr_both = snr(preds, target)

    metrics = {
        "si_snr": si_snr_both.mean().item(),
        "snr": snr_both.mean().item(),
    }

    if si_snr_both.shape[-1] == 2:
        metrics["si_snr_left"] = si_snr_both[..., 0].mean().item()
        metrics["si_snr_right"] = si_snr_both[..., 1].mean().item()
        metrics["snr_left"] = snr_both[..., 0].mean().item()
        metrics["snr_right"] = snr_both[..., 1].mean().item()

    if loss_fn is not None:
        # Call loss function like it is a batch
        audio_lens = (target.shape[-1],)
        loss = loss_fn(preds.unsqueeze(0), target.unsqueeze(0), audio_lens)
        metrics["loss"] = loss.item()

    if calculate_ild_itd:
        metrics["ild_diff"] = ild_diff_torch(preds, target)
        metrics["itd_diff"] = itd_diff_torch(preds, target, sr=sample_rate)

    if mix is not None and mix_metrics is None:
        mix_metrics = calculate_default_metrics(mix, target, sample_rate=sample_rate)

    if mix_metrics is not None:
        if "snr" in mix_metrics:
            metrics["snr_improvement"] = metrics["snr"] - mix_metrics["snr"]
        if "si_snr" in mix_metrics:
            metrics["si_snr_improvement"] = metrics["si_snr"] - mix_metrics["si_snr"]
        if "snr_left" in mix_metrics and "snr_left" in metrics:
            metrics["snr_improvement_left"] = metrics["snr_left"] - mix_metrics["snr_left"]
        if "snr_right" in mix_metrics and "snr_right" in metrics:
            metrics["snr_improvement_right"] = metrics["snr_right"] - mix_metrics["snr_right"]
        if "si_snr_left" in mix_metrics and "si_snr_left" in metrics:
            metrics["si_snr_improvement_left"] = metrics["si_snr_left"] - mix_metrics["si_snr_left"]
        if "si_snr_right" in mix_metrics and "si_snr_right" in metrics:
            metrics["si_snr_improvement_right"] = metrics["si_snr_right"] - mix_metrics["si_snr_right"]

    return metrics


############################
# Resource Tracking Utils #
############################


def get_allocated_cpus() -> int:
    """Returns the number of CPUs allocated to the current process, accounting for environments like SLURM and Docker."""

    # 1. Try SLURM environment variable
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)

    # 2. Try scheduler affinity (works for Docker/K8s/SLURM with cgroups)
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    # 3. Fallback to total system CPUs
    return os.cpu_count() or 1


def print_mem(label: str) -> None:
    print(f"{label}:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print()


####################
# Inference Utils #
####################


def perform_eval(
    model: "MisophoniaANCNet",
    data_loader: wds.WebLoader,
    *,
    device: torch.device,
    save_results_to: Path,
    save_aggregated_results_to: Path | None = None,
    aggregated_results_kwargs: dict | None = None,
    calculate_metrics_kwargs: dict | None = None,
    mono_to_stereo: bool = False,
    save_samples_to: Path | None = None,
    save_num_samples: int = 0,
    warm_up_iters: int = 10,
    mlflow_global_step: Optional["SimpleCounter"] = None,
    loss_fn: Callable | None = None,
    skip_subtraction: bool = False,
    split_name: SplitT | None = None,
) -> tuple[dict, dict | None]:
    """
    Run inference on the given model and dataloader and evaluate.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: A WebLoader that yields batches of data for evaluation.
        device: The torch.device to run inference on.
        save_results_to: A path to save the evaluation results as a JSON file. Will include metrics and metadata for each sample.
        save_aggregated_results_to: If not None, a path to save aggregated evaluation results (e.g. average metrics across all samples) as a JSON file.
        aggregated_results_kwargs: See aggregate_results() for details on the kwargs.
        calculate_metrics_kwargs: Additional kwargs to pass to the calculate_default_metrics() function when calculating metrics for each sample.
        mono_to_stereo: If True, combine every other channel in the batch into a stereo sample, i.e. (2B, T, 1) -> (B, T, 2).
        save_samples_to: If not None, a directory to save example audio files of the mixes, gts, and predictions. Will save as .flac files.
        save_num_samples: If save_samples_to is not None, the maximum number of samples to save to disk. If 0, do not save any.
        warm_up_iters: Number of iterations to run for warming up the model before measuring latency.
        mlflow_global_step: A counter to track the global step for MLflow logging.

    Returns:
        A tuple containing the individual sample results and the aggregated results.

    """

    if save_samples_to is not None:
        save_samples_to.mkdir(parents=True, exist_ok=True)
        assert save_samples_to.is_dir()

    model.eval()

    results = []

    samples_left_to_save = save_num_samples
    assert save_num_samples == 0 or save_samples_to is not None, (
        "If save_num_samples is greater than 0, save_samples_to must be provided."
    )
    has_wamed_up = False

    ground_truth_target = model.ground_truth_target

    log_to_mlflow = mlflow.active_run() is not None and mlflow_global_step is not None and split_name is not None
    mlflow_logger = CustomMlFlowLogger(allow_inactive=True)  # Allow it to do nothing if MLFlow is not active

    with mlflow_logger, torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Evaluating", unit=" batches"):
            if log_to_mlflow and (batch_idx % 1000 == 0 or batch_idx % 1000 == 1):
                _debug_to_mlflow(mlflow_logger, mlflow_global_step, device, prefix="val_")

            inputs = batch["inputs"]
            inputs["mix"] = inputs["mix"].to(device)
            inputs["label_vector"] = inputs["label_vector"].to(device)
            inputs["is_control"] = inputs["is_control"].to(device)
            batch_metrics = []

            # Warm up on the first round to get better latency measurements
            if has_wamed_up is False:
                _warm_up_model(model, inputs, num_iters=warm_up_iters)
                has_wamed_up = True

            # Run model and measure latency
            output, runtime_ms = _time_and_run_model(
                model,
                args=(inputs,),
                profiling=False,
            )

            batch = to_stereo_batch(batch) if mono_to_stereo else batch
            output = to_stereo_output(output) if mono_to_stereo else output
            output_items = output.items()

            batch_size = inputs["mix"].shape[0]

            for i in range(batch_size):
                sample_idx = batch["idxs"][i]
                valid_len = int(batch["audio_lens"][i].item())  # To remove padding
                mix_i = inputs["mix"][i, :, :valid_len]

                isolated_trigger_i = (
                    batch["isolated_trigger"][i, :, :valid_len] if "isolated_trigger" in batch else None
                )
                clean_mix_i = batch["clean_mix"][i, :, :valid_len] if "clean_mix" in batch else None

                sample_metdata = batch["metadata"][i] if "metadata" in batch else None
                sample_rate = sample_metdata.get("sample_rate", SAMPLE_RATE) if sample_metdata else SAMPLE_RATE

                if samples_left_to_save > 0:
                    save_sample = True
                    samples_left_to_save -= 1
                    mix_file = save_samples_to / f"sample_{sample_idx}_mix.flac"
                    clean_mix_file = (
                        save_samples_to / f"sample_{sample_idx}_clean_mix.flac" if clean_mix_i is not None else None
                    )
                    isolated_trigger_file = (
                        save_samples_to / f"sample_{sample_idx}_isolated_trigger.flac"
                        if isolated_trigger_i is not None
                        else None
                    )

                    _save_audio(mix_i, mix_file, sample_rate=sample_rate)
                    if clean_mix_i is not None:
                        _save_audio(clean_mix_i, clean_mix_file, sample_rate=sample_rate)
                    if isolated_trigger_i is not None:
                        _save_audio(isolated_trigger_i, isolated_trigger_file, sample_rate=sample_rate)
                else:
                    save_sample = False

                for pred_name, pred in output_items:
                    if skip_subtraction and pred_name != "x":
                        continue

                    pred_i = pred[i, :, :valid_len]

                    if pred_name == "x" and ground_truth_target == "isolated_trigger":  # is not subtracted
                        # precomputed_mix_metrics = (
                        #     sample_metdata.get("mix_vs_isolated_trigger_metrics", None) if sample_metdata else None
                        # )
                        # FIXME: Since we truncate in the dataloader now, we cannot use precomputed metrics
                        precomputed_mix_metrics = None
                        metrics = calculate_default_metrics(
                            pred_i.to(device),
                            isolated_trigger_i.to(device),
                            sample_rate=sample_rate,
                            mix=mix_i.to(device) if precomputed_mix_metrics is None else None,
                            mix_metrics=precomputed_mix_metrics,
                            loss_fn=loss_fn,
                            **(calculate_metrics_kwargs or {}),
                        )
                    else:  # Is subtracted
                        # precomputed_mix_metrics = (
                        #     sample_metdata.get("mix_vs_clean_mix_metrics", None) if sample_metdata else None
                        # )
                        # FIXME: Since we truncate in the dataloader now, we cannot use precomputed metrics
                        precomputed_mix_metrics = None
                        metrics = calculate_default_metrics(
                            pred_i.to(device),
                            clean_mix_i.to(device),
                            sample_rate=sample_rate,
                            mix=mix_i.to(device) if precomputed_mix_metrics is None else None,
                            mix_metrics=precomputed_mix_metrics,
                            loss_fn=loss_fn,
                            **(calculate_metrics_kwargs or {}),
                        )

                    if save_sample:
                        pred_file = save_samples_to / f"sample_{sample_idx}_{pred_name}.flac"
                        _save_audio(pred_i, pred_file, sample_rate=sample_rate)

                        sample_files = {
                            "mix_file": str(mix_file.name),
                            "clean_mix_file": str(clean_mix_file.name) if clean_mix_file else None,
                            "isolated_trigger_file": str(isolated_trigger_file.name) if isolated_trigger_file else None,
                            "pred_file": str(pred_file.name),
                        }
                    else:
                        sample_files = None

                    results.append(
                        {
                            "idx": sample_idx,
                            "pred_name": pred_name,
                            "runtime_ms": runtime_ms,
                            "metrics": metrics,
                            "batch_idx": batch_idx,
                            "batch_length": inputs["mix"].shape[-1],
                            "sample_length": valid_len,
                            "sample_metadata": sample_metdata,
                            "sample_files": sample_files,
                        }
                    )
                    if pred_name == "x":  # Only mlflow log the main prediction
                        batch_metrics.append(metrics)

            if log_to_mlflow:
                mlflow_global_step.increment()
                mlflow_logger.log_metrics(
                    {
                        "val/batch/si_snr_improvement": np.mean([m["si_snr_improvement"] for m in batch_metrics]),
                        "val/batch/si_snr": np.mean([m["si_snr"] for m in batch_metrics]),
                        "val/batch/snr_improvement": np.mean([m["snr_improvement"] for m in batch_metrics]),
                        "val/batch/snr": np.mean([m["snr"] for m in batch_metrics]),
                        "val/batch/loss": np.mean([m["loss"] for m in batch_metrics]),
                    },
                    step=mlflow_global_step.current,  # Batch step
                    synchronous=False,
                )

    eliot.log_message(f"Saving results to {save_results_to}", level="info")
    with save_results_to.open("w") as f:
        json.dump(results, f)

    if save_aggregated_results_to is None:
        return results, None

    eliot.log_message(f"Aggregating results and saving to {save_aggregated_results_to}", level="info")
    agg_res = aggregate_results(results, **(aggregated_results_kwargs or {}))
    with save_aggregated_results_to.open("w") as f:
        json.dump(agg_res, f, indent=4)
    return results, agg_res


class CustomMlFlowLogger:
    """
    Custom logger that only initilize the client once and keeps a queue to make batch requests to the MLflow server.

    NOTE: Not thread-safe.
    """

    def __init__(
        self,
        *,
        flush_queue_size: int = 512,
        flush_seconds: int = 30,
        allow_inactive: bool = True,
    ) -> None:
        # get current mlflow run
        active_run = mlflow.active_run()
        if active_run is None:
            if not allow_inactive:
                raise ValueError(
                    "No active MLflow run found. Please start an MLflow run before initializing CustomMlFlowLogger."
                )
            self._run_id = None
            return

        self._run_id = active_run.info.run_id
        self._client = mlflow.MlflowClient()
        self._queue = []
        self._flush_queue_size = flush_queue_size
        self._flush_seconds = flush_seconds
        self._last_flush_time = mlflow.utils.time.get_current_time_millis()

    def log_metrics(self, metrics: dict[str, float], step: int, *, synchronous: bool = False) -> None:
        if self._run_id is None:
            return

        timestamp = mlflow.utils.time.get_current_time_millis()
        metrics_arr = [
            mlflow.entities.Metric(
                key=key,
                value=value,
                timestamp=timestamp,
                step=step,
                run_id=self._run_id,
                model_id=None,
                dataset_name=None,
                dataset_digest=None,
            )
            for key, value in metrics.items()
        ]
        self._queue.extend(metrics_arr)

        if (
            len(self._queue) >= self._flush_queue_size
            or (timestamp - self._last_flush_time) >= self._flush_seconds * 1000
        ):
            self.flush(synchronous=synchronous, timestamp=timestamp)

    def flush(self, *, synchronous: bool = False, timestamp: int | None = None) -> None:
        if self._run_id is None:
            return

        self._last_flush_time = timestamp or mlflow.utils.time.get_current_time_millis()
        if len(self._queue) == 0:
            return

        self._client.log_batch(
            run_id=self._run_id,
            metrics=self._queue,
            params=[],
            tags=[],
            synchronous=synchronous,
        )
        self._queue = []

    def __enter__(self) -> "CustomMlFlowLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.flush(synchronous=True)


GroupSpec = str | Sequence[str]


_LEN_GROUP_RE = re.compile(r"^__len__\((?P<col>[^()]+)\)$")


def _parse_len_group_col(col: str) -> str | None:
    match = _LEN_GROUP_RE.match(col)
    if not match:
        return None
    return match.group("col")


def _group_len(value: Any) -> int:  # noqa: ANN401
    if value is None:
        return 0

    # Treat scalar strings as a single value, not character sequences.
    if isinstance(value, str):
        return 1

    if isinstance(value, Sized):
        return len(value)

    return 1


def aggregate_results(
    results: list[dict[str, object]],
    *,
    group_by: None | GroupSpec | Sequence[GroupSpec] = None,
) -> dict:
    """
    Aggregate the results from perform_eval into overall metrics.

    Examples:
        aggregate_results(results)

        aggregate_results(
            results,
            group_by=[
                ("fg_categories", "is_trigger"),
                ("__len__(fg_categories)",),
                "is_trigger",
            ],
        )

    Semantics:
        If fg_categories is ["a", "b"], it is treated as the exact group
        ("a", "b"). It is not counted in the "a" or "b" groups.
    """
    if not results:
        return {}

    rows = [
        {
            **result["metrics"],
            **(result.get("sample_metadata") or {}),
            "pred_name": result["pred_name"],
            "runtime_ms": result.get("runtime_ms"),
            "batch_length": result.get("batch_length"),
            "sample_length": result.get("sample_length"),
        }
        for result in results
    ]

    df = pd.DataFrame(rows)

    if "runtime_ms" in df.columns and "batch_length" in df.columns:
        df["runtime_ms_pr_length"] = df["runtime_ms"] / df["batch_length"]

    metric_keys = list(results[0]["metrics"].keys())

    agg_metrics = _agg_results_calc(df[metric_keys + ["pred_name"]])

    if group_by is None or len(group_by) == 0:
        return agg_metrics  # Return overall metrics without grouping

    # Group metrics by:
    metadata_keys = list((results[0].get("sample_metadata") or {}).keys())
    overlap = set(metric_keys).intersection(metadata_keys)
    if overlap:
        raise ValueError(f"Metric keys overlap with metadata keys: {overlap}")

    group_specs = _normalize_group_by(group_by)

    all_group_by_cols = {col for group in group_specs for col in group}

    derived_len_cols: dict[str, str] = {}
    plain_group_by_cols: set[str] = set()

    for col in all_group_by_cols:
        len_source_col = _parse_len_group_col(col)

        if len_source_col is None:
            plain_group_by_cols.add(col)
        else:
            derived_len_cols[col] = len_source_col

    missing_plain_keys = plain_group_by_cols.difference(metadata_keys)
    if missing_plain_keys:
        raise ValueError(f"Group by keys {missing_plain_keys} not found in metadata columns {metadata_keys}")

    missing_len_source_keys = set(derived_len_cols.values()).difference(metadata_keys)
    if missing_len_source_keys:
        raise ValueError(
            f"Group by __len__ source keys {missing_len_source_keys} not found in metadata columns {metadata_keys}"
        )

    for derived_col, source_col in derived_len_cols.items():
        df[derived_col] = df[source_col].map(_group_len)

    # Normalize list-valued / unhashable group columns.
    for col in all_group_by_cols:
        df[col] = df[col].map(_normalize_group_value)

    for group_cols in group_specs:
        for group_values, group_df in df.groupby(list(group_cols), dropna=False, sort=True):
            key = _format_group_key(group_cols, group_values)
            agg_metrics[key] = _agg_results_calc(group_df[metric_keys + ["pred_name"]])
            agg_metrics[key]["__count__"] = len(group_df)
            agg_metrics[key]["__grouped_by__"] = {
                _json_safe(col): _json_safe(group_values[i]) for i, col in enumerate(group_cols)
            }

    return agg_metrics


def _agg_results_calc(df: pd.DataFrame) -> dict:
    grouped = df.groupby("pred_name").agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    res = grouped.to_dict(orient="index")
    # apply _json_safe to all values in res
    for pred_name in res:
        for metric_stat in res[pred_name]:
            res[pred_name][metric_stat] = _json_safe(res[pred_name][metric_stat])
    return res


def _normalize_group_by(
    group_by: None | GroupSpec | Sequence[GroupSpec],
) -> tuple[tuple[str, ...], ...]:
    if group_by is None:
        return ()

    if isinstance(group_by, str):
        return ((group_by,),)

    # group_by=("fg_categories", "is_trigger")
    # Ambiguous: is this one compound group or two single groups?
    # I recommend requiring compound groups to be nested:
    # group_by=[("fg_categories", "is_trigger")]
    #
    # But for backward compatibility, treat a flat sequence of strings
    # as multiple single-column groupings.
    if all(isinstance(x, str) for x in group_by):
        return tuple((x,) for x in group_by)  # type: ignore[arg-type]

    out: list[tuple[str, ...]] = []
    for spec in group_by:  # type: ignore[union-attr]
        if isinstance(spec, str):
            out.append((spec,))
        else:
            out.append(tuple(spec))
    return tuple(out)


def _normalize_group_value(value: Any) -> Any:  # noqa: ANN401
    """
    Normalize group values so pandas can group by them.

    Important:
        ["a", "b"] is treated as the exact group ("a", "b"),
        not as membership in "a" and membership in "b".
    """
    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, set):
        return tuple(sorted(value))

    if isinstance(value, dict):
        return tuple(sorted(value.items()))

    return value


def _format_group_value(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, tuple):
        return ",".join(map(str, value))
    return str(value)


def _format_group_key(group_cols: tuple[str, ...], group_values: Any) -> str:  # noqa: ANN401
    if len(group_cols) == 1:
        group_values = (group_values,)
    elif not isinstance(group_values, tuple):
        group_values = (group_values,)

    parts = [f"{col}={_format_group_value(value)}" for col, value in zip(group_cols, group_values)]

    return "by:" + "&".join(parts)


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    if value is pd.NA:
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value)]

    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}

    return value


def _warm_up_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor], num_iters: int) -> None:
    """
    Run a few forward passes to warm up the model (e.g. for more accurate latency measurements).
    """
    model.eval()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(inputs)


def _save_audio(audio: torch.Tensor | np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Save audio tensor of shape [C, T] as wav.
    Assumes C is 1 or 2.
    """
    assert path.suffix == ".flac"
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()

    assert isinstance(audio, np.ndarray)
    assert len(audio.shape) == 2, f"Expected audio of shape [C, T], got {audio.shape}"
    assert audio.ndim in (1, 2), f"Expected audio with 1 or 2 channels, got {audio.shape}"

    audio = audio.T  # [T, C]
    sf.write(
        path,
        audio,
        samplerate=sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )


def mod_pad(x, chunk_size, pad):  # noqa: ANN202
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


def model_size(model) -> float:
    """
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_train_params / 1e6


def _time_and_run_model(model, args=(), kwargs={}, *, profiling: bool = False) -> tuple[torch.Tensor, float]:
    """
    Run a model while measuring the time taken for the forward pass. If `profiling` is True, also prints a detailed profiling report.

    Args:
        model: The PyTorch model to run.
        inputs: A dictionary of input tensors to pass to the model.
        profiling: Whether to print a detailed profiling report.

    Returns:
        A tuple of (model output, latency in milliseconds).

    """

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, acc_events=True) as prof:
        with record_function("model_inference"):
            output = model(*args, **kwargs)

    # Print profiling results
    if profiling:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    return output, prof.profiler.self_cpu_time_total / 1000


def get_git_sha() -> str | None:
    """Returns the current git commit SHA, or None if it cannot be determined."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        assert isinstance(sha, str) and len(sha) == 40, f"Unexpected git SHA format: {sha}"
        return sha
    except Exception:
        return None


##############
# Eval Utils #
##############


def compute_itd_torch(
    s_left: torch.Tensor,
    s_right: torch.Tensor,
    sr: int,
    t_max: int | None = None,
) -> torch.Tensor:
    """
    Estimate ITD in microseconds from two mono tensors.

    Inputs:
        s_left:  shape (T,)
        s_right: shape (T,)
        sr:      sampling rate
        t_max:   maximum lag in samples, e.g. int(round(1e-3 * sr))

    Returns:
        ITD in microseconds as a scalar tensor.

    Convention:
        Positive ITD means the right channel lags the left channel.
    """
    if s_left.ndim != 1 or s_right.ndim != 1:
        raise ValueError("compute_itd_torch expects 1D tensors with shape (T,).")

    if s_left.shape != s_right.shape:
        raise ValueError("Left and right signals must have the same shape.")

    T = s_left.shape[-1]  # noqa: N806

    if t_max is None:
        t_max = T - 1

    t_max = min(t_max, T - 1)

    # Remove DC offset to make correlation more stable.
    s_left = s_left - s_left.mean()
    s_right = s_right - s_right.mean()

    # Full linear cross-correlation via FFT.
    # Length of full correlation is 2T - 1.
    n_corr = 2 * T - 1
    n_fft = 1 << (n_corr - 1).bit_length()

    X_left = torch.fft.rfft(s_left, n=n_fft)  # noqa: N806
    X_right = torch.fft.rfft(s_right, n=n_fft)  # noqa: N806

    corr = torch.fft.irfft(X_left * torch.conj(X_right), n=n_fft)

    # Reorder FFT correlation output to match lags [-(T-1), ..., 0, ..., T-1].
    corr = torch.cat([corr[-(T - 1) :], corr[:T]])

    lags = torch.arange(
        -(T - 1),
        T,
        device=s_left.device,
        dtype=torch.long,
    )

    keep = torch.abs(lags) <= t_max
    corr = corr[keep]
    lags = lags[keep]

    lag = lags[torch.argmax(corr)]

    return lag.to(s_left.dtype) / sr * 1e6


def compute_ild_torch(
    s_left: torch.Tensor,
    s_right: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute ILD in dB.

    Inputs:
        s_left:  shape (T,)
        s_right: shape (T,)

    Returns:
        ILD in dB as a scalar tensor.

    Convention:
        Positive ILD means the left channel has higher energy than the right channel.
    """
    if s_left.ndim != 1 or s_right.ndim != 1:
        raise ValueError("compute_ild_torch expects 1D tensors with shape (T,).")

    if s_left.shape != s_right.shape:
        raise ValueError("Left and right signals must have the same shape.")

    e_left = torch.sum(s_left**2)
    e_right = torch.sum(s_right**2)

    return 10.0 * torch.log10((e_left + eps) / (e_right + eps))


def itd_diff_torch(
    s_est: torch.Tensor,
    s_gt: torch.Tensor,
    sr: int,
) -> float | None:
    """
    Compute absolute ITD error between estimate and ground truth.

    Inputs:
        s_est: shape (2, T)
        s_gt:  shape (2, T)

    Returns:
        Absolute ITD error in microseconds as a scalar tensor.
    """
    try:
        if s_est.ndim != 2 or s_gt.ndim != 2:
            raise ValueError("Expected inputs with shape (2, T).")

        if s_est.shape[0] != 2 or s_gt.shape[0] != 2:
            raise ValueError("Expected first dimension to contain left/right channels.")

        if s_est.shape != s_gt.shape:
            raise ValueError("Estimate and ground truth must have the same shape.")

        t_max = int(round(1e-3 * sr))

        itd_est = compute_itd_torch(s_est[0], s_est[1], sr, t_max)
        itd_gt = compute_itd_torch(s_gt[0], s_gt[1], sr, t_max)

        return torch.abs(itd_est - itd_gt).item()
    except Exception as e:
        eliot.log_message(f"Error computing ITD diff: {e}", level="error")
        return None


def ild_diff_torch(
    s_est: torch.Tensor,
    s_gt: torch.Tensor,
) -> float | None:
    """
    Compute absolute ILD error between estimate and ground truth.

    Inputs:
        s_est: shape (2, T)
        s_gt:  shape (2, T)

    Returns:
        Absolute ILD error in dB as a scalar tensor.
    """
    try:
        if s_est.ndim != 2 or s_gt.ndim != 2:
            raise ValueError("Expected inputs with shape (2, T).")

        if s_est.shape[0] != 2 or s_gt.shape[0] != 2:
            raise ValueError("Expected first dimension to contain left/right channels.")

        if s_est.shape != s_gt.shape:
            raise ValueError("Estimate and ground truth must have the same shape.")

        ild_est = compute_ild_torch(s_est[0], s_est[1])
        ild_gt = compute_ild_torch(s_gt[0], s_gt[1])

        return torch.abs(ild_est - ild_gt).item()
    except Exception as e:
        eliot.log_message(f"Error computing ILD diff: {e}", level="error")
        return None


# from scipy import signal

# def compute_itd(s_left, s_right, sr, t_max=None) -> float:
#     corr = signal.correlate(s_left, s_right)
#     corr /= np.max(corr)

#     mid = len(corr) // 2 + 1

#     cc = np.concatenate((corr[-mid:], corr[:mid]))

#     if t_max is not None:
#         cc = np.concatenate([cc[-t_max + 1 :], cc[: t_max + 1]])
#     else:
#         t_max = mid

#     tau = np.argmax(np.abs(cc))
#     tau -= t_max

#     return tau / sr * 1e6


# def compute_ild(s_left, s_right) -> np.ndarray:
#     sum_sq_left = np.sum(s_left**2, axis=-1)
#     sum_sq_right = np.sum(s_right**2, axis=-1)
#     return 10 * np.log10(sum_sq_left / sum_sq_right)


# def itd_diff(s_est: np.ndarray, s_gt: np.ndarray, sr: int) -> np.ndarray:
#     """
#     Computes the ITD error between model estimate and ground truth
#     input: (*, 2, T), (*, 2, T)
#     """
#     tmax = int(round(1e-3 * sr))
#     itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, tmax)
#     itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, tmax)
#     return np.abs(itd_est - itd_gt)


# def ild_diff(s_est: np.ndarray, s_gt: np.ndarray) -> np.ndarray:
#     """
#     Computes the ILD error between model estimate and ground truth
#     input: (*, 2, T), (*, 2, T)
#     """
#     ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
#     ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
#     return np.abs(ild_est - ild_gt)


def prepare_dir_or_file(target: Path, *, is_dir: bool, overwrite: bool) -> None:
    """Prepare a directory or file for writing. If it already exists, either overwrite it or raise an error based on the `overwrite` flag."""

    if target.exists():
        if is_dir and not target.is_dir():
            raise FileExistsError(f"A file already exists at {target}, but a directory is expected.")
        if not is_dir and not target.is_file():
            raise FileExistsError(f"A directory already exists at {target}, but a file is expected.")
        if is_dir and target.is_dir() and not any(target.iterdir()):
            pass  # Do nothing if the directory already exists and is empty
        else:
            if overwrite:
                if is_dir:
                    eliot.log_message(f"Deleting existing directory at {target}", level="warning")
                    for file in target.glob("*"):
                        file.unlink()
                else:
                    eliot.log_message(f"Overwriting existing file at {target}", level="warning")
                    target.unlink()
            else:
                raise FileExistsError(f"Directory already exists at {target}. Use --overwrite to overwrite.")

    if is_dir:
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)


def model_diffs(
    a: pydantic.BaseModel | dict,
    b: pydantic.BaseModel | dict,
    *,
    a_name: str = "a",
    b_name: str = "b",
) -> dict:
    """Recursively compute the differences between two pydantic models and return a dictionary of the differences."""
    a_dict = a.model_dump(mode="python") if isinstance(a, pydantic.BaseModel) else a
    b_dict = b.model_dump(mode="python") if isinstance(b, pydantic.BaseModel) else b
    diffs = {}
    for key in set(a_dict.keys()).union(b_dict.keys()):
        a_val = a_dict.get(key, "<MISSING>")
        b_val = b_dict.get(key, "<MISSING>")

        if isinstance(a_val, dict) and isinstance(b_val, dict):
            nested_diffs = model_diffs(a_val, b_val, a_name=a_name, b_name=b_name)
            if len(nested_diffs) > 0:
                diffs[key] = nested_diffs
        elif a_val != b_val:
            if isinstance(a_val, (list, tuple)) and isinstance(b_val, (list, tuple)):
                if len(a_val) == len(b_val) and all(x == y for x, y in zip(a_val, b_val)):
                    continue  # allow if all items are the same
            diffs[key] = {a_name: a_val, b_name: b_val}
    return diffs


def log_dataset_config_diffs(
    current: MisophoniaANCConfig | dict,
    preprocessed_file: Path,
    split: SplitT,
) -> None:
    """Log the differences between the current config and the config used to preprocess the dataset."""
    try:
        with preprocessed_file.open("r") as f:
            preprocessed_config = json.load(f)
    except:
        eliot.log_message(f"Could not load preprocessed config from {preprocessed_file}", level="error")
        return

    try:
        diffs = model_diffs(
            current.dataset_splits[split],
            preprocessed_config["config"]["dataset_splits"][split],
            a_name="current",
            b_name="preprocessed",
        )
        if len(diffs):
            pretty_diffs = json.dumps(diffs, indent=4)
            eliot.log_message(
                f"Config for {split} dataset has changed since preprocessing:\n{pretty_diffs}",
                level="warning",
            )
    except Exception as e:
        eliot.log_message(f"Error occurred while comparing dataset configs for {split}: {e}", level="error")


class SimpleCounter:
    """Simple class to keep track of a count that can be incremented. Useful for tracking global steps for MLflow logging, etc."""

    def __init__(self, start: int = 0) -> None:
        self._count = start

    def increment(self, n=1) -> None:
        """Increment the counter by n"""
        self._count += n

    @property
    def current(self) -> int:
        """The current count."""
        return self._count


def _debug_to_mlflow(
    mlflow_logger: CustomMlFlowLogger,
    step_counter: SimpleCounter,
    device: torch.device,
    prefix: str = "",
    **other_things: dict,
) -> None:
    if device == torch.device("cuda"):
        mlflow_logger.log_metrics(
            {
                f"debug/{prefix}batch_vram_allocated_gb": (torch.cuda.memory_allocated(device) / (1024**3)),
                f"debug/{prefix}batch_vram_reserved_gb": (torch.cuda.memory_reserved(device) / (1024**3)),
                f"debug/{prefix}batch_vram_free_gb": (
                    torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                )
                / (1024**3),
                f"debug/{prefix}batch_vram_total_gb": (
                    torch.cuda.get_device_properties(device).total_memory / (1024**3)
                ),
            },
            step=step_counter.current,
        )
    if len(other_things) > 0:
        mlflow_logger.log_metrics(
            {
                **{f"debug/{prefix}{k}": v for k, v in other_things.items()},
            },
            step=step_counter.current,
        )


#################################
# Spectrogram Visualization
#################################


def plot_average_spectrogram_by_trigger_category(
    model_dir: str,
    split: str,
    loader: wds.WebLoader,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    power: float = 2.0,
    max_length: int = 308700,
    device: str = "cpu",
    *,
    only_triggers: bool = True,
    find_average: bool = False,
) -> None:
    """
    Iterate over a WebLoader, group isolated trigger audio by trigger category,
    and plot the average spectrogram for each category.

    Assumes each batch contains:
        batch["isolated_trigger"]: Tensor shaped [B, C, T] or [B, T]
        batch["metadata"]: list[dict] or list[str] or dict of batched fields
    """
    num_classes = 8  # TODO: In future, don't hardcode this. Logic breaks if find_average is True and only_triggers is False.

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
    ).to(device)

    spec_sums = defaultdict(lambda: None)
    spec_counts = defaultdict(int)

    for batch in loader:
        isolated_trigger = batch["isolated_trigger"].to(device)

        # Shape: [B, T] -> [B, 1, T]
        if isolated_trigger.ndim == 2:
            isolated_trigger = isolated_trigger.unsqueeze(1)

        metadata = batch["metadata"]

        for x, meta in zip(isolated_trigger, metadata):
            if only_triggers and not meta.get("is_trigger", False):
                continue

            categories = meta.get("fg_categories", [])
            if isinstance(categories, str):
                categories = [categories]

            # Convert binaural audio to mono for one averaged spectrogram.
            x = x.mean(dim=0)

            if x.shape[0] < max_length:
                x = F.pad(x, (0, max_length - x.shape[0]))
            else:  # This should never happen but just as safeguard.
                x = x[:max_length]

            spec = spec_transform(x)  # [freq, time]

            for category in categories:
                category = str(category)

                if spec_sums[category] is None:
                    spec_sums[category] = spec.detach().clone()

                    spec_counts[category] = 1
                    if (
                        not find_average and len(spec_sums.keys()) == num_classes
                    ):  # We have all categories, no need to find more examples
                        break
                else:
                    if find_average:
                        spec_sums[category] += spec.detach()
                        spec_counts[category] += 1

    avg_specs = {
        category: spec_sums[category] / spec_counts[category] for category in spec_sums if spec_counts[category] > 0
    }

    _plot_avg_specs(model_dir, split, avg_specs, sample_rate=sample_rate, hop_length=hop_length, n_fft=n_fft, is_average=find_average)


def plot_average_spectogram_background(
    model_dir: str,
    split: str,
    loader: wds.WebLoader,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    power: float = 2.0,
    max_length: int = 308700,
    device: str = "cpu",
) -> None:
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
    ).to(device)

    spec_sums = None
    spec_counts = 0

    for batch in loader:
        if "clean_mix" not in batch:
            raise ValueError("Data loader was created without clean mix.")

        background = batch["clean_mix"].to(device)

        if background.ndim == 2:
            background = background.unsqueeze(1)

        for x in background:
            x = x.mean(dim=0)

            if x.shape[0] < max_length:
                x = F.pad(x, (0, max_length - x.shape[0]))
            else: # This should never happen but just as safeguard.
                x = x[:max_length]

            spec = spec_transform(x)

            if spec_sums is None:
                spec_sums = spec.detach().clone()
            else:
                spec_sums += spec.detach()

            spec_counts += 1

    if spec_counts == 0:
        raise ValueError("No background audio found in the dataset.")

    background_specs = {"background": spec_sums / spec_counts}
    _plot_avg_specs(model_dir, split, background_specs, sample_rate=sample_rate, hop_length=hop_length, n_fft=n_fft, is_background=True)


def _plot_avg_specs(
    model_dir: str,
    split: str,
    avg_specs: dict[str, torch.Tensor],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    *,
    is_average: bool = False,
    is_background: bool = False,
) -> None:
    """
    Plots and saves average spectrograms with real frequency/time axes.
    """

    n = len(avg_specs)
    if n == 0:
        print("No trigger categories found.")
        return

    os.makedirs(f"{model_dir}/spectrograms", exist_ok=True)

    items = []

    for category, spec in sorted(avg_specs.items()):
        display_category = "Chewing" if category == "chewing_gum" else category
        display_category = display_category.replace("_", " ").title()

        spec_db = torchaudio.functional.amplitude_to_DB(
            spec.cpu(),
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=0.0,
        )

        items.append((display_category, spec_db.numpy()))

    vmin = min(spec.min() for _, spec in items)
    vmax = max(spec.max() for _, spec in items)

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(20, 8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, (category, spec_db) in zip(axes, items):
        n_freqs, n_frames = spec_db.shape
        max_freq = sample_rate / 2
        duration_sec = (n_frames * hop_length) / sample_rate

        im = ax.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[0, duration_sec, 0, max_freq],
        )

        ax.set_title(category)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    # Hide unused subplots
    for ax in axes[len(items) :]:
        ax.axis("off")

    if is_average:
        fig.suptitle(f"Average spectrograms {split.title()}", fontsize=16)
    else:
        fig.suptitle(f"Example spectrograms: {split.title()}", fontsize=16)

    if not is_background:
        fig.subplots_adjust(right=0.88)
        fig.colorbar(
        im,
        ax=axes,
        label="dB",
        shrink=0.9,
        pad=0.04
        )
    else:
        fig.colorbar(im, ax=axes, label="dB", shrink=0.9)

    plt.tight_layout()

    filename = f"{model_dir}/spectrograms/{split}_avg_specs_grid.png" if not is_background else f"{model_dir}/spectrograms/{split}_background_spec.png"
    plt.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


# FIXME: Remove unused (commented out) function
# import numpy.fft as fft  # Semantic Hearinc also used mklfft, which is optimized for Intel
# import pyroomacoustics as pra  # Not installed, do we need it?
# import torch
# from scipy.fft import irfft, rfft


# def tdoa2(x1, x2, interp=1, fs=1, phat=True, t_max=None):
#     """
#     This function computes the time difference of arrival (TDOA)
#     of the signal at the two microphones. This in turns is used to infer
#     the direction of arrival (DOA) of the signal.
#     Specifically if s(k) is the signal at the reference microphone and
#     s_2(k) at the second microphone, then for signal arriving with DOA
#     theta we have
#     s_2(k) = s(k - tau)
#     with
#     tau = fs*d*sin(theta)/c
#     where d is the distance between the two microphones and c the speed of sound.
#     We recover tau using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
#     method. The reference is
#     Knapp, C., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay.
#     Parameters
#     ----------
#     x1 : nd-array
#         The signal of the reference microphone
#     x2 : nd-array
#         The signal of the second microphone
#     interp : int, optional (default 1)
#         The interpolation value for the cross-correlation, it can
#         improve the time resolution (and hence DOA resolution)
#     fs : int, optional (default 44100 Hz)
#         The sampling frequency of the input signal
#     Return
#     ------
#     theta : float
#         the angle of arrival (in radian (I think))
#     pwr : float
#         the magnitude of the maximum cross correlation coefficient
#     delay : float
#         the delay between the two microphones (in seconds)
#     """
#     # zero padded length for the FFT
#     n = x1.shape[-1] + x2.shape[-1] - 1
#     if n % 2 != 0:
#         n += 1

#     # Generalized Cross Correlation Phase Transform
#     # Used to find the delay between the two microphones
#     # up to line 71
#     X1 = fft.rfft(np.array(x1, dtype=np.float32), n=n, axis=-1)
#     X2 = fft.rfft(np.array(x2, dtype=np.float32), n=n, axis=-1)

#     if phat:
#         X1 /= np.abs(X1)
#         X2 /= np.abs(X2)

#     cc = fft.irfft(X1 * np.conj(X2), n=interp * n, axis=-1)

#     # maximum possible delay given distance between microphones

#     if t_max is None:
#         t_max = n // 2 + 1

#     # reorder the cross-correlation coefficients
#     cc = np.concatenate((cc[..., -t_max:], cc[..., :t_max]), axis=-1)

#     # import matplotlib.pyplot as plt

#     # t = np.arange(-t_max/fs, (t_max)/fs, 1/fs) * 1e6
#     # plt.plot(t, cc[15])
#     # plt.show()

#     # pick max cross correlation index as delay
#     tau = np.argmax(np.abs(cc), axis=-1)
#     tau -= t_max  # because zero time is at the center of the array

#     return tau / (fs * interp)


# from sklearn.utils.extmath import weighted_mode


# def framewise_gccphat(x, frame_dur, sr, window="tukey"):
#     TMAX = int(round(1e-3 * sr))
#     frame_width = int(round(frame_dur * sr))

#     # Total number of frames
#     T = 1 + (x.shape[-1] - 1) // frame_width

#     # Drop samples to get a multiple of frame size
#     if x.shape[-1] % T != 0:
#         x = x[..., -x.shape[-1] % T :]

#     assert x.shape[-1] % T == 0
#     frames = np.array(np.split(x, T, axis=-1))

#     window = signal.get_window(window, frame_width)
#     frames = frames * window

#     # Consider only frames that have energy above some threshold (ignore silence)
#     ENERGY_THRESHOLD = 5e-4
#     frame_energy = np.max(np.mean(frames**2, axis=-1) ** 0.5, axis=-1)
#     mask = frame_energy > ENERGY_THRESHOLD
#     frames = frames[mask]

#     fw_gccphat = tdoa2(frames[..., 0, :], frames[..., 1, :], fs=sr, t_max=TMAX)

#     # print(mask)
#     # print(fw_gccphat)
#     # print(frame_energy[mask])
#     itd = weighted_mode(fw_gccphat, frame_energy[mask], axis=-1)[0]
#     return itd[0]


# def fw_itd_diff(s_est, s_gt, sr, frame_duration=0.25):
#     """
#     Computes frame-wise delta ITD
#     """
#     # print("GT")
#     itd_gt = framewise_gccphat(s_gt, frame_duration, sr) * 1e6
#     # print("GT FW_ITD", itd_gt)
#     # print("EST")
#     itd_est = framewise_gccphat(s_est, frame_duration, sr) * 1e6
#     # print("EST FW_ITD", itd_est)
#     return np.abs(itd_est - itd_gt)


# def cal_interaural_error(predictions, targets, sr, debug=False):
#     """Compute ITD and ILD errors
#     input: (1, time, channel, speaker)
#     """

#     TMAX = int(round(1e-3 * sr))
#     EPS = 1e-8
#     s_target = targets[0]  # [T,E,C]
#     s_prediction = predictions[0]  # [T,E,C]

#     # ITD is computed with generalized cross-correlation phase transform (GCC-PHAT)
#     ITD_target = [
#         tdoa2(s_target[:, 0, i].cpu().numpy(), s_target[:, 1, i].cpu().numpy(), fs=sr, t_max=TMAX) * 10**6
#         for i in range(s_target.shape[-1])
#     ]
#     if debug:
#         print("TARGET ITD", ITD_target)

#     ITD_prediction = [
#         tdoa2(
#             s_prediction[:, 0, i].cpu().numpy(),
#             s_prediction[:, 1, i].cpu().numpy(),
#             fs=sr,
#             t_max=TMAX,
#         )
#         * 10**6
#         for i in range(s_prediction.shape[-1])
#     ]

#     if debug:
#         print("PREDICTED ITD", ITD_prediction)

#     ITD_error1 = np.mean(np.abs(np.array(ITD_target) - np.array(ITD_prediction)))
#     ITD_error2 = np.mean(np.abs(np.array(ITD_target) - np.array(ITD_prediction)[::-1]))
#     ITD_error = min(ITD_error1, ITD_error2)

#     # ILD  = 10 * log_10(||s_left||^2 / ||s_right||^2)
#     ILD_target_beforelog = torch.sum(s_target[:, 0] ** 2, dim=0) / (torch.sum(s_target[:, 1] ** 2, dim=0) + EPS)
#     ILD_target = 10 * torch.log10(ILD_target_beforelog + EPS)  # [C]
#     ILD_prediction_beforelog = torch.sum(s_prediction[:, 0] ** 2, dim=0) / (
#         torch.sum(s_prediction[:, 1] ** 2, dim=0) + EPS
#     )
#     ILD_prediction = 10 * torch.log10(ILD_prediction_beforelog + EPS)  # [C]

#     ILD_error1 = torch.mean(torch.abs(ILD_target - ILD_prediction))
#     ILD_error2 = torch.mean(torch.abs(ILD_target - ILD_prediction.flip(0)))
#     ILD_error = min(ILD_error1.item(), ILD_error2.item())

#     return ITD_error, ILD_error


# def compute_doa(mic_pos, s, sr, nfft=2048, num_sources=1):
#     # freq_range = [100, 20000]

#     X = pra.transform.stft.analysis(
#         s.T,
#         nfft,
#         nfft // 2,
#     )
#     X = X.transpose([2, 1, 0])

#     algo_names = ["SRP", "MUSIC", "FRIDA", "TOPS", "WAVES", "CSSM", "NormMUSIC"]

#     srp = pra.doa.algorithms["NormMUSIC"](mic_pos.T, sr, nfft, c=343, num_sources=num_sources)
#     srp.locate_sources(X)

#     values = srp.grid.values
#     phi = np.linspace(-np.pi, np.pi, 360)

#     values = np.roll(values, shift=180)

#     # plt.plot(phi * 180 / np.pi, values)
#     # plt.xlim([-90, 90])
#     # plt.show()

#     peak_idx = 90 + np.argmax(values[90:270])
#     return phi[peak_idx]


# def doa_diff(mic_pos, est, gt, sr):
#     doa_est = compute_doa(mic_pos, est, sr)
#     doa_gt = compute_doa(mic_pos, gt, sr)
#     return np.abs(doa_gt - doa_est)


# def gcc_phat(s_left, s_right, sr):
#     X = rfft(s_left)
#     Y = rfft(s_right)

#     Z = X * np.conj(Y)

#     y = irfft(np.exp(1j * np.angle(Z)))
#     center = (len(y) + 1) // 2
#     y = np.concatenate([y[center:], y[:center]])
#     lags = (np.linspace(0, len(y), len(y)) - ((len(y) + 1) / 2)) / sr
#     x = np.argmax(y)
#     tau = lags[x]

#     return lags, y


# def gcc_phat_diff(s_est, s_gt, sr):
#     TMAX = int(round(1e-3 * sr))
#     itd_est = tdoa2(s_est[..., 0, :], s_est[..., 1, :], fs=sr, t_max=TMAX)
#     itd_gt = tdoa2(s_gt[..., 0, :], s_gt[..., 1, :], fs=sr, t_max=TMAX)
#     return np.abs(itd_est - itd_gt) * 10**6


# def si_sdr(estimated_signal, reference_signals, scaling=True):
#     """
#     This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
#     or https://github.com/sigsep/bsseval/issues/3 for the motivation and
#     explanation
#     Input:
#         estimated_signal and reference signals are (N,) numpy arrays
#     Returns: SI-SDR as scalar
#     """

#     Rss = np.dot(reference_signals, reference_signals)
#     this_s = reference_signals

#     if scaling:
#         # get the scaling factor for clean sources
#         a = np.dot(this_s, estimated_signal) / Rss
#     else:
#         a = 1

#     e_true = a * this_s
#     e_res = estimated_signal - e_true

#     Sss = (e_true**2).sum()
#     Snn = (e_res**2).sum()

#     SDR = 10 * np.log10(Sss / Snn)

#     return SDR
