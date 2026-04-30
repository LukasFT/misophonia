"""A collection of useful helper functions"""

# ruff: noqa: ANN001 # FIXME: Improve quality

import json
import os
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

import eliot
import mlflow
import mlflow.entities
import mlflow.utils.time
import numpy as np
import pandas as pd
import pydantic
import soundfile as sf
import torch
import torch.nn.functional as F  # noqa: N812  # noqa: N812
import webdataset as wds
import yaml
from scipy import signal
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

    mlflow_experiment: str | None = pydantic.Field(
        None, description="MLflow experiment name to log training metrics to."
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, *, defaults={}) -> "MisophoniaANCConfig":
        conf = dict(defaults)
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Cannot load config since file does not exist: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        conf.update(data)
        return cls(**conf)


def make_custom_collate_fn(
    *,
    include_metadata: bool,
    include_isolated_trigger: bool,
    include_clean_mix: bool,
    max_length: int | None = None,
) -> callable:
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
) -> dict[str, float]:
    si_snr_both = si_snr(preds, target)
    snr_both = snr(preds, target)

    # FIXME: ild and itd encounter divide by zero etc.
    # # FIXME: Improve efficiency by implementing these functions using torch operations
    # preds_np, target_np = preds.cpu().numpy(), target.cpu().numpy()
    # ild = ild_diff(preds_np, target_np)
    # itd = itd_diff(preds_np, target_np, sr=sample_rate)

    metrics = {
        "si_snr": si_snr_both.mean().item(),
        "snr": snr_both.mean().item(),
        "si_snr_left": si_snr_both[..., 0].mean().item(),
        "si_snr_right": si_snr_both[..., 1].mean().item(),
        "snr_left": snr_both[..., 0].mean().item(),
        "snr_right": snr_both[..., 1].mean().item(),
        # "ild": ild,
        # "itd": itd,
    }

    if loss_fn is not None:
        # Call loss function like it is a batch
        audio_lens = (target.shape[-1],)
        loss = loss_fn(preds.unsqueeze(0), target.unsqueeze(0), audio_lens)
        metrics["loss"] = loss.item()

    if mix is not None and mix_metrics is None:
        mix_metrics = calculate_default_metrics(mix, target, sample_rate=sample_rate)

    if mix_metrics is not None:
        if "snr" in mix_metrics:
            metrics["snr_improvement"] = metrics["snr"] - mix_metrics["snr"]
        if "si_snr" in mix_metrics:
            metrics["si_snr_improvement"] = metrics["si_snr"] - mix_metrics["si_snr"]
        if "snr_left" in mix_metrics and "snr_right" in mix_metrics:
            metrics["snr_improvement_left"] = metrics["snr_left"] - mix_metrics["snr_left"]
            metrics["snr_improvement_right"] = metrics["snr_right"] - mix_metrics["snr_right"]
        if "si_snr_left" in mix_metrics and "si_snr_right" in mix_metrics:
            metrics["si_snr_improvement_left"] = metrics["si_snr_left"] - mix_metrics["si_snr_left"]
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
    mlflow_logger = CustomMlFlowLogger()

    with mlflow_logger, torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Evaluating", unit=" batches"):
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

            batch_size = inputs["mix"].shape[0]
            output_items = output.items()
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

                    _save_audio_stereo(mix_i, mix_file, sample_rate=sample_rate)
                    if clean_mix_i is not None:
                        _save_audio_stereo(clean_mix_i, clean_mix_file, sample_rate=sample_rate)
                    if isolated_trigger_i is not None:
                        _save_audio_stereo(isolated_trigger_i, isolated_trigger_file, sample_rate=sample_rate)
                else:
                    save_sample = False

                for pred_name, pred in output_items:
                    if skip_subtraction and pred_name != "x":
                        continue

                    pred_i = pred[i, :, :valid_len]

                    if pred_name == "x" and ground_truth_target == "isolated_trigger":  # is not subtracted
                        precomputed_mix_metrics = (
                            sample_metdata.get("mix_vs_isolated_trigger_metrics", None) if sample_metdata else None
                        )
                        metrics = calculate_default_metrics(
                            pred_i.to(device),
                            isolated_trigger_i.to(device),
                            sample_rate=sample_rate,
                            mix=mix_i.to(device) if precomputed_mix_metrics is None else None,
                            mix_metrics=precomputed_mix_metrics,
                            loss_fn=loss_fn,
                        )
                    else:  # Is subtracted
                        precomputed_mix_metrics = (
                            sample_metdata.get("mix_vs_clean_mix_metrics", None) if sample_metdata else None
                        )
                        metrics = calculate_default_metrics(
                            pred_i.to(device),
                            clean_mix_i.to(device),
                            sample_rate=sample_rate,
                            mix=mix_i.to(device) if precomputed_mix_metrics is None else None,
                            mix_metrics=precomputed_mix_metrics,
                            loss_fn=loss_fn,
                        )

                    if save_sample:
                        pred_file = save_samples_to / f"sample_{sample_idx}_{pred_name}.flac"
                        _save_audio_stereo(pred_i, pred_file, sample_rate=sample_rate)

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
    agg_res = aggregate_results(results)
    with save_aggregated_results_to.open("w") as f:
        json.dump(agg_res, f, indent=4)
    return results, agg_res


class CustomMlFlowLogger:
    """
    Custom logger that only initilize the client once and keeps a queue to make batch requests to the MLflow server.

    NOTE: Not thread-safe.
    """

    def __init__(self, *, flush_every: int = 512) -> None:
        # get current mlflow run
        active_run = mlflow.active_run()
        if active_run is None:
            raise ValueError(
                "No active MLflow run found. Please start an MLflow run before initializing CustomMlFlowLogger."
            )
        self._run_id = active_run.info.run_id
        self._client = mlflow.MlflowClient()
        self._queue = []
        self._flush_every = flush_every

    def log_metrics(self, metrics: dict[str, float], step: int, *, synchronous: bool = False) -> None:
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

        if len(self._queue) >= self._flush_every:
            self.flush(synchronous=synchronous)

    def flush(self, *, synchronous: bool = False) -> None:
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


def aggregate_results(results: list[dict[str, object]]) -> dict:
    """
    Aggregate the results from perform_eval into overall metrics.

    Args:
        results: A list of dictionaries containing metrics for each sample, as output by perform_eval.
    """
    df = pd.DataFrame(
        [
            {
                **result["metrics"],
                "pred_name": result["pred_name"],
                "runtime_ms": result.get("runtime_ms"),
                "batch_length": result.get("batch_length"),
                "sample_length": result.get("sample_length"),
            }
            for result in results
        ]
    )
    if "runtime_ms" in df.columns and "batch_length" in df.columns:
        df["runtime_ms_pr_length"] = (
            df["runtime_ms"] / df["batch_length"]
        )  # Model is run on batch-level, so normalize on that
    grouped = df.groupby("pred_name").agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    agg_metrics = grouped.to_dict(orient="index")
    return agg_metrics


def _warm_up_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor], num_iters: int) -> None:
    """
    Run a few forward passes to warm up the model (e.g. for more accurate latency measurements).
    """
    model.eval()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(inputs)


def _save_audio_stereo(audio: torch.Tensor | np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Save audio tensor of shape [C, T] as wav.
    Assumes C is 2.
    """
    assert path.suffix == ".flac"
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 2, f"Expected audio of shape [C, T], got {audio.shape}"

    # soundfile expects [T] or [T, C]
    if audio.ndim != 2:
        raise ValueError(f"Expected audio of shape [C, T], got {audio.shape}")

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


def compute_itd(s_left, s_right, sr, t_max=None) -> float:
    corr = signal.correlate(s_left, s_right)
    corr /= np.max(corr)

    mid = len(corr) // 2 + 1

    cc = np.concatenate((corr[-mid:], corr[:mid]))

    if t_max is not None:
        cc = np.concatenate([cc[-t_max + 1 :], cc[: t_max + 1]])
    else:
        t_max = mid

    tau = np.argmax(np.abs(cc))
    tau -= t_max

    return tau / sr * 1e6


def compute_ild(s_left, s_right) -> np.ndarray:
    sum_sq_left = np.sum(s_left**2, axis=-1)
    sum_sq_right = np.sum(s_right**2, axis=-1)
    return 10 * np.log10(sum_sq_left / sum_sq_right)


def itd_diff(s_est: np.ndarray, s_gt: np.ndarray, sr: int) -> np.ndarray:
    """
    Computes the ITD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    tmax = int(round(1e-3 * sr))
    itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, tmax)
    itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, tmax)
    return np.abs(itd_est - itd_gt)


def ild_diff(s_est: np.ndarray, s_gt: np.ndarray) -> np.ndarray:
    """
    Computes the ILD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
    return np.abs(ild_est - ild_gt)


def prepare_dir_or_file(target: Path, *, is_dir: bool, overwrite: bool) -> None:
    """Prepare a directory or file for writing. If it already exists, either overwrite it or raise an error based on the `overwrite` flag."""

    if target.exists():
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


# FIXME: Remove unused (commented out) functions

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
