import os
import subprocess
from datetime import datetime
from pathlib import Path

import eliot
import mlflow  # type: ignore
import torch
import typer
from dotenv import load_dotenv  # type: ignore
from typing_extensions import Annotated

from misophonia_dataset._log import setup_print_logging
from misophonia_dataset.interface import SplitT, get_data_dir
from misophonia_dataset.main import get_dataset_from_name
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import (
    MisophoniaANCConfig,
    _save_audio_stereo,
    get_allocated_cpus,
    make_dataloader,
    preprocess_to_webdataset_pt,
    print_mem,
)
from .model import MisophoniaANCNet
from .train import train_model

setup_print_logging()
load_dotenv()
app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")

# torch.backends.cudnn.enabled = False # To resolve memory issue


@app.command()
def preprocess(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    split: Annotated[SplitT, typer.Argument(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    *,
    overwrite: Annotated[bool, typer.Option(..., help="Whether to overwrite existing preprocessed shards.")] = False,
    data_base_dir: Annotated[Path | None, typer.Option(..., help="Base directory to load preprocessed audio.")] = None,
    num_workers: Annotated[
        int, typer.Option(..., help="Number of workers for data loading.", default_factory=lambda: get_allocated_cpus())
    ],
    samples_per_shard: Annotated[int, typer.Option(..., help="Number of samples per .tar shard.")] = 64,
    no_progress: Annotated[bool, typer.Option(..., help="Whether to disable progress bars.")] = False,
) -> None:
    """
    Pre-process data so it is as expected for the Misophonia ANC model.

    Will create a WebDataset with .tar shards containing the preprocessed data.
    It will contain torch files that are efficient to load during training and evaluation.
    """
    model_dir = get_data_dir(dataset_name=name, base_dir=data_base_dir)

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml")

    split_config = config.dataset_splits[split]
    shards_dir = model_dir / "webdataset" / split

    if shards_dir.exists():
        if overwrite:
            eliot.log_message(f"Deleting existing shards in {shards_dir}...", level="warning")
            for file in shards_dir.glob("*"):
                file.unlink()
            eliot.log_message(f"Deleted existing shards in {shards_dir}.", level="info")
        else:
            eliot.log_message(
                f"Preprocessed dataset already exists for split {split} at {shards_dir}.", level="warning"
            )
            eliot.log_message("Use --overwrite to overwrite existing preprocessed dataset.", level="warning")
            return

    if split_config.from_premade:
        assert split_config.generated_config is None, "generated_config should not be provided if from_premade is given"
        dataset = PremadeMisophoniaDataset(name=split_config.from_premade, base_save_dir=data_base_dir)
        dataset_split = dataset.get_split(split)
    else:
        assert split_config.generated_source_data is not None, (
            "source_data must be provided if from_premade is not given"
        )
        assert split_config.generated_config is not None, (
            "generated_config must be provided if from_premade is not given"
        )

        eliot.log_message("Generating using:", level="debug")
        eliot.log_message(f"{split_config.generated_source_data=}", level="debug")
        eliot.log_message(f"{split_config.generated_config=}", level="debug")
        eliot.log_message(
            f"Using {num_workers} workers for data loading during generation (total CPU count = {os.cpu_count()}, allocated = {get_allocated_cpus()}).",
            level="debug",
        )
        source_data = tuple(
            get_dataset_from_name(name, base_dir=data_base_dir) for name in split_config.generated_source_data
        )
        dataset = GeneratedMisophoniaDataset(source_data=source_data)
        dataset_split = dataset.get_split(split, **split_config.generated_config)

    dataset_glob = preprocess_to_webdataset_pt(
        shards_dir,
        dataset_split,
        num_workers=num_workers,
        show_progress=not no_progress,
        samples_per_shard=samples_per_shard,
    )
    eliot.log_message(f"Saved preprocessed data to: {dataset_glob}", level="info")


@app.command()
def train(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    num_workers: Annotated[
        int,
        typer.Option(
            ...,
            help="Number of workers for data loading. A few will suffice, too many will consume extensive memory.",
            default_factory=lambda: 8,
        ),
    ],
    data_base_dir: Annotated[Path | None, typer.Option(..., help="Base directory to load preprocessed audio.")] = None,
    fast_data_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            help="Move the preprocessed data to this directory before training. Useful on HPCs. WARNING: Data in this dir will be deleted.",
            envvar="FAST_DATA_DIR",
        ),
    ] = None,
    mlflow_uri: Annotated[
        str | None, typer.Option(..., help="MLflow tracking URI.", envvar="MLFLOW_TRACKING_URI")
    ] = None,
    mlflow_username: Annotated[
        str | None, typer.Option(..., help="MLflow tracking username.", envvar="MLFLOW_TRACKING_USERNAME")
    ] = None,
    mlflow_password: Annotated[
        str | None, typer.Option(..., help="MLflow tracking password.", envvar="MLFLOW_TRACKING_PASSWORD")
    ] = None,
) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
    print_mem("start")
    model_dir = get_data_dir(dataset_name=name, base_dir=data_base_dir)

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml")

    dataset_dir = model_dir / "webdataset"
    if fast_data_dir is not None:
        dataset_dir_orig = dataset_dir
        dataset_dir = Path(fast_data_dir) / name / "webdataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        eliot.log_message(
            f"Copying preprocessed data from {dataset_dir_orig} to {dataset_dir} using rsync... May take a while for nodes that have not yet loaded the data into the fast data dir.",
            level="info",
        )
        subprocess.run(["rsync", "-a", "--delete", str(dataset_dir_orig) + "/", str(dataset_dir) + "/"], check=True)
        eliot.log_message(f"Copied preprocessed data to {dataset_dir}.", level="debug")

    shards_train = (dataset_dir / "train").glob("data-*.tar")
    shards_val = (dataset_dir / "val").glob("data-*.tar")

    train_loader = make_dataloader(shards_train, batch_size=config.batch_size, num_workers=num_workers)
    val_loader = make_dataloader(shards_val, batch_size=config.batch_size, num_workers=num_workers)

    model = MisophoniaANCNet(**config.model_params)  # noqa: F841
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eliot.log_message(f"Using device: {device}", level="debug")

    if mlflow_uri is not None and config.mlflow_experiment is not None:
        if mlflow_username is None or mlflow_password is None:
            raise ValueError("MLflow username and password must be provided if MLflow URI is provided.")
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(config.mlflow_experiment)

            hostname = os.uname().nodename
            run_name = f"Train {name} on {hostname} with {device} at {datetime.now().isoformat()}"
            mlflow.start_run(run_name=run_name)

            mlflow.log_params(config.dict())

            run_link = f"{mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_experiment_by_name(config.mlflow_experiment).experiment_id}/runs/{mlflow.get_run(mlflow.active_run().info.run_id).info.run_id}"
            eliot.log_message(f"Started MLflow run with name '{run_name}': {run_link}", level="info")

    try:
        train_model(
            model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config.num_epochs,
            save_dir=Path(model_dir),
            **config.model_hyperparams,
        )
    finally:
        if mlflow_uri is not None:
            mlflow.end_run()


def infer(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    split: Annotated[SplitT, typer.Argument(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    *,
    checkpoint: Annotated[str, typer.Argument(..., help="Name of model checkpoint to load.")] = "best_weights.pt",
    num_samples: Annotated[
        int | None,
        typer.Option(..., help="Number of samples to examine model output"),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            ...,
            help="Number of workers for data loading. A few will suffice, too many will consume extensive memory.",
        ),
    ] = 2,
    data_base_dir: Annotated[Path | None, typer.Option(..., help="Base directory to load preprocessed audio.")] = None,
) -> None:
    """
    Function to compare sample gts and mixes to model outputs.
    """
    model_dir = get_data_dir(dataset_name=name, base_dir=data_base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_dir = model_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = model_dir / checkpoint
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file cannot be found at {checkpoint_file}")

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml")
    model = MisophoniaANCNet(**config.model_params)
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset_dir = model_dir / "webdataset"
    shards_test = sorted((dataset_dir / "test").glob("data-*.tar"))
    if not shards_test:
        raise FileNotFoundError(f"No test shards found in {dataset_dir / 'test'}")

    test_loader = make_dataloader(files=shards_test, batch_size=1, num_workers=num_workers)

    with torch.no_grad():
        for idx, (input, gt, audio_len) in enumerate(test_loader):
            if idx >= num_samples:
                break

            inputs = {k: v.to(device) for k, v in input.items()}
            gt = gt.to(device)

            output = model(inputs)
            pred = output["x"]

            valid_len = int(audio_len[0].item())
            gt_i = gt[0, :, :valid_len]
            pred_i = pred[0, :, :valid_len]
            mix_i = inputs["mix"][0, :, :valid_len]

            _save_audio_stereo(mix_i, samples_dir / f"sample_{idx:03d}_mix.wav")
            _save_audio_stereo(gt_i, samples_dir / f"sample_{idx:03d}_gt.wav")
            _save_audio_stereo(pred_i, samples_dir / f"sample_{idx:03d}_pred.wav")

    eliot.log_message(f"Saved {num_samples} samples to {samples_dir}", level="debug")


if __name__ == "__main__":
    app()
