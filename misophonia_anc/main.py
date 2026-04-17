import itertools
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import eliot
import mlflow
import torch
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from misophonia_dataset._log import setup_print_logging
from misophonia_dataset.interface import get_data_dir
from misophonia_dataset.main import get_dataset_from_name
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import (
    MisophoniaANCConfig,
    get_allocated_cpus,
    get_git_sha,
    log_dataset_config_diffs,
    make_dataloader,
    perform_eval,
    prepare_dir_or_file,
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
    *,
    splits: Annotated[
        list[str],  # list[SplitT] but that causes typer error
        typer.Option(
            ...,
            "--split",
            help="Dataset split to generate (e.g., 'train', 'val', 'test')",
        ),
    ] = ["train", "val", "test"],
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

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml", defaults={"mlflow_experiment": name})

    for split in splits:
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

        metadata = {
            "git_sha": get_git_sha(),
            "timestamp": datetime.now().isoformat(),
            "split": split,
            "name": name,
            "samples_per_shard": samples_per_shard,
            "config": config.model_dump(mode="json", round_trip=True),
        }

        if split_config.from_premade:
            assert split_config.generated_config is None, (
                "generated_config should not be provided if from_premade is given"
            )
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
            metadata=metadata,
        )
        eliot.log_message(f"Saved preprocessed data to: {dataset_glob}", level="info")

    eliot.log_message(f"Completed preprocessing for splits: {splits}", level="info")


@app.command()
def train(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    *,
    checkpoint: Annotated[
        str | None,
        typer.Option(
            ...,
            help="Name of model checkpoint file to load (e.g. 'weights_epoch_1.pt'). If 'init' or none given, a random untrained model will be used.",
        ),
    ] = None,
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

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml", defaults={"mlflow_experiment": name})

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

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    shards_train = train_dir.glob("data-*.tar")
    shards_val = val_dir.glob("data-*.tar")
    log_dataset_config_diffs(config, val_dir / "metadata.json", "val")
    log_dataset_config_diffs(config, train_dir / "metadata.json", "train")

    train_loader = make_dataloader(shards_train, batch_size=config.batch_size, num_workers=num_workers)
    val_loader = make_dataloader(shards_val, batch_size=config.batch_size, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eliot.log_message(f"Using device: {device}", level="debug")
    checkpoint = None if (checkpoint == "init" or checkpoint is None) else model_dir / "checkpoints" / checkpoint
    if checkpoint is None:
        eliot.log_message("No checkpoint provided. Initializing random model.", level="info")
    else:
        eliot.log_message(f"Loading model from checkpoint: {checkpoint}", level="info")
    model, checkpoint_metadata = MisophoniaANCNet.from_config(config, checkpoint=checkpoint, device=device)

    if mlflow_uri is not None and config.mlflow_experiment is not None:
        if mlflow_username is None or mlflow_password is None:
            raise ValueError("MLflow username and password must be provided if MLflow URI is provided.")
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(config.mlflow_experiment)

            mlflow_existing_id = checkpoint_metadata.get("mlflow_run_id", None)
            mlflow.start_run(
                run_id=mlflow_existing_id,  # If resuming from checkpoint, continue the same MLflow run
                run_name=f"Train at {datetime.now().isoformat()}",  # Name if starting new run
            )

            mlflow_artifact = f"parameters_{datetime.now().isoformat()}.json"
            mlflow_parameters = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint_metadata": checkpoint_metadata,
                "git_sha": get_git_sha(),
                "command": sys.argv,
                "hostname": os.uname().nodename,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "config": config.model_dump(mode="json", round_trip=True),
            }
            mlflow.log_dict(mlflow_parameters, mlflow_artifact)

            if mlflow_existing_id is None:
                mlflow.log_params(mlflow_parameters["config"])
            else:
                mlflow.set_tag("resumed", True)  # noqa: FBT003
                eliot.log_message(
                    f"Not updating MLflow parameters. Check {mlflow_artifact} artifact for checkpointed parameters.",
                    level="debug",
                )

            run_link = f"{mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_experiment_by_name(config.mlflow_experiment).experiment_id}/runs/{mlflow.get_run(mlflow.active_run().info.run_id).info.run_id}"
            run_name = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get(
                "mlflow.runName", "Unknown Run Name"
            )
            eliot.log_message(f"Tracking using MLflow '{run_name}': {run_link}", level="info")

    try:
        assert config.loss_option in ["time", "frequency", "combined"], (
            "Invalid loss option. Must be 'time', 'frequency' or 'combined'."
        )
        train_model(
            model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config.num_epochs,
            checkpoint_epoch=checkpoint_metadata.get("epoch", 0),
            loss_option=config.loss_option,
            save_dir=Path(model_dir),
            global_step_train=checkpoint_metadata.get("global_step_train", 0),
            global_step_val=checkpoint_metadata.get("global_step_val", 0),
            **config.model_hyperparams,
        )
    finally:
        if mlflow_uri is not None:
            mlflow.end_run()


@app.command()
def evaluate(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    *,
    splits: Annotated[
        list[str],  # list[SplitT] but that causes typer error
        typer.Option(
            ...,
            "--split",
            help="Dataset split to generate (e.g., 'train', 'val', 'test')",
        ),
    ] = ["train", "val", "test"],
    checkpoint: Annotated[
        str,
        typer.Option(..., help="Name of model checkpoint to load. If 'init', a random untrained model will be used."),
    ] = "best_weights.pt",
    overwrite: Annotated[bool, typer.Option(..., help="Whether to overwrite existing samples.")] = False,
    limit_samples: Annotated[
        int | None,
        typer.Option(..., help="Number of samples to examine model output"),
    ] = None,
    save_samples: Annotated[int, typer.Option(..., help="Number of examples to save to disk.")] = 50,
    batch_size: Annotated[int, typer.Option(..., help="Batch size for evaluation.")] = 4,
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

    for split in splits:
        checkpoint_without_pt = checkpoint.replace(".pt", "")

        filename_prefix = (
            f"{checkpoint_without_pt}_{split}{f'_{limit_samples}samples' if limit_samples is not None else ''}"
        )
        results_file = model_dir / "eval_results" / f"{filename_prefix}_results.json"
        aggregated_results_file = model_dir / "eval_results" / f"{filename_prefix}_aggregated_results.json"
        prepare_dir_or_file(results_file, overwrite=overwrite, is_dir=False)
        prepare_dir_or_file(aggregated_results_file, overwrite=True, is_dir=False)

        if save_samples == 0:
            samples_dir = None
        else:
            samples_dir = model_dir / "samples" / checkpoint_without_pt / split
            prepare_dir_or_file(samples_dir, overwrite=overwrite, is_dir=True)

        checkpoint_file = model_dir / "checkpoints" / checkpoint
        if checkpoint == "init":
            checkpoint_file = None
            eliot.log_message("Using random untrained model for inference.", level="info")

        config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml", defaults={"mlflow_experiment": name})
        model, _ = MisophoniaANCNet.from_config(config, checkpoint=checkpoint_file, device=device)
        model.eval()

        dataset_split_dir = model_dir / "webdataset" / split
        eliot.log_message(f"Loading {split} data from {dataset_split_dir}", level="debug")
        shards_split = dataset_split_dir.glob("data-*.tar")
        log_dataset_config_diffs(config, dataset_split_dir / "metadata.json", split)
        split_loader = make_dataloader(
            shards_split,
            batch_size=batch_size,
            num_workers=num_workers,
            include_metadata=True,
        )
        if limit_samples is not None:
            split_loader = itertools.islice(split_loader, math.ceil(limit_samples / batch_size))

        res, _ = perform_eval(
            model,
            split_loader,
            save_results_to=results_file,
            save_aggregated_results_to=aggregated_results_file,
            save_num_samples=save_samples,
            save_samples_to=samples_dir,
            device=device,
            subtract_using=config.model_hyperparams.get("subtract_using", None),
        )

        eliot.log_message(f"Evaluated {len(res)} {split} samples", level="info")

    eliot.log_message(f"Completed evaluation for checkpoint {checkpoint} on splits: {splits}", level="info")


if __name__ == "__main__":
    app()
