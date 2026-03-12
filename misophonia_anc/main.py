import glob
import os
import subprocess
from pathlib import Path

import eliot
import torch
import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_dataset._log import setup_print_logging
from misophonia_dataset.interface import SplitT, get_data_dir
from misophonia_dataset.main import get_dataset_from_name
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import MisophoniaANCConfig, get_allocated_cpus, preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import custom_collate_fn, train_model

setup_print_logging()
app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


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
        assert split_config.generated_source_data is None, (
            "generated_source_data should not be provided if from_premade is True"
        )
        assert split_config.generated_config is None, "generated_config should not be provided if from_premade is True"
        dataset = PremadeMisophoniaDataset(name=split_config.from_premade, base_save_dir=data_base_dir)
        dataset_split = dataset.get_split(split)
    else:
        assert split_config.generated_source_data is not None, "source_data must be provided if from_premade is False"
        assert split_config.generated_config is not None, "generated_config must be provided if from_premade is False"

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
            default_factory=lambda: 4,
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
) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
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

    shards_glob_train = glob.glob(str(dataset_dir / "train" / "data-*.tar"))
    eliot.log_message(f"Loading data from `{shards_glob_train[0]}` etc...", level="debug")
    eliot.log_message(
        f"Using {num_workers} workers loading WebDataset (total CPU count = {os.cpu_count()}, allocated = {get_allocated_cpus()}).",
        level="debug",
    )

    train_data = (
        wds.WebDataset(
            shards_glob_train,
            empty_check=False,
            shardshuffle=4,  # Number of shards to keep in memory at the time (as I understand it)
        )
        .shuffle(64)  # Number of samples to shuffle in memory at the time (as I understand it)
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
        .batched(
            config.batch_size,
            collation_fn=custom_collate_fn,  # Make batches of the same size
        )
    )

    train_loader = wds.WebLoader(
        train_data,
        batch_size=None,  # We set batch size in the WebDataset pipeline, so we set it to None here
        num_workers=num_workers,
    )

    model = MisophoniaANCNet(**config.model_params)  # noqa: F841
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eliot.log_message(f"Using device: {device}", level="debug")
    train_model(
        model,
        device=device,
        train_loader=train_loader,
        n_epochs=config.num_epochs,
    )


if __name__ == "__main__":
    app()
