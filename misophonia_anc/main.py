import glob
from pathlib import Path

import torch
import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_dataset.interface import SplitT, get_data_dir
from misophonia_dataset.main import get_dataset_from_name, get_default_datasets_names
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import load_config, preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import train_model

# from ._train_eval_utils import ... # TODO: Import actual functions needed for training and evaluation

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def preprocess(
    # General
    split_name: Annotated[SplitT, typer.Option(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    data_in_dir: Annotated[
        Path, typer.Option(..., help="Base directory to load audio if using Premade dataset")
    ] = None,
    save_dir: Annotated[
        Path, typer.Option(..., help="Directory to save preprocessed audio. Use data_in_dir if None")
    ] = None,
    num_workers: Annotated[
        int, typer.Option(..., help="Number of workers for parallel processing. Number of CPU cores if not given.")
    ] = None,
    name: Annotated[str, typer.Option(..., help="Name of the dataset to preprocess.")] = "demo-v1",
    overwrite: Annotated[bool, typer.Option(..., help="Whether to overwrite existing preprocessed shards.")] = False,  # noqa: FBT002
    # Generated
    source_data: Annotated[list[str], typer.Option(..., help="Name of source datasets to mix.")] = None,
    num_samples: Annotated[int, typer.Option(..., help="Number of samples to generate")] = 10,
) -> None:
    """
    Pre-process data so it is as expected for the Misophonia ANC model.

    Will create a WebDataset with .tar shards containing the preprocessed data.
    It will contain torch files that are efficient to load during training and evaluation.
    """
    data_in_dir = data_in_dir or get_data_dir()

    premade_dir = get_data_dir(dataset_name=name, base_dir=data_in_dir)
    if premade_dir.exists():
        print(f"Using premade dataset from {data_in_dir} with name {name} and split {split_name}")
        misophonia_dataset = PremadeMisophoniaDataset(name=name, base_save_dir=data_in_dir)
        split = misophonia_dataset.get_split(split_name)
    else:
        dataset_names = get_default_datasets_names() if source_data is None or len(source_data) == 0 else source_data
        datasets = tuple(get_dataset_from_name(name, base_dir=data_in_dir) for name in dataset_names)
        print("Generating dataset using source datasets:", datasets)
        misophonia_dataset = GeneratedMisophoniaDataset(source_data=datasets)
        split = misophonia_dataset.get_split(split_name, num_samples=num_samples)

    save_dir = Path(save_dir or data_in_dir)
    out_dir = save_dir / name / split_name
    try:
        dataset_glob = preprocess_to_webdataset_pt(
            out_dir,
            split,
            num_workers=num_workers,
            overwrite=overwrite,
        )
    except FileExistsError:
        print("File exists and --overwrite not set. Skipping preprocessing.")
        return

    print("Dataset glob: ", dataset_glob)


@app.command()
def train(config: Annotated[str, typer.Option(..., help="path to config file with training parameters.")]) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
    if config is None:
        raise ValueError("Please specify config file path for training.")

    config = load_config(config)

    shard_path = config.get("shard_path")
    num_epochs = config.get("train_params", {}).get("num_epochs", 10)
    batch_size = config.get("train_params", {}).get("batch_size", 1)
    num_workers = config.get("train_params", {}).get("num_workers", 0)
    model_params = config.get("model_params", {})

    if shard_path is None:
        raise ValueError("Path to webdataset shards must be provided for training.")
    shard_glob = glob.glob(shard_path)
    train_data = (
        wds.WebDataset(shard_glob)
        .shuffle(1000)  # optional
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
        .batched(batch_size)
    )

    train_loader = wds.WebLoader(train_data, batch_size=None, num_workers=num_workers)

    model = MisophoniaANCNet(**model_params)  # noqa: F841
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model,
        device=device,
        train_loader=train_loader,
        batch_size=batch_size,
        n_epochs=num_epochs,
        num_workers=num_workers,
        log_dir=None,
    )


@app.command()
def evaluate(
    some_param: int = typer.Option(..., help="Some parameter for evaluation"),  # TODO: Add actual parameters
) -> None:
    raise NotImplementedError("Evaluation function not implemented yet")


if __name__ == "__main__":
    app()
