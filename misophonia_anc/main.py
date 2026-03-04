import glob
import multiprocessing
import os
from pathlib import Path

import torch
import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_anc._train_eval_utils import custom_collate_fn
from misophonia_dataset.interface import SplitT
from misophonia_dataset.main import _get_default_datasets
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import load_config, preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import train_model

# from ._train_eval_utils import ... # TODO: Import actual functions needed for training and evaluation

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def preprocess(
    split_name: Annotated[SplitT, typer.Option(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    name: Annotated[str, typer.Option(..., help="Name of the dataset to preprocess")] = "demo-v1",
    base_dir: Annotated[Path, typer.Option(..., help="Base directory to load audio if using Premade dataset")] = None,
    source_data: Annotated[list[str], typer.Option(..., help="Name of source datasets to mix.")] = None,
    num_samples: Annotated[int, typer.Option(..., help="Number of samples to generate")] = 10,
    save_dir: Annotated[Path, typer.Option(..., help="Directory to save preprocessed audio")] = None,
    num_workers: Annotated[int, typer.Option(..., help="Number of workers for parallel processing")] = 8,
) -> None:

    if base_dir is None and source_data is None:
        raise ValueError("Either a premade dataset name or source data must be provided.")
    num_workers = min(num_workers, multiprocessing.cpu_count())

    if base_dir is not None:
        print(f"Using premade dataset from {base_dir} with name {name} and split {split_name}")
        misophonia_dataset = PremadeMisophoniaDataset(name=name, base_save_dir=base_dir)
        out_dir = Path(os.path.join(base_dir, name, split_name))

        split = misophonia_dataset.get_split(split_name)

    else:
        datasets = _get_default_datasets() if source_data is None or len(source_data) == 0 else source_data
        print("Generating dataset using source datasets:", datasets)
        misophonia_dataset = GeneratedMisophoniaDataset(source_data=datasets)
        out_dir = Path(os.path.join(save_dir, name, split_name))

        split = misophonia_dataset.get_split(split_name, num_samples=num_samples)

    out_dir = preprocess_to_webdataset_pt(
        out_dir,
        split,
        num_workers=num_workers,
    )

    print(out_dir)


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
        wds.WebDataset(shard_glob, shardshuffle=False)  # TODO: Consider enabling shard shuffling for better training
        .shuffle(1000)  # optional
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
    )

    train_loader = wds.WebLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate_fn
    )

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
