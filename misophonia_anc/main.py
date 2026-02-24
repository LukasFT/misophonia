import glob
import multiprocessing
import os
from pathlib import Path

import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_dataset.interface import SplitT
from misophonia_dataset.main import _get_default_datasets
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import train_model

# from ._train_eval_utils import ... # TODO: Import actual functions needed for training and evaluation

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def preprocess(
    split_name: Annotated[SplitT, typer.Argument(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    name: Annotated[str, typer.Argument(..., help="Name of the dataset to preprocess")] = "demo-v1",
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
def train(
    shard_path: Annotated[str, typer.Option(..., help="Path to webdataset shards to load")],
    batch_size: Annotated[int, typer.Option(..., help="Batch size for training")] = 16,
) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
    if shard_path is None:
        raise ValueError("Path to webdataset shards must be provided for training.")
    shard_glob = glob.glob(shard_path)
    train_data = (
        wds.WebDataset(shard_glob)
        .shuffle(1000)  # optional
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
        # .batched(batch_size)
    )
    for mix_batch, gt_batch, label_batch in train_data:
        print("Mix batch shape:", mix_batch.shape)
        break
    # model = MisophoniaANCNet()  # noqa: F841
    # train(model)


@app.command()
def evaluate(
    some_param: int = typer.Option(..., help="Some parameter for evaluation"),  # TODO: Add actual parameters
) -> None:
    raise NotImplementedError("Evaluation function not implemented yet")


if __name__ == "__main__":
    app()
