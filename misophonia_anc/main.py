import glob
import os
from pathlib import Path

import torch
import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_dataset.interface import SplitT, get_data_dir
from misophonia_dataset.main import get_dataset_from_name
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import MisophoniaANCConfig, preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import custom_collate_fn, train_model

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def preprocess(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    split: Annotated[SplitT, typer.Argument(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    *,
    overwrite: Annotated[bool, typer.Option(..., help="Whether to overwrite existing preprocessed shards.")] = False,
    data_base_dir: Annotated[Path | None, typer.Option(..., help="Base directory to load preprocessed audio.")] = None,
    num_workers: Annotated[
        int, typer.Option(..., help="Number of workers for data loading.", default_factory=lambda: os.cpu_count())
    ],
    samples_per_shard: Annotated[int, typer.Option(..., help="Number of samples per .tar shard.")] = 2048,
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
            print(f"Overwriting existing preprocessed dataset for split {split} with config: {split_config}")
            print(f"Deleting existing shards in {shards_dir}...")
            for file in shards_dir.glob("*"):
                file.unlink()
            print("Existing shards deleted.")
        else:
            print(f"Using existing preprocessed dataset for split {split} with config: {split_config}")
            print("Use --overwrite to overwrite existing preprocessed dataset.")
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
    print(f"Saved preprocessed data to: {dataset_glob}")


@app.command()
def train(
    name: Annotated[str, typer.Argument(..., help="Name of model directory.")],
    num_workers: Annotated[
        int, typer.Option(..., help="Number of workers for data loading.", default_factory=lambda: os.cpu_count())
    ],
    data_base_dir: Annotated[Path | None, typer.Option(..., help="Base directory to load preprocessed audio.")] = None,
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
    model_dir = get_data_dir(dataset_name=name, base_dir=data_base_dir)

    config = MisophoniaANCConfig.from_yaml(model_dir / "config.yaml")

    shards_glob_train = glob.glob(str(model_dir / "webdataset" / "train" / "data-*.tar"))

    train_data = (
        wds.WebDataset(
            shards_glob_train,
            empty_check=False,
            shardshuffle=25,  # Number of shards to keep in memory at the time (as I understand it)
        )
        .shuffle(1000)  # Number of samples to shuffle in memory at the time (as I understand it)
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
    train_model(
        model,
        device=device,
        train_loader=train_loader,
        n_epochs=config.num_epochs,
    )


if __name__ == "__main__":
    app()
