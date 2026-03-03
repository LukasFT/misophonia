import glob
from pathlib import Path

import torch
import typer
import webdataset as wds  # noqa: F401
from typing_extensions import Annotated

from misophonia_dataset.interface import SplitT, get_data_dir
from misophonia_dataset.main import get_dataset_from_name, get_default_datasets_names
from misophonia_dataset.misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset

from ._utils import MisophoniaANCConfig, get_shards_dir, preprocess_to_webdataset_pt
from .model import MisophoniaANCNet
from .train import train_model

# from ._train_eval_utils import ... # TODO: Import actual functions needed for training and evaluation

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def preprocess(
    config_file: Annotated[str, typer.Argument(..., help="path to config file with dataset parameters.")],
    split: Annotated[SplitT, typer.Option(..., help="Dataset split to generate (e.g., 'train', 'val', 'test')")],
    overwrite: Annotated[bool, typer.Option(..., help="Whether to overwrite existing preprocessed shards.")] = False,  # noqa: FBT002
) -> None:
    """
    Pre-process data so it is as expected for the Misophonia ANC model.

    Will create a WebDataset with .tar shards containing the preprocessed data.
    It will contain torch files that are efficient to load during training and evaluation.
    """
    config = MisophoniaANCConfig.from_yaml(config_file)
    split_config = config.dataset_splits[split]
    shards_dir = get_shards_dir(config, split)

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
        dataset = PremadeMisophoniaDataset(name=config.dataset_name, base_save_dir=config.dataset_base_save_dir)
        dataset_split = dataset.get_split(split)
    else:
        assert split_config.generated_source_data is not None, "source_data must be provided if from_premade is False"
        assert split_config.generated_config is not None, "generated_config must be provided if from_premade is False"

        source_data = tuple(
            get_dataset_from_name(name, base_dir=config.dataset_base_save_dir)
            for name in split_config.generated_source_data
        )
        dataset = GeneratedMisophoniaDataset(source_data=source_data)
        dataset_split = dataset.get_split(split, **split_config.generated_config)

    dataset_glob = preprocess_to_webdataset_pt(
        shards_dir,
        dataset_split,
        num_workers=config.num_workers,
    )
    print(f"Saved preprocessed data to: {dataset_glob}")


@app.command()
def train(
    config_file: Annotated[str, typer.Argument(..., help="path to config file with training parameters.")],
) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
    config = MisophoniaANCConfig.from_yaml(config_file)

    train_glob = glob.glob(str(get_shards_dir(config, "train") / "data-*.tar"))
    train_data = (
        wds.WebDataset(train_glob)
        .shuffle(1000)  # optional
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
        .batched(config.batch_size)
    )

    train_loader = wds.WebLoader(train_data, batch_size=None, num_workers=config.num_workers)

    model = MisophoniaANCNet(**config.model_params)  # noqa: F841
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model,
        device=device,
        train_loader=train_loader,
        batch_size=config.batch_size,
        n_epochs=config.num_epochs,
        num_workers=config.num_workers,
        log_dir=None,
    )


@app.command()
def evaluate(
    some_param: int = typer.Option(..., help="Some parameter for evaluation"),  # TODO: Add actual parameters
) -> None:
    raise NotImplementedError("Evaluation function not implemented yet")


if __name__ == "__main__":
    app()
