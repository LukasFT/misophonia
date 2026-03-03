"""
Usage:
> > python -m misophonia_dataset.main --help

"""

# noqa: ANN201
from pathlib import Path

import eliot
import pandas as pd
import typer
from typing_extensions import Annotated

from ._binamix import download_sadie
from ._log import setup_print_logging
from .interface import SourceData, get_data_dir
from .misophonia_dataset import GeneratedMisophoniaDataset, PremadeMisophoniaDataset, add_experimental_pairs_to_dataset
from .source_data.esc50 import Esc50Dataset
from .source_data.foams import FoamsDataset
from .source_data.fsd50k import Fsd50kDataset

setup_print_logging()

app = typer.Typer(help="Misophonia Dataset CLI")


@app.command()
def generate(
    name: Annotated[Path, typer.Argument(help="Name of the generated dataset")],
    split: Annotated[str, typer.Argument(help="Dataset split to generate")],
    *,
    if_exists: Annotated[str, typer.Option("--if-exists", help="Action if dataset exists")] = "error",
    datasets: Annotated[
        list[str], typer.Option("--source-dataset", "-d", help="Name(s) of source datasets use.")
    ] = None,
    source_base_dir: Annotated[Path, typer.Option("--source-dir", help="Directory to load datasets from")] = None,
    base_save_dir: Annotated[Path, typer.Option("--save-dir", "-s", help="Directory to save dataset to")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", "-n", help="Number of samples to generate")] = 1,
    trig_to_ctrl: Annotated[float, typer.Option("--trig-to-ctrl", help="Ratio of trigger to control sounds")] = 0.5,
    min_fgs_pr_item: Annotated[int, typer.Option("--min-fgs-pr-item", help="Minimum foregrounds per item")] = 1,
    max_fgs_pr_item: Annotated[int, typer.Option("--max-fgs-pr-item", help="Maximum foregrounds per item")] = 1,
    min_bgs_pr_item: Annotated[int, typer.Option("--min-bgs-pr-item", help="Minimum backgrounds per item")] = 1,
    max_bgs_pr_item: Annotated[int, typer.Option("--max-bgs-pr-item", help="Maximum backgrounds per item")] = 3,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for sampling")] = 42,
    add_experimental_pairs: Annotated[
        bool,
        typer.Option("--add-experimental-pairs", help="Add pairs used for the experimental validation of the dataset"),
    ] = False,
) -> None:
    if if_exists not in ("error", "replace", "append"):
        raise ValueError(f"Invalid value for if_exists: {if_exists}")
    if add_experimental_pairs and split != "test":
        raise ValueError("Experimental pairs can only be added to the 'test' split.")

    saved_generated = PremadeMisophoniaDataset(name=name, base_save_dir=base_save_dir)

    if num_samples > 0:
        datasets = get_default_datasets_names() if datasets is None or len(datasets) == 0 else datasets
        datasets = tuple(get_dataset_from_name(name, base_dir=source_base_dir) for name in datasets)

        misophonia_dataset = GeneratedMisophoniaDataset(source_data=datasets)

        eliot.log_message("Preparing source data", level="info")
        misophonia_dataset.prepare()

        eliot.log_message(f"Generating and saving {split} items", level="info")

        saved_generated.save_split(
            misophonia_dataset.get_split(
                split=split,
                num_samples=num_samples,
                random_seed=seed,
                foregrounds_per_item=(min_fgs_pr_item, max_fgs_pr_item),
                backgrounds_per_item=(min_bgs_pr_item, max_bgs_pr_item),
                trig_to_control_ratio=trig_to_ctrl,
            ),
            if_exists=if_exists,
            show_progress=True,
        )

    if add_experimental_pairs:
        eliot.log_message("Adding experimental pairs to test split", level="info")
        add_experimental_pairs_to_dataset(saved_generated, seed=seed)


@app.command()
def download(
    datasets: Annotated[list[str], typer.Argument(help="Name(s) of datasets to download.")] = None,
    base_save_dir: Annotated[Path, typer.Option("--datasets-dir", "-s", help="Directory to save datasets")] = None,
) -> None:
    """Downloads specified datasets."""
    if datasets is None or len(datasets) == 0:
        datasets = get_default_datasets_names() + ("sadie",)

    eliot.log_message(f"Downloading datasets: {datasets}", level="debug")

    for dataset_name in datasets:
        if dataset_name.lower().strip() == "sadie":
            download_sadie()
            continue

        dataset = get_dataset_from_name(dataset_name, base_dir=base_save_dir)

        if dataset.is_downloaded():
            eliot.log_message(f"Dataset {dataset_name} is already downloaded. Skipping.", level="info")
            continue

        eliot.log_message(f"Downloading dataset: {dataset_name}", level="info")
        dataset.download_data()
        eliot.log_message(f"Finished downloading dataset: {dataset_name}", level="info")


@app.command()
def search_metadata(
    q: Annotated[str, typer.Argument(help="Search query string in pandas query format.")],
    datasets: Annotated[list[str], typer.Argument(help="Name(s) of datasets to download.")] = None,
    base_save_dir: Annotated[Path, typer.Option("--datasets-dir", "-s", help="Directory to save datasets")] = None,
) -> None:
    """Search metadata in specified datasets."""
    eliot.log_message(f"Searching for '{q}' in datasets: {datasets}", level="info")

    metadata = []
    for dataset_name in datasets:
        dataset = get_dataset_from_name(dataset_name, base_dir=base_save_dir)
        assert dataset.is_downloaded(), f"Dataset {dataset_name} is not downloaded yet."
        meta = dataset.get_metadata()
        metadata.append(meta)

    metadata = pd.concat(metadata, ignore_index=True)

    res = metadata.query(q)
    print(f"Found {len(res)} matching entries:")
    print(res)


def get_dataset_from_name(name: str, base_dir: Path) -> SourceData:
    name = name.lower().strip()
    if name == "foams":
        return FoamsDataset(save_dir=get_data_dir(dataset_name="FOAMS", base_dir=base_dir))
    elif name == "esc50":
        return Esc50Dataset(save_dir=get_data_dir(dataset_name="ESC50", base_dir=base_dir))
    elif name == "fsd50k":
        return Fsd50kDataset(save_dir=get_data_dir(dataset_name="FSD50K", base_dir=base_dir))
    raise ValueError(f"Unknown dataset name: {name}")


def get_default_datasets_names() -> tuple[str]:
    return ("foams", "esc50", "fsd50k")


if __name__ == "__main__":
    app()
