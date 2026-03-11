import concurrent.futures
import itertools
import json
import os
import shutil
import uuid
import warnings
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Literal

import eliot
import numpy as np
import pandas as pd
import pydantic
import soundfile as sf
from tqdm import tqdm

from ._analysis import models_to_df
from .interface import (
    GlobalMixingParams,
    MisophoniaDataset,
    MisophoniaDatasetSplit,
    MisophoniaItem,
    SourceData,
    SourceDataItem,
    SplitT,
    get_data_dir,
)


class GeneratedMisophoniaDataset(MisophoniaDataset):
    """Mixed dataset that is generated on-the-fly for some given source datasets."""

    def __init__(self, source_data: Iterable[SourceData]) -> None:
        self._source_data = source_data
        self._items_by_split: dict[SplitT, list[SourceDataItem]] | None = None

    def prepare(self) -> None:
        if self._items_by_split is not None:
            return

        for ds in self._source_data:
            if not ds.is_downloaded():
                raise RuntimeError(
                    f"Source data {ds} is not downloaded. Please download it before preparing the dataset."
                )

        all_source_data = tuple(itertools.chain.from_iterable(ds.get_metadata() for ds in self._source_data))

        items_by_split: dict[SplitT, list[SourceDataItem]] = {"train": [], "val": [], "test": []}
        for item in all_source_data:
            items_by_split[item.split].append(item)

        self._items_by_split = items_by_split

    def get_split(
        self,
        split: SplitT,
        *,
        num_samples: int,
        foregrounds_per_item: tuple[int, int] = (1, 1),
        backgrounds_per_item: tuple[int, int] = (1, 3),
        trig_to_control_ratio: float = 0.5,
        random_seed: int = 42,
    ) -> MisophoniaDatasetSplit:
        """
        Return a split view for the dataset generated according to the specified options.

        Args:
            split: The dataset split to return. See SplitT for more details.
            num_samples: Number of samples to generate in this split.
            foregrounds_per_item: Tuple specifying the (min, max) number of foregrounds per mixed item.
            backgrounds_per_item: Tuple specifying the (min, max) number of backgrounds per mixed item.
            trig_to_control_ratio: Ratio of trigger to control sounds in the generated items.
                                        I.e, the percent of items that will contain a trigger sound rather than a control sound.
                                        Must be between 0 and 1.
            random_seed: Random seed for sampling.
                            Given the same seed, parameters, source data and code version, the same dataset will be generated.

        Returns:
            A MisophoniaDatasetSplit object representing the requested split. See MisophoniaDatasetSplit for more details.
        """
        from .mixing import binaural_mix, prepare_track_specs

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if split == "test":
            warnings.warn(
                """You are generating a new test dataset that is not the canonical version. """
                """Please do not use this for comparisons across different papers. """
                """See PremadeMisophoniaDataset for loading the canonical test dataset.""",
                UserWarning,
            )

        self.prepare()
        all_items: list[SourceDataItem] = self._items_by_split[split]

        all_trig_items = [it for it in all_items if it.label_type == "trigger"]
        all_ctrl_items = [it for it in all_items if it.label_type == "control"]
        all_bg_items = [it for it in all_items if it.label_type == "background"]

        def _make_sampling_plan():  # noqa: ANN202
            """
            Prepare sample indices and random seeds for dataset generation.

            Must be called in the main worker thread before any parallel generation starts.

            This means that the mixing can happen in parallel while still being reproducible.
            """
            rng_plan = np.random.default_rng(seed=random_seed)

            def make_idx_cycle(n: int):  # noqa: ANN202
                """Yield indices 0..n-1 in random order, then reshuffle and repeat."""
                order = np.arange(n)
                while True:
                    rng_plan.shuffle(order)
                    for idx in order:
                        yield int(idx)

            # Make cycles for each type of item to ensure we use all items before re-using any
            trig_cycle = make_idx_cycle(len(all_trig_items)) if all_trig_items else None
            ctrl_cycle = make_idx_cycle(len(all_ctrl_items)) if all_ctrl_items else None
            bg_cycle = make_idx_cycle(len(all_bg_items))

            if not trig_cycle and trig_to_control_ratio > 0:
                raise ValueError("No trigger items but trig_to_control_ratio > 0")
            if not ctrl_cycle and trig_to_control_ratio < 1:
                raise ValueError("No control items but trig_to_control_ratio < 1")

            is_trig_for_item: list[bool] = []
            fg_indices_for_item: list[list[int]] = []
            bg_indices_for_item: list[list[int]] = []
            seeds_for_item = rng_plan.integers(0, 2**32 - 1, size=num_samples, dtype=np.uint32)

            for _ in range(num_samples):
                num_fg = int(rng_plan.integers(foregrounds_per_item[0], foregrounds_per_item[1] + 1))
                num_bg = int(rng_plan.integers(backgrounds_per_item[0], backgrounds_per_item[1] + 1))

                # Decide trigger vs control
                is_trig = bool(rng_plan.random() < trig_to_control_ratio)
                is_trig_for_item.append(is_trig)

                # Foreground indices from the appropriate cycle
                fg_cycle = trig_cycle if is_trig else ctrl_cycle
                fg_indices = [next(fg_cycle) for _ in range(num_fg)]
                bg_indices = [next(bg_cycle) for _ in range(num_bg)]

                fg_indices_for_item.append(fg_indices)
                bg_indices_for_item.append(bg_indices)

            return is_trig_for_item, fg_indices_for_item, bg_indices_for_item, seeds_for_item

        is_trig_for_item, fg_indices_for_item, bg_indices_for_item, seeds_for_item = _make_sampling_plan()

        def _generate_one(index: int) -> MisophoniaItem:
            """Generate one mixed item."""

            # Use the the pre-computed sampling plan to ensure reproducability:
            rng = np.random.default_rng(int(seeds_for_item[index]))
            is_trig = is_trig_for_item[index]
            fg_idxs = fg_indices_for_item[index]
            bg_idxs = bg_indices_for_item[index]

            fg_pool = all_trig_items if is_trig else all_ctrl_items
            foreground_items = [fg_pool[j] for j in fg_idxs]
            background_items = [all_bg_items[j] for j in bg_idxs]

            # Generate the mixing specifications:
            global_params = GlobalMixingParams(_rng=rng)

            foreground_specs, background_specs = prepare_track_specs(  # Will also load the audio (I/O heavy)
                foreground_items,
                background_items,
                global_params=global_params,
                # Keep this the reference level for backgrounds.
                # In that way, the randomness in the foreground is always relative to the same background level.
                bg_track_options={"level": 0.7},
                rng=rng,
            )
            foreground_tracks, _ = tuple(zip(*foreground_specs))
            background_tracks, _ = tuple(zip(*background_specs))

            # Perform the mixing (heavy work happens here):
            mix, ground_truth = binaural_mix(
                fg_specs=foreground_specs,
                bg_specs=background_specs,
                global_params=global_params,
                is_trig=is_trig,
            )

            return MisophoniaItem(
                split=split,
                is_trigger=is_trig,
                mix=mix,
                ground_truth=ground_truth,
                length=mix.shape[1],
                global_mixing_params=global_params,
                foregrounds=foreground_tracks,
                backgrounds=background_tracks,
            )

        # Split view will call _generate_one as needed
        return MisophoniaDatasetSplit(
            split=split,
            num_samples=num_samples,
            get_one=_generate_one,
        )


class PremadeMisophoniaDataset(MisophoniaDataset):
    """Dataset that has been (or will be) pre-mixed and saved to disk."""

    def __init__(self, name: str, base_save_dir: Path | str | None = None) -> None:
        """
        Initialize a pre-made misophonia dataset that is stored on disk (or will be saved to disk).

        Args:
            name: Name of the dataset. Will be used to determine the save directory.
            base_save_dir: Base directory where the dataset is stored. Name and split is appended to this, e.g.:
                                base_save_dir / name / split / [... files ...]
                            See misophonia_dataset.interface.get_data_dir for details.
        """
        self.name = name
        self._base_save_dir = base_save_dir
        self._all_splits_dir = get_data_dir(dataset_name=self.name, base_dir=self._base_save_dir)
        self._items_by_split: dict[SplitT, list[MisophoniaItem] | None] | None = None

    def prepare(self) -> None:
        if self._items_by_split is not None:
            return

        self._items_by_split = {"train": [], "val": [], "test": []}

        for split in self._items_by_split.keys():
            split_dir = self._all_splits_dir / split
            metadata_file = split_dir / "metadata.jsonl"

            if not metadata_file.exists():
                self._items_by_split[split] = None
                continue

            def _handle_line(line: str) -> dict | None:
                obj = json.loads(line)
                obj["mix"] = split_dir / obj["mix"]
                obj["ground_truth"] = split_dir / obj["ground_truth"] if obj.get("ground_truth") is not None else None
                return obj

            with metadata_file.open("r", encoding="utf-8") as f:
                objs = (_handle_line(line) for line in f if line != "")
                t = pydantic.TypeAdapter(Generator[MisophoniaItem, None, None])
                self._items_by_split[split] = list(t.validate_python(objs))

    def get_split(self, split: SplitT) -> MisophoniaDatasetSplit:
        """
        Return a split view for a dataset saved on disk.

        Args:
            split: The dataset split to return. See SplitT for more details.

        Returns:
            A MisophoniaDatasetSplit object representing the requested split. See MisophoniaDatasetSplit for more details.
        """
        self.prepare()
        items = self._items_by_split[split]

        if items is None or len(items) == 0:
            raise ValueError(f"No data available for split '{split}'")

        return MisophoniaDatasetSplit(
            split=split,
            num_samples=len(items),
            get_one=items.__getitem__,  # Get the pre-computed item directly from the list
        )

    def save_split(
        self,
        split_data: MisophoniaDatasetSplit,
        *,
        n_workers: int | None = None,
        show_progress: bool = False,
        if_exists: Literal["error", "replace", "append"] = "error",
    ) -> None:
        """
        Save a dataset split to disk.

        Args:
            split_data: The dataset split to save.
                            E.g., one obtained from GeneratedMisophoniaDataset.get_split.
            n_workers: Number of parallel workers to use for saving. If None, will use os.cpu_count().
            show_progress: Whether to show a progress bar.
            if_exists: Behavior if the split directory already exists. Options:
                            "error": Raise an error.
                            "replace": Delete the existing directory and create a new one.
                            "append": Append new items to the existing directory (both audio data and metadata).
        """
        split = split_data.split

        split_dir = self._all_splits_dir / split
        mix_dir = split_dir / "mixes"
        gt_dir = split_dir / "ground_truths"
        metadata_file = split_dir / "metadata.jsonl"

        if split_dir.exists():
            if if_exists == "error":
                raise FileExistsError(f"Directory for split '{split}' already exists at {split_dir}")
            if if_exists == "replace":
                eliot.log_message(f"Replacing existing directory at {split_dir}", level="info")
                shutil.rmtree(split_dir)
            if if_exists == "append":
                eliot.log_message(f"Appending to existing directory at {split_dir}", level="info")

        mix_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        def _generate_and_save(i: int) -> str:
            item: MisophoniaItem = split_data[i]  # Heavy work (mixing + I/O) happens here

            mix_id = str(uuid.uuid4())

            mix_file = mix_dir / f"{mix_id}.flac"
            sf.write(
                mix_file,
                np.transpose(item.get_mix_audio()),
                samplerate=item.global_mixing_params.sample_rate,
                format="FLAC",
                subtype="PCM_24",
            )

            gt_file = None
            if item.ground_truth is not None:
                gt_file = gt_dir / f"{mix_id}.flac"
                sf.write(
                    gt_file,
                    np.transpose(item.get_ground_truth_audio()),
                    samplerate=item.global_mixing_params.sample_rate,
                    format="FLAC",
                    subtype="PCM_24",
                )

            item_with_paths = item.model_copy(
                update={
                    "uuid": mix_id,
                    "mix": mix_file.relative_to(split_dir),
                    "ground_truth": gt_file.relative_to(split_dir) if gt_file is not None else None,
                }
            )
            return item_with_paths.model_dump_json(round_trip=True)

        n_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
        size = len(split_data)

        with metadata_file.open("a", buffering=1, encoding="utf-8") as metadata_f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(_generate_and_save, range(size))
                if show_progress:
                    results = tqdm(results, total=size, desc=f"Saving {split} items")
                for row in results:
                    metadata_f.write(row + "\n")

    def __repr__(self) -> str:
        return f"<PremadeMisophoniaDataset from {self._all_splits_dir}>"


def add_experimental_pairs_to_dataset(
    original: PremadeMisophoniaDataset,
    *,
    seed: int = 42,
) -> None:
    """
    Post-hoc add experimental pairs to an existing premade dataset.

    It will sample trigger mixes (one foams and one non-foams per category) and create
    corresponding control mixes by replacing the trigger foreground with a control foreground,
    keeping everything else the same.

    It will furthermore add an anchor pair for the "chewing_gum" category.

    It will add an extra field `paired_uuid` to the generated control items.
    """
    # FIXME: This implementation is not very clean.
    #             It could be refactored into a GenerateExperimentalPairsDataset class that does not rely on sampling from PremadeMisophoniaDataset.
    #             But sampling directly from the source data.
    from .mixing import binaural_mix

    split = "test"  # Must be test split for experimental pairs

    eliot.log_message(f"Adding experimental pairs to dataset split '{split}' with seed {seed}", level="info")
    eliot.log_message(f"Loading base dataset from {original._all_splits_dir}", level="debug")
    original.prepare()

    original_split = original.get_split(split)

    # Only include single-foreground items
    all_items = models_to_df(original_split, flatten=True)
    all_items = all_items[all_items["len(foregrounds)"] == 1]
    all_items = all_items[all_items["len(foreground_categories)"] == 1]

    control_items = all_items[~all_items["is_trigger"]]
    trigger_items = all_items[all_items["is_trigger"]]

    assert "len(foregrounds[0][source_item][validated_by])" in trigger_items.columns, (
        "Cannot add experimental pairs since no sounds are validated."
    )
    assert trigger_items["len(foregrounds[0][source_item][validated_by])"].dropna().eq(1).all(), (
        "Some entries have multiple validators, which is not yet supported"
    )

    # Determine which sounds are FOAMS-validated
    is_foams = trigger_items["foregrounds[0][source_item][validated_by][0]"] == "FOAMS"

    foams_sounds = trigger_items[is_foams]
    non_foams_sounds = trigger_items[~is_foams]

    # Get trigger categories appearing in the dataset (sort to make reproducible)
    trigger_categories = tuple(sorted(trigger_items["foreground_categories[0]"].unique()))

    # Get samples for each category
    rng = np.random.default_rng(seed)
    existing_freesound_ids = set()
    trig_samples: list[MisophoniaItem] = []

    def _sample_non_used(subset: pd.DataFrame, n: int = 1) -> tuple[list[MisophoniaItem], set[int]] | bool:
        subset = subset.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        sampled_items = []
        sampled_freesound_ids = set()
        for _, row in subset.iterrows():
            item: MisophoniaItem = row["_model"]
            item_freesound_ids = {track.source_item.freesound_id for track in item.foregrounds + item.backgrounds}
            if not item_freesound_ids.intersection(existing_freesound_ids):
                sampled_items.append(item)
                sampled_freesound_ids.update(item_freesound_ids)
                if len(sampled_items) >= n:
                    break
        if len(sampled_items) < n:
            return False
        return sampled_items, sampled_freesound_ids

    for category in trigger_categories:
        # sample one FOAMS and one non-FOAMS sound from this category
        foams_in_category = foams_sounds[foams_sounds["foreground_categories[0]"] == category]
        non_foams_in_category = non_foams_sounds[non_foams_sounds["foreground_categories[0]"] == category]

        if len(foams_in_category) == 0 or len(non_foams_in_category) == 0:
            eliot.log_message(
                f"No FOAMS and non-FOAMS samples found in category '{category}' (FOAMS = {len(foams_in_category)}, non-FOAMS = {len(non_foams_in_category)})",
                level="warning",
            )
            continue

        f_samples = _sample_non_used(foams_in_category)
        nf_samples = _sample_non_used(non_foams_in_category)
        if not f_samples or not nf_samples:
            eliot.log_message(
                f"Could not sample non-repeating FOAMS and non-FOAMS samples in category '{category}'",
                level="warning",
            )
            continue

        trig_samples.extend(f_samples[0])
        trig_samples.extend(nf_samples[0])
        existing_freesound_ids.update(f_samples[1])
        existing_freesound_ids.update(nf_samples[1])

        if category.lower() == "chewing_gum":  # Add an extra anchor pair for the chewing_gum category
            anchor_subset = trigger_items[trigger_items["foreground_categories[0]"] == category]
            a_samples = _sample_non_used(anchor_subset)
            if not a_samples:
                eliot.log_message(
                    f"Could not sample non-repeating anchor sample in category '{category}'",
                    level="warning",
                )
            else:
                trig_samples.extend(a_samples[0])
                existing_freesound_ids.update(a_samples[1])

    assert len(control_items) >= len(trig_samples), "Not enough control items to match the number of trigger samples"

    ctrl_ss = _sample_non_used(control_items, n=len(trig_samples))
    if not ctrl_ss:
        raise RuntimeError("Could not sample non-repeating control samples for all trigger samples")
    existing_freesound_ids.update(ctrl_ss[1])
    control_samples = ctrl_ss[0]
    samples: list[tuple[MisophoniaItem, MisophoniaItem]] = list(zip(trig_samples, control_samples))

    def _get_paired_control_item(index: int) -> MisophoniaItem:
        """Replace the trigger foreground with a control foreground, keeping everything else the same."""
        # Use the sampling plan from above:
        original_trig, original_control = samples[index]

        assert original_control.uuid is not None, "Original control item must have a UUID"

        global_mixing_params = original_trig.global_mixing_params  # Keep the same
        sample_rate = global_mixing_params.sample_rate
        bg_specs = tuple(  # Load the audio for the background tracks (keeping them as is)
            (track, track.source_item.load_audio(sample_rate=sample_rate)[0]) for track in original_trig.backgrounds
        )
        bg_tracks, _ = zip(*bg_specs)

        control_item = original_control.foregrounds[0].source_item
        control_audio = control_item.load_audio(sample_rate=sample_rate)[0]

        fg_track = original_trig.foregrounds[0].model_copy(
            # Use the same config as the trigger foreground, but with the source item being a control
            # and the start/end also updated accordingly
            update={
                "source_item": control_item,
                # Start the control at the same time as the trig, and play it all out
                "start": original_trig.foregrounds[0].start,
                "end": original_trig.foregrounds[0].start + len(control_audio),
            }
        )

        mix, ground_truth = binaural_mix(
            fg_specs=((fg_track, control_audio),),
            bg_specs=bg_specs,
            global_params=global_mixing_params,
            is_trig=False,
        )

        return MisophoniaItem(
            split=split,
            is_trigger=False,
            mix=mix,
            ground_truth=ground_truth,
            length=mix.shape[1],
            global_mixing_params=global_mixing_params,
            foregrounds=(fg_track,),
            backgrounds=bg_tracks,
            paired_uuid=original_trig.uuid,
        )

    # Make the the split view to make it compatible with save_split API
    new_paired_items = MisophoniaDatasetSplit(
        split=split,
        num_samples=len(trig_samples),
        get_one=_get_paired_control_item,
    )
    original.save_split(new_paired_items, if_exists="append", show_progress=True)
