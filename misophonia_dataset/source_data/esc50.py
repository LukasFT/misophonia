import json
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from ..interface import License, MappingT, SourceData, SourceDataItem, get_data_dir
from ._downloading import download_and_unzip, is_unzipped
from ._freesound_license import generate_freesound_licenses
from ._splitting import is_validated_ids, train_valid_test_split


class Esc50Dataset(SourceData):
    """
    Class for the ESC50 dataset. Data is downloaded from "https://github.com/karoldvl/ESC-50/archive/master.zip".
    ESC50 is only used for trigger sounds, so the isTrig of the metadata column will have only 1s.
    """

    dataset_license = License(
        license_url="https://creativecommons.org/licenses/by-nc/3.0/",
        attribution_name="K. J. Piczak",
        attribution_url="http://dx.doi.org/10.1145/2733373.2806390",
    )

    def __init__(self, *, save_dir: Path | None = None, mapping: None | MappingT = None) -> None:
        if mapping is None:
            with (Path(__file__).parent / "esc50_mapping.json").open("r") as f:
                mapping = json.load(f)

        self.mapping: MappingT = mapping

        self._base_save_dir = save_dir if save_dir is not None else get_data_dir(dataset_name="ESC50")
        self._base_unzipped_dir = self._base_save_dir / "ESC-50-master"
        self._meta = None

    def is_downloaded(self) -> bool:
        return is_unzipped(file_path=self._base_save_dir / "ESC-50-master.zip")

    def download_data(self) -> None:
        """
        Downloads and extracts the ESC50 dataset from github

        Params:
            save_dir (str): directory to save the dataset

        """
        download_and_unzip(
            files=(
                {
                    # Latest commit per November 24, 2025
                    "url": "https://github.com/karolpiczak/ESC-50/archive/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6.zip",
                    "md5": "071b44018315e034b2c6e8064543d19c",
                    "filename": "ESC-50-master.zip",
                },
            ),
            save_dir=self._base_save_dir,
            delete_zip=True,
            rename_extracted_dir=self._base_unzipped_dir.name,
        )

    def get_metadata(self) -> Collection[SourceDataItem]:  # TODO: Change to not a dataframe
        """
        Get standardized metadata for ESC50 dataset.
        """
        if self._meta is not None:
            return self._meta

        assert self.is_downloaded(), "Dataset is not downloaded yet."
        meta = pd.read_csv(self._base_unzipped_dir / "meta" / "esc50.csv")
        meta = meta.add_prefix("esc50_")  # to avoid confusion with other datasets

        meta["source_dataset"] = "ESC50"

        base_audio_dir = (self._base_unzipped_dir / "audio").expanduser().resolve()
        cwd = Path.cwd()
        meta["file_path"] = meta["esc50_filename"].apply(lambda x: base_audio_dir / x)

        meta["labels"] = meta["esc50_category"].apply(lambda x: self.mapping.get(str(x), {}).get("foams_mapping", None))

        meta = meta[meta["labels"].notna()].copy()  # Only use trigger sounds from ESC50
        meta["label_type"] = "trigger"
        meta["labels"] = meta["labels"].apply(lambda x: [x])  # Make lists of labels

        meta = meta.rename(columns={"esc50_src_file": "freesound_id"})

        # The license of ESC50 depends on whether the sound is part of ESC10 or not
        assert len(meta[meta["esc50_esc10"]]) == 0, (
            "We assume no triggers are found in the ESC10 subset. If there is, please update the license handling code here."
        )
        # meta.loc[meta["esc50_esc10"] == True, "licensing"] = generate_freesound_licenses(
        #     meta.loc[meta["esc50_esc10"] == True, "freesound_id"],
        #     base_licenses=(
        #        License(
        #             license_url="https://creativecommons.org/licenses/by/3.0/",
        #             attribution_name="K. J. Piczak",
        #             attribution_url="http://dx.doi.org/10.1145/2733373.2806390",
        #         ),
        #     ),
        # )
        meta["sound_license"] = generate_freesound_licenses(meta["freesound_id"])
        meta["dataset_license"] = (self.dataset_license,) * len(meta)

        meta["validated_by"] = is_validated_ids(meta["freesound_id"])
        meta["split"] = train_valid_test_split(meta["freesound_id"], validated_by=meta["validated_by"])

        self._meta = [
            SourceDataItem(**row) for row in meta.to_dict(orient="records")
        ]  # Could probably be done more efficiently without pandas
        return self._meta

    def delete(self) -> None:
        self._base_save_dir.rmdir()
