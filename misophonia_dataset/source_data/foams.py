from collections.abc import Collection
from pathlib import Path

import pandas as pd

from ..interface import License, SourceData, SourceDataItem, get_data_dir
from ._downloading import download_and_unzip, download_single_file, is_downloaded, is_unzipped
from ._freesound_license import generate_freesound_licenses
from ._splitting import is_validated_ids, train_valid_test_split


class FoamsDataset(SourceData):
    """
    Class for FOAMS misophonia trigger sounds. Downloaded from https://zenodo.org/records/7109069
    """

    dataset_license = License(
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution_name="D. M. Orloff, D. Benesch & H. A. Hansen",
        attribution_url="https://doi.org/10.5334/jopd.94",
    )

    def __init__(self, save_dir: Path | None = None) -> None:
        self._base_save_dir = save_dir if save_dir is not None else get_data_dir(dataset_name="FOAMS")
        self._meta = None

    def is_downloaded(self) -> bool:
        return is_unzipped(file_path=self._base_save_dir / "FOAMS_processed_audio.zip") and is_downloaded(
            file_path=self._base_save_dir / "segmentation_info.csv"
        )

    def download_metadata(self) -> None:
        download_single_file(
            url="https://zenodo.org/records/8170225/files/segmentation_info.csv?download=1",
            md5="0ac1de8a66ffb52be34722ad8cd5e514",
            save_dir=self._base_save_dir,
        )

    def download_data(self) -> None:
        """
        Download 50 trigger samples from FOAMS at https://zenodo.org/records/7109069/files/. First checks if they have been downloaded alrady.
        Params:
            save_dir
        """
        self.download_metadata()
        download_and_unzip(
            files=(
                {
                    # Latest commit per November 24, 2025
                    "url": "https://zenodo.org/records/8170225/files/FOAMS_processed_audio.zip?download=1",
                    "md5": "89e717006cea3687384baa3c86d6307c",
                },
            ),
            save_dir=self._base_save_dir,
            delete_zip=True,
            rename_extracted_dir="processed_audio",
        )

    def get_metadata(self) -> Collection[SourceDataItem]:
        if self._meta is not None:
            return self._meta

        meta = self._get_base_metadata()

        meta["source_dataset"] = "FOAMS"

        meta = meta.rename(columns={"foams_label": "labels"})  # FOAMS labels are already aligned to FOAMS taxonomy
        meta = meta[
            ~meta["labels"].isin(["basketball_dribbling", "flipping_newspaper"])
        ]  # Remove these classes as they do not appear in other datasets

        meta["labels"] = meta["labels"].apply(lambda x: [x])  # Make singular lists to align with other datasets
        meta["label_type"] = "trigger"  # All FOAMS sounds are triggers
        cwd = Path.cwd()
        meta["file_path"] = meta["freesound_id"].apply(
            lambda x: (self._base_save_dir / "processed_audio" / f"{x}_processed.wav").relative_to(cwd)
        )

        meta["sound_license"] = generate_freesound_licenses(meta["freesound_id"])
        meta["dataset_license"] = (self.dataset_license,) * len(meta)

        meta["validated_by"] = is_validated_ids(meta["freesound_id"])
        meta["split"] = train_valid_test_split(meta["freesound_id"], validated_by=meta["validated_by"])

        self._meta = [
            SourceDataItem(**row) for row in meta.to_dict(orient="records")
        ]  # Could probably be done more efficiently without pandas
        return self._meta

    def get_all_sound_ids(self) -> pd.Series:
        """
        Returns a Series of FOAMS sound FreeSound IDs.
        """
        meta = self._get_base_metadata()
        return meta["freesound_id"]

    def _get_base_metadata(self) -> pd.DataFrame:
        meta = pd.read_csv(self._base_save_dir / "segmentation_info.csv")
        meta = meta.add_prefix("foams_")  # To avoid column name clashes
        meta = meta.rename(columns={"foams_id": "freesound_id"})  # FOAMS IDs correspond to FreeSound IDs
        return meta

    def delete(self) -> None:
        self._base_save_dir.rmdir()
