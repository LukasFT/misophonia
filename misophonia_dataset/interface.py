import ctypes.util
import itertools
import platform
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterator, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, overload

import numpy as np
import pydantic
import soundfile as sf

"""
This needs to be imported before librosa to avoid errors on Linux related to AudioToolbox.
Patch ctypes.util.find_library to suppress AudioToolbox check on Linux.
This prevents audioread from spawning gcc via tempfile for every librosa.load call.
"""
if platform.system() == "Linux":
    _original_find_library = ctypes.util.find_library

    def _find_library_patched(name):  # noqa: ANN001, ANN202
        if name == "AudioToolbox":
            return None
        return _original_find_library(name)

    ctypes.util.find_library = _find_library_patched
import librosa

DEFAULT_LABEL_ORDER = (
    # "basketball_dribbling",
    # "flipping newspaper",
    "chewing_gum",
    "clearing_throat",
    "human_breathing",
    "knife_cutting",
    "plastic_crumpling",
    "swallowing",
    "typing",
    "water_drops",
)

MappingT: TypeAlias = dict[str, dict[Literal["foams_mapping"], str]]
"""The structure of a mapping from dataset-specific classes to FOAMS classes."""


SplitT: TypeAlias = Literal["train", "val", "test"]
"""
The possible dataset splits, for both source datasets and mixed data.

See misophonia_dataset.source_data._splitting.train_valid_test_split for more information about how this is achieved by default.
"""


class BaseModel(pydantic.BaseModel):
    """
    Pydantic model that we will use to define our data classes.

    See https://docs.pydantic.dev/latest/ for more information.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,  # this means that instances are immutable
        extra="forbid",  # Do not allow extra fields
    )


class License(BaseModel):
    """A license for any sound, dataset or any other asset."""

    license_url: str
    attribution_name: str
    attribution_url: str


class SourceDataItem(BaseModel):
    """Metadata for a single audio file in a source dataset."""

    split: SplitT
    """Dataset split (see SplitT)."""

    source_dataset: str
    """Name of the source dataset that contains this audio and metadata."""

    file_path: Path
    """Path to the audio file."""

    freesound_id: int | None = None
    """FreeSound.org ID of the audio file, if available."""

    label_type: Literal["control", "trigger", "background"]
    """Type of sound."""
    labels: tuple[str, ...]
    """
    Label according to the FOAMS taxonomy (if label_type = 'trigger') and, otherwise, the AudioSet (FSD50K) taxonomy.

    There can be one more more labels. But all labels must be of the same sound_type.
    """

    validated_by: tuple[str, ...] | None = None
    """Names of studies (e.g., FOAMS) that have validated this data point to trigger misophoniacs. Only applicable for trigger sounds."""

    sound_license: License | None = None
    """License information for the specific sound."""
    dataset_license: License | None = None
    """License information for the source dataset containing the sound."""

    model_config = pydantic.ConfigDict(
        # Allow extra fields since different datasets have different metadata fields.
        # Extra fields should be prefixed by the dataset name to avoid conflicts.
        extra="allow",
    )

    def load_audio(self, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
        # TODO: Why are we using librosa.load here instead of soundfile.read?
        return librosa.load(self.file_path, sr=sample_rate, mono=True)


class SourceTrack(BaseModel):
    """A track contains a source item as well as the parameters specific to this used for mixing."""

    source_item: SourceDataItem
    """The source data item where the sound comes from."""

    start: int
    """Start offset (in samples, i.e. not time) of the clip."""
    end: int
    """End offset (in samples, i.e. not time) of the clip."""

    ### Binamix parameters ###
    # See https://github.com/QxLabIreland/Binamix/ for details.
    azimuth: float  # Random default
    """Azimuth angle for spatialization (in degrees)."""
    elevation: float  # Random default
    """Elevation angle for spatialization (in degrees)."""
    level: float  # Random default
    """Level (loudness) scaling factor for the sound. Applied after RMS normalization."""
    reverb: float  # Random default
    """Reverb amount for the sound."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fill_random_defaults(cls, values: list):  # noqa: ANN001, ANN206
        rng: np.random.Generator = values.pop("_rng", np.random.default_rng())

        if "azimuth" not in values:
            values["azimuth"] = rng.integers(-180, 181)

        if "elevation" not in values:
            values["elevation"] = rng.integers(-180, 181)

        if "level" not in values:
            values["level"] = 1

        if "reverb" not in values:
            values["reverb"] = round(rng.uniform(0.0, 1.0), 1)

        return values


ReverbT = Literal[
    "1",  # Theatre
    "2",  # Office
    "3",  # Small Room
    "4",  # Meeting Room
]
"""Type alias for reverb types. See https://github.com/QxLabIreland/Binamix/?tab=readme-ov-file#mix_tracks_binaural for details."""

InterpolationModeT = Literal["auto", "nearest", "planar", "two_point", "three_point"]
"""Type alias for interpolation modes. See https://github.com/QxLabIreland/Binamix/?tab=readme-ov-file#mix_tracks_binaural for details."""

ImpulseResponseT = Literal[
    "BRIR",  # Binaural Room Impulse Response
    "DRIR",  # Directional Room Impulse Response
]
"""Type alias for impulse response types. See https://github.com/QxLabIreland/Binamix/?tab=readme-ov-file#mix_tracks_binaural for details."""


class GlobalMixingParams(BaseModel):
    """
    Global mixing parameters relevant for the general mix (i.e., not specific to any one track).

    See https://github.com/QxLabIreland/Binamix/ for details.
    """

    subject_id: str  # Randomized default
    """ID of the subject used for binaural rendering, according to the SADIE II database."""
    speaker_layout: str = "none"  # Stereo
    """Layout of the output speakers / channels."""
    sample_rate: int = 44100
    """The target sample rate for the mix to be generated at."""
    reverb_type: ReverbT  # Randomized default
    """Type of reverb to use. See ReverbT for more."""
    mode: InterpolationModeT = "two_point"
    """Interpolation mode. See InterpolationModeT for more."""
    ir_type: ImpulseResponseT = "BRIR"
    """Type of impulse response to use."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fill_random_defaults(cls, values: list):  # noqa: ANN001, ANN206
        rng: np.random.Generator = values.pop("_rng", np.random.default_rng())

        if "subject_id" not in values:
            values["subject_id"] = rng.choice(
                [
                    "D1",
                    "D2",
                    "H3",
                    "H4",
                    "H5",
                    "H6",
                    "H7",
                    "H8",
                    "H9",
                    "H10",
                    "H11",
                    "H12",
                    "H13",
                    "H14",
                    "H15",
                    "H16",
                    "H17",
                    "H18",
                    "H19",
                    "H20",
                ]
            )

        if "reverb_type" not in values:
            values["reverb_type"] = rng.choice(["1", "2", "3", "4"])

        return values


DEFAULT_MIXED_DATASET_LICENSE = (
    License(
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution_name="Lukas Frimer Tholander & Tonio Ermakoff",
        attribution_url="https://github.com/LukasFT/misophonia-dataset",
    ),
    License(  # For the SADIE Database
        license_url="https://www.apache.org/licenses/LICENSE-2.0/",
        attribution_name="C. Armstrong, L. Thresh & G. Kearney",
        attribution_url="https://www.mdpi.com/2076-3417/8/11/2029",
    ),
)


class MisophoniaItem(BaseModel):
    """A single mixed item in a misophonia dataset."""

    split: SplitT
    """Dataset split (see SplitT)."""

    uuid: str | None = None
    """Unique identifier for the item. May be None if the item has not been saved."""

    is_trigger: bool
    """Whether this item contains trigger sounds. If False, it only contains control/background sounds."""
    foreground_categories: tuple[str, ...]
    """
    Categories of the foreground sounds in this item.

    Like SourceDataItem.labels, these are according to either the FOAMS taxonomy if is_trigger = True, else to the AudioSet (FSD50K) taxonomy.
    """
    background_categories: tuple[str, ...]
    """Categories of the background sounds in this item according to the AudioSet (FSD50K) taxonomy."""

    mix: np.ndarray | Path
    """
    Binaural mixed audio data for both foreground and background sounds.

    See get_mix_audio() to load the audio data from the file if it is a Path.
    """
    ground_truth: np.ndarray | Path | None
    """
    Ground truth audio data. If available, this is the isolated binaural audio for the trigger sound.

    See get_ground_truth_audio() to load the audio data from the file if it is a Path.
    """

    gt_is_isolated_trigger: bool = True
    """Whether the ground truth is the isolated trigger sound (if True) or the mix of all backgrounds (if False)."""

    length: int
    """Duration in number of samples."""

    foregrounds: tuple[SourceTrack, ...]
    """Foreground (trigger or control) tracks used in the mix."""
    backgrounds: tuple[SourceTrack, ...]
    """Background tracks used in the mix."""

    global_mixing_params: GlobalMixingParams
    """Global mixing parameters used to generate the mix."""

    mix_licensing: tuple[License, ...] = DEFAULT_MIXED_DATASET_LICENSE
    """The licensing for the mixing aspect. Note that this does not include the licensing for the individual source sounds."""

    ### Information relating to the experimental validation ###
    # See misophonia_dataset.misophonia_dataset.add_experimental_pairs_to_dataset for more details.
    paired_uuid: str | None = None
    """UUID of the paired version of this item, if any."""
    experimental_discomfort_level: float | None = pydantic.Field(None, ge=0, le=5)
    """Obtained average discomfort level from human experimental evaluation, on a Likert scale from 0 to 5."""

    @property
    def trigger_categories(self) -> tuple[str, ...] | None:
        return self.foreground_categories if self.is_trigger else None

    @property
    def duration(self) -> float:
        return self.length / self.global_mixing_params.sample_rate

    @property
    def all_licenses(self) -> tuple[License, ...]:
        """All licenses relevant for this item, including source sounds and mixing."""
        return tuple(
            itertools.chain(
                self.mix_licensing,
                *(
                    filter(
                        None,
                        (
                            track.source_item.sound_license,
                            track.source_item.dataset_license,
                        ),
                    )
                    for track in itertools.chain(self.foregrounds, self.backgrounds)
                ),
            )
        )

    def get_label_vector(self, label_order: tuple[str, ...] = DEFAULT_LABEL_ORDER) -> np.ndarray:
        """One-hot encoded vector of foreground_categories."""
        if not self.is_trigger:
            return np.zeros(len(label_order), dtype=np.float32)

        assert all(cat in label_order for cat in self.foreground_categories), (
            "All foreground categories must be in the label order."
        )
        return np.array(
            [1 if cat in self.foreground_categories else 0 for cat in label_order],
            dtype=np.float32,
        )

    def get_mix_audio(self) -> np.ndarray:
        """Load (if not already loaded) and return the mixed audio data."""
        if isinstance(self.mix, Path):
            return self._load_audio(self.mix)
        return self.mix

    def get_ground_truth_audio(self, *, control_as_zeros: bool = True) -> np.ndarray:
        """Load (if not already loaded) and return the ground truth audio data."""
        if self.ground_truth is None:
            return np.zeros((2, self.length)) if control_as_zeros else None
        if isinstance(self.ground_truth, Path):
            return self._load_audio(self.ground_truth)
        return self.ground_truth

    @pydantic.model_validator(mode="after")
    def _check_splits(self) -> "MisophoniaItem":
        """Validate that all forgrounds and background are of split"""
        if any(track.source_item.split != self.split for track in itertools.chain(self.foregrounds, self.backgrounds)):
            raise ValueError("All foreground and background items must match the MisophoniaItem split.")
        return self

    @pydantic.model_validator(mode="after")
    def _check_ground_truth(self) -> "MisophoniaItem":
        """Validate that iff is_trigger then ground_truth is not None"""
        if self.is_trigger and self.ground_truth is None:
            raise ValueError("If is_trigger is True, ground_truth must not be None.")
        if self.gt_is_isolated_trigger:
            if not self.is_trigger and self.ground_truth is not None:
                raise ValueError("If is_trigger is False, ground_truth must be None.")
        else:
            if not self.is_trigger and self.ground_truth is None:
                raise ValueError(
                    "If gt_is_isolated_trigger is False and is_trigger is False, ground_truth must be mix."
                )
        return self

    @pydantic.model_validator(mode="before")
    @classmethod
    def _auto_compute_categories(cls, values: dict) -> dict:
        """Auto-compute categories if none given. Create one-hot label vector."""
        if "foreground_categories" not in values or values["foreground_categories"] is None:
            if "foregrounds" not in values:
                raise ValueError("Cannot auto-compute foreground_categories without foregrounds.")
            values["foreground_categories"] = tuple(
                set(itertools.chain.from_iterable(it.source_item.labels for it in values["foregrounds"]))
            )
        if "background_categories" not in values or values["background_categories"] is None:
            if "backgrounds" not in values:
                raise ValueError("Cannot auto-compute background_categories without backgrounds.")
            values["background_categories"] = tuple(
                set(itertools.chain.from_iterable(it.source_item.labels for it in values["backgrounds"]))
            )

        if "is_trigger" not in values or values["is_trigger"] is None:
            raise ValueError("Cannot create label vector because audio does not have 'is_trigger' field.")

        return values

    @staticmethod
    def _load_audio(p: Path) -> np.ndarray:
        sound = sf.read(p)[0]
        sound = sound.T  # C, samples (like librosa)
        return sound


def get_data_dir(*, dataset_name: str | None = None, base_dir: Path | None = None) -> Path:
    """
    Get the default data directory for storing a dataset (either source or mixed).

    Args:
        dataset_name: Name of the dataset. If None, returns the base data directory.
        base_dir: Base directory to use. If None, uses the "data" directory next to this package.

    Returns:
        Path to the relevant data directory.
    """
    base_dir = base_dir or Path(__file__).parent.parent / "data"
    if dataset_name is None:
        return base_dir
    return base_dir / dataset_name


class SourceData(ABC):
    """Interface for source datasets to standaridize downloading and metadata extraction."""

    @abstractmethod
    def is_downloaded(self) -> bool:
        """
        Checks if the dataset has been downloaded.

        Returns:
            True if the dataset is downloaded, False otherwise.
        """
        pass

    @abstractmethod
    def download_data(self) -> None:
        """Downloads, extracts, and saves dataset data."""
        pass

    @abstractmethod
    def get_metadata(self) -> Collection[SourceDataItem]:
        """Get standardized metadata for the dataset."""
        pass

    @abstractmethod
    def delete(self) -> None:
        """Deletes the entire dataset. Useful after mixing sounds."""
        pass

    def __str__(self) -> str:
        return f"<SourceData: {self.__class__.__name__}>"


class MisophoniaDataset(ABC):
    @abstractmethod
    def prepare(self) -> None:
        """Download / index / precompute anything needed before being able to iterate over splits."""
        pass

    @abstractmethod
    def get_split(
        self,
        split: SplitT,
        **options: dict,
    ) -> "MisophoniaDatasetSplit":
        """
        Return a split view for this dataset. This split view is iterable and indexable (see MisophoniaDatasetSplit).

        Args:
            split: The dataset split to return. See SplitT for more details.
            **options: parameters custom to the specific dataset class (e.g. random_seed, num_samples, foregrounds_per_item, ...).

        Returns:
            A MisophoniaDatasetSplit object representing the requested split.
        """
        pass


class MisophoniaDatasetSplit(Sequence[MisophoniaItem]):
    """A view over a particular mixed dataset split with fixed parameters."""

    def __init__(
        self,
        *,
        split: SplitT,
        num_samples: int,
        get_one: Callable[[int], MisophoniaItem],
    ) -> None:
        """
        Make a view over a particular mixed dataset split with fixed parameters.

        Args:
            split: The dataset split this view represents. See SplitT for more details.
            num_samples: Number of samples in this split.
            get_one: A function that takes an index and returns the corresponding MisophoniaItem.
                        This should be where heavy logic for generating the item is implemented,
                        in order to make the class lightweight and parallelizable.
        """
        self._split = split
        self._num_samples = num_samples
        self._get_one = get_one

    @property
    def split(self) -> SplitT:
        """The dataset split this view represents. See SplitT for more details."""
        return self._split

    def __len__(self) -> int:
        """Number of samples in this split."""
        return self._num_samples

    @overload
    def __getitem__(self, idx: int) -> MisophoniaItem: ...

    @overload
    def __getitem__(self, idx: slice | list[int]) -> Sequence[MisophoniaItem]: ...

    def __getitem__(self, idx):
        """Get the item at the specified index."""
        if isinstance(idx, slice) or isinstance(idx, list):
            if isinstance(idx, slice):
                indices = range(*idx.indices(self._num_samples))
            else:
                indices = idx
            return [self._get_one(i) for i in indices]
        if idx < 0:  # Allow e.g. split[-1] indexing
            idx += self._num_samples
        if not (0 <= idx < self._num_samples):
            raise IndexError(idx)
        return self._get_one(idx)

    def __iter__(self) -> Iterator[MisophoniaItem]:
        """
        Helper to iterate over all items in the split.

        Note that this does not parallelize the generation of items. For parallel generation, use indexing.
        """
        for i in range(self._num_samples):
            yield self._get_one(i)
