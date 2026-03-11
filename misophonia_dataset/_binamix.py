import contextlib
import os
import subprocess
import sys
from pathlib import Path

from .source_data._downloading import download_and_unzip


def get_binamix_dir() -> Path:
    return Path(__file__).parent.parent / "Binamix"


def setup_binamix_import() -> None:
    binamix_repo = get_binamix_dir()

    if not binamix_repo.exists() or not any(binamix_repo.iterdir()):  # Binamix dir is empty
        print("Warning: Binamix repository is empty. Updating submodule...")
        subprocess.run(["git", "submodule", "update", "--init"], check=True)

    sys.path.append(str(binamix_repo))


def setup_binamix() -> None:
    """Adds the Binamix library to the system path for importing."""
    # TODO: Help Binamix distribute their package on PyPi or similar
    setup_binamix_import()

    try:
        import binamix  # type: ignore # noqa: F401
    except ImportError as e:
        raise ImportError("Failed to import Binamix after adding to path.") from e

    downloaded_sadie = False
    try:
        import binamix.sadie_utilities  # type: ignore
    except FileNotFoundError as e:
        if "SADIE" in str(e):
            print("Warning: Binamix SADIE database not found. Downloading it.")
            # binamix_repo = get_binamix_dir()
            # This should work but the files 404 (per November 26, 2025):
            # subprocess.run([sys.executable, "-m", "binamix.sadie_db_setup"], cwd=binamix_repo, check=True)
            download_sadie()
            downloaded_sadie = True

    if downloaded_sadie:
        try:
            import binamix.sadie_utilities  # type: ignore  # noqa: F401
        except FileNotFoundError as e:
            if "SADIE" in str(e):
                raise FileNotFoundError(
                    "Failed to find SADIE database even after downloading. Please check Binamix setup."
                ) from e


def download_sadie() -> None:
    binamix_repo = get_binamix_dir()
    if not binamix_repo.exists():
        raise FileNotFoundError(
            f"Binamix repository not found at expected location: {binamix_repo}. Hint: You might need to run setup_binamix()"
        )

    download_and_unzip(
        (
            {
                "url": "https://zenodo.org/records/12092466/files/Database-Master_V2-2.zip?download=1",
                "md5": "e598337f7c70af9ffa11d4a4f4f6740f",
            },
        ),
        save_dir=binamix_repo / "sadie",
        rename_extracted_dir="Database-Master_V2-2",  # Must  be named exactly "Database-Master_V1-4" to be found by Binamix (even if that issss not the version)
        delete_zip=True,
    )


def custom_mix_tracks_binaural(**kwargs) -> None:  # noqa: ANN003
    from binamix.sadie_utilities import mix_tracks_binaural  # type: ignore

    with contextlib.redirect_stdout(open(os.devnull, "w")):  # Silence their printing (can break in multiprocessing)
        return mix_tracks_binaural(**kwargs)


if __name__ == "__main__":
    setup_binamix()
