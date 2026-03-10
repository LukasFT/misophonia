import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import urlparse

import eliot
import requests
from tqdm import tqdm


def is_downloaded(*, file_path: Path, state_file: Path | None = None) -> bool:
    """
    Checks if a file has been downloaded based on the state file.
    """
    assert file_path is not None or state_file is not None, "Either file_path or state_file must be provided."
    state_file = state_file or _get_default_state_file(file_path)
    return _get_file_state(state_file).get("downloaded", False)


def is_unzipped(*, file_path: Path | None = None, state_file: Path | None = None) -> bool:
    """
    Checks if a file has been unzipped based on the state file.
    """
    assert file_path is not None or state_file is not None, "Either file_path or state_file must be provided."
    state_file = state_file or _get_default_state_file(file_path)
    return _get_file_state(state_file).get("unzipped", False)


def download_and_unzip(
    files: Iterable[dict[Literal["md5", "url", "filename"], str]],
    *,
    save_dir: Path,
    delete_zip: bool = False,
    rename_extracted_dir: str | None = None,
    state_file: Path | None = None,
    max_retries: int = 5,
) -> Path:
    """
    Downloads and unzips files from given URLs, with support for resuming and progress tracking, as well as multi-part zip files.

    Args:
        files: An iterable of dictionaries, each containing:
            - "url": The URL to download the file from.
            - "md5": The expected MD5 checksum of the file.
            - "filename" (optional): The name to save the file as. If not provided
                the filename is extracted from the URL.
        save_dir: Directory to save the downloaded files.
        delete_zip: Whether to delete the zip file after extraction.
        rename_extracted_dir: If provided, renames the extracted directory to this name.
        state_file: Path to a JSON file to track download and unzip state. If not provided
            a default state file will be created in the same directory as the downloaded file.
        max_retries: Maximum number of retries for downloading each file.

    Returns:
        Path to the extracted directory.

    """
    # Download all files
    save_paths = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                download_single_file,
                url=file_info["url"],
                save_dir=save_dir,
                md5=file_info.get("md5"),
                filename=file_info.get("filename"),
                state_file=state_file,
                max_retries=max_retries,
                tqdm_position=i,
            ): file_info
            for i, file_info in enumerate(files)
        }

        for future in as_completed(futures):
            try:
                save_paths.append(future.result())
            except Exception as exc:
                file_info = futures[future]
                raise RuntimeError(f"Failed to download {file_info['url']}") from exc

    zip_files = tuple(p for p in save_paths if p.suffix.lower() == ".zip")
    if len(zip_files) != 1:
        raise ValueError(f"Expected exactly one zip file to unzip, found {len(zip_files)}: {zip_files}")
    base_zip_path = zip_files[0]
    state_file = state_file or _get_default_state_file(base_zip_path)

    if not is_unzipped(state_file=state_file):
        # Find files that have the same base name with .z** extensions
        base_name = base_zip_path.stem
        part_file_paths = tuple(
            p
            for p in save_paths
            if p.name.startswith(base_name) and p != base_zip_path and p.suffix.lower().startswith(".z")
        )

        extract_to = save_dir / f"{base_name}_extracted"
        _unzip_file(
            base_zip_path,
            part_file_paths=part_file_paths,
            extract_to=extract_to,
            state_file=state_file,
            delete_zip=delete_zip,
        )

        # Move to
        # delete any __MACOSX directories if present
        macosx_dir = extract_to / "__MACOSX"
        if macosx_dir.exists() and macosx_dir.is_dir():
            shutil.rmtree(macosx_dir)

        if rename_extracted_dir:
            # Check if there is exactly one top-level directory
            extracted_items = list(extract_to.iterdir())
            top_level_dirs = [p for p in extracted_items if p.is_dir()]
            if len(top_level_dirs) == 1:
                original_path = top_level_dirs[0]
                new_path = extract_to.parent / rename_extracted_dir

                if new_path.exists():
                    raise RuntimeError(f"Cannot rename {original_path} to {new_path} because it already exists.")
                original_path.rename(new_path)
                extract_to.rmdir()  # Remove now-empty extracted_to directory
                eliot.log_message(f"Renamed {original_path.name} to {new_path.name}", level="debug")
            else:
                eliot.log_message(
                    f"Skipping rename: Extracted files do not have a single top-level directory (found {[p.name for p in top_level_dirs]})",
                    level="warning",
                )


def download_single_file(
    *,
    url: str,
    save_dir: Path,
    md5: str,
    filename: str | None = None,
    state_file: Path | None = None,
    max_retries: int = 5,
    tqdm_position: int = 0,
) -> Path:
    # Remove query params
    filename = os.path.basename(urlparse(url).path) if filename is None else filename
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    state_file = state_file or _get_default_state_file(save_path)

    if not _get_file_state(state_file).get("downloaded", False):
        for attempt in range(1, max_retries + 1):
            _set_file_state(state_file, downloaded=False)
            try:
                _download_single_inner(url, save_path, tqdm_position=tqdm_position)
                break  # Successful download, exit retry loop
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts.") from e

                sleep_time = min(1.5**attempt, 60*15)
                eliot.log_message(
                    f"Error downloading {filename} (attempt {attempt}/{max_retries}; retrying in {sleep_time:.1f} seconds): {e}",
                    level="warning",
                )
                time.sleep(sleep_time)

        # Check integrity of files
        if md5 is not None and not _is_correct_md5(save_path, md5):
            raise ValueError(f"MD5 checksum does not match for downloaded file: {save_path} (expected: `{md5}`)")

        _set_file_state(state_file, downloaded=True)

    return save_path


def _download_single_inner(url: str, save_path: Path, tqdm_position: int = 0) -> None:
    # Check for partial file
    expected_size = _get_expected_size(url)
    headers = {}
    if save_path.exists():
        existing = save_path.stat().st_size

        if expected_size is not None and existing >= expected_size:
            eliot.log_message(f"File {url} already fully downloaded at {save_path}", level="debug")
            return

        headers["Range"] = f"bytes={existing}-"
        eliot.log_message(f"Resuming download for {url} from byte {existing}", level="debug")
    else:
        existing = 0

    # Request (with streaming)
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()

    if "Range" in headers and response.status_code != 206:
        eliot.log_message(
            f"Server did not support resuming download for {url}. Restarting from beginning.", level="debug"
        )
        existing = 0
        if save_path.exists():
            save_path.unlink()  # Delete existing partial file

    total_size = int(response.headers.get("content-length", 0)) + existing
    chunk_size = 1024 * 1024  # 1MB

    with (
        save_path.open("ab" if existing else "wb") as f,
        tqdm(
            total=total_size,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {url}",
            ascii=True,
            position=tqdm_position,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def _get_default_state_file(save_path: Path) -> Path:
    return save_path.parent / f"state_{save_path.name}.json"


def _is_correct_md5(file: Path, md5: str) -> bool:
    with file.open("rb") as f:
        md5_hash = hashlib.md5()
        while chunk := f.read(8192):
            md5_hash.update(chunk)
        md5_hash = md5_hash.hexdigest()

    return md5_hash == md5


def _unzip_file(
    base_zip_path: Path,
    *,
    part_file_paths: Iterable[Path],
    extract_to: Path,
    state_file: Path | None = None,
    delete_zip: bool = False,
) -> None:
    state_file = state_file or _get_default_state_file(base_zip_path)

    if _get_file_state(state_file).get("unzipped", False):
        eliot.log_message(f"File {base_zip_path} has already been unzipped to {extract_to}", level="info")
        return

    base_zip_path = Path(base_zip_path).resolve()
    part_file_paths = tuple(Path(p).resolve() for p in part_file_paths)

    if len(part_file_paths) == 0:
        eliot.log_message(f"Unzipping file {base_zip_path} to {extract_to}", level="debug")
        _unzip_simple(zip_path=base_zip_path, extract_to=extract_to)
    else:
        eliot.log_message(f"Unzipping {base_zip_path} with parts {part_file_paths} to {extract_to}", level="debug")
        _unzip_with_parts(
            base_zip_path=base_zip_path,
            part_file_paths=part_file_paths,
            extract_to=extract_to,
        )

    if delete_zip:
        base_zip_path.unlink()  # Delete zip file after unzipping
        for part_path in part_file_paths:
            part_path.unlink()

    _set_file_state(state_file, unzipped=True)


def _unzip_simple(
    *,
    zip_path: Path,
    extract_to: Path,
) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def _unzip_with_parts(
    *,
    base_zip_path: Path,
    part_file_paths: Iterable[Path],
    extract_to: Path,
    rename_extracted_dir: str | None = None,
) -> None:
    """Use 7zip from command line to unzip a multi-part zip file."""

    # Ensure 7zip is installed
    if shutil.which("7z") is None:
        raise RuntimeError("7zip (7z) is not installed or not found in PATH. Cannot unzip multi-part zip files.")

    # Assert all parts are only differing by extension (z**)
    assert all(part.parent == base_zip_path.parent for part in part_file_paths), (
        "All parts must be in the same directory."
    )
    assert all(part.stem == base_zip_path.stem for part in part_file_paths), "All parts must have the same base name."
    assert all(part.suffix.lower().startswith(".z") for part in part_file_paths), "All parts must have .z** extensions."
    command = ["7z", "x", str(base_zip_path), f"-o{str(extract_to)}", "-y"]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"7zip failed to unzip files: {result.stderr}")

    if rename_extracted_dir:
        # Check if there is exactly one top-level directory
        extracted_items = list(extract_to.iterdir())
        top_level_dirs = [p for p in extracted_items if p.is_dir()]
        if len(top_level_dirs) == 1:
            original_path = top_level_dirs[0]
            new_path = extract_to / rename_extracted_dir

            if new_path.exists():
                shutil.rmtree(new_path)
            original_path.rename(new_path)
            eliot.log_message(f"Renamed {original_path.name} to {new_path.name}", level="debug")
        else:
            eliot.log_message(
                f"Skipping rename: Extracted files do not have a single top-level directory (found {[p.name for p in top_level_dirs]})",
                level="warning",
            )


def _get_file_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {}
    with state_file.open("r") as f:
        return json.load(f)


def _set_file_state(state_file: Path, **new_state: dict) -> None:
    state = _get_file_state(state_file)
    state.update(new_state)
    with state_file.open("w") as f:
        json.dump(state, f)


def _get_expected_size(url: str) -> int | None:
    response = requests.head(url, allow_redirects=True, timeout=10)
    if response.status_code != 200:
        return None
    content_length = response.headers.get("content-length")
    if content_length is None:
        return None
    return int(content_length)

