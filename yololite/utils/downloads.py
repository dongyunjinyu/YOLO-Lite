# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess
from pathlib import Path
from urllib import request
import torch
from yololite.utils import LOGGER, TQDM, clean_url, emojis, is_online, url2file


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # Decide to unzip directly or unzip into a directory
        unzip_as_dir = len(top_level_dirs) == 1  # (len(files) > 1 and not files[0].endswith("/"))
        if unzip_as_dir:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. extract coco8/ dir to ../datasets/
        else:
            # Zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. extract multiple files to ../datasets/coco8/

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f"WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)
    return path  # return unzip dir

def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.
        ```
    """
    f = Path(dir or ".") / (file or url2file(url))  # URL converted to filename
    if "://" not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        uri = clean_url(url).replace(  # cleaned and aliased url
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "https://ultralytics.com/assets/",  # assets alias
        )
        desc = f"Downloading {uri} to '{f}'"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = "sS" * (not progress)  # silent
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl return value {r}"
                else:  # urllib download
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Environment is not online.")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached.")) from e
                LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {uri}...")

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # unzip to dir if provided else unzip in place
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # unzip
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()  # remove zip
        return unzip_dir
