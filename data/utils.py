"""Image download helpers for dataset asset preparation notebooks/scripts."""

import multiprocessing
import os
import urllib.request
from functools import partial
from pathlib import Path
from time import sleep
from typing import Iterable

from tqdm import tqdm


def download_image_fast(image_link: str, savefolder: str) -> None:
    """Download one image without retries (high-throughput mode)."""
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)
    if os.path.exists(image_save_path):
        return

    try:
        urllib.request.urlretrieve(image_link, image_save_path)
    except Exception as ex:
        print(f"Warning: Not able to download - {image_link}\n{ex}")


def download_images_fast(image_links: Iterable[str], download_folder: str) -> None:
    """Download images using 100 worker processes without throttling."""
    image_links = list(image_links)
    os.makedirs(download_folder, exist_ok=True)
    download_image_partial = partial(download_image_fast, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))


def download_image(
    image_link: str,
    savefolder: str,
    delay: float = 0.05,
    max_retries: int = 3,
) -> None:
    """Download one image with retry/backoff to reduce transient failures."""
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)
    if os.path.exists(image_save_path):
        return

    sleep(delay)  # Per-worker throttle to avoid remote rate limiting.
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except Exception:
            if attempt < max_retries - 1:
                sleep(0.5 * (2 ** attempt))
            else:
                print(f"Warning: Failed after {max_retries} attempts - {filename}")


def download_images(
    image_links: Iterable[str],
    download_folder: str,
    num_workers: int = 40,
    delay: float = 0.1,
    max_retries: int = 5,
) -> None:
    """Download a list of image links with configurable multiprocessing.

    Args:
        image_links: Iterable of image URLs.
        download_folder: Directory where images will be saved.
        num_workers: Number of parallel worker processes.
        delay: Delay between requests per worker.
        max_retries: Retry count per failed URL.
    """
    image_links = list(image_links)
    os.makedirs(download_folder, exist_ok=True)
    download_image_partial = partial(
        download_image,
        savefolder=download_folder,
        delay=delay,
        max_retries=max_retries,
    )
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
