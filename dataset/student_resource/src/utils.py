import re
import os
import pandas as pd
import multiprocessing
from time import time as timer, sleep
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib

# Original fast version (renamed)
def download_image_fast(image_link, savefolder):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images_fast(image_links, download_folder):
    """Original fast version with 100 workers and no rate limiting."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image_fast, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

# New version with retry logic and reasonable speed
def download_image(image_link, savefolder, delay=0.05, max_retries=3):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            sleep(delay)  # Small delay to avoid overwhelming servers
            
            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(image_link, image_save_path)
                    return  # Success
                except Exception as ex:
                    if attempt < max_retries - 1:
                        # Wait progressively longer between retries
                        wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                        sleep(wait_time)
                    else:
                        # Final attempt failed, but don't spam output
                        if attempt == max_retries - 1:
                            print('Warning: Failed after {} attempts - {}'.format(max_retries, filename))
        else:
            return
    return

def download_images(image_links, download_folder, num_workers=40, delay=0.05, max_retries=5):
    """
    Download images with retry logic and balanced speed.
    
    For 75,000 images with 10 workers and 0.05s delay:
    - Theoretical time: ~6-7 hours (accounting for download time)
    - Retries handle connection resets automatically
    
    Args:
        image_links: List of image URLs
        download_folder: Folder to save images
        num_workers: Number of parallel workers (default: 10 - balanced)
        delay: Delay in seconds between downloads per worker (default: 0.05)
        max_retries: Number of retry attempts for failed downloads (default: 3)
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder, delay=delay, max_retries=max_retries)
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()