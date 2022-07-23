from appdirs import *
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

LFM_DATASET_PATH="https://cikm-evalrs-dataset.s3.us-west-2.amazonaws.com/lfm_1b_dataset.zip"

def download_with_progress(url, destination):
    """
    Downloads a file with a progress bar

    :param url: url from which to download from
    :destination: file path for saving data
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    with tqdm.wrapattr(open(destination, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)

def decompress_zipfile(file, outfile):
    with zipfile.ZipFile(file , 'r') as zip_ref:
        zip_ref.extractall(outfile)

def get_cache_directory():
    """
    Returns the cache directory on the system
    """
    appname = "evalrs"
    appauthor = "evalrs"
    cache_dir = user_cache_dir(appname, appauthor)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


