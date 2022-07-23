"""

    These utility functions are used to download the official dataset from a public bucket,
    and upload a submission to the Data Challenge S3, so that the leadearboard can be updated accordingly.

    You should not need to modify this script: if in doubt, ask the organizers first.

"""

from appdirs import *
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import os
import boto3
from datetime import datetime


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


def upload_submission(
        local_file: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        participant_id: str,
        bucket_name: str,

):
    """

    :param local_file: local path, may be only the file name or a full path
    :param task: rec or cart
    :return:
    """

    print("Starting submission at {}...\n".format(datetime.utcnow()))
    # instantiate boto3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id ,
        aws_secret_access_key=aws_secret_access_key,
        region_name='us-west-2'
    )
    s3_file_name = os.path.basename(local_file)
    # prepare s3 path according to the spec
    s3_file_path = '{}/{}'.format(participant_id, s3_file_name)  # it needs to be like e.g. "id/*.json"
    # upload file
    s3_client.upload_file(local_file, bucket_name, s3_file_path)
    # say bye
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))

    return