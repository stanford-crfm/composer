import json
import os
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple

from tqdm import tqdm
from webdataset import ShardWriter, WebDataset
from wurlitzer import pipes


def create_webdataset_meta(split_dir: str, n_samples: int, n_shards: int) -> None:
    '''Write a WebDataset meta file.'''
    samples_per_shard = n_samples // n_shards
    n_leftover = n_samples % samples_per_shard
    obj = {
        'n_shards': n_shards,
        'samples_per_shard': samples_per_shard,
        'n_leftover': n_leftover,
    }
    filename = os.path.join(split_dir, 'meta.json')
    json.dump(obj, open(filename, 'w'), sort_keys=True)


def create_webdataset(samples: Iterable[Dict[str, Any]],
                      dataset_dir: str,
                      split: str,
                      n_samples: int,
                      n_shards: int,
                      use_tqdm: int = 1) -> None:
    '''Write an entire WebDataset to a local directory, given an iterable of samples.'''
    split_dir = os.path.join(dataset_dir, split)
    os.makedirs(split_dir)
    pattern = os.path.join(split_dir, '%05d.tar')
    samples_per_shard = n_samples // n_shards
    with pipes():
        out = ShardWriter(pattern, maxcount=samples_per_shard)
        out.verbose = 0
    if use_tqdm:
        samples = tqdm(samples, total=n_samples, leave=False)
    for sample in samples:
        out.write(sample)
    out.close()
    create_webdataset_meta(split_dir, n_samples, n_shards)


def download_webdataset_meta(dataset_s3_bucket: str, split: str) -> bytes:
    '''Download a WebDataset meta file from S3.'''
    url = f's3://{dataset_s3_bucket}/{split}/meta.json'
    cmd = 'aws', 's3', 'cp', url, '-'
    return subprocess.run(cmd, capture_output=True).stdout


def load_webdataset(dataset_s3_bucket: str,
                    dataset_name: str,
                    split: str,
                    cache_dir: Optional[str] = None,
                    cache_verbose: bool = False) -> Tuple[WebDataset, dict]:
    '''Initialize a WebDataset pointed at S3 with an optional local cache dir.'''
    if cache_dir:
        split_dir = os.path.join(cache_dir, dataset_name, split)
        meta_file = os.path.join(split_dir, 'meta.json')
        if os.path.exists(meta_file):
            text = open(meta_file).read()
        else:
            text = download_webdataset_meta(dataset_s3_bucket, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            with open(meta_file, 'wb') as out:
                out.write(text)
    else:
        text = download_webdataset_meta(dataset_s3_bucket, split)
    meta = json.loads(text)
    max_shard = meta['n_shards'] - 1
    shards = f'{{{0:05d}..{max_shard:05d}}}.tar'
    urls = f'pipe: aws s3 cp s3://{dataset_s3_bucket}/{split}/{shards} -'
    dataset = WebDataset(urls, cache_dir=cache_dir, cache_verbose=cache_verbose)
    return dataset, meta