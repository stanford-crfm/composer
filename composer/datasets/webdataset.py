import json
import logging
import math
import os
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple

from tqdm import tqdm
from webdataset import ShardWriter, WebDataset
from wurlitzer import pipes

log = logging.getLogger(__name__)


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


def download_webdataset_meta(s3_bucket: str, split: str) -> bytes:
    '''Download a WebDataset meta file from S3.'''
    url = f's3://{s3_bucket}/{split}/meta.json'
    cmd = 'aws', 's3', 'cp', url, '-'
    return subprocess.run(cmd, capture_output=True).stdout


def load_webdataset(s3_bucket: str,
                    cache_name: str,
                    split: str,
                    cache_dir: Optional[str] = None,
                    cache_verbose: bool = False) -> Tuple[WebDataset, dict]:
    '''Initialize a WebDataset pointed at S3 with an optional local cache dir.'''
    if cache_dir:
        split_dir = os.path.join(cache_dir, cache_name, split)
        meta_file = os.path.join(split_dir, 'meta.json')
        if os.path.exists(meta_file):
            text = open(meta_file).read()
        else:
            text = download_webdataset_meta(s3_bucket, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            with open(meta_file, 'wb') as out:
                out.write(text)
    else:
        split_dir = None
        text = download_webdataset_meta(s3_bucket, split)
    meta = json.loads(text)
    max_shard = meta['n_shards'] - 1
    shards = f'{{{0:05d}..{max_shard:05d}}}.tar'
    urls = f'pipe: aws s3 cp s3://{s3_bucket}/{split}/{shards} -'
    dataset = WebDataset(urls, cache_dir=split_dir, cache_verbose=cache_verbose)
    return dataset, meta


def size_webdataset(dataset: WebDataset, n_shards: int, samples_per_shard: int, n_devices: int, workers_per_device: int,
                    batch_size: int, drop_last: bool) -> WebDataset:
    '''Calculate WebDataset with_epoch() and with_length().'''
    workers_per_device = max(1, workers_per_device)

    # Ensure that shards can be split among CPU workers
    n_workers_global = n_devices * workers_per_device
    if n_shards % n_workers_global != 0:
        raise ValueError(f"{n_shards=} must be divisible by {n_workers_global=}!")

    # Set IterableDataset epoch boundary and length for DDP, PyTorch Dataloader compatability
    shards_per_worker = n_shards // n_devices // workers_per_device
    expected_samples_per_worker = samples_per_shard * shards_per_worker
    if drop_last:
        samples_per_worker = (expected_samples_per_worker // batch_size) * batch_size
        samples_per_device = samples_per_worker * workers_per_device
        samples_total = samples_per_device * n_devices
        expected_samples_total = n_shards * samples_per_shard
        if samples_total != expected_samples_total:
            log.warning(
                f"Note that 'drop_last=True' with per-CPU-worker sharding will cause an incomplete batch to be dropped at the end of ** each CPU worker's sample list **. "
                f"Given your training configuration, we have calculated this will reduce samples_per_epoch from {expected_samples_total} -> {samples_total}."
            )
    else:
        samples_per_worker = expected_samples_per_worker
        samples_per_device = samples_per_worker * workers_per_device
        samples_total = samples_per_device * n_devices
        expected_batches_per_epoch = math.ceil(samples_per_worker * workers_per_device / batch_size)
        batches_per_epoch = math.ceil(samples_per_worker / batch_size) * workers_per_device
        if batches_per_epoch != expected_batches_per_epoch:
            log.warning(
                f"Note that 'drop_last=False' with per-CPU-worker sharding will lead to multiple incomplete batches to be read from each device, ** one for each CPU worker **. "
                f"Unfortunately, the PyTorch Dataloader does not handle this situation well in its __len__ implementation, so len(dataloader) will be an underestimate of batches_per_epoch. "
                f"(See https://github.com/pytorch/pytorch/blob/3d9ec11feacd69d0ff1bffe0b25a825cdf203b87/torch/utils/data/dataloader.py#L403-L411). "
                f"Given your training configuration, we have calculated this will increase batches_per_epoch from {expected_batches_per_epoch} -> {batches_per_epoch}."
            )
    # Set epoch boundary (per CPU worker).
    # Technically not needed if shards are constructed correctly, but used for safety
    dataset = dataset.with_epoch(samples_per_worker)
    # Set IterableDataset length (per device), to be read by PyTorch Dataloader
    return dataset.with_length(samples_per_device)
