# Copyright 2021 MosaicML. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core.types import DataSpec
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist
from composer.utils.data import NormalizationFn, pil_image_collate

import math
import sys
import webdataset as wds

# ImageNet normalization values from torchvision: https://pytorch.org/vision/stable/models.html
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


@dataclass
class ImagenetStreamingDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet dataset for image classification.
    
    Parameters:
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
    """
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams):

        if self.use_synthetic:
            total_dataset_size = 1_281_167 if self.is_train else 50_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, self.crop_size, self.crop_size],
                num_classes=1000,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None
        else:

            if self.is_train:
                # include fixed-size resize before RandomResizedCrop in training only
                # if requested (by specifying a size > 0)
                train_resize_size = self.resize_size
                train_transforms: List[torch.nn.Module] = []
                if train_resize_size > 0:
                    train_transforms.append(transforms.Resize(train_resize_size))
                # always include RandomResizedCrop and RandomHorizontalFlip
                train_transforms += [
                    transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                    transforms.RandomHorizontalFlip()
                ]
                transformation = transforms.Compose(train_transforms)
                split = "train"
            else:
                transformation = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                ])
                split = "validation"

            device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
            collate_fn = pil_image_collate

            cache = False
            cache_dir = f"./i1k_{split}_cache"
            cache_verbose =  True
            #########################################
            # TODO: read this info from S3 directly
            n_shards_map = {
                "train": 1024,
                "validation": 128,
            }
       
            size_map = {
                "train": 1281024,
                "validation": 49920,
            }
            #########################################
            
            dataset_size = size_map[split]
            device_dataset_size = dataset_size // dist.get_world_size()
            n_shards = n_shards_map[split]
            global_num_workers =  dist.get_world_size() * dataloader_hparams.num_workers
            assert n_shards % global_num_workers == 0, f"{n_shards=} not divisible by {global_num_workers=}"
            shard_digits = math.ceil(math.log(n_shards, 10))
            zeros = "0"*shard_digits
            
            urls = f"pipe: aws s3 cp s3://mosaicml-internal-dataset-i1k/i1k-{split}-{{{zeros}..{n_shards-1}}}.tar -"

            dataset = wds.WebDataset(
                urls=urls,
                cache_dir=(cache_dir if cache else None),
                cache_verbose=cache_verbose,
            )

            dataset = dataset.decode("pil").map_dict(jpg=transformation).to_tuple("jpg", "cls").with_epoch(device_dataset_size).with_length(device_dataset_size)


        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)
