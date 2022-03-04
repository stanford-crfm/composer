# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
from dataclasses import dataclass
from typing import List

import torch
import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist


@dataclass
class CIFAR10DatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification.

    Parameters:
        download (bool): Whether to download the dataset, if needed.
    """
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar10_mean, cifar10_std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
                from ffcv.loader import Loader, OrderOption
                from ffcv.transforms import (Convert, Cutout, RandomHorizontalFlip, RandomResizedCrop, RandomTranslate,
                                             ToDevice, ToTensor, ToTorchImage)
                from ffcv.transforms.common import Squeeze
            except ImportError:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, run `pip install mosaicml[ffcv]`"""))

            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_ffcv_mean = [125.307, 122.961, 113.8575]
            cifar10_ffcv_std = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend([
                    RandomHorizontalFlip(),
                    #RandomTranslate(padding=2, fill=tuple(map(int, cifar10_ffcv_mean))),
                    #Cutout(4, tuple(map(int, cifar10_ffcv_mean))),
                ])
            # Common transforms for train and test
            image_pipeline.extend([
                ToTensor(),
                ToTorchImage(channels_last=False, convert_back_int16=False),
                Convert(torch.float32),
                transforms.Normalize(cifar10_ffcv_mean, cifar10_ffcv_std),
            ])

            ordering = OrderOption.RANDOM if self.is_train else OrderOption.SEQUENTIAL

            filepath = "/cifar10_train.beton" if self.is_train else "/cifar10_test.beton"

            return Loader(self.datadir + filepath,
                          batch_size=batch_size,
                          num_workers=dataloader_hparams.num_workers,
                          order=ordering,
                          drop_last=self.drop_last,
                          pipelines={
                              'image': image_pipeline,
                              'label': label_pipeline
                          })
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            if self.is_train:
                transformation = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ])
            else:
                transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ])

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)
