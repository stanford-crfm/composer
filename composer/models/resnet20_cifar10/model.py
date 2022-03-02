# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnets import CIFAR_ResNet

__all__ = ["CIFAR10_ResNet20"]


class CIFAR10_ResNet20(ComposerClassifier):
    """A ResNet-20 model extending :class:`.ComposerClassifier`.

    From the paper Deep Residual Learning for Image Recognition `<https://arxiv.org/abs/1512.03385>`_.

    Example:

    .. testcode::

        from composer.models import CIFAR10_ResNet20

        model = CIFAR10_ResNet20()  # creates a resnet20 for cifar image classification

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: 10.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``).
    """

    def __init__(
        self,
        num_classes: int = 10,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = CIFAR_ResNet.get_model_from_name(
            "cifar_resnet_20",
            initializers,
            num_classes,
        )
        super().__init__(module=model)
