# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

from torchmetrics import Accuracy, MatthewsCorrcoef, MeanSquaredError, SpearmanCorrcoef
from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import BinaryF1Score, CrossEntropyLoss, MaskedAccuracy
from composer.models.transformer_shared import ComposerTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, BatchDict, Metrics, Tensors


class T5Model(ComposerTransformer):
    """
    Implements a T5 wrapper around a ComposerTransformer.
    """

    def __init__(self, module: transformers.T5Model, config: transformers.T5Config, tokenizer_name: str) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer_name=tokenizer_name)

        # we're going to remove the label from the expected inputs
        # since we will handle metric calculation with TorchMetrics instead of HuggingFace.
        # self.model_inputs.remove("labels")

        self.train_metrics = []
        self.val_metrics = []

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    # def validate(self, batch: BatchDict) -> Tuple[Tensors, Tensors]:
    #     """Runs the validation step.

    #     Args:
    #         batch (BatchDict): a dictionary of Dict[str, Tensor] of inputs
    #             that the model expects, as found in ComposerTransformer.get_model_inputs().

    #     Returns:
    #         A tuple of (Tensor, Tensor) with the output from the forward pass and the correct labels.
    #         This is fed into directly into the output of :meth:`metrics`.
    #     """

    #     assert self.training is False, "For validation, model must be in eval mode"

    #     # temporary hack until eval on multiple datasets is finished
    #     labels = batch.pop('labels')
    #     output = self.forward(batch)
    #     # output = output['logits']

    #     # # if we are in the single class case, then remove the classes dimension
    #     # if output.shape[1] == 1:
    #     #     output = output.squeeze(dim=1)

    #     return (output, labels)

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
