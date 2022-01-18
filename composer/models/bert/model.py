# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Tuple

import torch
import transformers
from torchmetrics import Accuracy, MatthewsCorrcoef, MeanSquaredError, SpearmanCorrcoef
from torchmetrics.collections import MetricCollection

# from composer.models.loss import CrossEntropyLoss, soft_cross_entropy
from composer.models.nlp_metrics import BinaryF1Score, CrossEntropyMetric, MaskedAccuracy
from composer.models.transformer_shared import MosaicTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, BatchDict, Metrics, Tensors


class BERTModel(MosaicTransformer):
    """
    Implements a BERT wrapper around a MosaicTransformer.
    """

    def __init__(self,
                 module: transformers.BertModel,
                 config: transformers.BertConfig,
                 tokenizer_name: str,
                 label_smoothing: Optional[float] = None) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer_name=tokenizer_name)

        # we're going to remove the label from the expected inputs
        # since we will handle metric calculation with TorchMetrics instead of HuggingFace.
        self.model_inputs.remove("labels")
        self.label_smoothing = label_smoothing

        self.train_metrics = []
        self.val_metrics = []

        self.num_classes = config.num_labels
        # TODO (Moin): make sure this is moved to be dataset-specific
        # if config.num_labels=1, then we are training a regression task, so we should update our loss functions
        if config.num_labels == 1:
            self.train_loss = MeanSquaredError()
            self.val_loss = MeanSquaredError()

            self.train_spearman = SpearmanCorrcoef()
            self.val_spearman = SpearmanCorrcoef()

            self.train_metrics.extend([self.train_loss, self.train_spearman])
            self.val_metrics.extend([self.val_loss, self.val_spearman])

        if config.num_labels == 2:
            self.train_f1 = BinaryF1Score()
            self.val_f1 = BinaryF1Score()

            self.train_metrics.extend([self.train_f1])
            self.val_metrics.extend([self.val_f1])

        if config.num_labels > 1 and config.num_labels != len(self.tokenizer):
            self.train_acc = Accuracy()
            self.val_acc = Accuracy()

            self.train_matthews = MatthewsCorrcoef(num_classes=self.num_classes)
            self.val_matthews = MatthewsCorrcoef(num_classes=self.num_classes)

            self.train_metrics.extend([self.train_acc, self.train_matthews])
            self.val_metrics.extend([self.val_acc, self.val_matthews])

        if config.num_labels == len(self.tokenizer):  # tests for MLM pre-training
            self.ignore_index = -100
            self.train_loss = CrossEntropyMetric(ignore_index=self.ignore_index, vocab_size=self.num_classes)
            self.val_loss = CrossEntropyMetric(ignore_index=self.ignore_index, vocab_size=self.num_classes)

            self.train_acc = MaskedAccuracy(ignore_index=self.ignore_index)
            self.val_acc = MaskedAccuracy(ignore_index=self.ignore_index)

            self.train_metrics.extend([self.train_loss, self.train_acc])
            self.val_metrics.extend([self.val_loss, self.val_acc])

    def loss(self, outputs: Mapping, batch: Batch, *args, **kwargs) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        elif self.label_smoothing is not None:
            log_probs = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            labels = outputs['labels']
            log_probs = -torch.nn.functional.log_softmax(log_probs, dim=-1)
            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)

            padding_mask = labels.eq(self.ignore_index)
            # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
            # will ignore them in any case.
            labels = torch.clamp(labels, min=0)
            nll_loss = log_probs.gather(dim=-1, index=labels)
            # works for fp16 input tensor too, by internally upcasting it to fp32
            smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

            nll_loss.masked_fill_(padding_mask, 0.0)
            smoothed_loss.masked_fill_(padding_mask, 0.0)

            # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            nll_loss = nll_loss.sum() / num_active_elements
            smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
            return (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smoothed_loss
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')
            # return soft_cross_entropy(outputs['logits'], outputs['labels'], *args, **kwargs)

    def forward(self, batch: Batch) -> Mapping:
        """Runs the forward pass of the model.
        TODO: update this docstring

        Args:
            batch (Batch): A dictionary of Dict[str, Tensor] of inputs that the
                model expects, as found in MosaicTransformer.get_model_inputs().

        Returns:
            A dictionary of model outputs as a ``Mapping``. It will include the loss
            if `labels` is passed as an input.
        """
        if self.label_smoothing is not None:
            labels = batch.pop("labels")

        output = super().forward(batch)

        if self.label_smoothing is not None:
            output['labels'] = labels
        return output

    def validate(self, batch: BatchDict) -> Tuple[Tensors, Tensors]:
        """Runs the validation step.

        Args:
            batch (BatchDict): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in MosaicTransformer.get_model_inputs().

        Returns:
            A tuple of (Tensor, Tensor) with the output from the forward pass and the correct labels.
            This is fed into directly into the output of :meth:`metrics`.
        """

        assert self.training is False, "For validation, model must be in eval mode"

        # temporary hack until eval on multiple datasets is finished
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']

        # if we are in the single class case, then remove the classes dimension
        if output.shape[1] == 1:
            output = output.squeeze(dim=1)

        output['labels'] = labels
        return (output, labels)

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection(self.train_metrics) if train else MetricCollection(self.val_metrics)
