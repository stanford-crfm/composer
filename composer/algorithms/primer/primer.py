# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import asdict, dataclass
from operator import attrgetter
from types import MethodType, ModuleType
from typing import Any, Callable, Optional, Union

import torch
import yahp as hp
from torch.nn.functional import relu

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


@dataclass
class PrimerHparams(AlgorithmHparams):
    """See :class:`Primer`"""

    def initialize_object(self) -> "Primer":
        return Primer(**asdict(self))


def apply_primer(model: torch.nn.Module) -> None:
    """
    Removes position embeddings and replaces the attention function and attention mask
    according to `AliBi <https://arxiv.org/abs/2108.12409>`_.

    Args:
        model: model to transform
        heads_per_layer: number of attention heads per layer
        max_sequence_length: maximum sequence length that the
            model will be able to accept without returning an error
        position_embedding_attribute: attribute for position
            embeddings. For example in HuggingFace's GPT2, the
            position embeddings are "transformer.wpe".
        attention_module: module/class that will have its
            self-attention function replaced. For example, in
            HuggingFace's GPT, the self-attention module is
            transformers.models.gpt2.modeling_gpt2.GPT2Attention.
        attr_to_replace: attribute that self-attention function will
            replace. For example, in HuggingFace's GPT2, the
            self-attention function is "_attn".
        alibi_attention: new self-attention function in which
            ALiBi is implemented. Used to replace
            "{attention_module}.{attr_to_replace}".
        mask_replacement_function: function to replace model's
            attention mask. This is sometimes necessary for evaluating
            on sequence lengths longer than the model was initialized to
            accommodate.
    """

    for idx in range(len(model.module.transformer.h)):
        model.module.transformer.h[idx].mlp.act = lambda x: relu(x)**2


class Primer(Algorithm):
    def __init__(self) -> None:

        self.hparams = PrimerHparams()

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None
            apply_primer(state.model)


def lazy_import(name: Union[str, None]) -> Any[Callable, ModuleType, None]:
    if not name:
        return None
    components = name.split('.')
    try:
        mod = importlib.import_module(components[0])
    except (ValueError, ModuleNotFoundError):
        log.exception(f"Module {components[0]} not found when attempting "
                      f"to import {name}. Please confirm the name and "
                      f"module path you're attempting to import.")
    try:
        mod = attrgetter('.'.join(components[1:]))(mod)  # type: ignore
    except (ValueError, AttributeError):
        log.exception(f"Unable to import {name}. "
                      f"Please confirm the name and module "
                      f" path you're attempting to import.")
    return mod  # type: ignore
