# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import asdict, dataclass
from operator import attrgetter
from types import MethodType, ModuleType
from typing import Any, Callable, List, Optional, Union

import torch
import yahp as hp
from torch.nn.functional import relu

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


@dataclass
class PrimerHparams(AlgorithmHparams):
    """See :class:`Primer`"""
    use_squared_relu: bool = hp.required("Whether to use squared ReLUs as the activation function or not.")
    use_dconv: bool = hp.required("Whether to add depth-wise convolutions after each multi-headed projection.")

    def initialize_object(self) -> "Primer":
        return Primer(**asdict(self))


def apply_primer(model: torch.nn.Module,
                 use_squared_relu: bool,
                 use_dconv: bool,
                 dconv_fns: Optional[List[torch.nn.Module]] = None) -> None:
    if use_squared_relu:
        print("Squaring the ReLU!")
        for idx in range(len(model.module.transformer.h)):
            model.module.transformer.h[idx].mlp.act = lambda x: relu(x)**2
    else:
        print("Not squaring the ReLU!")

    if use_dconv:
        print("Using the DConv!")
        # model_dim is the output of the embedding layer
        model_dim = model.module.transformer.wte.embedding_dim
        n_heads = model.config.n_head
        assert (model_dim % n_heads) == 0
        dim_per_head = model_dim // n_heads
        kernel_size = 3

        model.module.q_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)
        model.module.k_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)
        model.module.v_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)

        orig_attn_fn = model.module.transformer.h[0].attn._attn

        def dconv_attn(query, key, value, attention_mask=None, head_mask=None):
            # query shape is (bs x nhead x seq_len x head_dim)
            # the dconv expects (bs x seq_len x nhead x head_dim)
            query = model.module.q_dconv(query.transpose(1, 2)).transpose(1, 2)
            key = model.module.k_dconv(query.transpose(1, 2)).transpose(1, 2)
            value = model.module.v_dconv(query.transpose(1, 2)).transpose(1, 2)
            attn = orig_attn_fn(query, key, value, attention_mask=attention_mask, head_mask=head_mask)
            return attn

        for idx in range(len(model.module.transformer.h)):
            model.module.transformer.h[idx].attn._attn = dconv_attn
    else:
        print("Not using the DConv!")


def shift(x: torch.Tensor, amt: int, dim: int = -1):
    return torch.nn.functional.pad(x, (*((0, 0) * (-dim - 1)), amt, -amt), value=0.0)


# adapted from GPT Neo-X
class CausalDepthwiseConv(torch.nn.Module):

    def __init__(self, dim_per_head, n_heads, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.empty(size=(kernel_size, n_heads, dim_per_head)))
        # weird init from https://github.com/google-research/google-research/blob/3e1a06764ff52e33e3523d82ae836441df701c5d/primer/t5_models.py#L35
        torch.nn.init.constant_(self.weight, 0.5 / kernel_size)
        torch.nn.init.constant_(self.weight[0], 0.5)

    def forward(self, x, seq_dim=1):
        # x should be [b, s, np, hp]
        ret = x * self.weight[0]
        for shift_distance in range(1, self.kernel_size):
            x = shift(x, 1, dim=seq_dim)
            ret += x * self.weight[shift_distance]
        return ret


class Primer(Algorithm):

    def __init__(self, use_squared_relu: bool, use_dconv: bool) -> None:
        self.q_dconv = None
        self.k_dconv = None
        self.v_dconv = None
        self.use_squared_relu = use_squared_relu
        self.use_dconv = use_dconv

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None
            apply_primer(state.model, use_squared_relu=self.use_squared_relu, use_dconv=self.use_dconv)


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
