import itertools
from typing import TypeVar, Iterator

from torch.utils.data import functional_datapipe, IterDataPipe

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("cycle")
class CycleDataPipe(IterDataPipe[T_co]):
    r"""
    A data pipe tha loops over the data
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co]) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[T_co]:
        yield from itertools.cycle(self.source_datapipe)

    def __len__(self):
        raise ValueError("len is not supported on CycleDataPipe, as it is infinite")
