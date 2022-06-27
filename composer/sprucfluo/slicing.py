# Copyright 2022 The Board of Trustees of the Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import islice
from typing import Iterator, TypeVar, Sized, Optional

from torch.utils.data import functional_datapipe, IterDataPipe

T_co = TypeVar('T_co', covariant=True)

@functional_datapipe('slice')
class SliceIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, iterable: IterDataPipe[T_co], start_or_stop: int, stop: Optional[int] = None,
                 stride: Optional[int] = None) -> None:
        self.iterable = iterable
        if stop is None and stride is None:
            self._slice = slice(start_or_stop)
        else:
            self._slice = slice(start_or_stop, stop, stride)

    def __iter__(self) -> Iterator[T_co]:
        if self._slice.stop is not None and self._slice.stop < 0:
            stop = self._computed_base_len()
        else:
            stop = self._slice.stop
        return islice(self.iterable, self._slice.start, stop, self._slice.step)

    def _computed_base_len(self) -> Optional[int]:
        base_len = self._slice.stop
        if base_len is None or base_len < 0:
            if not isinstance(self.iterable, Sized):
                raise ValueError('Cannot compute length of unsized iterable')

            base_len = len(self.iterable)
            if self._slice.stop is not None:
                base_len = base_len + self._slice.stop

        return base_len

    def __len__(self) -> int:
        stop = self._computed_base_len()
        step = self._slice.step or 1
        start = self._slice.start or 0
        return (stop - start + (step - 1)) // step


def _take_data_pipe(data_pipe: IterDataPipe[T_co], num: int) -> IterDataPipe[T_co]:
    return data_pipe.slice(num)


def _drop_data_pipe(data_pipe: IterDataPipe[T_co], num: int) -> IterDataPipe[T_co]:
    return data_pipe.slice(num, None)


def _init_slicing():
    IterDataPipe.register_function("take", _take_data_pipe)
    IterDataPipe.register_function("drop", _drop_data_pipe)

_init_slicing()
