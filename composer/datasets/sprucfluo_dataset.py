from typing import List, Union, Any, Dict, Optional
from dataclasses import dataclass

import yahp as hp
import composer.sprucfluo as sprucfluo

from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams

from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe

def _split_dict_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if isinstance(batch, dict):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        for k, v in chunked.items():
            if len(v) != n_microbatches:
                raise ValueError(
                    f"Unable to split batch into microbatches. "
                    f"Key '{k}' has chunked list: {v} with length {len(v)}, but expected length {n_microbatches}. "
                )
        microbatches = []
        for idx in range(n_microbatches):
            mb = {k: v[idx] for k, v in chunked.items()}
            microbatches.append(mb)
        return microbatches
    else:
        raise ValueError(
            f"Expected batch to be of type Dict[str, Tensor], but got {type(batch)}"
        )


@dataclass
class SprucfluoDatasetSpecHparams(hp.Hparams):
    name: str = hp.required("name of the dataset")
    urls: List[str] = hp.required("urls of the dataset. Supports braceexpand")
    json_text_key: str = hp.optional("key of the json text", default="text")
    extra_fsspec_args: dict[str, Any] = hp.optional("fsspec args. Use for s3, gs, etc.", default_factory=lambda: {})

    def validate(self):
        pass

    def initialize_object(self):
        return sprucfluo.load_corpus(self.urls, **self.extra_fsspec_args)


@dataclass
class SprucfluoDatasetHparams(DatasetHparams):
  """
      Builds a DataSpec that uses Sprucfluo for data processing.

      Parameters:
          - datasets: List of SprucfluoDatasetSpecHparams
          - weights: dict of [str, float] for weights (optional)
  """
  hparams_registry = {  # type: ignore
      "datasets": {"dataset": SprucfluoDatasetSpecHparams},
  }
  datasets: List[SprucfluoDatasetSpecHparams] = hp.optional("list of SprucfluoDatasetSpec", default_factory=lambda: [])
  weights: Optional[Dict[str, float]] = hp.optional("dict of [str, float] for weights", default=None)

  num_samples: int = hp.optional(
        "The number of post-processed token samples, used to set epoch size of the IterableDataset.", default=None)
  tokenizer_name: str = hp.optional(
      "The name of the HuggingFace tokenizer to preprocess text with.", default=None
  )
  max_seq_len: int = hp.optional(
      "The max sequence length of each token sample.", default=1024
  )
  shuffle: bool = hp.optional(
      "Whether to shuffle the samples in the dataset. Currently, shards are assigned and consumed with deterministic "
      "per-device shard order, but shuffling affects the order of samples via (per-device) shuffle buffers.",
      default=True,
  )
  shuffle_buffer_size: int = hp.optional(
      "If `shuffle=True`, samples are read into a buffer of this size (per-device), and randomly sampled from there "
      "to produce shuffled samples.",
      default=10000,
  )
  seed: int = hp.optional(
      "If `shuffle=True`, what seed to use for shuffling operations.", default=5
  )
  drop_last: bool = hp.optional(
      "Whether to drop the last samples for the last batch.", default=True
  )

  def validate(self):
      assert len(self.datasets) > 0, "datasets must be a list of SprucfluoDatasetSpec"
      assert len(list(d.name for d in self.datasets)) == len(set(d.name for d in self.datasets)), "datasets must have unique names"

      if self.weights is not None:
          assert len(self.weights) == len(self.datasets), "weights must be a dict of [str, float] for weights"
          for dataset in self.datasets:
              assert dataset.name in self.weights, f"{dataset.name} not in weights"


  def initialize_object(self, batch_size:int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
      try:
          import transformers
      except ImportError:
          raise ImportError('HuggingFace transformers not installed. '
                            'Please install with `pip install composer[nlp]`')

      if dataloader_hparams.num_workers > 1:
          log.warning("Sprucfluo Dataset not compatible with num_workers > 1. Overwriting value to num_workers=1")
          dataloader_hparams.num_workers = 1

      datasets = {d.name: d.initialize_object() for d in self.datasets}

      tokenizer = AutoTokenizer.from_pretrained("gpt2")

      process_fun = functools.partial(tokenize_and_group_texts, tokenizer=tokenizer, seq_len=self.max_seq_len)

      datasets = {k: dataset.then(process_fun) for k, dataset in datasets.items()}

      if len(datasets) == 1:
          dataset = next(iter(datasets.values()))
      else:
          weights = self.weights
          if weights is None:
              weights = {d.name: 1.0 for d in self.datasets}
          weights = {datasets[d.name]: weights[d.name] for d in self.datasets if weights[d.name] > 0}
          dataset = SampleMultiplexerDataPipe(weights, seed=self.multiplex_seed)

      if self.shuffle:
          dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed)

      # Get collate_fn
      collate_fn = transformers.DataCollatorForLanguageModeling(
          tokenizer=tokenizer,
          mlm=False,
      )
      # Return DataSpec
      return DataSpec(
          dataloader=dataloader_hparams.initialize_object(
              dataset=dataset,
              batch_size=batch_size,
              sampler=None,
              drop_last=True,
              collate_fn=collate_fn,
          ),
          split_batch=_split_dict_fn,
      )
