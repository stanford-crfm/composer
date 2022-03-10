"""PubMed dataset from The Pile."""

from typing import Dict, List
import gzip
import json

import datasets


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION: str = """\
The Pile’s subsets of PubMed (abstracts and text).
Hosted at https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded.
"""

_CITATION: str = """
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles 
          and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
"""

_URL: str = "https://github.com/stanford-crfm/composer"

_ALL: str = "all"
_N_SHARDS_PER_SPLIT: Dict[str, Dict[str, int]] = {
    "Abs": {"train": 128, "val": 8},
    "C": {"train": 128, "val": 8},
}

_DATA_URL: str = (
    "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
    "pubmed{name}_{split}.{index}-of-{n_shards}.jsonl.gz"
)


class PubMed(datasets.GeneratorBasedBuilder):
    """PubMed dataset from The Pile"""

    BUILDER_CONFIGS = [datasets.BuilderConfig(name) for name in list(_N_SHARDS_PER_SPLIT) + [_ALL]]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls: Dict[str, List[str]] = {}
        if self.config.name == _ALL:
            logger.info(f"Pulling all: {list(_N_SHARDS_PER_SPLIT)}...")
            names: List[str] = list(_N_SHARDS_PER_SPLIT)
        else:
            names: List[str] = [self.config.name]
            logger.info(f"all was not set for name, pulling: {names}")

        for split in ["train", "val"]:
            for name in names:
                n_shards: int = _N_SHARDS_PER_SPLIT[name][split]
                data_urls[split] = [
                    _DATA_URL.format(
                        name=name,
                        split=split,
                        index=index + 1,
                        n_shards=n_shards,
                    )
                    for index in range(n_shards)
                ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": dl_manager.download(data_urls["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": dl_manager.download(data_urls["val"])},
            ),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_: int = 0
        for filepath in filepaths:
            logger.info(f"Generating examples from {filepath}")
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)
                        yield id_, example
                        id_ += 1
