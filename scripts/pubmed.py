# HuggingFace loading script for the PubMed portion of The Pile + plain medical text.
from collections import defaultdict
from typing import Dict, List
import gzip
import json
import random

import datasets

# Fix the random seed for reproducibility
random.seed(0)


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION: str = """\
The Pile’s subsets of PubMed (abstracts and text) + plain medical text.
Hosted at https://storage.googleapis.com/pubmed-mosaic.
"""

# TODO: add all the plain medical text citations:
#   MedlinePlus
#   Merck manuals
#   Other medical manuals from major healthcare institutions
#   (Mayo, Cleveland, JHU, Stanford, NHLBI, UCSF, UPMC, UW, SloanKettering)
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
_N_SHARDS_PER_SPLIT_PUBMED: Dict[str, Dict[str, int]] = {
    "Abs": {"train": 128, "val": 8},
    "C": {"train": 128, "val": 8},
}

_DATA_URL_PUBMED: str = (
    "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
    "pubmed{name}_{split}.{index}-of-{n_shards}.jsonl.gz"
)

_N_SHARDS_PER_SPLIT_MEDICAL_TEXT: Dict[str, int] = {"train": 128, "val": 8}
_DATA_URL_MEDICAL_TEXT: str = (
    "https://storage.googleapis.com/pubmed-mosaic/plain-medical-text-sharded/"
    "plain_medical_text_{split}.{index}-of-{n_shards}.jsonl.gz"
)


# TODO: rename this dataset - MedicalTextDataset?
class PubMed(datasets.GeneratorBasedBuilder):
    """
    PubMed dataset from The Pile + plain medical text from multiple sources:
        MedlinePlus (30MB)
        Merck manuals (30MB)
        Other medical manuals from major healthcare institutions
        (Mayo, Cleveland, JHU, Stanford, NHLBI, UCSF, UPMC, UW, SloanKettering) (140MB)
    """

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name)
        for name in list(_N_SHARDS_PER_SPLIT_PUBMED) + [_ALL]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls: Dict[str, List[str]] = defaultdict(list)

        # Add the sharded PubMed
        for name, shard_info in _N_SHARDS_PER_SPLIT_PUBMED.items():
            for split, n_shards in shard_info.items():
                data_urls[split].extend(
                    [
                        _DATA_URL_PUBMED.format(
                            name=name,
                            split=split,
                            index=index + 1,
                            n_shards=n_shards,
                        )
                        for index in range(n_shards)
                    ]
                )

        # Add the sharded plain medical text
        for split, n_shards in _N_SHARDS_PER_SPLIT_MEDICAL_TEXT.items():
            data_urls[split].extend(
                [
                    _DATA_URL_MEDICAL_TEXT.format(
                        split=split,
                        index=index + 1,
                        n_shards=n_shards,
                    )
                    for index in range(n_shards)
                ]
            )

        # TODO: is random.shuffle good enough to ensure we're interleaving data from different data sources?
        random.shuffle(data_urls["train"])
        random.shuffle(data_urls["val"])

        assert len(data_urls["train"]) == 128 * 3, f"Expected {128 * 3} files, but got {len(data_urls['train'])}" \
                                                   f"\n{data_urls['train']}"
        assert len(data_urls["val"]) == 8 * 3, f"Expected {8 * 3} files, but got {len(data_urls['val'])}" \
                                               f"\n{data_urls['val']}"

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
