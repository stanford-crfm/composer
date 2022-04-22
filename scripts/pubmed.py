# HuggingFace loading script for the PubMed portion of The Pile + plain medical text.
from collections import defaultdict
from typing import Dict, List
import gzip
import json
from random import Random

import datasets

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
N_SHARDS_PER_CORPUS_PER_SPLIT_PUBMED: Dict[str, Dict[str, int]] = {
    "Abs": {"train": 128, "val": 8},
    "C": {"train": 128, "val": 8},
    "randomized": {"train": 128, "val": 8},
    "medical": {"train": 128, "val": 8},
    "openwebtext": {"train": 128, "val": 8},
    "pubmed_plus_openweb": {"train": 128, "val": 8}
}

DATA_URL_BY_CORPUS: Dict[str, str] = {
    "Abs": "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
           "pubmedAbs_{split}.{index}-of-{n_shards}.jsonl.gz",
    "C": "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
         "pubmedC_{split}.{index}-of-{n_shards}.jsonl.gz",
    "randomized": "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
         "pubmedRandomized_{split}.{index}-of-{n_shards}.jsonl.gz",
    # TODO: split this into its own dataset, add mixing
    "medical": "https://storage.googleapis.com/pubmed-mosaic/plain-medical-text-sharded/"
               "plain_medical_text_{split}.{index}-of-{n_shards}.jsonl.gz",
    "openwebtext": "https://storage.googleapis.com/pubmed-mosaic/openwebtext-sharded/"
               "openwebtext_{split}.{index}-of-{n_shards}.jsonl.gz",
    "pubmed_plus_openweb": "https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/"
               "pubmed_plus_openweb_{split}.{index}-of-{n_shards}.jsonl.gz"
}

NAMES_TO_CORPORA: Dict[str, str] = {
    "Abs": ["Abs"],
    "C": ["C"],
    "randomized": ["randomized"],
    "medical": ["medical"],
    _ALL: ["Abs", "C", "medical"],
    "pubmed": ["Abs", "C"],
    "pubmed_randomized": ["randomized"],
    "openwebtext": ["openwebtext"],
    "pubmed_plus_openweb": ["pubmed_plus_openweb"]
}


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
        for name in NAMES_TO_CORPORA.keys()
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
        for corpus in NAMES_TO_CORPORA[self.config.name]:
            shard_info = N_SHARDS_PER_CORPUS_PER_SPLIT_PUBMED[corpus]
            for split, n_shards in shard_info.items():
                data_urls[split].extend(
                    [
                        DATA_URL_BY_CORPUS[corpus].format(
                            name=corpus,
                            split=split,
                            index=index + 1,
                            n_shards=n_shards,
                        )
                        for index in range(n_shards)
                    ]
                )

        # TODO: is random.shuffle good enough to ensure we're interleaving data from different data sources?
        # pretty sure no, so need to revisit
        gen = Random(0)
        gen.shuffle(data_urls["train"])
        gen.shuffle(data_urls["val"])

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
