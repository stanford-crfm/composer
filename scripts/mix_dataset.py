import argparse
import json
import math
import os
import subprocess

from random import shuffle
from tqdm import tqdm
from typing import List, Tuple

"""
Concat dataset files, randomize, output to final mixture files.
We currently only support jsonl files.

Usage:

    python3 scripts/mix_dataset.py --name <Output file name> \
    --train-files <List of train files> --val-files <List of val files>

Example usage:

    python3 scripts/mix_dataset.py --name pubmedShuffled --output-path shuffled_data
    --train-files pubmed_data/pubmedAbs_train.jsonl pubmed_data/pubmedC_train.jsonl \
    --val-files pubmed_data/pubmedAbs_validation.jsonl pubmed_data/pubmedC_validation.jsonl 

"""


class LocalDatasetMixer:
    """Mix and potentially randomize local dataset files. Only works with jsonl files for now."""

    def __init__(
        self,
        output_path: str,
        name: str,
        train_files: List[str],
        val_files: List[str],
        randomize: bool = True,
        debug: bool = False,
    ):
        output_path = os.path.expanduser(output_path)
        os.makedirs(output_path, exist_ok=True)
        self.output_path: str = output_path
        self.name: str = name
        self.train_files: List[str] = train_files
        self.val_files: List[str] = val_files
        self.randomize: bool = randomize
        self.debug: bool = debug

    def mix_files(self, name: str, files_list: List[str], randomize: bool = True):
        print(f"Mixing files: {'\n'.join(files_list)}")
        all_docs = []
        print(f"Loading files for {name}")
        for f in files_list:
            f_docs = open(f, "r")
            if self.debug:
                f_docs = [json.loads(doc.strip("\n")) for doc in f_docs]
                for idx, doc in enumerate(f_docs):
                    doc["source"] = f
                    doc["orig_idx"] = idx
                f_docs = [json.dumps(doc) + "\n" for doc in f_docs]
            all_docs += f_docs
        if self.randomize:
            shuffle(all_docs)
            if self.debug:
                for doc in all_docs:
                    doc_json = json.loads(doc.strip("\n"))
                    print(doc_json["source"] + "\t" + str(doc_json["orig_idx"]))
        with open(f"{self.output_path}/{name}.jsonl", "w") as f_write:
            for doc in all_docs:
                f_write.write(doc)

    def mix(self):
        print("Processing train files...")
        self.mix_files(
            name=f"{self.name}_train",
            files_list=self.train_files,
            randomize=self.randomize,
        )

        print("Processing validation files...")
        self.mix_files(
            name=f"{self.name}_validation",
            files_list=self.val_files,
            randomize=self.randomize,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", help="Where to output the sharded files", default="~"
    )
    parser.add_argument("--name", help="Name for mixture files", default="~")
    parser.add_argument(
        "--train-files", help="Files for the training set", nargs="+", default=[]
    )
    parser.add_argument(
        "--val-files", help="Files for the validation set", nargs="+", default=[]
    )
    parser.add_argument(
        "--debug",
        help="Run steps for testing correctness of script",
        action="store_true",
    )

    args = parser.parse_args()

    mixer = LocalDatasetMixer(
        output_path=args.output_path,
        name=args.name,
        train_files=args.train_files,
        val_files=args.val_files,
        debug=args.debug,
    )
    mixer.mix()
    print("\nDone.")


if __name__ == "__main__":
    main()
