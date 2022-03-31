import argparse
import math
import os
import subprocess

from tqdm import tqdm
from typing import List, Tuple

"""
Shard the training and validation sets of a given dataset into n_shards_train and n_shards_val shards.
We currently only support jsonl files.

Usage:

    python3 scripts/shard_dataset.py \
    --n-shards-train <Number of shards for training set> --n-shards-val <Number of shards for validation set> \
    --train-files <List of train files> --val-files <List of val files>

Example usage:

    python3 scripts/shard_dataset.py --output-path plain_medical_text --n-shards-train 128 --n-shards-val 8 \
    --train-files plain_medical_text/plain_medical_text_train.jsonl \
    --val-files plain_medical_text/plain_medical_text_val.jsonl 

Use --compress option to compress files after sharding.
"""


class LocalDatasetSharder:
    """Shard local dataset files. Only works with jsonl files for now."""

    @staticmethod
    def get_file_info(path: str) -> Tuple[str, str]:
        """Given a path, return the name of the file and file extension."""
        file_name, file_extension = os.path.splitext(path)
        return file_name.split(os.path.sep)[-1], file_extension

    def __init__(
        self,
        output_path: str,
        n_shards_train: int,
        n_shards_val: int,
        train_files: List[str],
        val_files: List[str],
    ):
        output_path = os.path.expanduser(output_path)
        os.makedirs(output_path, exist_ok=True)
        self.output_path: str = output_path
        self.n_shards_train: int = n_shards_train
        self.n_shards_val: int = n_shards_val
        self.train_files: List[str] = train_files
        self.val_files: List[str] = val_files

    def shard_file(self, file_path: str, num_shards: int):
        file_name, file_extension = LocalDatasetSharder.get_file_info(file_path)
        output_paths: List[str] = []

        # Read in original file and shard it and shard it to `num_shards` files
        with open(file_path, "r") as f:
            json_list: List[str] = list(f)
            print(f"Read {len(json_list)} lines from {file_path}.")

            lines_per_file: int = math.ceil(len(json_list) / num_shards)
            print(
                f"{num_shards} total files where each sharded file will have {lines_per_file} examples."
            )

        for i in tqdm(range(num_shards)):
            output_file: str = f"{file_name}.{i+1}-of-{num_shards}{file_extension}"
            output_path: str = os.path.join(self.output_path, output_file)
            output_paths.append(output_path)

            print(f"Writing out shard to path {output_path}...")
            with open(output_path, "w+") as f_write:
                for line in range(lines_per_file):
                    actual_line_number: int = i * lines_per_file + line
                    if actual_line_number >= len(json_list):
                        break

                    f_write.write(json_list[actual_line_number])

        # Verify shards were created properly
        total_lines: int = 0
        for output_path in output_paths:
            with open(output_path, "r") as f:
                json_list: List[str] = list(f)
                total_lines += len(json_list)
        print(f"Read {total_lines} total lines from shards.")

    def shard(self):
        print("Processing train files...")
        for file in self.train_files:
            self.shard_file(file, self.n_shards_train)

        print("Processing validation files...")
        for file in self.val_files:
            self.shard_file(file, self.n_shards_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", help="Where to output the sharded files", default="~"
    )
    parser.add_argument(
        "--n-shards-train", type=int, help="Number of shards for training set"
    )
    parser.add_argument(
        "--n-shards-val", type=int, help="Number of shards for validation set"
    )
    parser.add_argument(
        "--train-files", help="Files for the training set", nargs="+", default=[]
    )
    parser.add_argument(
        "--val-files", help="Files for the validation set", nargs="+", default=[]
    )
    parser.add_argument(
        "--compress", help="Compress files after sharding", action="store_true"
    )

    args = parser.parse_args()

    sharder = LocalDatasetSharder(
        output_path=args.output_path,
        n_shards_train=args.n_shards_train,
        n_shards_val=args.n_shards_val,
        train_files=args.train_files,
        val_files=args.val_files,
    )
    sharder.shard()
    if args.compress:
        print(f"Compressing *.jsonl files in {args.output_path}")
        subprocess.call(f"touch {args.output_path}/*.jsonl ; gzip {args.output_path}/*.jsonl", shell=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
