from typing import List, Tuple
import argparse
import os
import random

# Fix the random seed for reproducibility
random.seed(0)

"""
Create train and val splits for a given dataset 

Example usage:

    python3 scripts/split_dataset.py --dataset-path plain_medical_text.jsonl --output-path plain_medical_text
"""


class DatasetSplitter:
    """Split a given dataset file into tran and val sets."""

    @staticmethod
    def get_file_info(path: str) -> Tuple[str, str]:
        """Given a path, return the name of the file and file extension."""
        file_name, file_extension = os.path.splitext(path)
        return file_name.split(os.path.sep)[-1], file_extension

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        val_size_fraction: float,
    ):
        output_path = os.path.expanduser(output_path)
        os.makedirs(output_path, exist_ok=True)

        self.dataset_path: str = dataset_path
        self.output_path: str = output_path
        self.val_size_fraction: float = val_size_fraction

    def split(self):
        def write(file_name: str, split: List[str]):
            with open(os.path.join(self.output_path, file_name), "w+") as f:
                for line in split:
                    f.write(line)
            print(f"Outputted {len(split)} lines to {file_name}.")

        print("Processing train files...")
        with open(self.dataset_path, "r") as f:
            json_list: List[str] = list(f)
            print(f"Read {len(json_list)} lines from {self.dataset_path}.")

            # Randomly shuffle the list of examples and get the first `self.val_size_fraction` worth of examples
            random.shuffle(json_list)
            num_val_examples: int = int(len(json_list) * self.val_size_fraction)
            val_set: List[str] = json_list[: num_val_examples]
            train_set: List[str] = json_list[num_val_examples:]
            assert len(val_set) + len(train_set) == len(json_list)

        # Write out the new splits to new files
        file_name, file_extension = DatasetSplitter.get_file_info(self.dataset_path)
        write(f"{file_name}_val{file_extension}", val_set)
        write(f"{file_name}_train{file_extension}", train_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", help="Where to output the new split files", default="~"
    )
    parser.add_argument(
        "--dataset-path", help="Path of the dataset"
    )
    parser.add_argument(
        "--val-size-fraction",
        type=float,
        default=0.1,
        help="What percentage of the dataset should be part of the validation split"
    )

    args = parser.parse_args()
    splitter = DatasetSplitter(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        val_size_fraction=args.val_size_fraction,
    )
    splitter.split()
    print("\nDone.")


if __name__ == "__main__":
    main()
