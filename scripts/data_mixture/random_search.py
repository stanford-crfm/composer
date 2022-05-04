from collections import defaultdict
from typing import List, Dict
import argparse
import copy
import os
import yaml

import random
random.seed(0)  # Fix for reproducibility

"""
Generates config files with weights for the different data sources for 
Sprucfluo (https://github.com/stanford-crfm/sprucfluo).

Usage:

    python3 scripts/data_mixture/random_search.py
"""


class RandomSearchDataMixture:

    @staticmethod
    def round_weight(weight: float) -> float:
        """Round weights to 2 decimal places."""
        return round(weight, 2)

    def __init__(
        self, output_path: str, base_config_path: str, data_sources: List[str],
    ):
        os.makedirs(output_path, exist_ok=True)
        self.output_path: str = output_path
        self.base_config_path: str = base_config_path
        self.data_sources: List[str] = data_sources

        with open(base_config_path, "r") as f:
            self.base_config: Dict = yaml.safe_load(f)

    def generate_name(self, data_sources_to_weights: Dict[str, float]) -> str:
        """
        Generate a name given weights of different data sources.
        Names can only have the following characters: [abcdefghijklmnopqrstuvwxyz0123456789\-]
        """
        name: str = self.base_config["name"].replace("-", "")
        for data_source, weight in data_sources_to_weights.items():
            name += f"-{data_source}{RandomSearchDataMixture.round_weight(weight)}"
        name = name.replace("0.", "")
        return name.lower()

    def generate(self, num_configs: int):
        run_commands: List[str] = []

        # Generate `num_configs` configs with varying weights
        for _ in range(num_configs):
            total_weight: float = 1.0
            data_sources_to_weights: Dict[str, float] = defaultdict(lambda: 0.0)

            # for i, data_source in enumerate(self.data_sources):
            #     if i == len(self.data_sources) - 1:
            #         # Give the remaining weight to the last data source
            #         data_sources_to_weights[data_source] = RandomSearchDataMixture.round_weight(total_weight)
            #     else:
            #         weight: float = RandomSearchDataMixture.round_weight(random.uniform(0, total_weight))
            #         data_sources_to_weights[data_source] = weight
            #         total_weight -= weight

            total_weight = 0.0
            for data_source in self.data_sources:
                weight: float = random.uniform(0, 10)
                data_sources_to_weights[data_source] = weight
                total_weight += weight

            for data_source, weight in data_sources_to_weights.items():
                data_sources_to_weights[data_source] = weight / total_weight

            # Ensure sampled weights add up to 1
            total_weight: float = RandomSearchDataMixture.round_weight(
                sum([weight for weight in data_sources_to_weights.values()])
            )
            assert total_weight == 1.0, f"Weights did not add up to 1.0. Total was {total_weight}."

            config_name: str = self.generate_name(data_sources_to_weights)
            new_config: Dict = copy.deepcopy(self.base_config)

            # Update the config with the new name and weights
            # Names must be less than 63 characters. We also have to take into account the generated id.
            # Names cannot end with a non-alphanumeric character
            short_name: str = config_name[:50]
            if short_name[-1] == "-":
                short_name = short_name[:-1]
            new_config["name"] = short_name

            weights: Dict[str, float] = new_config["parameters"]["train_dataset"]["sprucfluo"]["weights"]
            for data_source in weights.keys():
                weights[data_source] = data_sources_to_weights[data_source]

            # Write out the new config to a file
            config_path: str = os.path.join(self.output_path, f"{config_name}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            run_commands.append(f"mcli sweep -f {config_path}")

        print("Run:")
        for run_command in run_commands:
            # We can automatically run the command, but due to compute constraints
            # just print the commands for now.
            print(run_command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        help="Where to output the config files",
        default="mini-weighted-runs/generated",
    )
    parser.add_argument(
        "--base-config-path",
        help="Base config file",
        default="mini-weighted-runs/mini_weighted_all_pubmed.yaml",
    )
    parser.add_argument(
        "--data-sources",
        help="List of data sources",
        nargs="+",
        default=[
            "pubmedAbs",
            "pubmedC",
            "medical",
            "nih",
            "wikipedia",
            "subtitles",
            "webtext",
        ],
    )
    parser.add_argument("--num-configs", help="Number of configs to generate", type=int, default=10)
    args = parser.parse_args()

    RandomSearchDataMixture(
        output_path=args.output_path,
        base_config_path=args.base_config_path,
        data_sources=args.data_sources,
    ).generate(args.num_configs)


if __name__ == "__main__":
    main()
