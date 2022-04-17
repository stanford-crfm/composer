from typing import Dict, List, Optional
import re
import tarfile
import yaml

import wandb


def initialize_wandb(api_key: str, project: str, entity: str) -> wandb.Api:
    """
    Authenticate and initialize wandb given the API key, project and entity.
    """
    # Login and return the initialized `wandb.Api` object
    wandb.login(key=api_key)
    return wandb.Api(overrides={"project": project, "entity": entity})


def read_configs_from_yaml(path: str) -> Dict:
    """Read configs from a yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_tar_file(tar_file_path: str, output_path: str):
    """Extracts contents of tar file."""
    tar = tarfile.open(tar_file_path)
    tar.extractall(path=output_path)
    tar.close()


def clean_path(path: str) -> str:
    """Colons in paths causes trouble, so replace them."""
    return path.replace(":", "-")


def extract_largest_number(input: str) -> Optional[int]:
    """
    Extract the largest integer from a string. Returns `None` if there are no valid integers.
    Example: rank_0-checkpoints-ep0-ba380000-rank0.tar:v1 => 380,000
    """
    numbers: List[int] = list(map(int, re.findall("\d+", input)))
    return max(numbers) if len(numbers) > 0 else None
