import argparse

import os
import shutil
import subprocess
import torch
import transformers

from transformers import AutoConfig, AutoTokenizer


"""
Converts DeepSpeed-format checkpoint to Torch-format checkpoint.

Example usage:

python3 scripts/convert_deepspeed_to_torch_checkpoint.py --input-path ./deepspeed/mp_rank_00_model_states.pt
"""


def convert_checkpoint(deepspeed_checkpoint_path: str):
    """
    Converts DeepSpeed-format checkpoint to Torch-format state_dict.
    Args:
        deepspeed_checkpoint_path: str

    Returns:
        Torch state dict
    """
    model_checkpoint = torch.load(
        deepspeed_checkpoint_path, map_location=torch.device("cpu")
    )
    state_dict = model_checkpoint["module"]
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, prefix="module."
    )

    return state_dict


def extract_hf_config_from_yaml(yaml_path: str):
    import yaml

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    assert len(config["model"]) == 1
    model_name, model_config = next(iter(config["model"].items()))
    model_config = model_config["model_config"]
    return model_name, AutoConfig.for_model(**model_config)


def create_hf_model(config, torch_state_dict):
    model = transformers.AutoModelForCausalLM.from_config(config)
    # Load FP16 weights, convert model to FP32 if desired
    model.load_state_dict(torch_state_dict)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", help="Path to DeepSpeed checkpoint")
    parser.add_argument("--wandb-id", help="Artifact url")
    parser.add_argument("--config", help="Path to Composer yaml config")
    parser.add_argument(
        "--output-dir", help="Output directory for the HF Model", default="model"
    )

    args = parser.parse_args()
    if args.wandb_id:
        work_dir = "wandb_download_tmp"
        os.makedirs(work_dir, exist_ok=True)
        subprocess.call(f"wandb artifact get --root {work_dir} {args.wandb_id}", shell=True)
        shutil.unpack_archive(f"{work_dir}/{os.listdir(work_dir)[0]}", work_dir)
        wandb_model_path = f"deepspeed/mp_rank_00_model_states.pt"
        args.input_checkpoint = f"{work_dir}/{wandb_model_path}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    state_dict = convert_checkpoint(args.input_checkpoint)
    model_name, config = extract_hf_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_hf_model(config, state_dict)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.wandb_id:
        shutil.rmtree("wandb_download_tmp")

    # Load HF model to verify the conversion went okay.
    try:
        transformers.AutoModelForCausalLM.from_pretrained(args.output_dir)
        print("Done and verified.")
    except Exception as e:
        print(f"Something went wrong: {str(e)}")


if __name__ == "__main__":
    main()
