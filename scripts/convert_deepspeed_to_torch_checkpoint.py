import argparse
import os

import torch

"""
Converts DeepSpeed-format checkpoint to Torch-format checkpoint.

Example usage:

python3 scripts/convert_deepspeed_to_torch_checkpoint.py --input-path ./deepspeed/mp_rank_00_model_states.pt
"""


def convert_checkpoint(deepspeed_checkpoint_path: str, output_dir: str) -> str:
    """
    Converts DeepSpeed-format checkpoint to Torch-format checkpoint.
    Args:
        deepspeed_checkpoint_path: str
        output_dir: str

    Returns:
        Torch checkpoint path
    """
    model_checkpoint = torch.load(
        deepspeed_checkpoint_path, map_location=torch.device("cpu")
    )
    state_dict = model_checkpoint["module"]
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, prefix="module."
    )

    file_name: str = deepspeed_checkpoint_path.split(os.path.sep)[-1]
    output_path: str = os.path.join(output_dir, f"torch_{file_name}")
    torch.save(state_dict, output_path)
    return output_path


def load_hf_model(torch_checkpoint_path: str):
    """
    An example of how to load HuggingFace model from a Torch checkpoint.
    """
    import transformers

    # Recreate your custom model...
    config = transformers.GPT2Config(n_positions=1024)
    model = transformers.AutoModelForCausalLM.from_config(config)

    # Read state dict from Pytorch checkpoint
    state_dict = torch.load(torch_checkpoint_path, map_location=torch.device("cpu"))

    # Load FP16 weights, convert model to FP32 if desired
    model.load_state_dict(state_dict)
    model = model.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to DeepSpeed checkpoint")
    parser.add_argument(
        "--output-dir", help="Output directory for the Torch checkpoint", default="."
    )

    args = parser.parse_args()
    output_path: str = convert_checkpoint(args.input_path, args.output_dir)

    # Load HF model to verify the conversion went okay.
    try:
        load_hf_model(output_path)
        print("Done and verified.")
    except Exception as e:
        print(f"Something went wrong: {str(e)}")


if __name__ == "__main__":
    main()
