import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import transformers
import wandb
from transformers import AutoTokenizer, AutoConfig
from wandb.apis.public import Run, Artifact

from convert_deepspeed_to_hf_model import convert_checkpoint, create_hf_model


def extract_hf_config_from_run_params_yaml(yaml_path: str):
    import yaml

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    assert len(config["model"]) == 1
    model_name, model_config = next(iter(config["model"].items()))
    model_config = model_config["model_config"]
    return model_name, AutoConfig.for_model(**model_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="WandB run ID. i.e. the thing that comes after runs/ in the URL")
    # parser.add_argument("--config", help="Path to Composer yaml config")
    parser.add_argument("--key", help="wandb key. ENV variable if not specified", default=None)
    parser.add_argument("--entity", help="wandb entity", default="stanford-mercury")
    parser.add_argument("--project", help="wandb project", default="mosaic-gpt2")
    parser.add_argument(
        "--output-dir", help="Output directory for the HF Model", default=None
    )

    args = parser.parse_args()

    key = args.key

    wandb.login(key=key)

    api = wandb.Api(overrides={"project": args.project, "entity": args.entity})
    run: Run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    print(run)
    a: Artifact
    config_artifact = None
    checkpoint_artifact = None
    for a in run.logged_artifacts():
        if "hparams.yaml" in a.name:
            config_artifact = a
        elif "checkpoint" in a.name:
            if checkpoint_artifact is None:
                checkpoint_artifact = a
            elif checkpoint_artifact.created_at < a.created_at:
                checkpoint_artifact = a

    if config_artifact is None:
        raise ValueError("No config artifact found")
    if checkpoint_artifact is None:
        raise ValueError("No checkpoint artifact found")

    print("Found checkpoint artifact:", checkpoint_artifact.name)

    if args.output_dir is None:
        args.output_dir = f"{run.name}-checkpoint-{checkpoint_artifact.version}"

    temp_dir = TemporaryDirectory()
    temp_config = f"{temp_dir.name}/config"
    temp_checkpoint = f"{temp_dir.name}/checkpoint"
    config_artifact.download(temp_config)
    checkpoint_artifact.download(temp_checkpoint)

    # wandb downloads files into directories, so we need to get the one file from each dir:
    temp_checkpoint = [f for f in Path(temp_checkpoint).iterdir()][0]
    temp_config = [f for f in Path(temp_config).iterdir()][0]

    shutil.unpack_archive(temp_checkpoint, str(temp_dir.name), "tar")
    wandb_model_path = f"deepspeed/mp_rank_00_model_states.pt"
    checkpoint = f"{temp_dir.name}/{wandb_model_path}"


    state_dict = convert_checkpoint(checkpoint)
    model_name, config = extract_hf_config_from_run_params_yaml(str(temp_config))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_hf_model(config, state_dict)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    try:
        transformers.AutoModelForCausalLM.from_pretrained(args.output_dir)
        print("Done and verified.")
    except Exception as e:
        print(f"Something went wrong: {str(e)}")


if __name__ == "__main__":
    main()