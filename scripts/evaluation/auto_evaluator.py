import os
import traceback
import time
import argparse
import pdb
from datetime import datetime
from typing import List, Dict, Optional

from wandb.apis.public import Run, Runs, Artifact, RunArtifacts
import wandb

from scripts.evaluation.evaluator_state import EvaluatorState
from scripts.evaluation.hierarchical_logger import htrack_block, hlog
from scripts.evaluation.util import (
    clean_path,
    initialize_wandb,
    read_configs_from_yaml,
    extract_tar_file,
    extract_largest_number,
)
from scripts.convert_deepspeed_to_hf_model import (
    convert_deepspeed_checkpoint_to_hf_model,
)
from scripts.evaluation.tasks import all_evaluators

"""
Have a process that runs infinitely observing runs that belongs to a wandb entity using the wandb api. Whenever a new 
checkpoint is uploaded to wandb, the server downloads the checkpoint, untars it, convert it to a HF checkpoint, 
runs the evaluation and uploads the result to wandb. 

The process will update the state of each checkpoint and keep track of the ones the artifacts that already 
have been evaluated.

Usage: 

    python3 scripts/evaluation/auto_evaluator.py --config_path <Path to YAML file> 

"""


class AutoEvaluator:
    def __init__(self, config_path: str):
        with htrack_block("Initializing AutoEvaluator"):
            hlog(f"Reading evaluator config values from {config_path}...")
            config: Dict = read_configs_from_yaml(config_path)

            # The contents of the wandb configuration file should look like the following:
            # apiKey: <Your secret wandb API key>
            # project: <Name of your wandb project>
            # entity: <Name of your wandb entity>
            wandb_config: Dict[str, str] = config["wandb"]
            self.wandb_api: wandb.Api = initialize_wandb(
                api_key=wandb_config["apiKey"],
                project=wandb_config["source_project"],
                entity=wandb_config["entity"],
            )
            if "date" not in wandb_config:
                wandb_config["date"] = "1970-01-01"
            self.wandb_start: int = time.mktime(
                datetime.strptime(wandb_config["date"], "%Y-%m-%d").timetuple()
            )
            self.wandb_filters: list = wandb_config["prefix_filters"]

            # Create the output directory if it doesn't exist already
            self.output_dir: str = config["outputDir"]
            os.makedirs(config["outputDir"], exist_ok=True)

            self.evaluation_frequency_steps: int = config["evaluationFrequencySteps"]
            self.check_frequency_seconds: int = config["checkFrequencySeconds"]
            self.downstream_configs: Dict[str, str] = config["downstreamTaskConfigs"]
            # add wandb info for task evaluators
            self.downstream_configs.update(
                {
                    "wandb_entity": wandb_config["entity"],
                    "wandb_project": wandb_config["target_project"],
                }
            )
            self.state = EvaluatorState(os.path.join(self.output_dir, "state.json"))
            hlog("Initialized the AutoEvaluator.")

    def run(self):
        hlog("Running the AutoEvaluator infinitely...")
        while True:
            self.run_once()
            time.sleep(self.check_frequency_seconds)

    def run_once(self):
        datetime_now: str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        with htrack_block(f"Processing runs from wandb ({datetime_now})"):
            runs: Runs = self.wandb_api.runs()
            hlog(f"Found {len(runs)} runs...")

            for run in runs:
                # check if run passes config filters
                # check timestamp of yaml creation
                time_check = False
                for artifact in run.logged_artifacts():
                    if "yaml" in artifact.name:
                        try:
                            created_date = artifact.created_at.split("T")[0]
                            created_date = time.mktime(
                                datetime.strptime(created_date, "%Y-%m-%d").timetuple()
                            )
                            if created_date > self.wandb_start:
                                time_check = True
                        except:
                            pass
                if not time_check:
                    continue
                # check prefix match
                prefix_match = True in [
                    run.name.startswith(prefix) for prefix in self.wandb_filters
                ]
                if not prefix_match:
                    continue
                with htrack_block(f"Run: {run.name}"):
                    artifacts: RunArtifacts = run.logged_artifacts()
                    hlog(f"Found {len(artifacts)} artifacts...")

                    for artifact in artifacts:
                        step: Optional[int] = extract_largest_number(artifact.name)
                        # If it's a step we don't care about, then skip it
                        if (
                            not step
                            or step < self.evaluation_frequency_steps
                            or step % self.evaluation_frequency_steps != 0
                        ):
                            continue

                        # Read from state to ensure we haven't process the same artifact before
                        with htrack_block(
                            f"Processing artifact: {artifact.name} for step: {step}"
                        ):
                            try:
                                # Banking on the fact that the hyperparameter config file is pushed first to wandb
                                checkpoint_path: str = self.prepare(
                                    run, artifact, hyperparameter_artifact=artifacts[0]
                                )
                                # Evaluate on downstream tasks
                                tasks_to_eval = [
                                    k.lower() for k in self.downstream_configs
                                ]
                                for evaluatorClass in all_evaluators:
                                    task_key = evaluatorClass.__name__[
                                        : -len("TaskEvaluator")
                                    ].lower()
                                    if task_key in tasks_to_eval:
                                        task_config = dict(
                                            self.downstream_configs[
                                                evaluatorClass.task_name
                                            ]
                                        )
                                        evaluatorClass(
                                            evaluator_state=self.state,
                                            run=run,
                                            artifact=artifact,
                                            step=step,
                                            downstream_config=task_config,
                                            checkpoint_path=checkpoint_path,
                                        ).evaluate()
                            except Exception as e:
                                # Catch exception and try again
                                hlog(traceback.format_exc())
                                hlog("Trying again in a future iteration.")

    def prepare(
        self, run: Run, artifact: Artifact, hyperparameter_artifact: Artifact
    ) -> str:
        """
        Prepares artifact by following these steps:
        1. Downloads the artifact from wandb.
        2. Downloads the hyperparameter artifact from wandb (necessary for step 4)
        3. Extracts the content from the tar file
        4. Converts the DeepSpeed checkpoint to a HF checkpoint
        5. Marks that this specific artifact has been already prepped.
        """

        # Create the sub directories and download the hyperparameters used for the particular run
        run_dir_path: str = clean_path(os.path.join(self.output_dir, run.name))
        checkpoint_dir_path: str = clean_path(os.path.join(run_dir_path, artifact.name))

        # If we already prepared the artifact, then do nothing
        if self.state.has_prepared(run.name, artifact.name):
            hlog("Already downloaded artifact and converted to HuggingFace model.")
            return checkpoint_dir_path

        os.makedirs(checkpoint_dir_path, exist_ok=True)
        hyperparameter_artifact.download(root=run_dir_path)

        # Download the artifact from wandb
        artifact.download(root=checkpoint_dir_path)

        files: List[str] = os.listdir(checkpoint_dir_path)
        if len(files) == 0:
            hlog("Nothing found in the directory.")
            return checkpoint_dir_path

        tar_file_path: str = os.path.join(checkpoint_dir_path, files[0])
        extract_tar_file(tar_file_path, checkpoint_dir_path)

        deepspeed_checkpoint_path: str = os.path.join(checkpoint_dir_path, "deepspeed")
        files: List[str] = os.listdir(deepspeed_checkpoint_path)
        if len(files) == 0:
            hlog("Nothing found in the directory.")
            return checkpoint_dir_path

        # Convert the DeepSpeed checkpoint to a HuggingFace model
        convert_deepspeed_checkpoint_to_hf_model(
            checkpoint_path=os.path.join(deepspeed_checkpoint_path, files[0]),
            config_path=os.path.join(run_dir_path, "hparams.yaml"),
            output_path=checkpoint_dir_path,
        )

        # Mark that we have this preprocessed this particular run's artifact
        self.state.mark_prepared(run.name, artifact.name)
        return checkpoint_dir_path


def main():
    auto_evaluator = AutoEvaluator(args.config_path)
    auto_evaluator.run_once()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="evaluator_config.yaml",
        help="Path to auto evaluator configuration file.",
    )
    args = parser.parse_args()
    # TODO: add dry run flag -Tony

    main()
