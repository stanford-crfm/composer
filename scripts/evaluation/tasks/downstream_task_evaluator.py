import os
import subprocess
from abc import ABC, abstractmethod

from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from scripts.evaluation.hierarchical_logger import hlog, htrack_block


class DownstreamTaskEvaluator(ABC):
    def __init__(
        self,
        evaluator_state: EvaluatorState,
        run: Run,
        artifact: Artifact,
        step: int,
        downstream_config: dict,
        checkpoint_path: str,
    ):
        self.run: Run = run
        self.artifact: Artifact = artifact
        self.step: int = step
        self.evaluator_state: EvaluatorState = evaluator_state
        # generate all requested hparam settings for the task
        self.hparams: dict = dict(downstream_config)
        for k in ["environment", "path"]: self.hparams.pop(k, None)
        self.sweep_configs: list = DownstreamTaskEvaluator.config_cartesian(self.hparams)
        self.downstream_dir: str = downstream_config["path"]
        self.environment: str = downstream_config.get("environment", None)
        self.original_work_dir: str = os.path.abspath(os.getcwd())
        self.checkpoint_path: str = os.path.abspath(checkpoint_path)
        self.results_dir: str = os.path.abspath(
            os.path.join(self.checkpoint_path, "result", self.task_name)
        )
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def config_to_str(
        config: dict,
        delimiter: str = " ",
        k_prefix: str = " ",
        v_prefix: str = " ",
        ignore_keys: list = [],
    ) -> str:
        """
        Turn a config dict into a str. Add prefixes to keys and values, potentially ignore keys.
        Example:
            input: config = {"learning_rate": 0.01, "epochs": 20}
                   delimiter=" ",
                   k_prefix="--",
                   v_prefix=" "
            output: "--learning_rate 0.01 --epochs 20"
        """
        keys = sorted([k for k in config.keys() if k not in ignore_keys])
        return delimiter.join([f"{k_prefix}{k}{v_prefix}{config[k]}" for k in keys])

    @staticmethod
    def config_cartesian(config):
        """
        Expand a config to a cartesian product of every key with list value.
        Example: {"a": 0, "b": [1,2]} => [{"a": 0, "b": 1}, {"a": 0, "b": 2}]
        """
        if len(config) == 0:
            return [{}]
        else:
            config_cp = dict(config)
            head = list(config_cp.keys())[0]
            config_cp.pop(head)
            tails = cartesian(config_cp)
            head_options = (
                [config[head]] if not type(config[head]) == list else config[head]
            )
            return_list = [
                {**{head: h}, **tail} for h in head_options for tail in tails
            ]
            return return_list

    @abstractmethod
    def task_name(self) -> str:
        pass

    @abstractmethod
    def command(self) -> str:
        pass

    @property
    def run_name(self) -> str:
        return f"{self.run.name}_step={self.step}_task={self.task_name}"

    def evaluate(self):
        if self.evaluator_state.has_evaluated(
            self.run.name, self.artifact.name, self.task_name
        ):
            hlog(
                f"Already evaluated {self.artifact.name} for {self.run.name} on task {self.task_name}"
            )
            return

        with htrack_block(f"Evaluating {self.artifact.name} of {self.run.name}"):
            hlog(f"cd {self.downstream_dir}")
            os.chdir(self.downstream_dir)

            try:
                for hparam_config in self.sweep_configs:
                    hparam_cmd = self.command(config=sweep_config) 
                    if self.environment:
                        hparam_cmd = f"conda activate {self.environment} ; {hparam_cmd}"
                    hlog(hparam_cmd)
                    subprocess.check_call(hparam_cmd, shell=True)
            except subprocess.CalledProcessError as e:
                hlog(
                    f"There was an error executing the downstream evaluation script: {e.output}"
                )
                raise

            self.wrap_up()

    def wrap_up(self):
        os.chdir(self.original_work_dir)
        self.evaluator_state.mark_evaluated(
            self.run.name, self.artifact.name, self.task_name
        )
