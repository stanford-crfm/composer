import itertools
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
        self.downstream_config = downstream_config
        # generate all requested hparam settings for the task
        self.hparams: dict = dict(downstream_config["hparams"])
        for k in ["environment", "path"]:
            self.hparams.pop(k, None)
        # task params are keyed with 2 attributes
        # 1. model_tag (e.g. mini)
        # 2. executable (e.g. run_multiple_choice.py)
        # This is to allow for different configs per model size/type and to handle
        # tasks with multiple steps
        # Expand each executable's configs to cartesian product of all options
        for model_tag in self.hparams:
            for executable in self.hparams[model_tag]:
                self.hparams[model_tag][
                    executable
                ] = DownstreamTaskEvaluator.config_cartesian(
                    self.hparams[model_tag][executable]
                )
        self.downstream_dir: str = downstream_config["path"]
        self.environment: str = downstream_config.get("environment", None)
        self.original_work_dir: str = os.path.abspath(os.getcwd())
        self.checkpoint_path: str = os.path.abspath(checkpoint_path)
        self.results_dir: str = os.path.abspath(
            os.path.join(self.checkpoint_path, "result", self.task_base_name)
        )
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def config_to_str(
        config: dict,
        delimiter: str = " ",
        k_prefix: str = " ",
        v_prefix: str = " ",
        ignore_keys: list = [],
        sort_keys=True,
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
        keys = [k for k in config.keys() if k not in ignore_keys]
        if sort_keys:
            keys = sorted(keys)
        return delimiter.join([f"{k_prefix}{k}{v_prefix}{config[k]}" for k in keys])

    @staticmethod
    def config_cartesian(config):
        """
        Expand a config to a cartesian product of every key with list value.
        Example: {"a": 0, "b": [1,2]} => [{"a": 0, "b": 1}, {"a": 0, "b": 2}]
        """
        config_copy = dict(config)
        for k in config_copy:
            if type(config_copy[k]) != list:
                config_copy[k] = [config_copy[k]]
        return [
            dict(zip(config.keys(), value_combo))
            for value_combo in itertools.product(*config_copy.values())
        ]

    def task_name(self, config) -> str:
        options_label = DownstreamTaskEvaluator.config_to_str(
            config=config, delimiter="_", k_prefix="", v_prefix="="
        )
        return f"{self.task_base_name}_{options_label}"

    @abstractmethod
    def command(self, config) -> str:
        pass

    def build_command(self, executables: list, defaults: dict, config: dict):
        """
        Given list of executables, defaults, and updates, build final cmd
        that concats cmds via ";"
        """
        sub_cmds = []
        for executable in executables:
            executable_config = dict(defaults[executable])
            executable_config.update(config[executable])
            sub_cmds.append(
                f"python -u {executable} "
                + DownstreamTaskEvaluator.config_to_str(
                    config=executable_config, delimiter=" ", k_prefix="--", v_prefix=" "
                )
            )
        return " ; ".join(sub_cmds)

    def run_name(self, config) -> str:
        return f"{self.run.name}_step={self.step}_task={self.task_name(config)}"

    @property
    def model_type(self) -> str:
        """
        Return model type (e.g. xs, mini, small, etc...) by looking at artifact's metadata
        """
        return self.artifact.metadata.get("model_type", None)

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
                if self.model_type and self.model_type in self.hparams:
                    model_type_config = self.model_type
                elif "all" in self.hparams:
                    model_type_config = "all"
                else:
                    model_type_config = None
                if model_type_config:
                    cmd_config_combos = DownstreamTaskEvaluator.config_cartesian(
                        self.hparams[model_type_config]
                    )
                    for cmd_config in cmd_config_combos:
                        hparam_cmd = self.command(config=cmd_config)
                        if self.environment:
                            # determine location of conda.sh
                            conda_prefix = os.environ["CONDA_EXE"][: -len("/bin/conda")]
                            conda_shell = f"{conda_prefix}/etc/profile.d/conda.sh"
                            assert os.path.exists(
                                conda_shell
                            ), f"Error: {conda_shell} does not exist"
                            # add conda set up to task command
                            setup_env_cmd = (
                                f". {conda_shell} ; conda activate {self.environment}"
                            )
                            hparam_cmd = f"{setup_env_cmd} ; {hparam_cmd}"
                        hlog(hparam_cmd)
                        subprocess.check_call(hparam_cmd, shell=True)
                        self.wrap_up(config=cmd_config)
            except subprocess.CalledProcessError as e:
                hlog(
                    f"There was an error executing the downstream evaluation script: {e.output}"
                )
                raise


    def wrap_up(self, config):
        os.chdir(self.original_work_dir)
        self.evaluator_state.mark_evaluated(
            self.run.name, self.artifact.name, self.task_name(config)
        )
