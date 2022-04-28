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
        self.downstream_config: dict = downstream_config
        self.downstream_dir: str = downstream_config["path"]
        self.original_work_dir: str = os.path.abspath(os.getcwd())
        self.checkpoint_path: str = os.path.abspath(checkpoint_path)
        self.results_dir: str = os.path.abspath(
            os.path.join(self.checkpoint_path, "result", self.task_name)
        )
        os.makedirs(self.results_dir, exist_ok=True)

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
                hlog(self.command)
                subprocess.check_call(self.command, shell=True)
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
