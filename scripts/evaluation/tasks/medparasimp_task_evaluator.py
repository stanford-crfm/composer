import os
from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from .downstream_task_evaluator import DownstreamTaskEvaluator


class MedParaSimpTaskEvaluator(DownstreamTaskEvaluator):

    def __init__(
        self,
        evaluator_state: EvaluatorState,
        run: Run,
        artifact: Artifact,
        step: int,
        downstream_config: dict,
        checkpoint_path: str,
    ):
        super().__init__(
            evaluator_state, run, artifact, step, downstream_dir_path, checkpoint_path
        )

    @property
    def task_name(self) -> str:
        # TODO: add hyperparameter values to the name
        return "MedParaSimp"

    def command(self, config=None) -> str:
        task_name: str = "medparasimp"
        uid: str = f"{self.run.name}-{self.artifact.name}"
        finetune_executable: str = "train_e2e.py"
        gen_executable: str = "gen_batch.py"
        data_dir: str = os.path.join("data", f"{task_name}")

        # TODO: Hardcoded to default parameters for now. We can make this configurable too.
        # TODO: add --fp16
        finetune_cmd: str = " ".join(
            [
                "python",
                "-u",
                finetune_executable,
                "--mode",
                f"{task_name}",
                "--tuning_mode",
                "finetune",
                "--epoch",
                "100",
                "--learning_rate",
                "1e-4",
                "--bsz",
                "16",
                "--gradient_accumulation_step",
                "2",
                "--seed",
                "101",
                "--model_name_or_path",
                f"{self.checkpoint_path}",
                "--dir_name",
                uid,
                "--warmup_steps",
                "1000",
            ]
        )
        gen_cmd: str = " ".join(
            [
                "python",
                "-u",
                gen_executable,
                "--mode",
                f"{task_name}",
                "--batch_size",
                "9",
                "--length",
                "400",
                "--no_repeat_ngram_size",
                "6",
                "--control_mode",
                "no",
                "--use_prefixtuning",
                "0",
                "--eval_split",
                "test",
                "--base_model_name_or_path",
                f"{self.checkpoint_path}",
                "--load_checkpoint_path",
                os.path.join(f"runs_{task_name}", uid)
            ]
        )
        command: str = " ; ".join([finetune_cmd, gen_cmd])
        return command
