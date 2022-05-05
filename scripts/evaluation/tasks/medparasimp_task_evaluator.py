import os
from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from .downstream_task_evaluator import DownstreamTaskEvaluator


class MedParaSimpTaskEvaluator(DownstreamTaskEvaluator):

    task_base_name = "MedParaSimp"

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
            evaluator_state, run, artifact, step, downstream_config, checkpoint_path
        )


    def command(self, config=None) -> str:
        gen_task_name = "medparasimp"
        uid: str = f"{self.run.name}-{self.artifact.name}-finetune"
        executables: list = ["train_e2e.py", "gen_batch.py"]
        defaults: dict = {
            "train_e2e.py": {
                "mode": gen_task_name,
                "tuning_mode": "finetune",
                "epoch": 100,
                "learning_rate": 1e-4,
                "bsz": 16,
                "gradient_accumulation_step": 2,
                "seed": 101,
                "model_name_or_path": self.checkpoint_path,
                "dir_name": uid,
                "warmup_steps": 1000
            },
            "gen_batch.py": {
                "mode": "medparasimp",
                "batch_size": 9,
                "length": 400,
                "no_repeat_ngram_size": 6,
                "control_mode": "no",
                "use_prefixtuning": 0,
                "eval_split": "test",
                "base_model_name_or_path": self.checkpoint_path,
                "load_checkpoint_path": os.path.join(f"runs_{gen_task_name}", f"{uid}_uctd=no_o=1_o=1"),
                "wandb_entity": self.downstream_config["wandb_entity"],
                "wandb_project": self.downstream_config["wandb_project"],
                "wandb_run_name": self.run_name(config["gen_batch.py"])
           }
        }
        return self.build_command(executables=executables,defaults=defaults,config=config)
        
