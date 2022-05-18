import os
from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from .downstream_task_evaluator import DownstreamTaskEvaluator


class MedQATaskEvaluator(DownstreamTaskEvaluator):

    task_base_name = "MedQA"

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


    def command(self, config: dict={}) -> str:
        qa_task_name = "medqa_usmle_hf"
        executables: list = ["run_multiple_choice.py"]
        data_dir: str = os.path.join("data", "medqa_usmle_hf")
        defaults: dict = {
            "run_multiple_choice.py": {
                "tokenizer_name": "gpt2",
                "model_name_or_path": self.checkpoint_path,
                "train_file": f"{data_dir}/train.json",
                "validation_file": f"{data_dir}/dev.json",
                "test_file": f"{data_dir}/test.json",
                "do_train": "",
                "do_eval": "",
                "do_predict": "",
                "fp16": "",
                "per_device_train_batch_size": "8",
                "per_device_eval_batch_size": "1",
                "gradient_accumulation_steps": "32",
                "learning_rate": "3e-5",
                "warmup_ratio": "0.5",
                "num_train_epochs": "30",
                "max_seq_length": "512",
                "logging_first_step": "",
                "logging_steps": "20",
                "save_strategy": "no",
                "evaluation_strategy": "steps",
                "eval_steps": "100",
                "run_name": self.run_name(config["run_multiple_choice.py"]),
                "output_dir": f"{self.results_dir}",
            }
        }
        return self.build_command(executables=executables,defaults=defaults,config=config)

