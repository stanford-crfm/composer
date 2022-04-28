import os
from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from .downstream_task_evaluator import DownstreamTaskEvaluator


class PubMedQATaskEvaluator(DownstreamTaskEvaluator):
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


    def task_name(self) -> str:
        # TODO: add hyperparameter values to the name
        return "PubMedQA"

    def command(self, config: dict={}) -> str:
        executable: str = "run_seqcls_gpt.py"
        data_dir: str = os.path.join("data", "pubmedqa_hf")
        final_config: dict = {
            "tokenizer_name": "gpt2",
            "model_name_or_path": self.checkpoint_path,
            "train_file": f"{data_dir}/train.json",
            "validation_file": f"{data_dir}/dev.json",
            "test_file": f"{data_dir}/test.json",
            "do_train": "",
            "do_eval": "",
            "do_predict": "",
            "fp16": "",
            "per_device_train_batch_size": "16",
            "gradient_accumulation_steps": "1",
            "learning_rate": "2e-5",
            "warmup_steps": "100",
            "num_train_epochs": "30",
            "max_seq_length": "512",
            "logging_steps": "100",
            "save_strategy": "no",
            "evaluation_strategy": "no",
            "run_name": f"{self.run_name}",
            "output_dir": f"{self.results_dir}",
        }
        final_config.update(config)
        params_str: str = DownstreamTaskEvaluator.config_to_str(
            final_config, delimiter=" ", k_prefix="--", v_prefix=" "
        )
        command: str = f"python3 {executable} {params_str}"
        return command
