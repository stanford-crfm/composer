import os
from wandb.apis.public import Run, Artifact

from scripts.evaluation.evaluator_state import EvaluatorState
from .downstream_task_evaluator import DownstreamTaskEvaluator


class NEREBMPICOTaskEvaluator(DownstreamTaskEvaluator):
    def __init__(
        self,
        evaluator_state: EvaluatorState,
        run: Run,
        artifact: Artifact,
        step: int,
        downstream_dir_path: str,
        checkpoint_path: str,
    ):
        super().__init__(
            evaluator_state, run, artifact, step, downstream_dir_path, checkpoint_path
        )

    @property
    def task_name(self) -> str:
        # TODO: add hyperparameter values to the name
        return "NER_EBM_PICO"

    @property
    def command(self) -> str:
        executable: str = "run_ner.py"
        data_dir: str = os.path.join("data", "ebmnlp_hf")

        # TODO: Hardcoded to default parameters for now. We can make this configurable too.
        # TODO: add --fp16
        command: str = " ".join(
            [
                "python3",
                executable,
                "--tokenizer_name gpt2",
                f"--model_name_or_path {self.checkpoint_path}",
                f"--train_file {data_dir}/train.json",
                f"--validation_file {data_dir}/dev.json",
                f"--test_file {data_dir}/test.json",
                "--do_train",
                "--do_eval",
                "--do_predict",
                "--per_device_train_batch_size 32",
                "--gradient_accumulation_steps 1",
                "--learning_rate 5e-5",
                "--warmup_steps 100",
                "--num_train_epochs 1",
                "--max_seq_length 512",
                "--logging_steps 100",
                "--save_strategy no",
                "--evaluation_strategy no",
                "--return_macro_metrics",
                f"--run_name {self.run_name}",
                f"--output_dir {self.results_dir}",
            ]
        )
        return command
