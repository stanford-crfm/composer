import json
import os
from typing import Dict

from scripts.evaluation.hierarchical_logger import hlog


class EvaluatorState:
    PREPARED_KEY: str = "prepared"

    """
    Keeps track of the artifacts and runs we have processed so far.
    The states are kept in memory for quick access, but we also write to a JSON file as a back-up.
    """

    def __init__(self, state_path: str):
        self.state_path: str = state_path
        # TODO: turn the state into objects
        self.state: Dict

        if not os.path.isfile(self.state_path):
            hlog(
                f"The state JSON file does not exist at {self.state_path}. Initializing from an empty state."
            )
            self.state = {}
            self._write()
        else:
            hlog(f"Reading state from {self.state_path}...")
            with open(self.state_path) as f:
                self.state = json.load(f)

    def mark_prepared(self, run_name: str, artifact_name: str):
        if run_name not in self.state:
            self.state[run_name] = dict()
        if artifact_name not in self.state[run_name]:
            self.state[run_name][artifact_name] = dict()

        self.state[run_name][artifact_name][EvaluatorState.PREPARED_KEY] = True
        self._write()

    def has_prepared(self, run_name: str, artifact_name: str) -> bool:
        """Returns a boolean to indicate that we have prepped the artifact."""
        return (
            run_name in self.state
            and artifact_name in self.state[run_name]
            and EvaluatorState.PREPARED_KEY in self.state[run_name][artifact_name]
        )

    def mark_evaluated(self, run_name: str, artifact_name: str, task_name: str):
        self.state[run_name][artifact_name][task_name] = True
        self._write()

    def has_evaluated(self, run_name: str, artifact_name: str, task_name: str) -> bool:
        """
        Returns a boolean to indicate if we have already evaluated an artifact for a particular downstream task.
        """
        return (
            run_name in self.state
            and artifact_name in self.state[run_name]
            and task_name in self.state[run_name][artifact_name]
        )

    def _write(self):
        """Writes out the state to disk as a backup."""
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=4)
