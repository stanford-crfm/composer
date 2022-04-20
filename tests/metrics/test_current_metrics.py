# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Tuple
from unittest.mock import MagicMock

import pytest

from composer.core import Callback, State
from composer.core.types import DataLoader
from composer.loggers import Logger
from composer.trainer import Trainer
from tests.fixtures.models import SimpleBatchPairModel


class TestMetricsCallback(Callback):

    def __init__(self, compute_training_metrics: bool, compute_val_metrics: bool) -> None:
        self.compute_training_metrics = compute_training_metrics
        self.compute_val_metrics = compute_val_metrics
        self._train_batch_end_train_accuracy = None
        self._eval_batch_end_accuracy = None

    def init(self, state: State, logger: Logger) -> None:
        # on init, the `current_metrics` should be empty
        del logger  # unused
        assert state.current_metrics == {}, "no metrics should be defined on init()"

    def batch_end(self, state: State, logger: Logger) -> None:
        # The metric should be computed and updated on state every batch.
        del logger  # unused
        if self.compute_training_metrics:
            # assuming that at least one sample was correctly classified
            assert state.current_metrics["train"]["Accuracy"] != 0.0
            self._train_batch_end_train_accuracy = state.current_metrics["train"]["Accuracy"]

    def epoch_end(self, state: State, logger: Logger) -> None:
        # The metric at epoch end should be the same as on batch end.
        del logger  # unused
        if self.compute_training_metrics:
            assert state.current_metrics["train"]["Accuracy"] == self._train_batch_end_train_accuracy

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        # The validation accuracy should be defined after each eval batch
        if self.compute_val_metrics:
            # assuming that at least one sample was correctly classified
            assert state.current_metrics["eval"]["Accuracy"] != 0.0
            self._eval_batch_end_accuracy = state.current_metrics["eval"]["Accuracy"]

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.compute_val_metrics:
            assert state.current_metrics["eval"]["Accuracy"] == self._eval_batch_end_accuracy


@pytest.mark.parametrize('compute_training_metrics', [True, False])
@pytest.mark.parametrize('validate_every_n_batches', [-1, 1])
@pytest.mark.parametrize('validate_every_n_epochs', [-1, 1])
def test_current_metrics(
    dummy_train_dataloader: DataLoader,
    dummy_val_dataloader: DataLoader,
    dummy_num_classes: int,
    dummy_in_shape: Tuple[int, ...],
    compute_training_metrics: bool,
    validate_every_n_batches: int,
    validate_every_n_epochs: int,
):
    # Configure the trainer
    num_channels = dummy_in_shape[0]
    mock_logger_destination = MagicMock()
    model = SimpleBatchPairModel(num_channels=num_channels, num_classes=dummy_num_classes)
    compute_val_metrics = validate_every_n_batches == 1 or validate_every_n_epochs == 1
    train_subset_num_batches = 2
    eval_subset_num_batches = 2
    num_epochs = 2
    metrics_callback = TestMetricsCallback(
        compute_training_metrics=compute_training_metrics,
        compute_val_metrics=compute_val_metrics,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
        max_duration=num_epochs,
        compute_training_metrics=compute_training_metrics,
        train_subset_num_batches=train_subset_num_batches,
        eval_subset_num_batches=eval_subset_num_batches,
        loggers=[mock_logger_destination],
        callbacks=[metrics_callback],
        validate_every_n_batches=validate_every_n_batches,
        validate_every_n_epochs=validate_every_n_epochs,
    )

    # Train the model
    trainer.fit()

    if not compute_training_metrics and not compute_val_metrics:
        return

    # Validate the metrics
    if compute_training_metrics:
        assert trainer.state.current_metrics["train"]["Accuracy"] != 0.0
    else:
        assert "train" not in trainer.state.current_metrics

    if compute_val_metrics:
        assert trainer.state.current_metrics["eval"]["Accuracy"] != 0.0
    else:
        assert "eval" not in trainer.state.current_metrics

    # Validate that the logger was called the correct number of times for metric calls
    num_expected_calls = 0
    if compute_training_metrics:
        # computed once per batch
        # and again at epoch end
        num_expected_calls += (train_subset_num_batches + 1) * num_epochs
    if compute_val_metrics:
        num_calls_per_eval = eval_subset_num_batches + 1
        num_evals = 0
        if validate_every_n_batches == 1:
            num_evals += train_subset_num_batches * num_epochs
        if validate_every_n_epochs == 1:
            num_evals += num_epochs
        num_expected_calls += (num_calls_per_eval) * num_evals
    num_actual_calls = 0

    # Need to filter out non-metrics-related calls
    for call in mock_logger_destination.log_data.mock_calls:
        data = call[1][2]
        for k in data:
            if k.startswith("metrics/"):
                num_actual_calls += 1
                break

    assert num_actual_calls == num_expected_calls
