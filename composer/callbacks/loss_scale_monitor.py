# Copyright 2021 MosaicML. All Rights Reserved.

"""Monitor learning rate during training."""
from composer.core import Callback, Logger, State

__all__ = ["LossScaleMonitor"]


class LossScaleMonitor(Callback):
    """Logs the learning rate.

    This callback iterates over all optimizers that have a loss scale and logs it under
    ``loss_scale-{OPTIMIZER_NAME}/loss_scale`` key.

    Example
       >>> # constructing trainer object with this callback
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_dataloader,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ...     callbacks=[callbacks.LossScaleMonitor()],
       ... )

    The learning rate is logged by the :class:`~composer.core.logging.logger.Logger` to the following key as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    |                                             | Learning rate for each optimizer and  |
    | ``loss_scale-{OPTIMIZER_NAME}}``            | parameter group for that optimizer is |
    |                                             | logged to a separate key              |
    +---------------------------------------------+---------------------------------------+
    """

    def __init__(self) -> None:
        super().__init__()

    def batch_end(self, state: State, logger: Logger):
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in state.optimizers:
            if hasattr(optimizer, "loss_scale"):
                name = optimizer.__class__.__name__
                logger.metric(f"loss_scale-{name}", optimizer.loss_scale)
