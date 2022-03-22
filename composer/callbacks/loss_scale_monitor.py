# Copyright 2021 MosaicML. All Rights Reserved.

"""Monitor loss scale during training."""

from composer.core import Callback, State
from composer.loggers import Logger


class LossScaleMonitor(Callback):
    """Logs the learning rate.

    This callback iterates over all optimizers that have a loss scale and logs it under
    ``loss_scale-{OPTIMIZER_NAME}`` key.

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

    @staticmethod
    def _find_scale(optimizer):
        options = [lambda: optimizer.loss_scale, lambda: optimizer.cur_scale]
        for o in options:
            try:
                return o()
            except AttributeError:
                pass

    def after_train_batch(self, state: State, logger: Logger):
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in state.optimizers:
            scale = LossScaleMonitor._find_scale(optimizer)
            if scale:
                name = optimizer.__class__.__name__
                logger.data_batch({f"loss_scale-{name}": scale})
