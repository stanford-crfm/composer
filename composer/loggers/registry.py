# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Type

from composer.loggers.logger_hparams import (LoggerBackendHparams, FileLoggerHparams, TQDMLoggerHparams,
                                             WandBLoggerHparams)

logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": TQDMLoggerHparams,
}


def get_logger_hparams(name: str) -> Type[LoggerBackendHparams]:
    return logger_registry[name]
