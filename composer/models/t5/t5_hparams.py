# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.models.transformer_hparams import TransformerHparams

if TYPE_CHECKING:
    from composer.models.transformer_shared import ComposerTransformer


@dataclass
class T5Hparams(TransformerHparams):

    def initialize_object(self) -> "ComposerTransformer":
        try:
            import transformers
        except ImportError as e:
            raise ImportError('transformers is not installed. '
                              'Please install with `pip install \'mosaicml[nlp]\'`') from e

        from composer.models.t5.model import T5Model
        self.validate()

        if self.model_config:
            config = transformers.T5Config.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.T5Config.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        # set the number of labels ot the vocab size, used for measuring MLM accuracy
        config.num_labels = config.vocab_size

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = transformers.AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        else:
            model = transformers.AutoModelForMaskedLM.from_config(config)  #type: ignore (thirdparty)

        return T5Model(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer_name=self.tokenizer_name,
        )
