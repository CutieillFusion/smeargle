import json


class EagleConfig:
    """Draft-specific configuration for the Eagle3 model.

    Only contains parameters that differ from the target model. Shared parameters
    (hidden_size, num_attention_heads, etc.) come from the target model's config,
    which is passed separately to the model.
    """

    def __init__(
        self,
        draft_vocab_size=32000,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.draft_vocab_size = draft_vocab_size
        self.attention_dropout = attention_dropout
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
