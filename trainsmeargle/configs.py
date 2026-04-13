import json


class SmeargleConfig:
    """Draft-specific configuration for the Smeargle (Mamba2-based) model.

    Only contains Mamba2-specific and draft-specific parameters. Shared parameters
    (hidden_size, intermediate_size, vocab_size, etc.) come from the target model's
    config, which is passed separately to the model.
    """

    def __init__(
        self,
        draft_vocab_size=32000,
        num_heads=128,
        head_dim=64,
        state_size=256,
        expand=2,
        conv_kernel=4,
        n_groups=1,
        chunk_size=256,
        **kwargs,
    ):
        self.draft_vocab_size = draft_vocab_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
