from transformers.configuration_utils import PretrainedConfig
from typing import Union, Tuple


class EConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        ssm_state_size (`int`, *optional*, defaults to 16):
            State dimension for MAMBA selective state space model.
        ssm_conv_kernel (`int`, *optional*, defaults to 4):
            Convolution kernel size for token mixing in MAMBA.
        ssm_expand (`int`, *optional*, defaults to 2):
            Expansion factor for channel mixing in MAMBA (hidden_size * expand).
        ssm_dt_rank (`int` or `str`, *optional*, defaults to "auto"):
            Rank for discretization step (Δ). If "auto", uses max(16, d_model // 16).
        ssm_dt_scale (`float`, *optional*, defaults to 1.0):
            Scale for discretization step.
        ssm_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum discretization step.
        ssm_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum discretization step.
        ssm_dt_init (`str`, *optional*, defaults to "random"):
            Initialization method for discretization step ("random" or "constant").
        ssm_A_init_range (`tuple`, *optional*, defaults to (1, 16)):
            Initialization range for A matrix.
        ssm_use_fast_path (`bool`, *optional*, defaults to True):
            Use optimized CUDA kernels if available.
        draft_vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size for draft model (reduced vocabulary for efficiency).

        Example:

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        # MAMBA-specific parameters
        ssm_state_size=16,
        ssm_conv_kernel=4,
        ssm_expand=2,
        ssm_dt_rank="auto",
        ssm_dt_scale=1.0,
        ssm_dt_min=0.001,
        ssm_dt_max=0.1,
        ssm_dt_init="random",
        ssm_A_init_range=(1, 16),
        ssm_use_fast_path=True,
        draft_vocab_size=32000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        
        # MAMBA-specific parameters
        self.ssm_state_size = ssm_state_size
        self.ssm_conv_kernel = ssm_conv_kernel
        self.ssm_expand = ssm_expand
        self.ssm_dt_rank = ssm_dt_rank
        self.ssm_dt_scale = ssm_dt_scale
        self.ssm_dt_min = ssm_dt_min
        self.ssm_dt_max = ssm_dt_max
        self.ssm_dt_init = ssm_dt_init
        self.ssm_A_init_range = ssm_A_init_range
        self.ssm_use_fast_path = ssm_use_fast_path
        self.draft_vocab_size = draft_vocab_size
        
        # Validate MAMBA parameters
        self._mamba_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
    
    def _mamba_validation(self):
        """
        Validate MAMBA-specific configuration parameters.
        """
        # Validate ssm_state_size
        if not isinstance(self.ssm_state_size, int) or self.ssm_state_size <= 0:
            raise ValueError(f"`ssm_state_size` must be a positive integer, got {self.ssm_state_size}")
        
        # Validate ssm_conv_kernel
        if not isinstance(self.ssm_conv_kernel, int) or self.ssm_conv_kernel <= 0:
            raise ValueError(f"`ssm_conv_kernel` must be a positive integer, got {self.ssm_conv_kernel}")
        
        # Validate ssm_expand
        if not isinstance(self.ssm_expand, int) or self.ssm_expand < 1:
            raise ValueError(f"`ssm_expand` must be a positive integer (typically 2-4), got {self.ssm_expand}")
        if self.ssm_expand > 8:
            raise ValueError(f"`ssm_expand` should be reasonable (typically 2-4), got {self.ssm_expand}")
        
        # Validate ssm_dt_rank
        if isinstance(self.ssm_dt_rank, str) and self.ssm_dt_rank != "auto":
            raise ValueError(f"`ssm_dt_rank` must be an integer or 'auto', got {self.ssm_dt_rank}")
        if isinstance(self.ssm_dt_rank, int) and self.ssm_dt_rank <= 0:
            raise ValueError(f"`ssm_dt_rank` must be a positive integer, got {self.ssm_dt_rank}")
        
        # Validate discretization step bounds
        if not isinstance(self.ssm_dt_min, (int, float)) or self.ssm_dt_min <= 0:
            raise ValueError(f"`ssm_dt_min` must be a positive number, got {self.ssm_dt_min}")
        if not isinstance(self.ssm_dt_max, (int, float)) or self.ssm_dt_max <= 0:
            raise ValueError(f"`ssm_dt_max` must be a positive number, got {self.ssm_dt_max}")
        if self.ssm_dt_min >= self.ssm_dt_max:
            raise ValueError(f"`ssm_dt_min` ({self.ssm_dt_min}) must be less than `ssm_dt_max` ({self.ssm_dt_max})")
        
        # Validate ssm_dt_init
        if self.ssm_dt_init not in ["random", "constant"]:
            raise ValueError(f"`ssm_dt_init` must be 'random' or 'constant', got {self.ssm_dt_init}")
        
        # Validate ssm_A_init_range
        if not isinstance(self.ssm_A_init_range, (tuple, list)) or len(self.ssm_A_init_range) != 2:
            raise ValueError(f"`ssm_A_init_range` must be a tuple of length 2, got {self.ssm_A_init_range}")
        if self.ssm_A_init_range[0] >= self.ssm_A_init_range[1]:
            raise ValueError(f"`ssm_A_init_range` first value must be less than second, got {self.ssm_A_init_range}")
        
        # Validate ssm_use_fast_path
        if not isinstance(self.ssm_use_fast_path, bool):
            raise ValueError(f"`ssm_use_fast_path` must be a boolean, got {self.ssm_use_fast_path}")