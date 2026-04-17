from typing import Any, Optional

import torch

from transformers.cache_utils import Cache, CacheLayerMixin


class KVCacheLayer(CacheLayerMixin):
    """Wraps a single [key_KVCache, value_KVCache] pair as a CacheLayerMixin
    for compatibility with the modern transformers Cache API."""

    is_sliding = False
    is_compileable = False

    def __init__(self, key_cache: "KVCache", value_cache: "KVCache"):
        # Skip CacheLayerMixin.__init__ to avoid overwriting our cache references
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor):
        pass  # Already initialized with preallocated memory

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_out = self.key_cache.cat(key_states, dim=2)
        value_out = self.value_cache.cat(value_states, dim=2)
        return key_out, value_out

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.key_cache.current_length.item()

    def get_max_cache_shape(self) -> int:
        return self.key_cache.data.shape[2]  # max_length dimension


class KVCacheAdapter(Cache):
    """Adapts List[List[KVCache]] to the modern transformers Cache API.

    The existing speculative decoding code uses preallocated KVCache objects
    organized as a list of [key_cache, value_cache] pairs per layer. This
    adapter wraps that structure so the modern HuggingFace modeling code
    (which expects a Cache object) can use it transparently.
    """

    def __init__(self, kv_caches: list):
        layer_adapters = [
            KVCacheLayer(kv_pair[0], kv_pair[1])
            for kv_pair in kv_caches
        ]
        super().__init__(layers=layer_adapters)


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model, max_length=2200):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.
        max_length (int): Maximum sequence length to preallocate.

    Returns:
        tuple:
            - past_key_values (KVCacheAdapter): A Cache-compatible adapter wrapping preallocated KVCache objects.
            - past_key_values_data_list (list[torch.Tensor]): Raw tensors for direct cache manipulation (one per device).
            - current_length_data (torch.Tensor): CPU tensor tracking current length of keys/values per layer.
    """
    config = model.config
    batch_size = 1
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    layers = getattr(model, "model", model).layers

    # Detect which device each layer is on
    devices = [layers[i].self_attn.q_proj.weight.device for i in range(config.num_hidden_layers)]

    # Allocate one big tensor per device, grouping consecutive layers
    past_key_values_data_list = []
    group_start = 0
    for i in range(1, len(devices)):
        if devices[i] != devices[group_start]:
            count = i - group_start
            past_key_values_data_list.append(torch.zeros(
                count * 2, batch_size, config.num_key_value_heads, max_length, head_dim,
                device=devices[group_start], dtype=model.dtype,
            ))
            group_start = i
    # Final group
    count = len(devices) - group_start
    past_key_values_data_list.append(torch.zeros(
        count * 2, batch_size, config.num_key_value_heads, max_length, head_dim,
        device=devices[group_start], dtype=model.dtype,
    ))

    # CPU tensor for fast length tracking across all layers
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )

    # Create KVCache pairs, mapping each layer to its device-grouped tensor
    past_key_values = []
    bias = 0
    current_device_idx = devices[0].index
    first_device_idx = devices[0].index
    for i in range(config.num_hidden_layers):
        device_idx = devices[i].index
        if device_idx != current_device_idx:
            bias = 0
            current_device_idx = device_idx
        data_tensor = past_key_values_data_list[device_idx - first_device_idx]
        past_key_values.append([
            KVCache(data_tensor[2 * bias + j], current_length_data[i * 2 + j])
            for j in range(2)
        ])
        bias += 1

    return KVCacheAdapter(past_key_values), past_key_values_data_list, current_length_data
