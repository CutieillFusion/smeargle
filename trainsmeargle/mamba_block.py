# coding=utf-8
"""
MAMBA Block implementation for EAGLE model using official mamba-ssm package.
Replaces attention mechanism with selective state space models.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from mamba_ssm import Mamba2


class MambaBlock(nn.Module):
    """MAMBA block replacing attention mechanism using official Mamba2 implementation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Input dimension is hidden_size * 2 (concatenated input_emb and hidden_states)
        d_model = config.hidden_size * 2
        
        # Get Mamba2 parameters from config with defaults
        d_state = getattr(config, 'ssm_state_size', 16)
        d_conv = getattr(config, 'ssm_conv_kernel', 4)
        expand = getattr(config, 'ssm_expand', 2)
        
        # Initialize official Mamba2 block
        # Note: Mamba2 combines token mixing, SSM, and channel mixing internally
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Output projection: (hidden_size * 2) -> hidden_size
        self.out_proj = nn.Linear(
            d_model,
            config.hidden_size,
            bias=False
        )
        
        # Store d_state for state caching compatibility
        self.d_state = d_state
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Ignored but kept for compatibility
        position_ids: Optional[torch.LongTensor] = None,  # Ignored but kept for compatibility
        past_key_value: Optional[Tuple] = None,  # Ignored but kept for compatibility
        output_attentions: bool = False,  # Always False for MAMBA
        use_cache: bool = True,  # Controls state caching
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of MAMBA block.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size * 2) - concatenated input_emb and hidden_states
            cache_state: (batch, d_state) or None - previous SSM state (currently not used by Mamba2)
            attention_mask: Ignored (kept for compatibility)
            position_ids: Ignored (kept for compatibility)
            past_key_value: Ignored (kept for compatibility)
            output_attentions: Always False for MAMBA
            use_cache: Whether to return updated state
        
        Returns:
            output: (batch, seq_len, hidden_size)
            cache_state: (batch, d_state) or None
        """
        # Forward pass through Mamba2
        # Mamba2 internally handles token mixing, SSM, and channel mixing
        # Note: cache_state parameter is ignored as Mamba2 manages state internally
        x = self.mamba2(hidden_states)
        
        # Output projection: (hidden_size * 2) -> hidden_size
        output = self.out_proj(x)
        
        # Note: The official Mamba2 manages state internally and doesn't expose
        # it in the same way as the custom implementation. For compatibility with
        # the existing interface, we return None for state. If state caching is
        # required, Mamba2's internal state management should be sufficient for
        # sequential processing within a single forward pass.
        if use_cache:
            # Return None as Mamba2 handles state internally
            # The caller should not rely on this state for cross-forward-pass caching
            return output, None
        else:
            return output, None
