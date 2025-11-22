import torch
import torch.nn as nn

from typing import Optional
from .RMSNorm import RMSNorm
from .FeedForward import FeedForward
from .LowRankMultiHeadAttention import LowRankMultiHeadAttention

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, rank:int=32):
        """
        Complete transformer block with attention and feed-forward

        Args:
            hidden_size: Model dimension
            num_heads: Number of attention heads
            intermediate_size: FFN hidden dimension
        """
        super().__init__()

        self.rms_norm1 = RMSNorm(hidden_size=hidden_size)
        self.rms_norm2 = RMSNorm(hidden_size=hidden_size)

        self.multi_head_attn = LowRankMultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, rank=rank)
        self.ffn = FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer block

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask

        Returns:
            hidden_states: Output tensor (batch_size, seq_len, hidden_size)
            attn_out: Attention matrix tensor
        """

        norm_1 = self.rms_norm1(hidden_states=hidden_states)
        attn_out, _ = self.multi_head_attn(hidden_states=norm_1, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        norm_2 = self.rms_norm2(hidden_states=hidden_states)
        ffn_output = self.ffn(norm_2)
        hidden_states = hidden_states + ffn_output

        return hidden_states, attn_out
