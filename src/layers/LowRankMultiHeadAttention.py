import torch
import torch.nn as nn
from typing import Tuple, Optional

from .LowRankLinear import LowRankLinear

"""
Contains the class implementation of Low Rank Multi-Headed Attention
"""


class LowAttentionHead(nn.Module):

    def __init__(self, hidden_size: int, head_dim: int,  rank:int=32):
        """
        Single attention head implementation

        Args:
            hidden_size: Input dimension
            head_dim: Dimension of each attention head
            rank: Rank of decomposed A and B
        """
        super().__init__()

        self.head_dim = head_dim
        self.W_q = LowRankLinear(hidden_size, head_dim, rank, bias=False)
        self.W_k = LowRankLinear(hidden_size, head_dim, rank, bias=False)
        self.W_v = LowRankLinear(hidden_size, head_dim, rank, bias=False)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forward pass for attention head

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attn_mask: Attention mask (batch_size, seq_len, seq_len) - 1 for attend, 0 for mask

        Returns:
            attention_output: (batch_size, seq_len, head_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_score = torch.einsum("bij,bkj->bik", Q, K)
        attn_score /= torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float64))

        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)
            attn_score = attn_score.masked_fill(attn_mask==0, float("-inf"))

        attn_weight = torch.softmax(attn_score, dim=-1)
        output = torch.einsum("bij, bjk->bik", attn_weight, V)

        return output, attn_weight


class LowRankMultiHeadAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, rank=32):
        """
        Multi-head attention implementation

        Args:
            hidden_size: Model dimension
            num_heads: Number of attention heads
            rank: Rank Decomposition of A and B -> Default set to 32
        """

        super().__init__()
        head_dim = hidden_size // num_heads

        self.multi_attn_block = nn.ModuleList()
        for _ in range(num_heads):
          attn_block = LowAttentionHead(hidden_size=hidden_size, head_dim=head_dim, rank=rank)
          self.multi_attn_block.append(attn_block)

        self.out_proj = nn.Linear(head_dim * num_heads, hidden_size, bias=False)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (1, seq_len, seq_len)

        Returns:
            attention_output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """

        attn_output = []
        attn_weight = []

        for attn in self.multi_attn_block:
          output, weight = attn(x=hidden_states, attn_mask=attention_mask)
          attn_output.append(output)
          attn_weight.append(weight)

        attn_output = torch.cat(attn_output, dim=-1)
        attn_output = self.out_proj(attn_output)

        attn_weight = torch.stack(attn_weight, dim=1)

        return attn_output, attn_weight
