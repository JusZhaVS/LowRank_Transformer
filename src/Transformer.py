import torch
import torch.nn as nn

from typing import Optional
from .config.TransformerConfig import TransformerConfig
from .layers.RMSNorm import RMSNorm
from .layers.TransformerBlock import TransformerBlock
from .utils.create_causal_mask import create_causal_mask


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Complete transformer model for causal language modeling
        """
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
          transformer_block = TransformerBlock(
              hidden_size=config.hidden_size,
              num_heads=config.num_attention_heads,
              intermediate_size=config.intermediate_size,
              rank=config.rank
          )

          self.layers.append(transformer_block)

        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer model

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len, seq_len)

        Returns:
            hidden_states: Final hidden states (batch_size, seq_len, hidden_size)
        """

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.token_embeddings(input_ids) + self.pos_embeddings(positions)

        if self.config.use_causal_mask and attention_mask is None:
          attention_mask = create_causal_mask(seq_len, device=device)

        for trans_block in self.layers:
          hidden_states = trans_block(hidden_states=hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states
