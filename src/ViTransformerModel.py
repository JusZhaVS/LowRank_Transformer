import torch
import torch.nn as nn

from typing import Optional, Tuple
from .layers.TransformerBlock import TransformerBlock
from .config.ViTransformerConfig import ViTransformerConfig


class ViTransformerModel(nn.Module):
    def __init__(self, config: ViTransformerConfig):
        super().__init__()

        self.network = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            transformer_block = TransformerBlock(
              hidden_size=config.hidden_size,
              num_heads=config.num_attention_heads,
              intermediate_size=config.intermediate_size,
              rank=config.rank
            )

            self.network.append(transformer_block)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, return_attn=False)-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_state, collected_attns = x, []

        for trans_block in self.network:
            hidden_state, attn_out = trans_block(hidden_states=hidden_state, attention_mask=attn_mask)
            if return_attn:
                collected_attns.append(attn_out)

        if return_attn:
            collected_attns = torch.stack(collected_attns, dim=1)
        else:
            collected_attns = None

        return hidden_state, collected_attns
