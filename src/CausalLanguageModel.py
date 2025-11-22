import torch
import torch.nn as nn
from .TransformerModel import TransformerModel

from typing import Optional, Union, Tuple
from .config.TransformerConfig import TransformerConfig


class CausalLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Causal language model with transformer backbone"""
        super().__init__()

        self.transformer = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for language model

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Target labels for loss computation (batch_size, seq_len)

        Returns:
            If labels provided: (loss, logits)
            Else: logits only
        """

        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)

        if labels is not None:
          shift_logits = logits[:, :-1, :].contiguous()
          shift_labels = labels[:, 1:].contiguous()

          loss = self.loss_fn(
              shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1)
          )

          return loss, logits

        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text using the language model

        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """

        for _ in range(max_new_tokens):

          logits = self(input_ids)
          next_token_logits = logits[:, -1, :] / temperature

          probs = torch.softmax(next_token_logits, dim=-1)
          next_token = torch.multinomial(probs, num_samples=1)

          input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
