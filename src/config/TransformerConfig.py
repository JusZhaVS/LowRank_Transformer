from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration class for transformer model"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 64
    num_hidden_layers: int = 32
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    use_causal_mask: bool = True

    # Rank Size
    rank: int = 32
