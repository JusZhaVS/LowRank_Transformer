from dataclasses import dataclass


@dataclass
class ViTransformerConfig:
    """Configuration class for vision transformer model"""
    in_channels = 3
    out_channels = 10
    img_size = 32
    patch_size = 4
    hidden_size = 128
    hidden_dim = 64
    intermediate_size = 128
    num_attention_heads = 3
    num_hidden_layers = 6

    rank = 32
