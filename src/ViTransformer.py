import torch
import torch.nn as nn

from typing import Optional, Tuple

from .layers.PatchEmbedding import PatchEmbedding
from .config.ViTransformerConfig import ViTransformerConfig
from .ViTransformerModel import ViTransformerModel


class VisionTransformer(nn.Module):

    def __init__(self, config: ViTransformerConfig):
        """
        Vision Transformer Implementaiton using Low Rank Transformer blocks
        """

        super().__init__()

        in_channels = config.in_channels
        out_channels = config.out_channels
        hidden_size = config.hidden_size
        img_size = config.img_size
        patch_size = config.patch_size

        self.patch_embed = PatchEmbedding(img_size=img_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          out_channels=hidden_size
                                          )

        self.pos_E = nn.Embedding((img_size // patch_size) ** 2, hidden_size) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.transformer = ViTransformerModel(config)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            img: Image Tensors
            return_attn: Return attention for each layer
        Return:
            attn_out (Optional): Attention matrix tensors
        """

        embs = self.patch_embed(img) 
        B, T, _ = embs.shape
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs += self.pos_E(pos_ids)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, embs], dim=1)

        x, attn_out = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]

        return out, attn_out
