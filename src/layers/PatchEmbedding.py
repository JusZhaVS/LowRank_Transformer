import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_channels: int, out_channels: int):
        """
        Patch Embedding for Vision Transformer

        Args:
            img_size: Dimension of image
            patch_size: Dimension of patch, image must be divisible by patch
            in_channels: number of input channels
            out_channels: number of output channels
        """
        super().__init__()

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
          )
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
