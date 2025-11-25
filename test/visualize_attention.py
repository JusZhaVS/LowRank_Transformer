import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def visualize_attention(self, model):
    """Visualize Attention Matrices for an Vision Transformer"""

    inv_transform = transforms.Compose([
            transforms.Normalize(
                mean = [ 0., 0., 0. ],
                std = 1 / np.array(self.std)),
            transforms.Normalize(
                mean = -np.array(self.mean),
                std = [ 1., 1., 1. ]),
            transforms.ToPILImage(),
        ])

    val_batch = self.val_dataloader[0]
    model.eval()
    with torch.no_grad():
        img, _ = val_batch
        img = img.cuda()
        _, attns = model(img, return_attn=True)

    fig, ax = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        flattened_attns = attns.flatten(1,2)[:, :, 0, 1:].mean(1).reshape(-1, 8, 8).cpu().numpy()
        ax[0, i].imshow(inv_transform(img[i]))
        ax[1, i].imshow(flattened_attns[i])
        ax[0, i].axis(False)
        ax[1, i].axis(False)
