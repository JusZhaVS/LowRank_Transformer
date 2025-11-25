import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from typing import Optional, List


class CIFAR10Dataset:

    def __init__(self, mean: Optional[List[int]] = None, std: Optional[List[int]] = None):
        self.mean = mean if mean is not None else [0.4914, 0.4822, 0.4465]
        self.std = std if std is not None else [0.2470, 0.2435, 0.2616]

        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        
    def get_dataloaders(self):
        train_dataset = torchvision.datasets.CIFAR10(train=True,
                                                     root='data',
                                                     transform=self.img_transform,
                                                     download=True
                                                     )

        val_dataset = torchvision.datasets.CIFAR10(train=False,
                                                   root='data',
                                                   transform=self.img_transform
                                                   )
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10)

        return train_dataloader, val_dataloader
