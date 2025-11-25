import torch
import os
import torch.nn as nn
import torch.optim as optim

from src.config.ViTrainingConfig import ViTrainingConfig
from training_utils import AverageMeter
from tqdm import tqdm


class ViTrainer(nn.Module):

    def __init__(self, model, train_dataloader, val_dataloader, config: ViTrainingConfig):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def evaluate_cifar_model(self):
        """CIFAR Testing Evaluation"""

        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            for img, labels in self.val_loader:
                img = img.cuda()
                labels = labels.cuda()
                outputs, _ = self.model(img)
                loss_meter.update(self.loss_fn(outputs, labels).item(), len(img))
                acc = (outputs.argmax(-1) == labels).float().mean().item()
                acc_meter.update(acc, len(img))

        self.model.train(training)

        return loss_meter.calculate(), acc_meter.calculate()

    def train(self) -> nn.Module:
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            for img, labels in tqdm(self.train_dataloader):

                img = img.cuda()
                labels = labels.cuda()

                self.optimizer.zero_grad()

                outputs, _ = self.model(img)
                loss = self.loss_fn(outputs, labels)
                loss_meter.update(loss.item(), len(img))

                acc = (outputs.argmax(-1) == labels).float().mean().item()
                acc_meter.update(acc, len(img))

                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            if epoch % 10 == 0:
                val_loss, val_acc = self.evaluate_cifar_model(self.model, self.loss_fn, self.val_dataloader)
                print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")

        val_loss, val_acc = self.evaluate_cifar_model(self.model, self.loss_fn, self.val_dataloader)
        print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")
        print('Finished Training')

        return self.model
