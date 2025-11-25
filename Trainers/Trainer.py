import torch
import torch.nn as nn
import numpy as np
import os

from tqdm import tqdm

from src.config.TrainingConfig import TrainingConfig
from src.utils.save_model import save_model

from training_utils import get_lr_scheduler
from training_utils import evaluate_model
from training_utils import TrainingMetrics


class Trainer:
    def __init__(self, model, train_dataset, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Setup dataset for on-the-fly tokenization
        self.train_dataset = train_dataset
        self.batch_size = config.batch_size

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.total_steps = (len(self.train_dataset) // self.batch_size) * config.num_epochs
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.warmup_steps,
            self.total_steps
        )

        # Metrics
        self.metrics = TrainingMetrics()
        self.global_step = 0
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train_step(self, batch) -> float:
        """
        Single training step

        Args:
            batch: Batch of data

        Returns:
            loss: Training loss for this step
        """

        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        loss, output = self.model(input_ids=input_ids, labels=labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate_step(self, batch) -> float:
        """Evaluation step"""
        self.model.eval()

        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss, logits = self.model(input_ids, labels)
            return loss.item()


    def train(self) -> nn.Module:
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total steps: {self.total_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Create indices for shuffling
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)

            epoch_loss = 0
            num_batches = len(indices) // self.batch_size

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")

            for batch_idx in progress_bar:
                # Get batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch = {
                    'input_ids': [],
                    'labels': []
                }

                for idx in batch_indices:
                    sample = self.train_dataset[idx]
                    batch['input_ids'].append(sample['input_ids'])
                    batch['labels'].append(sample['labels'])


                batch['input_ids'] = torch.stack(batch['input_ids'])
                batch['labels'] = torch.stack(batch['labels'])

                loss = self.train_step(batch)
                epoch_loss += loss

                current_lr = self.scheduler.get_last_lr()[0]
                self.metrics.update(loss, current_lr)
                self.global_step += 1

                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{self.metrics.get_avg_loss():.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                # Evaluate model
                if self.global_step % self.config.eval_steps == 0:
                    print(f"\nEvaluating model at step {self.global_step}:")
                    print("-" * 50)
                    evaluate_model(self.model, self.tokenizer, ["Once upon a time", "The little girl"])
                    print("-" * 50)
                    checkpoint_path = os.path.join(
                        self.config.output_dir,
                        f"checkpoint-{self.global_step}"
                    )
                    save_model(self.model, self.tokenizer, checkpoint_path)


            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        save_model(self.model, self.tokenizer, self.config.output_dir)
        print("Training completed!")

        return self.model
