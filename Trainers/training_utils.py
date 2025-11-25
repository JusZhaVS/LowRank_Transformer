import torch
import numpy as np

from typing import List


class TrainingMetrics:
    """Track training metrics"""
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.step = 0
    
    def update(self, loss: float, lr: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step += 1
    
    def get_avg_loss(self, last_n: int = 100):
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses[-last_n:])
    

class AverageMeter:
    """Simple utility class for calculating running average"""
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val: float, sz: float):
        self.num += val*sz
        self.tot += sz

    def calculate(self) -> float:
        return self.num / self.tot


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Get learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, tokenizer, test_prompts: List[str], temperature: float = 0.7):
    """Evaluate model with test prompts"""
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    print("Generating samples from trained model:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        print("-" * 40)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with different temperatures
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids, 
                max_new_tokens=150,
                temperature=temperature
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Temperature {temperature}: {generated_text}")
            print()


def evaluate_cifar_model(model, criterion, val_loader):
    is_train = model.training
    model.eval()
    with torch.no_grad():
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for img, labels in val_loader:
            # move all img, labels to device (cuda)
            img = img.cuda()
            labels = labels.cuda()
            outputs, _ = model(img)
            loss_meter.update(criterion(outputs, labels).item(), len(img))
            acc = (outputs.argmax(-1) == labels).float().mean().item()
            acc_meter.update(acc, len(img))
    model.train(is_train)
    return loss_meter.calculate(), acc_meter.calculate()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
