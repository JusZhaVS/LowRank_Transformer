from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500

    # Paths
    output_dir: str = "./tiny_lowrank_model"
    log_dir: str = "./logs"
