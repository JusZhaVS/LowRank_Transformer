from dataclasses import dataclass


@dataclass
class ViTrainingConfig:
    """Configuration class for vision transformer model"""
    num_epochs = 10
    learning_rate = 1e-3

    # Output Directories
    output_dir = "./cifar10_lowrank_model"
    log_dir = "./logs"
