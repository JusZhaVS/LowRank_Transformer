from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model hyperparameters
    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    use_causal_mask: bool = True

    # Rank Size
    rank: int = 32

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
