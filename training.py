import datasets

from transformers import AutoTokenizer
from Trainers.training_utils import count_parameters
from Trainers.Trainer import Trainer

from src.CausalLanguageModel import CausalLanguageModel
from src.config.TrainingConfig import TrainingConfig
from src.config.TransformerConfig import TransformerConfig

from dataset.TinyStoriesDataset import TinyStoriesDataset


def main():
    # Set up Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 

    # Load Dataset
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    print(f"Dataset loaded with {len(dataset)} examples")

    train_dataset = TinyStoriesDataset(dataset, tokenizer, max_length=512, max_samples=10000)

    # Load Training Configuration
    training_config = TrainingConfig(vocab_size=tokenizer.vocab_size)

    # Create model config and initialize model
    model_config = TransformerConfig(rank=32)

    # Initialize model
    model = CausalLanguageModel(model_config)
    print(f"Model initialized with {count_parameters(model):,} parameters")

    # Initialize trainer
    trainer = Trainer(model, train_dataset, tokenizer, training_config)
    model = trainer.train()


if __name__ == '__main__':
    main()
