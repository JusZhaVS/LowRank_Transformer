from src.ViTransformerModel import ViTransformerModel
from src.config.ViTrainingConfig import ViTrainingConfig
from src.config.ViTransformerConfig import ViTransformerConfig

from Trainers.ViTrainer import ViTrainer
from dataset.CIFAR10Dataset import CIFAR10Dataset

from test.visualize_attention import visualize_attention

def main():
    # Create CIFAR Dataset Object
    cifar_dataset = CIFAR10Dataset()
    train_dataloader, val_dataloader = cifar_dataset.get_dataloaders()

    # Default ViTraining and ViTransformer Config
    vit_training_config = ViTrainingConfig()
    vit_transformer_config = ViTransformerConfig()

    # Set up Model
    model = ViTransformerModel(config=vit_transformer_config)
    vitrainer = ViTrainer(model=model,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          config=vit_training_config)
    
    model = vitrainer.train()
    visualize_attention(model=model)


if __name__ == '__main__':
    main()
