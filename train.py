"""
Training for the multimodal price prediction model.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.data import create_dataloaders
from src.utils import TextPreprocessor


def train(model, train_loader, text_preprocessor, criterion, optimizer, device, epochs):
    """
    Train for number of epochs.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        text_preprocessor: Text preprocessing utility
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of epochs to train
    
    Returns:
        Average training loss
    """
    


def evaluate(model, data_loader, text_preprocessor, criterion, device):
    """
    evaluate the model.
    
    Args:
        model: The neural network model
        data_loader: data loader
        text_preprocessor: Text preprocessing utility
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average loss
    """


    


def main():
    """Main training function."""
    
    print("=" * 80)
    print("Training Multimodal Price Prediction Model")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        csv_path=str(config.CSV_PATH),
        images_dir=str(config.IMAGES_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        log_transform_price=False
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Initialize text preprocessor
    print("\nInitializing text preprocessor...")
    text_preprocessor = TextPreprocessor(
        model_name='bert-base-uncased',
        max_length=config.MAX_TEXT_LENGTH
    )
    
    # Initialize model
    print("\nInitializing model...")

    model = config.MODEL_NAME(
        text_preprocessor=text_preprocessor,
        image_model_name=config.IMAGE_MODEL_NAME,
        pretrained=config.PRETRAINED_IMAGE_MODEL,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    print(model)

    # Create WandB run
    run = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "model_name": config.MODEL_NAME
        }
    )
    
    # Train
    
    # Evaluate on test set
    if(config.EVALUATE_TEST):
        print("\nEvaluating on test set...")
        test_loss = evaluate(
            model=model,
            data_loader=test_loader,
            text_preprocessor=text_preprocessor,
            criterion=config.LOSS_FUNCTION,
            device=device
        )
        print(f"Test Loss: {test_loss:.4f}")
        wandb.log({"Test Loss": test_loss})

if __name__ == "__main__":
    main()
