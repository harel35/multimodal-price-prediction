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
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.data import create_dataloaders
from src.utils import TextPreprocessor


def train(
    model=None,
    train_loader=None,
    text_preprocessor=None,
    criterion=None,
    optimizer=None,
    device=None,
    epochs=None
):
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
        std=config.IMAGE_STD
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    


def evaluate(
    model=None,
    data_loader=None,
    text_preprocessor=None,
    criterion=None,
    device=None
):
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
    # Create WandB run
    run = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        config={
            "model_name": config.MODEL_NAME
        }
    )
    print("\nEvaluating on test set...")
    test_loss = evaluate(
        model=model,
        data_loader=data_loader,
        text_preprocessor=text_preprocessor,
        criterion=config.LOSS_FUNCTION,
        device=device
    )
    print(f"Test Loss: {test_loss:.4f}")
    wandb.log({"Test Loss": test_loss})


def main():
    '''
    Main Function which parses arguments and calls relevant functions
    '''
    parser = argparse.ArgumentParser(
        description="Train or evaluate the multimodal price prediction model."
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")

    args = parser.parse_args()

    if not (args.train or args.evaluate):
        parser.print_help()
        return

    if args.train:
        train()
    if args.evaluate:
        evaluate()
        

if __name__ == "__main__":
    main()
