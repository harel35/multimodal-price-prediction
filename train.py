"""
Training template for the multimodal price prediction model.

This is a skeleton template showing how to structure your training loop.
You'll need to implement the model architecture in src/models/model.py
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.data import create_dataloaders
from src.utils import TextPreprocessor


def train_epoch(model, train_loader, text_preprocessor, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        text_preprocessor: Text preprocessing utility
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for images, texts, prices, metadata in pbar:
        # Move data to device
        images = images.to(device)
        prices = prices.to(device)
        
        # Encode texts
        encoded_texts = text_preprocessor.encode_batch(texts)
        input_ids = encoded_texts['input_ids'].to(device)
        attention_mask = encoded_texts['attention_mask'].to(device)
        
        # Forward pass
        # TODO: Implement your model forward pass
        # predictions = model(images, input_ids, attention_mask)
        
        # For now, just a placeholder
        # predictions = torch.randn_like(prices)  # Replace with actual model output
        
        # Compute loss
        # loss = criterion(predictions, prices)
        
        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # total_loss += loss.item()
        # num_batches += 1
        
        # pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # return avg_loss
    
    print("TODO: Implement model and training loop")
    return 0.0


def validate(model, val_loader, text_preprocessor, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        text_preprocessor: Text preprocessing utility
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, texts, prices, metadata in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = images.to(device)
            prices = prices.to(device)
            
            # Encode texts
            encoded_texts = text_preprocessor.encode_batch(texts)
            input_ids = encoded_texts['input_ids'].to(device)
            attention_mask = encoded_texts['attention_mask'].to(device)
            
            # Forward pass
            # TODO: Implement your model forward pass
            # predictions = model(images, input_ids, attention_mask)
            
            # Compute loss
            # loss = criterion(predictions, prices)
            # total_loss += loss.item()
            # num_batches += 1
    
    # avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # return avg_loss
    
    return 0.0


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
        log_transform_price=True
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
    
    # TODO: Initialize your model
    print("\nInitializing model...")
    print("TODO: Create your model in src/models/model.py")
    print("Example:")
    print("  from src.models.model import MultimodalPricePredictor")
    print("  model = MultimodalPricePredictor(...).to(device)")
    
    # model = YourModel().to(device)
    
    # Loss function and optimizer
    # criterion = nn.MSELoss()  # or nn.L1Loss() or nn.HuberLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training Loop")
    print("=" * 80)
    
    # best_val_loss = float('inf')
    # patience_counter = 0
    
    # for epoch in range(config.NUM_EPOCHS):
    #     print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
    #     
    #     # Train
    #     train_loss = train_epoch(
    #         model, train_loader, text_preprocessor,
    #         criterion, optimizer, device
    #     )
    #     
    #     # Validate
    #     val_loss = validate(
    #         model, val_loader, text_preprocessor,
    #         criterion, device
    #     )
    #     
    #     print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    #     
    #     # Learning rate scheduling
    #     scheduler.step(val_loss)
    #     
    #     # Early stopping
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
    #         # Save best model
    #         torch.save(model.state_dict(), config.CHECKPOINT_DIR / 'best_model.pth')
    #         print(f"Saved best model (val_loss: {val_loss:.4f})")
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= config.EARLY_STOPPING_PATIENCE:
    #             print(f"Early stopping after {epoch + 1} epochs")
    #             break
    
    print("\nTODO: Implement your model architecture to start training!")
    print("\nSteps to get started:")
    print("  1. Create src/models/model.py with your model class")
    print("  2. Uncomment the training code in this file")
    print("  3. Run: python train.py")


if __name__ == "__main__":
    main()
