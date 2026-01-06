"""
Example script demonstrating how to use the dataset and dataloader.

This script shows:
1. How to load and inspect the dataset
2. How to create dataloaders
3. How to iterate through batches
4. How to use the text preprocessor
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.data import create_dataloaders
from src.utils import TextPreprocessor


def main():
    """Main function demonstrating dataset and dataloader usage."""
    
    print("=" * 80)
    print("Multimodal Price Prediction - Dataset & DataLoader Example")
    print("=" * 80)
    
    # Configuration
    csv_path = str(config.CSV_PATH)
    images_dir = str(config.IMAGES_DIR)
    
    print(f"\nData paths:")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {images_dir}")
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("Creating DataLoaders...")
    print("=" * 80)
    
    dataloaders = create_dataloaders(
        csv_path=csv_path,
        images_dir=images_dir,
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
    
    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize text preprocessor
    print("\n" + "=" * 80)
    print("Initializing Text Preprocessor...")
    print("=" * 80)
    
    text_preprocessor = TextPreprocessor(
        model_name='bert-base-uncased',
        max_length=config.MAX_TEXT_LENGTH
    )
    print("Text preprocessor initialized with BERT tokenizer")
    
    # Inspect a batch
    print("\n" + "=" * 80)
    print("Inspecting Training Batch...")
    print("=" * 80)
    
    for batch_idx, (images, texts, prices, metadata) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of texts: {len(texts)}")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Number of metadata items: {len(metadata)}")
        
        # Show sample text
        print(f"\n  Sample text (first item):")
        print(f"    Raw: {texts[0][:100]}...")
        
        # Encode texts
        encoded = text_preprocessor.encode_batch(texts)
        print(f"\n  Encoded text shapes:")
        print(f"    Input IDs: {encoded['input_ids'].shape}")
        print(f"    Attention mask: {encoded['attention_mask'].shape}")
        
        # Show price statistics
        print(f"\n  Batch price statistics (log-transformed):")
        print(f"    Min: {prices.min().item():.4f}")
        print(f"    Max: {prices.max().item():.4f}")
        print(f"    Mean: {prices.mean().item():.4f}")
        print(f"    Std: {prices.std().item():.4f}")
        
        # Show original prices
        original_prices = torch.tensor([m['original_price'] for m in metadata])
        print(f"\n  Original price statistics:")
        print(f"    Min: ${original_prices.min().item():.2f}")
        print(f"    Max: ${original_prices.max().item():.2f}")
        print(f"    Mean: ${original_prices.mean().item():.2f}")
        
        # Show sample metadata
        print(f"\n  Sample metadata (first item):")
        for key, value in metadata[0].items():
            print(f"    {key}: {value}")
        
        # Only process first batch for demo
        break
    
    # Validation batch
    print("\n" + "=" * 80)
    print("Inspecting Validation Batch...")
    print("=" * 80)
    
    for batch_idx, (images, texts, prices, metadata) in enumerate(val_loader):
        print(f"\nValidation Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Texts count: {len(texts)}")
        print(f"  Prices shape: {prices.shape}")
        
        # Only process first batch
        break
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Create a model architecture (in src/models/)")
    print("  2. Implement training loop")
    print("  3. Add evaluation metrics")
    print("  4. Implement inference pipeline")


if __name__ == "__main__":
    main()
