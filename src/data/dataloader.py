"""
DataLoader utilities for creating train, validation, and test loaders.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from .dataset import MultimodalPriceDataset


def get_transforms(mode: str = 'train', image_size: Tuple[int, int] = (224, 224),
                   mean: list = [0.485, 0.456, 0.406],
                   std: list = [0.229, 0.224, 0.225]) -> transforms.Compose:
    """
    Get image transforms for different modes.
    
    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    if mode == 'train':
        # Training augmentations
        return transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def split_dataset_indices(
    dataset_size: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into train, validation, and test sets.
    
    Args:
        dataset_size: Total number of samples
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    indices = np.arange(dataset_size)
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_seed
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_split / (train_split + val_split)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=random_seed
    )
    
    return train_indices, val_indices, test_indices


def create_dataloaders(
    csv_path: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        csv_path: Path to CSV file
        images_dir: Directory containing images
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        pin_memory: Whether to use pinned memory (faster for GPU)
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create full dataset with training transforms first to get indices
    full_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=None,  # We'll apply transforms per split
        mode='full'
    )
    
    # Split indices
    train_indices, val_indices, test_indices = split_dataset_indices(
        dataset_size=len(full_dataset),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    
    print(f"Dataset splits - Train: {len(train_indices)}, "
          f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create separate datasets with appropriate transforms
    train_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms('train', image_size, mean, std),
        mode='train'
    )
    
    val_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms('val', image_size, mean, std),
        mode='val'
    )
    
    test_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms('test', image_size, mean, std),
        mode='test'
    )
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for training
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    stats = full_dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn(batch):
    """
    Custom collate function for batching multimodal data.
    
    Args:
        batch: List of tuples (image, text, price, metadata)
    
    Returns:
        Batched data
    """
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    prices = torch.stack([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    
    return images, texts, prices, metadata
