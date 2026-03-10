"""DataLoader and tokenization utilities for multimodal training."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from .dataset import MultimodalPriceDataset


_TEXT_ENCODER_DEFAULTS = {
    "bert": {
        "tokenizer_name": "bert-base-uncased",
        "max_length": 512
    },
    "clip": {
        "tokenizer_name": "openai/clip-vit-base-patch32",
        "max_length": 77
    },
    "fasttext": {
        "tokenizer_name": None,
        "max_length": None,
        "lowercase": True
    }
}

def get_transforms(
    mode: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """
    Get image transforms for different modes.
    
    Args:
        mode: ``train``, ``val``, or ``test``
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Validation/test transforms intentionally avoid augmentation.
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
    if abs(train_split + val_split + test_split - 1.0) >= 1e-6:
        raise ValueError("train_split + val_split + test_split must sum to 1.0.")

    indices = np.arange(dataset_size)

    # First split: hold out the test set.
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_seed
    )

    # Second split: split remaining data into train/validation.
    val_size_adjusted = val_split / (train_split + val_split)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=random_seed
    )
    
    return train_indices, val_indices, test_indices


def _resolve_text_encoder_config(
    text_encoder: str,
    text_encoder_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Resolve text encoder config with defaults for the chosen encoder.

    User-provided values override defaults only when value is not ``None``.
    """
    encoder_key = (text_encoder or "").lower()
    if encoder_key not in _TEXT_ENCODER_DEFAULTS:
        raise ValueError(
            f"Unknown text encoder '{text_encoder}'. "
            f"Choose from {sorted(_TEXT_ENCODER_DEFAULTS.keys())}."
        )

    resolved = dict(_TEXT_ENCODER_DEFAULTS[encoder_key])
    if text_encoder_config:
        for key, value in text_encoder_config.items():
            if value is not None:
                resolved[key] = value

    return resolved


def _build_text_tokenizer(
    text_encoder: str,
    text_encoder_config: Dict[str, Any]
) -> Callable[[List[str]], Any]:
    """
    Build a tokenizer callable for the selected text encoder.

    Returns:
        Callable that receives a list of raw strings and returns model-ready
        text inputs:
        - BERT/CLIP: dictionary of tensors
        - FastText: list of token lists
    """
    encoder_key = (text_encoder or "").lower()

    if encoder_key in {"bert", "clip"}:
        tokenizer_name = text_encoder_config.get("tokenizer_name")
        max_length = text_encoder_config.get("max_length")
        if not tokenizer_name:
            raise ValueError(f"tokenizer_name must be set for text encoder '{encoder_key}'.")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        def tokenize(texts: List[str]) -> Dict[str, torch.Tensor]:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            return dict(encoded)

        return tokenize

    if encoder_key == "fasttext":
        lowercase = bool(text_encoder_config.get("lowercase", True))

        def tokenize(texts: List[str]) -> List[List[str]]:
            tokens = []
            for text in texts:
                cleaned = text.lower().strip() if lowercase else text.strip()
                tokens.append(cleaned.split())
            return tokens

        return tokenize

    raise ValueError(f"Unsupported text encoder '{text_encoder}'.")


def build_collate_fn(
    text_encoder: str = "bert",
    text_encoder_config: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Build a collate function that tokenizes text according to ``text_encoder``.
    """
    resolved_config = _resolve_text_encoder_config(text_encoder, text_encoder_config)
    tokenizer = _build_text_tokenizer(text_encoder, resolved_config)

    def collate(batch):
        images = torch.stack([item[0] for item in batch])
        raw_texts = [item[1] for item in batch]
        prices = torch.stack([item[2] for item in batch])
        metadata = [
            {**item[3], "raw_text": raw_texts[idx]}
            for idx, item in enumerate(batch)
        ]
        text_inputs = tokenizer(raw_texts)
        return images, text_inputs, prices, metadata

    return collate


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
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    text_encoder: str = "bert",
    text_encoder_config: Optional[Dict[str, Any]] = None,
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
        text_encoder: Text encoder type ('bert', 'clip', 'fasttext')
        text_encoder_config: Encoder-specific tokenizer config
        pin_memory: Whether to use pinned memory (faster for GPU)
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    # Build one dataset instance to compute valid sample set and split indices.
    full_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=None,
        mode="full"
    )

    train_indices, val_indices, test_indices = split_dataset_indices(
        dataset_size=len(full_dataset),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    
    print(
        f"Dataset splits - Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Each split gets its own transform pipeline. Dataset filtering remains
    # deterministic because all splits read the same CSV/images.
    train_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms("train", image_size, mean, std),
        mode="train"
    )

    val_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms("val", image_size, mean, std),
        mode="val"
    )

    test_dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_transforms("test", image_size, mean, std),
        mode="test"
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    text_collate_fn = build_collate_fn(
        text_encoder=text_encoder,
        text_encoder_config=text_encoder_config
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": text_collate_fn,
    }

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        drop_last=True,  # Keep stable tensor shapes in training statistics.
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print("\nDataset Statistics:")
    stats = full_dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def collate_fn(batch):
    """
    Legacy collate function that returns raw text without tokenization.

    Args:
        batch: List of tuples (image, text, price, metadata)

    Returns:
        Batched data
    """
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    prices = torch.stack([item[2] for item in batch])
    metadata = [
        {**item[3], "raw_text": texts[idx]}
        for idx, item in enumerate(batch)
    ]
    
    return images, texts, prices, metadata
