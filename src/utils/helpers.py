"""
Helper utilities for the deep learning project.
"""
import os
import random
import numpy as np
import torch
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
):
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint


def visualize_batch(
    images: torch.Tensor,
    texts: list,
    prices: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    figsize: tuple = (15, 10),
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    save_path: Optional[str] = None
):
    """
    Visualize a batch of samples.
    
    Args:
        images: Batch of images [B, C, H, W]
        texts: List of text descriptions
        prices: True prices
        predictions: Optional predicted prices
        num_samples: Number of samples to show
        figsize: Figure size
        mean: Normalization mean
        std: Normalization std
        save_path: Optional path to save figure
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(num_samples):
        # Denormalize image
        img = images[idx].cpu().clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        # Plot image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Title with text and price
        text_short = texts[idx][:50] + '...' if len(texts[idx]) > 50 else texts[idx]
        title = f"{text_short}\nTrue: ${np.exp(prices[idx].item()) - 1:.2f}"
        
        if predictions is not None:
            pred_price = np.exp(predictions[idx].item()) - 1
            title += f"\nPred: ${pred_price:.2f}"
        
        axes[idx].set_title(title, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
    
    Returns:
        Dictionary of metrics
    """
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (avoiding division by zero)
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for printing
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for name, value in metrics.items():
        if name == 'mape':
            print(f"  {name.upper()}: {value:.2f}%")
        else:
            print(f"  {name.upper()}: {value:.4f}")
    print("-" * 50)


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def create_exp_directory(base_dir: str = "./experiments") -> Path:
    """
    Create a unique experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    
    return exp_dir
