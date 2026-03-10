"""General helper utilities for training, evaluation, and visualization."""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """Save model and optimizer state to disk."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
):
    """Load checkpoint into model (and optionally optimizer)."""
    map_location = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        map_location = "cpu"

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    return checkpoint


def _to_price(value: float, values_are_log1p: bool) -> float:
    """Convert model/log-space value into displayed USD price."""
    return float(np.expm1(value)) if values_are_log1p else float(value)


def visualize_batch(
    images: torch.Tensor,
    texts: list,
    prices: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    figsize: tuple = (15, 10),
    mean: Optional[list] = None,
    std: Optional[list] = None,
    save_path: Optional[str] = None,
    prices_are_log1p: bool = False,
):
    """Visualize a mini-batch with optional predictions."""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    if isinstance(texts, dict):
        texts = ["<tokenized>"] * len(images)

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for idx in range(num_samples):
        img = images[idx].detach().cpu().clone()
        for channel in range(3):
            img[channel] = img[channel] * std[channel] + mean[channel]
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        axes[idx].imshow(img)
        axes[idx].axis("off")

        text_item = texts[idx]
        if isinstance(text_item, (list, tuple)):
            text_item = " ".join(text_item)
        text_short = text_item[:70] + "..." if len(text_item) > 70 else text_item

        true_price = _to_price(prices[idx].item(), prices_are_log1p)
        title = f"{text_short}\nTrue: ${true_price:.2f}"

        if predictions is not None:
            pred_price = _to_price(predictions[idx].item(), prices_are_log1p)
            title += f"\nPred: ${pred_price:.2f}"

        axes[idx].set_title(title, fontsize=8)

    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    plt.show()


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute standard regression metrics (MAE/MSE/RMSE/MAPE/R2)."""
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    non_zero_mask = targets != 0
    if np.any(non_zero_mask):
        mape = np.mean(
            np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])
        ) * 100
    else:
        mape = float("nan")

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty-print a metric dictionary."""
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for name, value in metrics.items():
        if name == "mape":
            if np.isnan(value):
                print(f"  {name.upper()}: N/A")
            else:
                print(f"  {name.upper()}: {value:.2f}%")
        else:
            print(f"  {name.upper()}: {value:.4f}")
    print("-" * 50)


def get_device():
    """Return the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS device")
        return torch.device("mps")

    print("Using CPU device")
    return torch.device("cpu")


def create_exp_directory(base_dir: str = "./experiments") -> Path:
    """Create and return a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir
