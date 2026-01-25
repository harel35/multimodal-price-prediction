"""
Training for the multimodal price prediction model.
"""
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.data import create_dataloaders
from src.models.image_encoders import (
    create_resnet_encoder,
    create_vit_encoder,
    create_dinov3_encoder
)
from src.models.text_encoders import (
    create_bert_text_encoder,
    create_clip_text_encoder,
    create_fasttext_encoder
)
from src.models.fusion import create_mlp_fusion
from src.utils.helpers import set_seed, save_checkpoint, load_checkpoint, count_parameters
from src.utils.metrics import smape_loss


class MultimodalPriceModel(nn.Module):
    """
    Simple wrapper for image encoder + text encoder + fusion head.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion_head: nn.Module
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_head = fusion_head

    def forward(self, images: torch.Tensor, text_inputs) -> torch.Tensor:
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_inputs)
        return self.fusion_head(image_features, text_features)


def _resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available() and getattr(config, "DEVICE", "cuda") == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _serialize_for_wandb(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_wandb(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_wandb(val) for val in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_wandb_config(
    model_config: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    base_config = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper()
    }
    payload = _serialize_for_wandb(base_config)
    if model_config:
        payload["resolved_model"] = _serialize_for_wandb(model_config)
    if runtime_config:
        payload["runtime"] = _serialize_for_wandb(runtime_config)
    return payload


def _move_text_inputs(text_inputs, device: torch.device):
    if isinstance(text_inputs, dict):
        return {key: value.to(device) for key, value in text_inputs.items()}
    if isinstance(text_inputs, torch.Tensor):
        return text_inputs.to(device)
    return text_inputs


def _resolve_image_encoder() -> Tuple[nn.Module, Dict[str, Any]]:
    encoder_name = getattr(config, "IMAGE_ENCODER", "resnet").lower()
    variant = getattr(config, "IMAGE_ENCODER_VARIANT", None)
    pretrained = getattr(config, "IMAGE_ENCODER_PRETRAINED", True)
    pretrained_name = getattr(config, "IMAGE_ENCODER_PRETRAINED_NAME", None)
    output_dim = getattr(config, "IMAGE_EMBEDDING_DIM", None)
    freeze_backbone = getattr(config, "IMAGE_ENCODER_FREEZE", False)
    dropout_rate = getattr(config, "IMAGE_DROPOUT_RATE", getattr(config, "DROPOUT_RATE", 0.3))

    if encoder_name in {"resnet", "resnet-18", "resnet-26", "resnet-34", "resnet-50", "resnet-101", "resnet-152"}:
        variant = variant or ("resnet-50" if encoder_name == "resnet" else encoder_name)
        encoder = create_resnet_encoder(
            variant=variant,
            pretrained=pretrained,
            pretrained_name=pretrained_name,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif encoder_name in {"vit", "vit-base", "vit-large"}:
        variant = variant or "vit-base-patch16-224"
        encoder = create_vit_encoder(
            variant=variant,
            pretrained=pretrained,
            pretrained_name=pretrained_name,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif encoder_name in {"dinov3", "dino", "dino-v3"}:
        variant = variant or "dinov3-vitb16-pretrain-lvd1689m"
        encoder = create_dinov3_encoder(
            variant=variant,
            pretrained=pretrained,
            pretrained_name=pretrained_name,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported IMAGE_ENCODER '{encoder_name}'.")

    encoder_config = {
        "image_encoder": encoder_name,
        "image_encoder_variant": variant,
        "image_encoder_pretrained": pretrained,
        "image_encoder_output_dim": output_dim
    }
    return encoder, encoder_config


def _resolve_text_encoder() -> Tuple[nn.Module, Dict[str, Any]]:
    encoder_name = getattr(config, "TEXT_ENCODER", "bert").lower()
    output_dim = getattr(config, "TEXT_EMBEDDING_DIM", None)
    freeze_backbone = getattr(config, "TEXT_ENCODER_FREEZE", False)
    dropout_rate = getattr(config, "TEXT_DROPOUT_RATE", getattr(config, "DROPOUT_RATE", 0.3))

    if encoder_name == "bert":
        variant = getattr(
            config,
            "TEXT_ENCODER_VARIANT",
            config.TEXT_ENCODER_CONFIG.get("tokenizer_name", "bert-base-uncased")
        )
        encoder = create_bert_text_encoder(
            variant=variant,
            pretrained=True,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif encoder_name == "clip":
        tokenizer_name = config.TEXT_ENCODER_CONFIG.get(
            "tokenizer_name",
            "openai/clip-vit-base-patch32"
        )
        variant = getattr(config, "TEXT_ENCODER_VARIANT", tokenizer_name.split("/", 1)[-1])
        encoder = create_clip_text_encoder(
            variant=variant,
            pretrained=True,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif encoder_name == "fasttext":
        variant = getattr(config, "TEXT_ENCODER_VARIANT", "cc.en.300")
        model_path = getattr(config, "FASTTEXT_MODEL_PATH", None)
        lowercase = bool(config.TEXT_ENCODER_CONFIG.get("lowercase", True))
        encoder = create_fasttext_encoder(
            variant=variant,
            pretrained=True,
            model_path=model_path,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            lowercase=lowercase
        )
    else:
        raise ValueError(f"Unsupported TEXT_ENCODER '{encoder_name}'.")

    encoder_config = {
        "text_encoder": encoder_name,
        "text_encoder_variant": encoder.variant,
        "text_encoder_output_dim": output_dim
    }
    return encoder, encoder_config


def _resolve_fusion_head(image_dim: int, text_dim: int) -> Tuple[nn.Module, Dict[str, Any]]:
    hidden_dims = getattr(config, "FUSION_HIDDEN_DIMS", (512, 256))
    fusion_method = getattr(config, "FUSION_METHOD", "concat")
    dropout_rate = getattr(config, "DROPOUT_RATE", 0.3)
    activation = getattr(config, "FUSION_ACTIVATION", "relu")
    use_batch_norm = getattr(config, "FUSION_USE_BATCH_NORM", True)
    fusion_dim = getattr(config, "FUSION_DIM", None)
    output_activation = getattr(config, "OUTPUT_ACTIVATION", None)

    fusion = create_mlp_fusion(
        image_dim=image_dim,
        text_dim=text_dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        fusion_method=fusion_method,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batch_norm=use_batch_norm,
        fusion_dim=fusion_dim,
        output_activation=output_activation
    )

    fusion_config = {
        "fusion_method": fusion_method,
        "fusion_hidden_dims": hidden_dims,
        "fusion_dim": fusion_dim,
        "fusion_output_activation": output_activation
    }
    return fusion, fusion_config


def _build_model() -> Tuple[nn.Module, Dict[str, Any]]:
    image_encoder, image_config = _resolve_image_encoder()
    text_encoder, text_config = _resolve_text_encoder()
    fusion_head, fusion_config = _resolve_fusion_head(
        image_dim=image_encoder.output_dim,
        text_dim=text_encoder.output_dim
    )
    model = MultimodalPriceModel(image_encoder, text_encoder, fusion_head)
    print("\nEmbedding dimensions:")
    print(f"  Image: {image_encoder.output_dim}")
    print(f"  Text: {text_encoder.output_dim}")
    model_config = {**image_config, **text_config, **fusion_config}
    return model, model_config


def _average_metrics(loss_sum: float, mae_sum: float, mse_sum: float, count: int) -> Dict[str, float]:
    count = max(count, 1)
    loss = loss_sum / count
    mae = mae_sum / count
    rmse = math.sqrt(mse_sum / count)
    return {"loss": loss, "mae": mae, "rmse": rmse}




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
        Tuple of (model, history)
    """
    print("=" * 80)
    print("Training Multimodal Price Prediction Model")
    print("=" * 80)

    set_seed(getattr(config, "RANDOM_SEED", 42))
    device = _resolve_device(device)
    print(f"\nUsing device: {device}")

    val_loader = None
    if train_loader is None:
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
            text_encoder=config.TEXT_ENCODER,
            text_encoder_config=config.TEXT_ENCODER_CONFIG
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
    elif isinstance(train_loader, dict):
        val_loader = train_loader.get("val")
        train_loader = train_loader.get("train")

    if model is None:
        model, model_config = _build_model()
    else:
        model_config = {}

    model = model.to(device)

    if criterion is None:
        criterion = smape_loss

    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=getattr(config, "LEARNING_RATE", 1e-4),
            weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4)
        )

    epochs = epochs or getattr(config, "NUM_EPOCHS", 10)
    trainable_params = count_parameters(model)
    print(f"\nTrainable parameters: {trainable_params:,}")

    if wandb.run is None:
        runtime_config = {
            "epochs": epochs,
            "trainable_params": trainable_params,
            "optimizer": optimizer.__class__.__name__,
            "device": str(device)
        }
        wandb_kwargs = {
            "project": getattr(config, "WANDB_PROJECT_NAME", "deep-learning-project"),
            "config": _collect_wandb_config(
                model_config=model_config,
                runtime_config=runtime_config
            )
        }
        wandb_entity = getattr(config, "WANDB_ENTITY", None)
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        run_name = getattr(config, "WANDB_RUN_NAME", None)
        if run_name:
            wandb_kwargs["name"] = run_name
        wandb_mode = getattr(config, "WANDB_MODE", None)
        if wandb_mode:
            wandb_kwargs["mode"] = wandb_mode
        wandb.init(**wandb_kwargs)

    checkpoint_path = Path(getattr(config, "CHECKPOINT_PATH", config.CHECKPOINT_DIR / "best_model.pt"))
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        loss_sum = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        count = 0

        for images, text_inputs, prices, _ in tqdm(train_loader, desc="train", leave=False):
            images = images.to(device, non_blocking=True)
            prices = prices.to(device, non_blocking=True).view(-1)
            text_inputs = _move_text_inputs(text_inputs, device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images, text_inputs).squeeze(-1)
            loss = criterion(preds, prices)
            loss.backward()
            optimizer.step()

            batch_size = prices.size(0)
            loss_sum += loss.item() * batch_size
            mae_sum += (preds - prices).abs().sum().item()
            mse_sum += ((preds - prices) ** 2).sum().item()
            count += batch_size

        train_metrics = _average_metrics(loss_sum, mae_sum, mse_sum, count)
        history["train"].append(train_metrics)

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                split_name="val",
                log_to_wandb=False
            )
            history["val"].append(val_metrics)

        log_payload = {
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/mae": train_metrics["mae"],
            "train/rmse": train_metrics["rmse"]
        }
        if val_metrics is not None:
            log_payload.update({
                "val/loss": val_metrics["loss"],
                "val/mae": val_metrics["mae"],
                "val/rmse": val_metrics["rmse"]
            })
        wandb.log(log_payload, step=epoch + 1)

        if val_metrics is not None:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_metrics["loss"],
                path=str(checkpoint_path),
                metrics=val_metrics
            )

    return model, history


def evaluate(
    model=None,
    data_loader=None,
    text_preprocessor=None,
    criterion=None,
    device=None,
    split_name: str = "test",
    log_to_wandb: bool = True
):
    """
    Evaluate the model.

    Args:
        model: The neural network model
        data_loader: data loader
        text_preprocessor: Text preprocessing utility
        criterion: Loss function
        device: Device to run on

    Returns:
        Metrics dictionary
    """
    device = _resolve_device(device)

    if data_loader is None:
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
            text_encoder=config.TEXT_ENCODER,
            text_encoder_config=config.TEXT_ENCODER_CONFIG
        )
        data_loader = dataloaders["test"]

    if model is None:
        model, _ = _build_model()
        checkpoint_path = Path(getattr(config, "CHECKPOINT_PATH", config.CHECKPOINT_DIR / "best_model.pt"))
        if checkpoint_path.exists():
            load_checkpoint(str(checkpoint_path), model, optimizer=None, device=str(device))
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Evaluating with fresh weights.")

    model = model.to(device)

    if criterion is None:
        criterion = smape_loss

    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for images, text_inputs, prices, _ in tqdm(data_loader, desc=split_name, leave=False):
            images = images.to(device, non_blocking=True)
            prices = prices.to(device, non_blocking=True).view(-1)
            text_inputs = _move_text_inputs(text_inputs, device)

            preds = model(images, text_inputs).squeeze(-1)
            loss = criterion(preds, prices)

            batch_size = prices.size(0)
            loss_sum += loss.item() * batch_size
            mae_sum += (preds - prices).abs().sum().item()
            mse_sum += ((preds - prices) ** 2).sum().item()
            count += batch_size

    metrics = _average_metrics(loss_sum, mae_sum, mse_sum, count)

    print(f"\n{split_name.capitalize()} metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")

    if log_to_wandb:
        if wandb.run is None:
            wandb_kwargs = {
                "project": getattr(config, "WANDB_PROJECT_NAME", "multimodal-price-prediction"),
                "config": _collect_wandb_config(runtime_config={
                    "split": split_name,
                    "device": str(device)
                })
            }
            wandb_entity = getattr(config, "WANDB_ENTITY", None)
            if wandb_entity:
                wandb_kwargs["entity"] = wandb_entity
            run_name = getattr(config, "WANDB_RUN_NAME", None)
            if run_name:
                wandb_kwargs["name"] = run_name
            wandb_mode = getattr(config, "WANDB_MODE", None)
            if wandb_mode:
                wandb_kwargs["mode"] = wandb_mode
            wandb.init(**wandb_kwargs)

        wandb.log({
            f"{split_name}/loss": metrics["loss"],
            f"{split_name}/mae": metrics["mae"],
            f"{split_name}/rmse": metrics["rmse"]
        })

    return metrics


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
