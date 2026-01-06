"""
Vision Transformer (ViT) Image Encoder Module using HuggingFace Transformers.

This module implements a configurable ViT-based image encoder for feature extraction.
It supports:
- Multiple ViT variants with pretrained weights
- Training from scratch with custom configuration
- Custom output dimensions via projection layer
- Optional dropout for regularization
- Checkpoint saving and loading
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) based image encoder using HuggingFace transformers.

    Args:
        variant (str): ViT variant to use. Example: 'vit-base-patch16-224'.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        pretrained_name (Optional[str]): Custom pretrained model name/path.
        output_dim (Optional[int]): Output feature dimension. If None, uses hidden_size.
        freeze_backbone (bool): Whether to freeze the ViT backbone. Default: False.
        pooling_type (str): Pooling method: 'cls', 'mean', or 'max'. Default: 'cls'.
        dropout_rate (float): Dropout rate applied before projection. Default: 0.3.
        image_size (Optional[int]): Image size for scratch training. Overrides variant.
        patch_size (Optional[int]): Patch size for scratch training. Overrides variant.
        hidden_size (Optional[int]): Hidden size for scratch training. Overrides variant.
        num_hidden_layers (Optional[int]): Transformer layers for scratch training.
        num_attention_heads (Optional[int]): Attention heads for scratch training.
        mlp_dim (Optional[int]): MLP dimension for scratch training.
        use_bias (bool): Whether to use bias in QKV projections. Default: True.
    """

    _VARIANT_CONFIGS = {
        'vit-base-patch16-224': {
            'image_size': 224,
            'patch_size': 16,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'mlp_dim': 3072
        },
        'vit-base-patch32-224': {
            'image_size': 224,
            'patch_size': 32,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'mlp_dim': 3072
        },
        'vit-base-patch16-384': {
            'image_size': 384,
            'patch_size': 16,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'mlp_dim': 3072
        },
        'vit-large-patch16-224': {
            'image_size': 224,
            'patch_size': 16,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'mlp_dim': 4096
        },
        'vit-large-patch32-224': {
            'image_size': 224,
            'patch_size': 32,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'mlp_dim': 4096
        },
        'vit-large-patch16-384': {
            'image_size': 384,
            'patch_size': 16,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'mlp_dim': 4096
        }
    }

    _PRETRAINED_MAP = {
        'vit-base-patch16-224': 'google/vit-base-patch16-224',
        'vit-base-patch16-224-in21k': 'google/vit-base-patch16-224-in21k',
        'vit-base-patch32-224': 'google/vit-base-patch32-224',
        'vit-base-patch16-384': 'google/vit-base-patch16-384',
        'vit-large-patch16-224': 'google/vit-large-patch16-224',
        'vit-large-patch32-224': 'google/vit-large-patch32-224',
        'vit-large-patch16-384': 'google/vit-large-patch16-384'
    }

    def __init__(
        self,
        variant: str = 'vit-base-patch16-224',
        pretrained: bool = True,
        pretrained_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        pooling_type: str = 'cls',
        dropout_rate: float = 0.3,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        use_bias: bool = True
    ):
        super(ViTEncoder, self).__init__()

        self.variant = variant.lower()
        self.pretrained = pretrained
        self.pretrained_name = pretrained_name
        self.freeze_backbone = freeze_backbone
        self.pooling_type = pooling_type.lower()
        self.dropout_rate = dropout_rate

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.use_bias = use_bias

        valid_pooling = ['cls', 'mean', 'max']
        if self.pooling_type not in valid_pooling:
            raise ValueError(
                f"Invalid pooling type '{pooling_type}'. "
                f"Must be one of {valid_pooling}"
            )

        if not self.pretrained and self.variant not in self._VARIANT_CONFIGS:
            if any(val is None for val in [
                self.image_size,
                self.patch_size,
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.mlp_dim
            ]):
                raise ValueError(
                    "Unknown ViT variant for scratch initialization. "
                    "Provide image_size, patch_size, hidden_size, num_hidden_layers, "
                    "num_attention_heads, and mlp_dim."
                )

        self.model = self._create_model()
        self._sync_model_config()

        self.feature_dim = self._get_feature_dim()
        self.output_dim = output_dim if output_dim is not None else self.feature_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        if output_dim is not None and output_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.projection = nn.Identity()

        if freeze_backbone:
            self._freeze_backbone()

    def _get_pretrained_name(self) -> str:
        """
        Resolve the pretrained model name based on variant or explicit name.
        """
        if self.pretrained_name:
            return self.pretrained_name

        if self.variant in self._PRETRAINED_MAP:
            return self._PRETRAINED_MAP[self.variant]

        if '/' in self.variant:
            return self.variant

        raise ValueError(
            f"Invalid ViT variant '{self.variant}'. "
            f"Known variants: {sorted(self._VARIANT_CONFIGS.keys())}"
        )

    def _get_variant_config(self) -> Optional[Dict[str, int]]:
        """
        Get default config for a known variant.
        """
        return self._VARIANT_CONFIGS.get(self.variant)

    def _create_model(self) -> ViTModel:
        """
        Create or load ViT model.
        """
        if self.pretrained:
            model_name = self._get_pretrained_name()
            try:
                model = ViTModel.from_pretrained(model_name)
                print(f"Loaded pretrained {self.variant} from {model_name}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load pretrained model '{model_name}': {e}\n"
                    f"Initializing from scratch instead."
                )
                model = self._create_from_scratch()
        else:
            model = self._create_from_scratch()
            print(f"Initialized {self.variant} from scratch (random weights)")

        return model

    def _create_from_scratch(self) -> ViTModel:
        """
        Create ViT model from scratch with random initialization.
        """
        defaults = self._get_variant_config() or {}

        image_size = self.image_size or defaults.get('image_size')
        patch_size = self.patch_size or defaults.get('patch_size')
        hidden_size = self.hidden_size or defaults.get('hidden_size')
        num_hidden_layers = self.num_hidden_layers or defaults.get('num_hidden_layers')
        num_attention_heads = self.num_attention_heads or defaults.get('num_attention_heads')
        mlp_dim = self.mlp_dim or defaults.get('mlp_dim')

        missing = [
            name for name, value in [
                ('image_size', image_size),
                ('patch_size', patch_size),
                ('hidden_size', hidden_size),
                ('num_hidden_layers', num_hidden_layers),
                ('num_attention_heads', num_attention_heads),
                ('mlp_dim', mlp_dim)
            ] if value is None
        ]
        if missing:
            raise ValueError(
                f"Missing configuration for scratch ViT: {missing}. "
                "Provide these parameters explicitly."
            )

        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=mlp_dim,
            qkv_bias=self.use_bias,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate
        )

        return ViTModel(config)

    def _sync_model_config(self) -> None:
        """
        Sync encoder attributes from the underlying model configuration.
        """
        if not hasattr(self.model, 'config'):
            return

        config = self.model.config
        self.image_size = getattr(config, 'image_size', self.image_size)
        self.patch_size = getattr(config, 'patch_size', self.patch_size)
        self.hidden_size = getattr(config, 'hidden_size', self.hidden_size)
        self.num_hidden_layers = getattr(config, 'num_hidden_layers', self.num_hidden_layers)
        self.num_attention_heads = getattr(config, 'num_attention_heads', self.num_attention_heads)
        self.mlp_dim = getattr(config, 'intermediate_size', self.mlp_dim)
        self.use_bias = getattr(config, 'qkv_bias', self.use_bias)

    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension of the ViT model.
        """
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        return self.hidden_size or 768

    def _freeze_backbone(self) -> None:
        """
        Freeze all parameters in the ViT backbone.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Frozen entire {self.variant} backbone")

    def unfreeze_all(self) -> None:
        """
        Unfreeze all parameters in the encoder.
        """
        for param in self.parameters():
            param.requires_grad = True

        print(f"Unfrozen all parameters in {self.variant}")

    def unfreeze_last_n_layers(self, n: int) -> None:
        """
        Unfreeze the last N transformer layers.

        Args:
            n (int): Number of layers to unfreeze from the end.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'layer'):
            raise AttributeError("ViT model does not expose encoder layers to unfreeze.")

        layers = self.model.encoder.layer
        if n > len(layers):
            raise ValueError(f"n must be <= {len(layers)}, got {n}")

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Unfrozen last {n} layer(s) of {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT encoder.

        Args:
            x (torch.Tensor): Input images of shape (batch, 3, height, width).

        Returns:
            torch.Tensor: Encoded image features of shape (batch, output_dim).
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch, channels, height, width), "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        if x.size(1) != 3:
            raise ValueError(
                f"Expected 3 input channels (RGB), got {x.size(1)} channels"
            )

        outputs = self.model(pixel_values=x)
        tokens = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if self.pooling_type == 'cls':
            if outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = tokens[:, 0]
        elif self.pooling_type == 'mean':
            if tokens.size(1) > 1:
                features = tokens[:, 1:, :].mean(dim=1)
            else:
                features = tokens.mean(dim=1)
        else:
            if tokens.size(1) > 1:
                features = tokens[:, 1:, :].max(dim=1).values
            else:
                features = tokens.max(dim=1).values

        features = self.dropout(features)
        features = self.projection(features)

        return features

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for the encoder.
        """
        return {
            'variant': self.variant,
            'pretrained': self.pretrained,
            'pretrained_name': self.pretrained_name,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'freeze_backbone': self.freeze_backbone,
            'pooling_type': self.pooling_type,
            'dropout_rate': self.dropout_rate,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'mlp_dim': self.mlp_dim,
            'use_bias': self.use_bias,
            'has_projection': not isinstance(self.projection, nn.Identity)
        }

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Get count of trainable and total parameters.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save encoder checkpoint with model state and training information.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'model_class': self.__class__.__name__
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if additional_info is not None:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load encoder checkpoint and restore model state.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu')

        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            current_config = self.get_config()

            critical_params = ['variant', 'feature_dim', 'output_dim']
            for param in critical_params:
                if saved_config.get(param) != current_config.get(param):
                    warning_msg = (
                        f"Configuration mismatch for '{param}': "
                        f"checkpoint has {saved_config.get(param)}, "
                        f"current model has {current_config.get(param)}"
                    )
                    if strict:
                        raise RuntimeError(warning_msg)
                    else:
                        warnings.warn(warning_msg)

        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Model state loaded from: {path}")

        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")

        metadata = {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
            'additional_info': checkpoint.get('additional_info'),
            'config': checkpoint.get('config')
        }

        return metadata

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        map_location: Optional[str] = None,
        strict: bool = True
    ) -> 'ViTEncoder':
        """
        Create a ViTEncoder instance from a checkpoint file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location or 'cpu')

        if 'config' not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'config' key. "
                "Cannot reconstruct model architecture."
            )

        config = checkpoint['config']

        model = cls(
            variant=config['variant'],
            pretrained=False,
            pretrained_name=config.get('pretrained_name'),
            output_dim=config['output_dim'] if config['output_dim'] != config['feature_dim'] else None,
            freeze_backbone=False,
            pooling_type=config.get('pooling_type', 'cls'),
            dropout_rate=config['dropout_rate'],
            image_size=config.get('image_size'),
            patch_size=config.get('patch_size'),
            hidden_size=config.get('hidden_size'),
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            mlp_dim=config.get('mlp_dim'),
            use_bias=config.get('use_bias', True)
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        print(f"Loaded {config['variant']} encoder from: {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")

        return model

    def __repr__(self) -> str:
        trainable, total = self.get_trainable_parameters()
        return (
            f"ViTEncoder(\n"
            f"  variant={self.variant},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pretrained={self.pretrained},\n"
            f"  pooling_type={self.pooling_type},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )


def create_vit_encoder(
    variant: str = 'vit-base-patch16-224',
    pretrained: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> ViTEncoder:
    """
    Factory function to create a ViT encoder with common configurations.
    """
    return ViTEncoder(
        variant=variant,
        pretrained=pretrained,
        output_dim=output_dim,
        **kwargs
    )
