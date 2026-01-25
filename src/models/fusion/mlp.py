"""
MLP Fusion Module for multimodal price prediction.

This module fuses image and text features and predicts a scalar price.
It supports simple fusion strategies and a configurable MLP head.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterable, List, Callable
import warnings

import torch
import torch.nn as nn


class MLPFusion(nn.Module):
    """
    MLP-based fusion model for image/text features.

    Args:
        image_dim (int): Dimension of image features.
        text_dim (int): Dimension of text features.
        hidden_dims (Iterable[int]): Hidden layer sizes for the MLP.
        output_dim (int): Output dimension, default 1 for price prediction.
        fusion_method (str): 'concat', 'addition', 'attention', or 'gated'.
        dropout_rate (float): Dropout rate applied between layers.
        activation (str): Activation name: 'relu', 'gelu', or 'tanh'.
        use_batch_norm (bool): Whether to add BatchNorm between layers.
        use_layer_norm (bool): Whether to add LayerNorm between layers.
        use_residual (bool): Whether to add residual connections when dims match.
        fusion_dim (Optional[int]): Shared dimension for addition/attention if dims differ.
        output_activation (Optional[str]): Output activation: 'relu', 'softplus', 'sigmoid', or None.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        hidden_dims: Iterable[int] = (512, 256),
        output_dim: int = 1,
        fusion_method: str = 'concat',
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_residual: bool = True,
        fusion_dim: Optional[int] = None,
        output_activation: Optional[str] = None
    ):
        super(MLPFusion, self).__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.fusion_method = fusion_method.lower()
        self.dropout_rate = dropout_rate
        self.activation = activation.lower()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.fusion_dim = fusion_dim
        self.output_activation = output_activation.lower() if output_activation else None

        valid_fusion = ['concat', 'addition', 'attention', 'gated']
        if self.fusion_method not in valid_fusion:
            raise ValueError(
                f"Invalid fusion method '{fusion_method}'. "
                f"Must be one of {valid_fusion}"
            )
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("Enable only one of use_batch_norm or use_layer_norm.")

        self.image_proj = nn.Identity()
        self.text_proj = nn.Identity()
        self.gate = None

        fused_dim = self._get_fused_dim()
        self.fused_dim = fused_dim

        self.mlp = self._build_mlp()
        self.output_layer = nn.Linear(self.hidden_dims[-1] if self.hidden_dims else fused_dim, output_dim)
        self.output_act = self._get_output_activation()
        if self.fusion_method == 'gated':
            self.gate = nn.Linear(2 * fused_dim, fused_dim)

    def _get_fused_dim(self) -> int:
        """
        Configure projections and determine the fused dimension.
        """
        if self.fusion_method == 'concat':
            return self.image_dim + self.text_dim

        if self.image_dim != self.text_dim:
            if self.fusion_dim is None:
                raise ValueError(
                    "fusion_dim must be set when image_dim != text_dim "
                    "for addition/attention/gated fusion."
                )
            self.image_proj = nn.Linear(self.image_dim, self.fusion_dim)
            self.text_proj = nn.Linear(self.text_dim, self.fusion_dim)
            return self.fusion_dim

        return self.image_dim

    def _get_activation(self) -> nn.Module:
        """
        Select activation module.
        """
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        if self.activation == 'gelu':
            return nn.GELU()
        if self.activation == 'tanh':
            return nn.Tanh()
        raise ValueError(f"Unsupported activation '{self.activation}'")

    def _get_output_activation(self) -> nn.Module:
        """
        Select output activation module.
        """
        if self.output_activation is None:
            return nn.Identity()
        if self.output_activation == 'relu':
            return nn.ReLU(inplace=True)
        if self.output_activation == 'softplus':
            return nn.Softplus()
        if self.output_activation == 'sigmoid':
            return nn.Sigmoid()
        raise ValueError(f"Unsupported output activation '{self.output_activation}'")

    def _get_norm(self, dim: int) -> nn.Module:
        """
        Select normalization module.
        """
        if self.use_batch_norm:
            return nn.BatchNorm1d(dim)
        if self.use_layer_norm:
            return nn.LayerNorm(dim)
        return nn.Identity()

    def _build_mlp(self) -> nn.Module:
        """
        Build MLP head layers.
        """
        class MLPBlock(nn.Module):
            def __init__(
                self,
                in_dim: int,
                out_dim: int,
                activation_factory: Callable[[], nn.Module],
                norm_factory: Callable[[int], nn.Module],
                dropout_rate: float,
                use_residual: bool
            ):
                super().__init__()
                self.use_residual = use_residual and (in_dim == out_dim)
                self.linear = nn.Linear(in_dim, out_dim)
                self.norm = norm_factory(out_dim)
                self.activation = activation_factory()
                self.dropout = nn.Dropout(p=dropout_rate)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.linear(x)
                out = self.norm(out)
                out = self.activation(out)
                out = self.dropout(out)
                if self.use_residual:
                    out = out + x
                return out

        layers: List[nn.Module] = []
        in_dim = self.fused_dim
        for hidden_dim in self.hidden_dims:
            layers.append(
                MLPBlock(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    activation_factory=self._get_activation,
                    norm_factory=self._get_norm,
                    dropout_rate=self.dropout_rate,
                    use_residual=self.use_residual
                )
            )
            in_dim = hidden_dim
        if not layers:
            return nn.Identity()
        return nn.Sequential(*layers)

    def _fuse(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse image and text features according to the configured method.
        """
        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        if self.fusion_method == 'concat':
            return torch.cat([image_features, text_features], dim=-1)

        if self.fusion_method == 'addition':
            return image_features + text_features

        if self.fusion_method == 'gated':
            if self.gate is None:
                raise RuntimeError("Gated fusion requires a gate layer.")
            merged = torch.cat([image_features, text_features], dim=-1)
            weights = torch.sigmoid(self.gate(merged))
            return weights * text_features + (1.0 - weights) * image_features

        # attention-based weighting between modalities
        logits = torch.stack([image_features, text_features], dim=1)  # (B, 2, D)
        weights = torch.softmax(logits.mean(dim=-1), dim=1).unsqueeze(-1)
        return (logits * weights).sum(dim=1)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion MLP.

        Args:
            image_features (torch.Tensor): Image feature tensor (batch, image_dim).
            text_features (torch.Tensor): Text feature tensor (batch, text_dim).

        Returns:
            torch.Tensor: Predicted prices (batch, output_dim).
        """
        if image_features is None or text_features is None:
            raise ValueError("Both image_features and text_features are required.")

        if image_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError(
                "Expected 2D feature tensors (batch, feature_dim). "
                f"Got {image_features.shape} and {text_features.shape}."
            )

        fused = self._fuse(image_features, text_features)
        hidden = self.mlp(fused)
        output = self.output_layer(hidden)
        output = self.output_act(output)
        return output

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for the fusion module.
        """
        return {
            'image_dim': self.image_dim,
            'text_dim': self.text_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'fusion_method': self.fusion_method,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_residual': self.use_residual,
            'fusion_dim': self.fusion_dim,
            'output_activation': self.output_activation
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
        Save fusion checkpoint with model state and training information.
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
        Load fusion checkpoint and restore model state.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu')

        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            current_config = self.get_config()

            critical_params = ['image_dim', 'text_dim', 'output_dim']
            for param in critical_params:
                if saved_config.get(param) != current_config.get(param):
                    warning_msg = (
                        f"Configuration mismatch for '{param}': "
                        f"checkpoint has {saved_config.get(param)}, "
                        f"current model has {current_config.get(param)}"
                    )
                    if strict:
                        raise RuntimeError(warning_msg)
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
    ) -> 'MLPFusion':
        """
        Create an MLPFusion instance from a checkpoint file.
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
            image_dim=config['image_dim'],
            text_dim=config['text_dim'],
            hidden_dims=config.get('hidden_dims', (512, 256)),
            output_dim=config.get('output_dim', 1),
            fusion_method=config.get('fusion_method', 'concat'),
            dropout_rate=config.get('dropout_rate', 0.3),
            activation=config.get('activation', 'relu'),
            use_batch_norm=config.get('use_batch_norm', True),
            use_layer_norm=config.get('use_layer_norm', False),
            use_residual=config.get('use_residual', True),
            fusion_dim=config.get('fusion_dim'),
            output_activation=config.get('output_activation')
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        print(f"Loaded fusion model from: {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")

        return model

    def __repr__(self) -> str:
        trainable, total = self.get_trainable_parameters()
        return (
            f"MLPFusion(\n"
            f"  image_dim={self.image_dim},\n"
            f"  text_dim={self.text_dim},\n"
            f"  fused_dim={self.fused_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  fusion_method={self.fusion_method},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )


def create_mlp_fusion(
    image_dim: int,
    text_dim: int,
    output_dim: int = 1,
    **kwargs
) -> MLPFusion:
    """
    Factory function to create an MLP fusion head.
    """
    return MLPFusion(
        image_dim=image_dim,
        text_dim=text_dim,
        output_dim=output_dim,
        **kwargs
    )
