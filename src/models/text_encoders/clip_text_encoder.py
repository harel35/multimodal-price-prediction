"""
CLIP Text Encoder Module using HuggingFace Transformers.

This module implements a configurable CLIP-based text encoder for feature extraction.
It supports:
- Multiple CLIP text variants with pretrained weights
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
from transformers import CLIPTextModel, CLIPTextConfig


class CLIPTextEncoder(nn.Module):
    """
    CLIP-based text encoder using HuggingFace transformers.

    Args:
        variant (str): CLIP variant to use.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        pretrained_name (Optional[str]): Custom pretrained model name/path.
        output_dim (Optional[int]): Output feature dimension. If None, uses hidden_size.
        freeze_backbone (bool): Whether to freeze the backbone. Default: False.
        pooling_type (str): Pooling method: 'eos', 'cls', 'mean', or 'max'.
        dropout_rate (float): Dropout rate applied before projection. Default: 0.1.
        vocab_size (Optional[int]): Vocabulary size for scratch training.
        max_position_embeddings (Optional[int]): Context length for scratch training.
        hidden_size (Optional[int]): Hidden size for scratch training.
        num_hidden_layers (Optional[int]): Transformer layers for scratch training.
        num_attention_heads (Optional[int]): Attention heads for scratch training.
        intermediate_size (Optional[int]): MLP dimension for scratch training.
        pad_token_id (Optional[int]): Padding token ID.
        bos_token_id (Optional[int]): BOS token ID.
        eos_token_id (Optional[int]): EOS token ID.
        layer_norm_eps (float): Layer norm epsilon. Default: 1e-5.
    """

    _VARIANT_CONFIGS = {
        'clip-vit-base-patch32': {
            'vocab_size': 49408,
            'max_position_embeddings': 77,
            'hidden_size': 512,
            'num_hidden_layers': 12,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'pad_token_id': 1,
            'bos_token_id': 49406,
            'eos_token_id': 49407
        },
        'clip-vit-base-patch16': {
            'vocab_size': 49408,
            'max_position_embeddings': 77,
            'hidden_size': 512,
            'num_hidden_layers': 12,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'pad_token_id': 1,
            'bos_token_id': 49406,
            'eos_token_id': 49407
        },
        'clip-vit-large-patch14': {
            'vocab_size': 49408,
            'max_position_embeddings': 77,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'pad_token_id': 1,
            'bos_token_id': 49406,
            'eos_token_id': 49407
        },
        'clip-vit-large-patch14-336': {
            'vocab_size': 49408,
            'max_position_embeddings': 77,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'pad_token_id': 1,
            'bos_token_id': 49406,
            'eos_token_id': 49407
        }
    }

    _PRETRAINED_MAP = {
        'clip-vit-base-patch32': 'openai/clip-vit-base-patch32',
        'clip-vit-base-patch16': 'openai/clip-vit-base-patch16',
        'clip-vit-large-patch14': 'openai/clip-vit-large-patch14',
        'clip-vit-large-patch14-336': 'openai/clip-vit-large-patch14-336'
    }

    def __init__(
        self,
        variant: str = 'clip-vit-base-patch32',
        pretrained: bool = True,
        pretrained_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        pooling_type: str = 'eos',
        dropout_rate: float = 0.1,
        vocab_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        layer_norm_eps: float = 1e-5
    ):
        super(CLIPTextEncoder, self).__init__()

        self.variant = variant.lower()
        self.pretrained = pretrained
        self.pretrained_name = pretrained_name
        self.freeze_backbone = freeze_backbone
        self.pooling_type = pooling_type.lower()
        self.dropout_rate = dropout_rate

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.layer_norm_eps = layer_norm_eps

        valid_pooling = ['eos', 'cls', 'mean', 'max']
        if self.pooling_type not in valid_pooling:
            raise ValueError(
                f"Invalid pooling type '{pooling_type}'. "
                f"Must be one of {valid_pooling}"
            )

        if not self.pretrained and self.variant not in self._VARIANT_CONFIGS:
            if any(val is None for val in [
                self.vocab_size,
                self.max_position_embeddings,
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size
            ]):
                raise ValueError(
                    "Unknown CLIP variant for scratch initialization. "
                    "Provide vocab_size, max_position_embeddings, hidden_size, "
                    "num_hidden_layers, num_attention_heads, and intermediate_size."
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
            f"Invalid CLIP variant '{self.variant}'. "
            f"Known variants: {sorted(self._VARIANT_CONFIGS.keys())}"
        )

    def _get_variant_config(self) -> Optional[Dict[str, int]]:
        """
        Get default config for a known variant.
        """
        return self._VARIANT_CONFIGS.get(self.variant)

    def _create_model(self) -> CLIPTextModel:
        """
        Create or load CLIP text model.
        """
        if self.pretrained:
            model_name = self._get_pretrained_name()
            try:
                model = CLIPTextModel.from_pretrained(model_name)
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

    def _create_from_scratch(self) -> CLIPTextModel:
        """
        Create CLIP text model from scratch with random initialization.
        """
        defaults = self._get_variant_config() or {}

        vocab_size = self.vocab_size or defaults.get('vocab_size')
        max_position_embeddings = self.max_position_embeddings or defaults.get('max_position_embeddings')
        hidden_size = self.hidden_size or defaults.get('hidden_size')
        num_hidden_layers = self.num_hidden_layers or defaults.get('num_hidden_layers')
        num_attention_heads = self.num_attention_heads or defaults.get('num_attention_heads')
        intermediate_size = self.intermediate_size or defaults.get('intermediate_size')
        pad_token_id = self.pad_token_id or defaults.get('pad_token_id')
        bos_token_id = self.bos_token_id or defaults.get('bos_token_id')
        eos_token_id = self.eos_token_id or defaults.get('eos_token_id')

        missing = [
            name for name, value in [
                ('vocab_size', vocab_size),
                ('max_position_embeddings', max_position_embeddings),
                ('hidden_size', hidden_size),
                ('num_hidden_layers', num_hidden_layers),
                ('num_attention_heads', num_attention_heads),
                ('intermediate_size', intermediate_size)
            ] if value is None
        ]
        if missing:
            raise ValueError(
                f"Missing configuration for scratch CLIP: {missing}. "
                "Provide these parameters explicitly."
            )

        config = CLIPTextConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            layer_norm_eps=self.layer_norm_eps,
            dropout=self.dropout_rate,
            attention_dropout=self.dropout_rate
        )

        return CLIPTextModel(config)

    def _sync_model_config(self) -> None:
        """
        Sync encoder attributes from the underlying model configuration.
        """
        if not hasattr(self.model, 'config'):
            return

        config = self.model.config
        self.vocab_size = getattr(config, 'vocab_size', self.vocab_size)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', self.max_position_embeddings)
        self.hidden_size = getattr(config, 'hidden_size', self.hidden_size)
        self.num_hidden_layers = getattr(config, 'num_hidden_layers', self.num_hidden_layers)
        self.num_attention_heads = getattr(config, 'num_attention_heads', self.num_attention_heads)
        self.intermediate_size = getattr(config, 'intermediate_size', self.intermediate_size)
        self.pad_token_id = getattr(config, 'pad_token_id', self.pad_token_id)
        self.bos_token_id = getattr(config, 'bos_token_id', self.bos_token_id)
        self.eos_token_id = getattr(config, 'eos_token_id', self.eos_token_id)
        self.layer_norm_eps = getattr(config, 'layer_norm_eps', self.layer_norm_eps)

    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension of the CLIP text model.
        """
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        return self.hidden_size or 512

    def _freeze_backbone(self) -> None:
        """
        Freeze all parameters in the CLIP text backbone.
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

        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'layers'):
            raise AttributeError("CLIP text model does not expose encoder layers to unfreeze.")

        layers = self.model.encoder.layers
        if n > len(layers):
            raise ValueError(f"n must be <= {len(layers)}, got {n}")

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Unfrozen last {n} layer(s) of {self.variant}")

    def _pool_eos(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Pool using the end-of-sequence token.
        """
        batch_size, seq_len, _ = last_hidden_state.shape
        device = last_hidden_state.device

        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1)
            indices = (lengths - 1).to(device)
        elif input_ids is not None and self.eos_token_id is not None:
            eos_mask = input_ids.eq(self.eos_token_id)
            if eos_mask.any(dim=1).all():
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                indices = (eos_mask.long() * positions).max(dim=1).values
            else:
                indices = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
        else:
            indices = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)

        batch_indices = torch.arange(batch_size, device=device)
        return last_hidden_state[batch_indices, indices]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the CLIP text encoder.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch, seq_len).

        Returns:
            torch.Tensor: Encoded text features of shape (batch, output_dim).
        """
        inputs = None
        if isinstance(input_ids, dict):
            inputs = input_ids
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
        elif 'inputs' in kwargs:
            inputs = kwargs.pop('inputs')
            input_ids = inputs.get('input_ids', input_ids)
            attention_mask = inputs.get('attention_mask', attention_mask)

        if input_ids is None and inputs is None:
            raise ValueError("input_ids or inputs dict must be provided.")

        if inputs is None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        else:
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        pooler_output = getattr(outputs, 'pooler_output', None)

        if self.pooling_type == 'eos':
            if pooler_output is not None:
                features = pooler_output
            else:
                features = self._pool_eos(last_hidden_state, input_ids, attention_mask)
        elif self.pooling_type == 'cls':
            features = last_hidden_state[:, 0]
        elif self.pooling_type == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1).clamp(min=1.0)
                features = (last_hidden_state * mask).sum(dim=1) / denom
            else:
                features = last_hidden_state.mean(dim=1)
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                masked = last_hidden_state.masked_fill(mask == 0, float('-inf'))
                features = masked.max(dim=1).values
            else:
                features = last_hidden_state.max(dim=1).values

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
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'layer_norm_eps': self.layer_norm_eps,
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
    ) -> 'CLIPTextEncoder':
        """
        Create a CLIPTextEncoder instance from a checkpoint file.
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
            pooling_type=config.get('pooling_type', 'eos'),
            dropout_rate=config['dropout_rate'],
            vocab_size=config.get('vocab_size'),
            max_position_embeddings=config.get('max_position_embeddings'),
            hidden_size=config.get('hidden_size'),
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            pad_token_id=config.get('pad_token_id'),
            bos_token_id=config.get('bos_token_id'),
            eos_token_id=config.get('eos_token_id'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5)
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
            f"CLIPTextEncoder(\n"
            f"  variant={self.variant},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pretrained={self.pretrained},\n"
            f"  pooling_type={self.pooling_type},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )


def create_clip_text_encoder(
    variant: str = 'clip-vit-base-patch32',
    pretrained: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> CLIPTextEncoder:
    """
    Factory function to create a CLIP text encoder with common configurations.
    """
    return CLIPTextEncoder(
        variant=variant,
        pretrained=pretrained,
        output_dim=output_dim,
        **kwargs
    )
