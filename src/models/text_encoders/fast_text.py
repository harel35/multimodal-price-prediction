"""
FastText Text Encoder Module.

This module implements a configurable FastText-based text encoder for feature extraction.
It supports:
- Loading pretrained Meta FastText models (.bin/.ftz)
    - Training from scratch on a text corpus
- Custom output dimensions via projection layer
- Optional dropout for regularization
- Checkpoint saving and loading
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import warnings

import torch
import torch.nn as nn

try:
    import fasttext
except Exception:  # pragma: no cover - handled at runtime
    fasttext = None


class FastTextEncoder(nn.Module):
    """
    FastText-based text encoder.

    Args:
        variant (str): FastText variant name or path to model file.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        model_path (Optional[str]): Path to FastText model (.bin/.ftz).
        output_dim (Optional[int]): Output feature dimension. If None, uses vector dim.
        freeze_backbone (bool): Whether to freeze the backbone. Default: True.
        pooling_type (str): Pooling method: 'sentence', 'mean', or 'max'.
        dropout_rate (float): Dropout rate applied before projection. Default: 0.1.
        train_data_path (Optional[str]): Text file path for training from scratch.
        train_mode (str): Training mode: 'skipgram' or 'cbow'. Default: 'skipgram'.
        train_args (Optional[Dict[str, Any]]): Additional args for fasttext training.
        lowercase (bool): Whether to lowercase text for token pooling. Default: True.
    """

    def __init__(
        self,
        variant: str = 'cc.en.300',
        pretrained: bool = True,
        model_path: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        pooling_type: str = 'sentence',
        dropout_rate: float = 0.1,
        train_data_path: Optional[str] = None,
        train_mode: str = 'skipgram',
        train_args: Optional[Dict[str, Any]] = None,
        lowercase: bool = True
    ):
        super(FastTextEncoder, self).__init__()

        self.variant = variant
        self.pretrained = pretrained
        self.model_path = model_path
        self.freeze_backbone = freeze_backbone
        self.pooling_type = pooling_type.lower()
        self.dropout_rate = dropout_rate
        self.train_data_path = train_data_path
        self.train_mode = train_mode
        self.train_args = train_args or {}
        self.lowercase = lowercase

        valid_pooling = ['sentence', 'mean', 'max']
        if self.pooling_type not in valid_pooling:
            raise ValueError(
                f"Invalid pooling type '{pooling_type}'. "
                f"Must be one of {valid_pooling}"
            )

        if self.pretrained:
            if self._resolve_model_path() is None:
                raise ValueError(
                    "pretrained=True requires a valid model_path or variant "
                    "pointing to a FastText .bin/.ftz file."
                )
        else:
            if self.train_data_path is None:
                raise ValueError(
                    "pretrained=False requires train_data_path for FastText training."
                )

        self.model = self._create_model()
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

        self.register_buffer("_device_tensor", torch.empty(0), persistent=False)

        if freeze_backbone:
            self._freeze_backbone()

    def _ensure_fasttext(self) -> None:
        """
        Ensure the fasttext library is available.
        """
        if fasttext is None:
            raise ImportError(
                "fasttext is required for FastTextEncoder. "
                "Install it with `pip install fasttext`."
            )

    def _resolve_model_path(self) -> Optional[str]:
        """
        Resolve the model path for pretrained FastText.
        """
        if self.model_path:
            return self.model_path

        if self.variant and Path(self.variant).exists():
            return self.variant

        return None

    def _create_model(self):
        """
        Create or load FastText model.
        """
        self._ensure_fasttext()

        if self.pretrained:
            model_path = self._resolve_model_path()
            try:
                model = fasttext.load_model(str(model_path))
                print(f"Loaded pretrained FastText model from {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load FastText model '{model_path}': {e}"
                )
        else:
            train_args = dict(self.train_args)
            if 'model' not in train_args:
                train_args['model'] = self.train_mode
            try:
                model = fasttext.train_unsupervised(
                    input=self.train_data_path,
                    **train_args
                )
                print(f"Trained FastText model from {self.train_data_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to train FastText model from '{self.train_data_path}': {e}"
                )

        return model

    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension of the FastText model.
        """
        if hasattr(self.model, 'get_dimension'):
            return int(self.model.get_dimension())
        return self.train_args.get('dim', 300)

    def _freeze_backbone(self) -> None:
        """
        Freeze the FastText backbone (no trainable parameters).
        """
        print("FastText backbone is non-trainable; projection remains trainable.")

    def unfreeze_all(self) -> None:
        """
        Unfreeze all parameters in the encoder (projection only).
        """
        for param in self.parameters():
            param.requires_grad = True

        print("Unfrozen all parameters in FastText encoder")

    def _tokenize(self, text: str) -> List[str]:
        """
        Basic whitespace tokenizer for FastText pooling.
        """
        if self.lowercase:
            text = text.lower()
        return text.strip().split()

    def _sentence_vector(self, text: str) -> torch.Tensor:
        """
        Get sentence vector from FastText.
        """
        vec = self.model.get_sentence_vector(text)
        return torch.from_numpy(vec)

    def _pool_tokens(self, tokens: List[str]) -> torch.Tensor:
        """
        Pool token vectors into a single sentence embedding.
        """
        if not tokens:
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        vectors = [torch.from_numpy(self.model.get_word_vector(tok)) for tok in tokens]
        stacked = torch.stack(vectors, dim=0)
        if self.pooling_type == 'mean':
            return stacked.mean(dim=0)
        return stacked.max(dim=0).values

    def forward(self, texts: Union[str, List[str], List[List[str]]]) -> torch.Tensor:
        """
        Forward pass through the FastText encoder.

        Args:
            texts (str, List[str], or List[List[str]]): Input text(s) or token lists.

        Returns:
            torch.Tensor: Encoded text features of shape (batch, output_dim).
        """
        if texts is None:
            raise ValueError("texts must be provided for FastTextEncoder.")

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a string or a list/tuple of strings.")

        is_tokenized = len(texts) > 0 and isinstance(texts[0], (list, tuple))
        if is_tokenized and self.lowercase:
            texts = [[str(token).lower() for token in tokens] for tokens in texts]

        if self.pooling_type == 'sentence':
            if is_tokenized:
                joined = [" ".join(tokens) for tokens in texts]
                features = torch.stack([self._sentence_vector(text) for text in joined], dim=0)
            else:
                features = torch.stack([self._sentence_vector(text) for text in texts], dim=0)
        else:
            if is_tokenized:
                features = torch.stack([self._pool_tokens(tokens) for tokens in texts], dim=0)
            else:
                features = torch.stack([self._pool_tokens(self._tokenize(text)) for text in texts], dim=0)

        device = self._device_tensor.device
        features = features.to(device=device, dtype=torch.float32)

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
            'model_path': self.model_path,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'freeze_backbone': self.freeze_backbone,
            'pooling_type': self.pooling_type,
            'dropout_rate': self.dropout_rate,
            'train_data_path': self.train_data_path,
            'train_mode': self.train_mode,
            'train_args': self.train_args,
            'lowercase': self.lowercase,
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
    ) -> 'FastTextEncoder':
        """
        Create a FastTextEncoder instance from a checkpoint file.
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
            pretrained=config['pretrained'],
            model_path=config.get('model_path'),
            output_dim=config['output_dim'] if config['output_dim'] != config['feature_dim'] else None,
            freeze_backbone=config.get('freeze_backbone', True),
            pooling_type=config.get('pooling_type', 'sentence'),
            dropout_rate=config.get('dropout_rate', 0.1),
            train_data_path=config.get('train_data_path'),
            train_mode=config.get('train_mode', 'skipgram'),
            train_args=config.get('train_args'),
            lowercase=config.get('lowercase', True)
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        print(f"Loaded FastText encoder from: {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")

        return model

    def __repr__(self) -> str:
        trainable, total = self.get_trainable_parameters()
        return (
            f"FastTextEncoder(\n"
            f"  variant={self.variant},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pretrained={self.pretrained},\n"
            f"  pooling_type={self.pooling_type},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )


def create_fasttext_encoder(
    variant: str = 'cc.en.300',
    pretrained: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> FastTextEncoder:
    """
    Factory function to create a FastText encoder with common configurations.
    """
    return FastTextEncoder(
        variant=variant,
        pretrained=pretrained,
        output_dim=output_dim,
        **kwargs
    )
