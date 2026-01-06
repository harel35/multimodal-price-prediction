"""
ResNet Image Encoder Module using HuggingFace Transformers

This module implements ResNet-based image encoders using the transformers library.
Supports multiple ResNet variants with options for pretrained weights, training from scratch,
and custom output dimensions.
"""

from transformers import ResNetModel, AutoImageProcessor, ResNetConfig
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import warnings


class ResNetEncoder(nn.Module):
    """
    ResNet-based image encoder using HuggingFace transformers library.
    
    This encoder wraps HuggingFace's ResNet models and provides a flexible interface
    for extracting image features. It supports:
    - Multiple ResNet variants (18, 26, 34, 50, 101, 152)
    - Pretrained weights from Microsoft's ResNet models
    - Training from scratch with custom initialization
    - Custom output dimension via projection layer
    - Optional dropout for regularization
    - Checkpoint saving and loading
    
    Args:
        variant (str): ResNet variant to use. Options: 'resnet-18', 'resnet-26', 
                      'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152'. 
                      Default: 'resnet-50'.
        pretrained (bool): Whether to use pretrained weights from Microsoft.
                          If False, model is randomly initialized. Default: True.
        pretrained_name (Optional[str]): Custom pretrained model name/path.
                                         If None, uses default microsoft/resnet-{variant}.
        output_dim (Optional[int]): Dimension of output features. If None, uses
                                    the natural ResNet output dimension (2048 for 
                                    ResNet50/101/152, 512 for ResNet18/26/34).
                                    If specified, adds a projection layer.
        freeze_backbone (bool): Whether to freeze the ResNet backbone. 
                               Useful for transfer learning. Default: False.
        pooling_type (str): Type of pooling to apply. Options: 'mean', 'max', 'cls'.
                           'mean' averages spatial dimensions, 'max' takes max,
                           'cls' uses the pooler output. Default: 'mean'.
        dropout_rate (float): Dropout rate to apply before final layer. Default: 0.3.
    
    Attributes:
        variant (str): The ResNet variant being used.
        feature_dim (int): Dimension of features before projection.
        output_dim (int): Final output dimension after projection (if any).
        model (ResNetModel): The HuggingFace ResNet model.
        projection (nn.Module): Optional projection layer to map to output_dim.
        dropout (nn.Dropout): Dropout layer for regularization.
    
    Example:
        >>> # Basic usage with pretrained ResNet50
        >>> encoder = ResNetEncoder(variant='resnet-50', pretrained=True)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> features = encoder(images)  # Shape: (8, 2048)
        
        >>> # Train from scratch with custom output dimension
        >>> encoder = ResNetEncoder(variant='resnet-50', pretrained=False, output_dim=512)
        >>> features = encoder(images)  # Shape: (8, 512)
        
        >>> # Use different pretrained model
        >>> encoder = ResNetEncoder(
        ...     variant='resnet-50',
        ...     pretrained=True,
        ...     pretrained_name='custom/resnet-model'
        ... )
    """
    
    def __init__(
        self,
        variant: str = 'resnet-50',
        pretrained: bool = True,
        pretrained_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        pooling_type: str = 'mean',
        dropout_rate: float = 0.3
    ):
        super(ResNetEncoder, self).__init__()
        
        self.variant = variant.lower()
        self.pretrained = pretrained
        self.pretrained_name = pretrained_name
        self.freeze_backbone = freeze_backbone
        self.pooling_type = pooling_type.lower()
        self.dropout_rate = dropout_rate
        
        # Validate variant
        valid_variants = ['resnet-18', 'resnet-26', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152']
        if self.variant not in valid_variants:
            raise ValueError(
                f"Invalid ResNet variant '{variant}'. "
                f"Must be one of {valid_variants}"
            )
        
        # Validate pooling type
        valid_pooling = ['mean', 'max', 'cls']
        if self.pooling_type not in valid_pooling:
            raise ValueError(
                f"Invalid pooling type '{pooling_type}'. "
                f"Must be one of {valid_pooling}"
            )
        
        # Load or create ResNet model
        self.model = self._create_model()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Set output dimension
        self.output_dim = output_dim if output_dim is not None else self.feature_dim
        
        # Add dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Add projection layer if output_dim is specified and different from feature_dim
        if output_dim is not None and output_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.projection = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _create_model(self) -> ResNetModel:
        """
        Create or load ResNet model.
        
        Returns:
            ResNetModel: HuggingFace ResNet model.
        """
        if self.pretrained:
            # Use custom pretrained name or default microsoft model
            model_name = self.pretrained_name or f"microsoft/{self.variant}"
            
            try:
                model = ResNetModel.from_pretrained(model_name)
                print(f"✓ Loaded pretrained {self.variant} from {model_name}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load pretrained model '{model_name}': {e}\n"
                    f"Initializing from scratch instead."
                )
                model = self._create_from_scratch()
        else:
            # Train from scratch
            model = self._create_from_scratch()
            print(f"✓ Initialized {self.variant} from scratch (random weights)")
        
        return model
    
    def _create_from_scratch(self) -> ResNetModel:
        """
        Create ResNet model from scratch with random initialization.
        
        Returns:
            ResNetModel: Randomly initialized ResNet model.
        """
        # Map variant to number of layers
        variant_to_layers = {
            'resnet-18': [2, 2, 2, 2],
            'resnet-26': [2, 2, 2, 2],  # Similar to ResNet-18
            'resnet-34': [3, 4, 6, 3],
            'resnet-50': [3, 4, 6, 3],
            'resnet-101': [3, 4, 23, 3],
            'resnet-152': [3, 8, 36, 3]
        }
        
        # Determine if we should use bottleneck (deeper models use bottleneck)
        use_bottleneck = self.variant in ['resnet-50', 'resnet-101', 'resnet-152']
        
        # Create configuration
        config = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048] if use_bottleneck else [64, 128, 256, 512],
            depths=variant_to_layers[self.variant],
            layer_type='bottleneck' if use_bottleneck else 'basic',
            hidden_act='relu',
            downsample_in_first_stage=False
        )
        
        # Create model from config
        model = ResNetModel(config)
        
        return model
    
    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension of the ResNet model.
        
        Returns:
            int: Feature dimension.
        """
        # For transformers ResNet, check the hidden_sizes
        if hasattr(self.model.config, 'hidden_sizes'):
            # Last hidden size is the feature dimension
            return self.model.config.hidden_sizes[-1]
        else:
            # Fallback based on variant
            feature_dims = {
                'resnet-18': 512,
                'resnet-26': 512,
                'resnet-34': 512,
                'resnet-50': 2048,
                'resnet-101': 2048,
                'resnet-152': 2048
            }
            return feature_dims[self.variant]
    
    def _freeze_backbone(self) -> None:
        """
        Freeze all parameters in the ResNet backbone.
        Only the projection layer (if present) will be trainable.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✓ Frozen entire {self.variant} backbone")
    
    def unfreeze_all(self) -> None:
        """
        Unfreeze all parameters in the encoder (backbone + projection).
        Useful for fine-tuning after initial training with frozen backbone.
        """
        for param in self.parameters():
            param.requires_grad = True
        
        print(f"✓ Unfrozen all parameters in {self.variant}")
    
    def unfreeze_last_n_stages(self, n: int) -> None:
        """
        Unfreeze the last N stages of the ResNet backbone.
        
        Args:
            n (int): Number of stages to unfreeze from the end (1-4).
        """
        if n < 1 or n > 4:
            raise ValueError(f"n must be between 1 and 4, got {n}")
        
        # ResNet has 4 stages: encoder.stages[0-3]
        stages = self.model.encoder.stages
        
        # Unfreeze last n stages
        for i in range(len(stages) - n, len(stages)):
            for param in stages[i].parameters():
                param.requires_grad = True
        
        print(f"✓ Unfrozen last {n} stage(s) of {self.variant}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, height, width).
                             Expected to be normalized appropriately (e.g., ImageNet stats).
        
        Returns:
            torch.Tensor: Encoded image features of shape (batch_size, output_dim).
        
        Raises:
            ValueError: If input tensor doesn't have 4 dimensions or wrong number of channels.
        """
        # Validate input
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch, channels, height, width), "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )
        
        if x.size(1) != 3:
            raise ValueError(
                f"Expected 3 input channels (RGB), got {x.size(1)} channels"
            )
        
        # Forward through ResNet model
        outputs = self.model(x)
        
        # Extract features based on pooling type
        if self.pooling_type == 'cls':
            # Use the pooler output (if available)
            features = outputs.pooler_output  # Shape: (batch, feature_dim)
        else:
            # Get last hidden state
            features = outputs.last_hidden_state  # Shape: (batch, feature_dim, H, W)
            
            # Apply pooling
            if self.pooling_type == 'mean':
                features = features.mean(dim=[2, 3])  # Shape: (batch, feature_dim)
            elif self.pooling_type == 'max':
                features = torch.max(torch.max(features, dim=2)[0], dim=2)[0]  # Shape: (batch, feature_dim)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Apply projection if present
        features = self.projection(features)  # Shape: (batch, output_dim)
        
        return features
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for the encoder.
        
        Returns:
            Dict[str, Any]: Configuration parameters.
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
            'has_projection': not isinstance(self.projection, nn.Identity)
        }
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Get count of trainable and total parameters.
        
        Returns:
            Tuple[int, int]: (trainable_params, total_params)
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
        
        Args:
            path (str): Path where to save the checkpoint.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer state to save.
            epoch (Optional[int]): Current training epoch.
            metrics (Optional[Dict[str, float]]): Training/validation metrics.
            additional_info (Optional[Dict[str, Any]]): Any additional information.
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'model_class': self.__class__.__name__
        }
        
        # Add optional components
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if additional_info is not None:
            checkpoint['additional_info'] = additional_info
        
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to: {path}")
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load encoder checkpoint and restore model state.
        
        Args:
            path (str): Path to the checkpoint file.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into.
            strict (bool): Whether to strictly enforce state_dict keys match. Default: True.
            load_optimizer (bool): Whether to load optimizer state. Default: True.
        
        Returns:
            Dict[str, Any]: Checkpoint metadata (epoch, metrics, etc.).
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Validate model configuration
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            current_config = self.get_config()
            
            # Check critical parameters
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
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Model state loaded from: {path}")
        
        # Load optimizer state if present and requested
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        
        # Return metadata
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
    ) -> 'ResNetEncoder':
        """
        Create a ResNetEncoder instance from a checkpoint file.
        
        Args:
            path (str): Path to the checkpoint file.
            map_location (Optional[str]): Device to map tensors to ('cpu', 'cuda', etc.).
            strict (bool): Whether to strictly enforce state_dict keys match.
        
        Returns:
            ResNetEncoder: Loaded encoder instance.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=map_location or 'cpu')
        
        # Extract configuration
        if 'config' not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'config' key. "
                "Cannot reconstruct model architecture."
            )
        
        config = checkpoint['config']
        
        # Create model instance with saved configuration
        model = cls(
            variant=config['variant'],
            pretrained=False,  # Don't load pretrained, we'll load checkpoint weights
            pretrained_name=config.get('pretrained_name'),
            output_dim=config['output_dim'] if config['output_dim'] != config['feature_dim'] else None,
            freeze_backbone=False,
            pooling_type=config.get('pooling_type', 'mean'),
            dropout_rate=config['dropout_rate']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        print(f"✓ Loaded {config['variant']} encoder from: {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")
        
        return model
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        trainable, total = self.get_trainable_parameters()
        return (
            f"ResNetEncoder(\n"
            f"  variant={self.variant},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pretrained={self.pretrained},\n"
            f"  pooling_type={self.pooling_type},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )


def create_resnet_encoder(
    variant: str = 'resnet-50',
    pretrained: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> ResNetEncoder:
    """
    Factory function to create a ResNet encoder with common configurations.
    
    Args:
        variant (str): ResNet variant.
        pretrained (bool): Whether to use pretrained weights.
        output_dim (Optional[int]): Output feature dimension.
        **kwargs: Additional arguments passed to ResNetEncoder.
    
    Returns:
        ResNetEncoder: Configured ResNet encoder.
    """
    return ResNetEncoder(
        variant=variant,
        pretrained=pretrained,
        output_dim=output_dim,
        **kwargs
    )