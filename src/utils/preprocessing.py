"""Preprocessing helpers for text normalization and price transforms."""

import re
from typing import List

import torch
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    Text preprocessor for legacy workflows that need explicit tokenization.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512
    ):
        """
        Args:
            model_name: Hugging Face model name for tokenizer.
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        text = "" if text is None else str(text)

        # Remove extra whitespace.
        text = re.sub(r'\s+', ' ', text)

        # Keep alphanumeric content and basic punctuation.
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?-]', '', text)

        text = text.strip()

        return text
    
    def encode_text(
        self,
        text: str,
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Encode text using tokenizer.
        
        Args:
            text: Input text
            return_tensors: Format for tensors ('pt' for PyTorch)
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        cleaned_text = self.clean_text(text)
        
        encoded = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def encode_batch(
        self,
        texts: List[str],
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts
            return_tensors: Format for tensors
        
        Returns:
            Dictionary with batched encodings
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        encoded = self.tokenizer(
            cleaned_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
        
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class PriceTransformer:
    """
    Optional transform for price targets (legacy utility).
    """
    
    def __init__(self, method: str = 'log1p'):
        """
        Args:
            method: Transformation method ('log1p', 'log', 'sqrt', 'none')
        """
        self.method = method
        self.mean = None
        self.std = None
    
    def fit(self, prices: torch.Tensor):
        """
        Fit transformer on training data.
        
        Args:
            prices: Tensor of prices
        """
        positive_prices = prices.clamp_min(0.0)
        if self.method == 'log1p':
            transformed = torch.log1p(positive_prices)
        elif self.method == 'log':
            transformed = torch.log(positive_prices + 1e-8)
        elif self.method == 'sqrt':
            transformed = torch.sqrt(positive_prices)
        else:
            transformed = prices
        
        self.mean = transformed.mean()
        self.std = transformed.std()
    
    def transform(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Transform prices.
        
        Args:
            prices: Tensor of prices
        
        Returns:
            Transformed prices
        """
        positive_prices = prices.clamp_min(0.0)
        if self.method == 'log1p':
            transformed = torch.log1p(positive_prices)
        elif self.method == 'log':
            transformed = torch.log(positive_prices + 1e-8)
        elif self.method == 'sqrt':
            transformed = torch.sqrt(positive_prices)
        else:
            transformed = prices
        
        # Optionally normalize
        if self.mean is not None and self.std is not None:
            transformed = (transformed - self.mean) / (self.std + 1e-8)
        
        return transformed
    
    def inverse_transform(self, transformed_prices: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform prices back to original scale.
        
        Args:
            transformed_prices: Transformed prices
        
        Returns:
            Original scale prices
        """
        # Denormalize if needed
        if self.mean is not None and self.std is not None:
            transformed_prices = transformed_prices * self.std + self.mean
        
        if self.method == 'log1p':
            prices = torch.expm1(transformed_prices)
        elif self.method == 'log':
            prices = torch.exp(transformed_prices) - 1e-8
        elif self.method == 'sqrt':
            prices = transformed_prices ** 2
        else:
            prices = transformed_prices
        
        return prices


def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Denormalize an image tensor.
    
    Args:
        image: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized image tensor
    """
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    
    denormalized = image * std_tensor + mean_tensor
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized
