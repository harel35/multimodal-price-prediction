"""
Custom PyTorch Dataset for multimodal (image + text) price prediction.
"""
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultimodalPriceDataset(Dataset):
    """
    Custom Dataset for loading images and text data for price prediction.
    
    This dataset handles:
    - Loading images from file paths
    - Extracting text features from catalog content
    - Preprocessing both modalities
    - Returning normalized price targets
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train'
    ):
        """
        Args:
            csv_path: Path to the CSV file containing data
            images_dir: Directory containing images
            transform: Torchvision transforms for images
            mode: 'train', 'val', or 'test'
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.mode = mode
        
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Filter out samples with missing images or prices
        self.df = self._filter_valid_samples()
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples in {mode} mode")
        
    def _filter_valid_samples(self) -> pd.DataFrame:
        """Filter samples that have valid images and prices."""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            # Check if price is valid
            if pd.isna(row['price']) or row['price'] <= 0:
                continue
            
            # Extract image filename and check if it exists
            image_filename = self._extract_image_filename(row['image_link'])
            image_path = self.images_dir / image_filename
            
            if image_path.exists():
                valid_indices.append(idx)
        
        return self.df.loc[valid_indices]
    
    def _extract_image_filename(self, image_link: str) -> str:
        """
        Extract image filename from URL.
        Example: https://m.media-amazon.com/images/I/51mo8htwTHL.jpg -> 51mo8htwTHL.jpg
        """
        # Extract the part after the last '/'
        filename = image_link.split('/')[-1]
        return filename
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    def _parse_catalog_content(self, catalog_content: str) -> Dict[str, str]:
        """
        Parse catalog content to extract structured information.
        
        Returns:
            Dictionary with 'item_name', 'value', 'unit'
        """
        result = {
            'item_name': '',
            'value': '',
            'unit': ''
        }
        
        lines = catalog_content.strip().split('\n')
        
        for line in lines:
            if line.startswith('Item Name:'):
                result['item_name'] = line.replace('Item Name:', '').strip()
            elif line.startswith('Value:'):
                result['value'] = line.replace('Value:', '').strip()
            elif line.startswith('Unit:'):
                result['unit'] = line.replace('Unit:', '').strip()
        
        return result
    
    def _create_text_representation(self, catalog_content: str) -> str:
        """
        Create a clean text representation from catalog content.
        """
        parsed = self._parse_catalog_content(catalog_content)
        
        # Combine all fields into a single text
        text_parts = []
        
        if parsed['item_name']:
            text_parts.append(parsed['item_name'])
        
        if parsed['value'] and parsed['unit']:
            text_parts.append(f"{parsed['value']} {parsed['unit']}")
        
        return ' '.join(text_parts)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, Dict]:
        """
        Returns:
            image: Preprocessed image tensor
            text: Text representation
            price: Target price
            metadata: Dictionary with additional info
        """
        row = self.df.iloc[idx]
        
        # Load image
        image_filename = self._extract_image_filename(row['image_link'])
        image_path = self.images_dir / image_filename
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Use raw text; tokenization happens in the text encoder.
        text = str(row['catalog_content'])
        
        # Process price (target)
        price = float(row['price'])
        price = torch.tensor(price, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'sample_id': row['sample_id'],
            'image_link': row['image_link'],
            'original_price': float(row['price'])
        }
        
        return image, text, price, metadata
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        prices = self.df['price'].values
        
        return {
            'num_samples': len(self.df),
            'price_mean': float(np.mean(prices)),
            'price_std': float(np.std(prices)),
            'price_min': float(np.min(prices)),
            'price_max': float(np.max(prices)),
            'price_median': float(np.median(prices))
        }
