"""Dataset definition for multimodal price regression.

Expected CSV columns:
- ``image_link``: URL used to derive local image filename.
- ``catalog_content``: raw product text.
- ``price``: regression target in USD.
- ``sample_id``: optional identifier used in metadata.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MultimodalPriceDataset(Dataset):
    """
    PyTorch dataset returning image, raw text, and price target.

    Tokenization is intentionally deferred to ``build_collate_fn`` in
    ``src/data/dataloader.py`` so each text encoder can control how text is
    prepared (BERT/CLIP token IDs vs. FastText token lists).
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        mode: str = "train",
    ):
        """
        Args:
            csv_path: Path to the CSV file containing data
            images_dir: Directory containing images
            transform: Torchvision transforms for images
            mode: split label used for logging/debugging.
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.mode = mode

        self.df = pd.read_csv(csv_path)
        self._validate_columns()
        self.df = self._filter_valid_samples()
        self.df = self.df.reset_index(drop=True)

        print(f"Loaded {len(self.df)} samples in {mode} mode")

    def _validate_columns(self) -> None:
        """Validate expected CSV schema early with a clear error."""
        required_columns = {"image_link", "catalog_content", "price"}
        missing = required_columns.difference(set(self.df.columns))
        if missing:
            raise ValueError(
                f"CSV at '{self.csv_path}' is missing required columns: {sorted(missing)}"
            )

    def _filter_valid_samples(self) -> pd.DataFrame:
        """Keep rows with positive price and an existing local image file."""
        valid_indices = []

        for idx, row in self.df.iterrows():
            if pd.isna(row["price"]) or row["price"] <= 0:
                continue

            image_filename = self._extract_image_filename(row["image_link"])
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                continue

            valid_indices.append(idx)

        return self.df.loc[valid_indices]

    def _extract_image_filename(self, image_link: str) -> str:
        """
        Extract image filename from URL.

        Example: https://m.media-amazon.com/images/I/51mo8htwTHL.jpg -> 51mo8htwTHL.jpg
        """
        if not isinstance(image_link, str):
            return ""
        filename = image_link.split("/")[-1]
        return filename

    def _can_open_image(self, image_path: Path) -> bool:
        """Check whether an image can be opened and converted to RGB."""
        try:
            with Image.open(image_path) as image:
                image.convert("RGB")
            return True
        except Exception:
            return False

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Keep training resilient to occasional corrupted image files.
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def _parse_catalog_content(self, catalog_content: str) -> Dict[str, str]:
        """
        Parse catalog content to extract structured information.

        Returns:
            Dictionary with 'item_name', 'value', 'unit'
        """
        result = {
            "item_name": "",
            "value": "",
            "unit": "",
        }

        lines = catalog_content.strip().split("\n")

        for line in lines:
            if line.startswith("Item Name:"):
                result["item_name"] = line.replace("Item Name:", "").strip()
            elif line.startswith("Value:"):
                result["value"] = line.replace("Value:", "").strip()
            elif line.startswith("Unit:"):
                result["unit"] = line.replace("Unit:", "").strip()

        return result

    def _create_text_representation(self, catalog_content: str) -> str:
        """
        Create a compact text string from catalog fields.

        Note: main training pipeline uses raw catalog text directly. This helper
        remains for backwards compatibility with older experiments.
        """
        parsed = self._parse_catalog_content(catalog_content)

        text_parts = []

        if parsed["item_name"]:
            text_parts.append(parsed["item_name"])

        if parsed["value"] and parsed["unit"]:
            text_parts.append(f"{parsed['value']} {parsed['unit']}")

        return " ".join(text_parts)

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

        image_filename = self._extract_image_filename(row["image_link"])
        image_path = self.images_dir / image_filename
        image = self._load_image(image_path)

        if self.transform:
            image = self.transform(image)

        # Tokenization is handled in the collate function.
        text = str(row["catalog_content"])

        price = float(row["price"])
        price = torch.tensor(price, dtype=torch.float32)

        metadata = {
            "sample_id": row.get("sample_id", idx),
            "image_link": row["image_link"],
            "original_price": float(row["price"]),
        }

        return image, text, price, metadata

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        prices = self.df["price"].values

        return {
            "num_samples": len(self.df),
            "price_mean": float(np.mean(prices)),
            "price_std": float(np.std(prices)),
            "price_min": float(np.min(prices)),
            "price_max": float(np.max(prices)),
            "price_median": float(np.median(prices)),
        }
