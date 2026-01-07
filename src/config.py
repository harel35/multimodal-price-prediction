"""
Configuration file for the deep learning project.
Contains all hyperparameters, paths, and settings.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "student_resource"
IMAGES_DIR = DATA_ROOT / "images"
CSV_PATH = DATA_ROOT / "dataset" / "data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Data parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Image parameters
IMAGE_SIZE = (224, 224)  # Standard size for pretrained models
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Text parameters
TEXT_ENCODER = "bert"  # Options: "bert", "clip", "fasttext"
TEXT_ENCODER_CONFIGS = {
    "bert": {
        "tokenizer_name": "bert-base-uncased",
        "max_length": 512
    },
    "clip": {
        "tokenizer_name": "openai/clip-vit-base-patch32",
        "max_length": 77
    },
    "fasttext": {
        "tokenizer_name": None,
        "max_length": None,
        "lowercase": True
    }
}
TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS[TEXT_ENCODER]
TEXT_EMBEDDING_DIM = 768  # BERT base dimension

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Model parameters
DROPOUT_RATE = 0.3
FUSION_METHOD = "concat"  # Options: "concat", "attention", "addition"

# Device
DEVICE = "cuda"  # Will be set to "cpu" if CUDA is not available
