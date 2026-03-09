"""
Configuration file for the deep learning project.
Contains all hyperparameters, paths, and settings.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" 
IMAGES_DIR = DATA_ROOT / "images"
CSV_PATH = DATA_ROOT / "data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Create directoriesd if they don't exist
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

# Image encoder parameters
IMAGE_ENCODER = "dinov3"  # Options: "resnet", "vit", "dinov3"
IMAGE_ENCODER_VARIANT = None  # Uses encoder default if None
IMAGE_ENCODER_PRETRAINED = True
IMAGE_EMBEDDING_DIM = 768  # Set to override encoder output dim
IMAGE_ENCODER_FREEZE = False
IMAGE_DROPOUT_RATE = 0.1

# Text parameters

TEXT_ENCODER = "clip"  # Options: "bert", "clip", "fasttext"
TEXT_ENCODER_CONFIGS = { # tokenizer and max length for each encoder
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
TEXT_EMBEDDING_DIM = 768  # Set to override encoder output dim  
TEXT_ENCODER_VARIANT = "openai/clip-vit-base-patch32"  # Uses encoder default if None
TEXT_ENCODER_PRETRAINED = True
TEXT_ENCODER_FREEZE = False
TEXT_DROPOUT_RATE = 0.1

# FastText parameters
FASTTEXT_VARIANT = "cc.en.300"
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "cc.en.300.bin"

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10

# Model parameters
# When experimenting with fusion architectures we provide three predefined MLP
# variants. You can choose one using FUSION_MLP_TYPE.  The settings for each
# variant are hard-coded in _resolve_fusion_head; if you want full control you
# can still override individual parameters like FUSION_HIDDEN_DIMS,
# FUSION_ACTIVATION, etc., but the variant will take precedence.
#   * mlp1: baseline network [512,256], ReLU, dropout=0.3, no residuals, batch norm.
#   * mlp2: identical to mlp1; only change is three hidden layers [512,512,256].
#   * mlp3: same width as mlp2 [512,512,256], GELU, residuals, layer norm,
#           and higher dropout.
FUSION_MLP_TYPE = "mlp3"  # Options: "mlp1", "mlp2", "mlp3"; set to None for
                           # backward-compatible manual configuration.

DROPOUT_RATE = 0.2
FUSION_METHOD = "concat"  # Options: "concat", "attention", "addition", "gated"
FUSION_HIDDEN_DIMS = (512, 512, 256)
FUSION_ACTIVATION = "relu"
FUSION_USE_BATCH_NORM = True
FUSION_USE_LAYER_NORM = False
FUSION_USE_RESIDUAL = False
FUSION_DIM = None
OUTPUT_ACTIVATION = "softplus"
SMAPE_EPS = 1e-4
GRAD_CLIP_NORM = 1.0
LR_SCHEDULER = "plateau"  # Options: "plateau", "cosine", None
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LR = 1e-6

# Checkpoint parameters
CHECKPOINT_PATH = CHECKPOINT_DIR / "dino_fasttext_unfreeze_concat_mlp1.pt"

# Weights & Biases
WANDB_PROJECT_NAME = "deep-learning-project"
WANDB_ENTITY = "deep-learning-project-technion"
WANDB_RUN_NAME = None
WANDB_MODE = None

# Device configuration
DEVICE = "cuda"  # cuda or cpu
