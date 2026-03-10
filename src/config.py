"""Central configuration for multimodal price prediction experiments.

This module intentionally keeps all project-level knobs in one place so
experiments can be reproduced by editing a single file.
"""

from pathlib import Path
from typing import Any, Dict


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_ROOT / "images"
CSV_PATH = DATA_ROOT / "data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data split configuration
# -----------------------------------------------------------------------------
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Image encoder configuration
# -----------------------------------------------------------------------------
IMAGE_ENCODER = "dinov3"  # "resnet" | "vit" | "dinov3"
IMAGE_ENCODER_VARIANT = None  # Defaults are resolved in main.py
IMAGE_ENCODER_PRETRAINED = True
IMAGE_EMBEDDING_DIM = 768  # Set to None to keep backbone native dimension
IMAGE_ENCODER_FREEZE = False
IMAGE_DROPOUT_RATE = 0.1


# -----------------------------------------------------------------------------
# Text encoder configuration
# -----------------------------------------------------------------------------
TEXT_ENCODER = "clip"  # "bert" | "clip" | "fasttext"
TEXT_ENCODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bert": {
        "tokenizer_name": "bert-base-uncased",
        "max_length": 512,
    },
    "clip": {
        "tokenizer_name": "openai/clip-vit-base-patch32",
        "max_length": 77,
    },
    "fasttext": {
        "tokenizer_name": None,
        "max_length": None,
        "lowercase": True,
    },
}
if TEXT_ENCODER not in TEXT_ENCODER_CONFIGS:
    raise ValueError(
        f"Unsupported TEXT_ENCODER '{TEXT_ENCODER}'. "
        f"Expected one of {sorted(TEXT_ENCODER_CONFIGS.keys())}."
    )
TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS[TEXT_ENCODER]

TEXT_EMBEDDING_DIM = 768  # Set to None to keep backbone native dimension
TEXT_ENCODER_VARIANT = "openai/clip-vit-base-patch32"
TEXT_ENCODER_PRETRAINED = True
TEXT_ENCODER_FREEZE = False
TEXT_DROPOUT_RATE = 0.1


# -----------------------------------------------------------------------------
# FastText assets
# -----------------------------------------------------------------------------
FASTTEXT_VARIANT = "cc.en.300"
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "cc.en.300.bin"


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP_NORM = 1.0

LR_SCHEDULER = "plateau"  # "plateau" | "cosine" | None
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LR = 1e-6


# -----------------------------------------------------------------------------
# Fusion head configuration
# -----------------------------------------------------------------------------
# Preset variants are resolved in main.py:
# - mlp1: hidden=(512, 256), ReLU, dropout=0.2
# - mlp2: hidden=(512, 512, 256), ReLU, dropout=0.2
# - mlp3: hidden=(512, 512, 256), GELU, residual, layer norm, dropout=0.3
FUSION_MLP_TYPE = "mlp3"  # Set to None to use manual settings below

DROPOUT_RATE = 0.2
FUSION_METHOD = "concat"  # "concat" | "addition" | "attention" | "gated"
FUSION_HIDDEN_DIMS = (512, 512, 256)
FUSION_ACTIVATION = "relu"
FUSION_USE_BATCH_NORM = True
FUSION_USE_LAYER_NORM = False
FUSION_USE_RESIDUAL = False
FUSION_DIM = None  # Required for addition/attention/gated if dims differ
OUTPUT_ACTIVATION = "softplus"


# -----------------------------------------------------------------------------
# Optimization target and checkpointing
# -----------------------------------------------------------------------------
SMAPE_EPS = 1e-4
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"


# -----------------------------------------------------------------------------
# W&B tracking
# -----------------------------------------------------------------------------
WANDB_PROJECT_NAME = "deep-learning-project"
WANDB_ENTITY = "deep-learning-project-technion"
WANDB_RUN_NAME = None
WANDB_MODE = None  # e.g., "offline"


# -----------------------------------------------------------------------------
# Qwen baseline defaults
# -----------------------------------------------------------------------------
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_PROMPT_TEMPLATE = (
    "Estimate the product price in USD from the image and catalog text below.\n"
    "Respond with only a single numeric value.\n\n"
    "{text}"
)
QWEN_MAX_NEW_TOKENS = 40
QWEN_TEMPERATURE = 0.0
QWEN_TOP_P = 1.0


# -----------------------------------------------------------------------------
# Runtime device
# -----------------------------------------------------------------------------
DEVICE = "cuda"  # "cuda" or "cpu"
