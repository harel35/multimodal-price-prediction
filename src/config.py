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

# Image encoder parameters
# ResNet
IMAGE_ENCODER = "resnet"  # Options: "resnet", "vit", "dinov3"
IMAGE_ENCODER_VARIANT = "resnet-50"
IMAGE_ENCODER_PRETRAINED = True
IMAGE_ENCODER_PRETRAINED_NAME = "microsoft/resnet-50"
IMAGE_EMBEDDING_DIM = None  # Set to override encoder output dim
IMAGE_ENCODER_FREEZE = True
IMAGE_DROPOUT_RATE = 0.1
# ViT
# IMAGE_ENCODER = "vit"
# IMAGE_ENCODER_VARIANT = "vit-base-patch16-224"
# IMAGE_ENCODER_PRETRAINED = True
# IMAGE_ENCODER_PRETRAINED_NAME = "google/vit-base-patch16-224"
# IMAGE_EMBEDDING_DIM = None
# IMAGE_ENCODER_FREEZE = True
# IMAGE_DROPOUT_RATE = 0.1
# DINOv3
# IMAGE_ENCODER = "dinov3"
# IMAGE_ENCODER_VARIANT = "dinov3-vits16-pretrain-lvd1689m"
# IMAGE_ENCODER_PRETRAINED = True
# IMAGE_ENCODER_PRETRAINED_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
# IMAGE_EMBEDDING_DIM = None
# IMAGE_ENCODER_FREEZE = True
# IMAGE_DROPOUT_RATE = 0.1

# Text parameters
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

TEXT_ENCODER = "fasttext"  # Options: "bert", "clip", "fasttext"
TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS[TEXT_ENCODER]
TEXT_EMBEDDING_DIM = 300
TEXT_ENCODER_VARIANT = "cc.en.300"
TEXT_ENCODER_FREEZE = True
TEXT_DROPOUT_RATE = 0.1
FASTTEXT_MODEL_PATH = "/home/projects/sipl-prj10268/DeepLearning/Project/ProjectDeepLearning/cc.en.300.bin"

# Quick-switch presets (uncomment a block and comment the active settings)
# BERT
# TEXT_ENCODER = "bert"
# TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS["bert"]
# TEXT_EMBEDDING_DIM = None
# TEXT_ENCODER_VARIANT = "bert-base-uncased"
# TEXT_ENCODER_FREEZE = True
# TEXT_DROPOUT_RATE = 0.1
# CLIP
# TEXT_ENCODER = "clip"
# TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS["clip"]
# TEXT_EMBEDDING_DIM = None
# TEXT_ENCODER_VARIANT = "clip-vit-base-patch32"
# TEXT_ENCODER_FREEZE = True
# TEXT_DROPOUT_RATE = 0.1
# FastText
# TEXT_ENCODER = "fasttext"
# TEXT_ENCODER_CONFIG = TEXT_ENCODER_CONFIGS["fasttext"]
# TEXT_EMBEDDING_DIM = 300
# TEXT_ENCODER_VARIANT = "cc.en.300"
# FASTTEXT_MODEL_PATH = "/home/projects/sipl-prj10268/DeepLearning/Project/ProjectDeepLearning/cc.en.300.bin"
# TEXT_ENCODER_FREEZE = True
# TEXT_DROPOUT_RATE = 0.1

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10

# Model parameters
DROPOUT_RATE = 0.2
FUSION_METHOD = "concat"  # Options: "concat", "attention", "addition"
FUSION_HIDDEN_DIMS = (2048, 1024, 512, 256)
FUSION_ACTIVATION = "relu"
FUSION_USE_BATCH_NORM = True
FUSION_DIM = None
OUTPUT_ACTIVATION = None

# Checkpoint parameters
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"

# Weights & Biases
WANDB_PROJECT_NAME = "deep-learning-project"
WANDB_ENTITY = "deep-learning-project-technion"
WANDB_RUN_NAME = None
WANDB_MODE = None

# Qwen evaluation (image-text-to-text)
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 40
QWEN_TEMPERATURE = 0.0
QWEN_TOP_P = 1.0
QWEN_PROMPT_TEMPLATE = (
    "Predict the product price as a single number using the image and description. "
    "Description: {text} "
    "Only output the number."
)

# Device
DEVICE = "cuda"  # cuda or cpu
