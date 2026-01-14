# ProjectDeepLearning

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Shueyr/ProjectDeepLearning.git
cd ProjectDeepLearning
```

Use the provided conda environment to install all dependencies:

```bash
conda env create -f environment.yml
conda activate dl_project
```

### Dataset And Assets Setup

Use the notebook to download the dataset images and the FastText model:

- `setup_assets.ipynb`

The image download step can take a few tries depending on network stability and source availability, so rerun the relevant cells if needed.

## Configuration

Edit `ProjectDeepLearning/src/config.py` to set encoder choices, paths, and training hyperparameters. Key settings include:

- `IMAGE_ENCODER` and `IMAGE_ENCODER_VARIANT`
- `TEXT_ENCODER` and `TEXT_ENCODER_VARIANT`
- `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- `FUSION_METHOD` and `FUSION_HIDDEN_DIMS`

## Usage

Train the model:

```bash
python ProjectDeepLearning/main.py --train
```

Evaluate the model:

```bash
python ProjectDeepLearning/main.py --evaluate
```

## Configuration Reference

All configuration lives in `ProjectDeepLearning/src/config.py`. Use the tables below to adjust behavior.

### Paths

| Parameter | Description |
| --- | --- |
| `PROJECT_ROOT` | Project root directory (derived from the config file location). |
| `DATA_ROOT` | Base directory for dataset files. |
| `IMAGES_DIR` | Directory containing image files. |
| `CSV_PATH` | data CSV file with metadata and labels. |
| `MODELS_DIR` | Directory for model artifacts (created if missing). |
| `CHECKPOINT_DIR` | Directory for checkpoints (created if missing). |

### Data Splits

| Parameter | Description |
| --- | --- |
| `TRAIN_SPLIT` | Fraction of data for training. (e.g 0.8)|
| `VAL_SPLIT` | Fraction of data for validation. (e.g 0.1)|
| `TEST_SPLIT` | Fraction of data for testing. (e.g 0.1)|
| `RANDOM_SEED` | Seed for deterministic splits and training. |

### Image Preprocessing

| Parameter | Description |
| --- | --- |
| `IMAGE_SIZE` | Target image size used in transforms. |
| `IMAGE_MEAN` | Normalization mean (per channel). |
| `IMAGE_STD` | Normalization standard deviation (per channel). |

### Image Encoder

| Parameter | Description |
| --- | --- |
| `IMAGE_ENCODER` | Image encoder type (`resnet`, `vit`, `dinov3`). |
| `IMAGE_ENCODER_VARIANT` | Model variant name |
| `IMAGE_ENCODER_PRETRAINED` | Load pretrained weights if `True`. |
| `IMAGE_EMBEDDING_DIM` | Override image embedding dim (projection). |
| `IMAGE_ENCODER_FREEZE` | Freeze image backbone if `True`. |
| `IMAGE_DROPOUT_RATE` | Dropout used in the image encoder. |

### Text Encoder

| Parameter | Description |
| --- | --- |
| `TEXT_ENCODER` | Text encoder type (`bert`, `clip`, `fasttext`). |
| `TEXT_ENCODER_CONFIGS` | Tokenizer settings per encoder. |
| `TEXT_ENCODER_CONFIG` | Active tokenizer config selected from `TEXT_ENCODER_CONFIGS`. |
| `TEXT_EMBEDDING_DIM` | Override text embedding dim (projection). |
| `TEXT_ENCODER_VARIANT` | Model variant name for BERT/CLIP. |
| `TEXT_ENCODER_FREEZE` | Freeze text backbone if `True`. |
| `TEXT_DROPOUT_RATE` | Dropout used in the text encoder. |

### FastText

| Parameter | Description |
| --- | --- |
| `FASTTEXT_VARIANT` | FastText variant name or path string. |
| `FASTTEXT_MODEL_PATH` | Path to a `.bin` or `.ftz` FastText model. |

### Training

| Parameter | Description |
| --- | --- |
| `BATCH_SIZE` | Batch size for training/evaluation. |
| `NUM_WORKERS` | DataLoader worker processes. |
| `LEARNING_RATE` | Optimizer learning rate. |
| `WEIGHT_DECAY` | Optimizer weight decay. |
| `NUM_EPOCHS` | Training epochs. |
| `EARLY_STOPPING_PATIENCE` | Patience for early stopping (if enabled). |

### Fusion Head

| Parameter | Description |
| --- | --- |
| `DROPOUT_RATE` | Default dropout for fusion layers. |
| `FUSION_METHOD` | Fusion strategy (`concat`, `attention`, `addition`). |
| `FUSION_HIDDEN_DIMS` | Hidden layer sizes for the fusion MLP. |
| `FUSION_ACTIVATION` | Activation function used in fusion. |
| `FUSION_USE_BATCH_NORM` | Enable batch norm in fusion layers. |
| `FUSION_DIM` | Shared dim for `addition`/`attention` when encoder dims differ. |
| `OUTPUT_ACTIVATION` | Optional activation on the final output. |

### Checkpointing

| Parameter | Description |
| --- | --- |
| `CHECKPOINT_PATH` | Path to the best model checkpoint. |

### Weights And Biases

| Parameter | Description |
| --- | --- |
| `WANDB_PROJECT_NAME` | W&B project name. |
| `WANDB_ENTITY` | W&B entity/team name. |
| `WANDB_RUN_NAME` | Optional run name. |
| `WANDB_MODE` | W&B mode (e.g., `offline`). |

### Device

| Parameter | Description |
| --- | --- |
| `DEVICE` | Compute device (`cuda` or `cpu`). |
