# Multimodal Price Prediction from Product Images and Text

Predicting a continuous USD product price from two modalities:
- product image
- catalog text description

This repository contains the training/evaluation code used to compare multimodal design choices for the Amazon ML Challenge 2025-style data format.

## Quickstart (Minimal)

```bash
# 1) Create environment
conda env create -f environment.yml
conda activate multimodal-price-prediction

# 2) Prepare assets (images + FastText model)
jupyter lab setup_assets.ipynb

# 3) Train
python main.py --train

# 4) Evaluate
python main.py --evaluate
```

Shell wrappers are also available:

```bash
bash scripts/train.sh
bash scripts/evaluate.sh
bash scripts/qwen_eval.sh --split val --max-samples 200
```

## Motivation

E-commerce price prediction is a naturally multimodal regression problem. Product photos encode visual quality/brand/category cues, while catalog text captures product attributes, quantity, and marketing context.
This project studies how encoder choices and fusion strategies affect price prediction quality.

## Task Definition

- **Input**: `(image, catalog_content)`
- **Target**: scalar product `price` (USD)
- **Task type**: supervised regression
- **Primary metric**: SMAPE
- **Secondary metrics**: MAE, RMSE

## Repository Highlights

- Modular image encoders: **ResNet**, **ViT**, **DINOv3** (Hugging Face).
- Modular text encoders: **BERT**, **CLIP**, **FastText**.
- Configurable fusion head: concatenation/addition/attention/gated + MLP variants.
- Optional Qwen VLM baseline (`qwen_eval.py`) for comparison.
- W&B integrated for experiment tracking.
- Reproducible split control and fixed random seed support.

## Main Features / Capabilities

- Consistent train/val/test split creation from one CSV.
- Per-encoder text tokenization in collate stage.
- Checkpoint save/load utilities.
- Config-driven experiments from one central file (`src/config.py`).
- Optional notebook-based setup for external assets.

## Repository Organization

```text
ProjectDeepLearning/
├── README.md
├── environment.yml
├── main.py                        # Main train/evaluate CLI entry point
├── qwen_eval.py                   # Qwen VLM baseline evaluator
├── setup_assets.ipynb             # Notebook to download images + FastText model
├── scripts/
│   ├── train.sh                   # Wrapper for python main.py --train
│   ├── evaluate.sh                # Wrapper for python main.py --evaluate
│   └── qwen_eval.sh               # Wrapper for python qwen_eval.py
├── data/
│   ├── data.csv                   # Main dataset CSV
│   ├── images/                    # Product images (downloaded locally)
│   ├── utils.py                   # Image download helpers
│   └── example.ipynb              # Exploratory notebook
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central experiment configuration
│   ├── data/
│   │   ├── dataset.py             # Dataset class (returns raw text)
│   │   ├── dataloader.py          # Split/tokenization/dataloader logic
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_encoders/
│   │   │   ├── resnet_encoder.py
│   │   │   ├── vit_encoder.py
│   │   │   ├── dinov3_encoder.py
│   │   │   └── __init__.py
│   │   ├── text_encoders/
│   │   │   ├── bert_text_encoding.py
│   │   │   ├── clip_text_encoder.py
│   │   │   ├── fast_text.py
│   │   │   └── __init__.py
│   │   └── fusion/
│   │       ├── mlp.py
│   │       └── __init__.py
│   └── utils/
│       ├── helpers.py
│       ├── metrics.py
│       ├── preprocessing.py
│       └── __init__.py
├── models/
│   └── checkpoints/               # Saved model checkpoints (git-ignored)
└── outputs/                       # Prediction CSVs and outputs (git-ignored)
```

## Setup Instructions

### Python Version

- Recommended: **Python 3.10**

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate multimodal-price-prediction
```

`environment.yml` is the single dependency source of truth for this repository.

### System / Hardware Notes

- CPU training/evaluation is supported but slow.
- GPU is strongly recommended for transformer-based encoders.
- Qwen baseline requires significant VRAM for comfortable inference.

## Data Instructions

### Dataset Source

- Dataset format follows the Amazon 2025 challenge style: image + catalog text + price.
- Main CSV expected at: `data/data.csv`.

### Required Local Layout

```text
data/
├── data.csv
└── images/
    ├── <image_name_1>.jpg
    ├── <image_name_2>.jpg
    └── ...
```

Image filename is derived from `image_link` in the CSV (last URL path component).

### Preparation Steps

1. Ensure `data/data.csv` exists.
2. Download images into `data/images/`.
3. Download FastText binary (`cc.en.300.bin`) if using FastText encoder.
4. You can use `setup_assets.ipynb` for steps 2-3.

### What Is Not Bundled in Git

- Downloaded dataset images (`data/images/`)
- Trained checkpoints (`models/checkpoints/*.pt`)
- W&B logs and output predictions (`wandb/`, `outputs/*`)
- Large FastText binaries (`cc.en.300.bin`, `cc.en.300.bin.gz`)

## How to Run

### Train

```bash
python main.py --train
```

### Evaluate (Loads checkpoint from config if available)

```bash
python main.py --evaluate
```

### Train + Evaluate in One Session

```bash
python main.py --train --evaluate
```

### Qwen Baseline Evaluation

```bash
python qwen_eval.py --split val --max-samples 500
```

Useful Qwen flags:

```bash
python qwen_eval.py \
  --split test \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --max-new-tokens 40 \
  --temperature 0.0 \
  --save-preds outputs/qwen_test_preds.csv
```

## Configuration Guide (`src/config.py`)

All experiment knobs live in one file.

### Core Paths

- `DATA_ROOT`, `CSV_PATH`, `IMAGES_DIR`
- `MODELS_DIR`, `CHECKPOINT_DIR`, `CHECKPOINT_PATH`

### Data / Reproducibility

- `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT`
- `RANDOM_SEED`

### Image Encoder

- `IMAGE_ENCODER`: `resnet` | `vit` | `dinov3`
- `IMAGE_ENCODER_VARIANT`
- `IMAGE_ENCODER_PRETRAINED`
- `IMAGE_EMBEDDING_DIM`
- `IMAGE_ENCODER_FREEZE`
- `IMAGE_DROPOUT_RATE`

### Text Encoder

- `TEXT_ENCODER`: `bert` | `clip` | `fasttext`
- `TEXT_ENCODER_CONFIGS`
- `TEXT_ENCODER_CONFIG`
- `TEXT_ENCODER_VARIANT`
- `TEXT_ENCODER_PRETRAINED`
- `TEXT_ENCODER_FREEZE`
- `TEXT_EMBEDDING_DIM`

### Fusion / Regression Head

- `FUSION_METHOD`: `concat` | `addition` | `attention` | `gated`
- `FUSION_MLP_TYPE`: `mlp1` | `mlp2` | `mlp3` | `None`
- `FUSION_HIDDEN_DIMS`, `FUSION_ACTIVATION`
- `FUSION_USE_BATCH_NORM`, `FUSION_USE_LAYER_NORM`, `FUSION_USE_RESIDUAL`
- `FUSION_DIM` (required for non-concat fusion when dims differ)
- `OUTPUT_ACTIVATION`

### Optimization

- `LEARNING_RATE`, `WEIGHT_DECAY`, `NUM_EPOCHS`
- `EARLY_STOPPING_PATIENCE`
- `LR_SCHEDULER`, `LR_FACTOR`, `LR_PATIENCE`, `MIN_LR`
- `GRAD_CLIP_NORM`
- `SMAPE_EPS`

### W&B

- `WANDB_PROJECT_NAME`, `WANDB_ENTITY`, `WANDB_RUN_NAME`, `WANDB_MODE`

### Qwen Baseline Defaults

- `QWEN_MODEL_NAME`
- `QWEN_PROMPT_TEMPLATE`
- `QWEN_MAX_NEW_TOKENS`, `QWEN_TEMPERATURE`, `QWEN_TOP_P`

## Model Options Summary

### Image Encoders

- ResNet (`src/models/image_encoders/resnet_encoder.py`)
- ViT (`src/models/image_encoders/vit_encoder.py`)
- DINOv3 (`src/models/image_encoders/dinov3_encoder.py`)

### Text Encoders

- BERT (`src/models/text_encoders/bert_text_encoding.py`)
- CLIP text (`src/models/text_encoders/clip_text_encoder.py`)
- FastText (`src/models/text_encoders/fast_text.py`)

### Fusion and Head

- Fusion implemented in `src/models/fusion/mlp.py`
- Supports concat/addition/attention/gated fusion
- MLP output head regresses to a single scalar price

## Metrics

### Primary Metric: SMAPE

Used both as training loss and the primary leaderboard-style metric.

\[
\text{SMAPE} = \frac{1}{N}\sum_i \frac{|p_i - y_i|}{(|p_i| + |y_i|)/2}
\]

### Secondary Metrics

- **MAE**: mean absolute error (USD)
- **RMSE**: root mean squared error (USD)

## Reproducibility Notes

- Seed control via `RANDOM_SEED` and `set_seed(...)`.
- Splits are deterministic through `split_dataset_indices(...)`.
- Checkpoint path is explicit (`CHECKPOINT_PATH`).
- Log runs to W&B for config and metric traceability.
- Reproducibility still depends on hardware/CUDA/cuDNN settings.

## Output Artifacts

- **Checkpoints**: `models/checkpoints/`
- **Prediction CSVs**: `outputs/`
- **W&B logs**: `wandb/`

`main.py` saves the best validation checkpoint.
`qwen_eval.py` can optionally dump per-sample outputs with `--save-preds`.

## Troubleshooting

### 1) Missing images / tiny dataset after filtering

- Ensure `data/images/` contains files named exactly as URL basenames from `image_link`.
- Dataset class filters rows with missing images or non-positive prices.

### 2) FastText file not found

- Verify `FASTTEXT_MODEL_PATH` in `src/config.py`.
- Confirm `cc.en.300.bin` exists locally.

### 3) CUDA out-of-memory

- Lower `BATCH_SIZE`.
- Freeze encoders (`IMAGE_ENCODER_FREEZE`, `TEXT_ENCODER_FREEZE`) if needed.
- Prefer smaller variants (`vit-base`, `bert-base`, etc.).

### 4) Hugging Face model download/network issues

- Retry after network stabilization.
- Pre-download/cache models in your environment when possible.

### 5) Tokenizer/encoder mismatch

- Keep `TEXT_ENCODER` aligned with `TEXT_ENCODER_CONFIG`.
- For FastText, collate returns token lists; for BERT/CLIP, it returns tensor dicts.

### 6) No checkpoint loaded during evaluation

- Ensure `CHECKPOINT_PATH` points to an existing `.pt` file.
- Run training first or set a valid checkpoint path manually.

## Development / Maintainer Notes

- This is **research-oriented student code** with polished entry points.
- Priority is experiment clarity and reproducibility, not production serving.
- Main extension points:
  - add new encoder module under `src/models/...`
  - extend config with new options in `src/config.py`
  - keep train/eval loop interfaces stable in `main.py`
- Keep output/data/model binaries out of git (already covered by `.gitignore`).

## Research Code vs. Polished Scripts

- **Polished / reusable paths**:
  - `main.py`, `qwen_eval.py`, `src/`, `scripts/`
- **Exploratory assets**:
  - notebooks (`setup_assets.ipynb`, `data/example.ipynb`)

Use scripts/CLI for reproducible runs; use notebooks for setup and quick analysis.

## Connection to Report / Experiments

This repository is structured to support reporting across encoder/fusion ablations:

- image encoder variants (ResNet / ViT / DINOv3)
- text encoder variants (BERT / CLIP / FastText)
- fusion strategy variants (concat / addition / attention / gated)
- baseline comparison with Qwen VLM (`qwen_eval.py`)

When preparing your report, record:
- exact config values from `src/config.py`
- checkpoint used
- split and seed
- W&B run URL/ID
