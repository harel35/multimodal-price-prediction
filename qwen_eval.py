"""
Evaluate Qwen/Qwen2.5-VL-7B-Instruct on the dataset and log metrics to W&B.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm
import wandb
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys

sys.path.append(str(Path(__file__).parent))

from src import config
from src.data.dataset import MultimodalPriceDataset
from src.data.dataloader import split_dataset_indices
from src.utils.helpers import set_seed


_PRICE_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available() and getattr(config, "DEVICE", "cuda") == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_price(text: str) -> Optional[float]:
    cleaned = text.replace(",", "")
    match = _PRICE_PATTERN.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _build_prompt(template: str, raw_text: str) -> str:
    return template.replace("{text}", raw_text)


def _prepare_inputs(processor, messages):
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        process_vision_info = None

    if process_vision_info is None:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )


def _load_model(model_name: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    return processor, model


def _init_wandb(config_payload: Dict, run_name: Optional[str]):
    wandb_kwargs = {
        "project": getattr(config, "WANDB_PROJECT_NAME", "deep-learning-project"),
        "config": config_payload,
    }
    wandb_entity = getattr(config, "WANDB_ENTITY", None)
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if run_name:
        wandb_kwargs["name"] = run_name
    wandb_mode = getattr(config, "WANDB_MODE", None)
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    wandb.init(**wandb_kwargs)


def evaluate_qwen(
    csv_path: str,
    images_dir: str,
    split: str,
    model_name: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: Optional[str] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
):
    set_seed(seed)
    torch_device = _resolve_device(device)

    dataset = MultimodalPriceDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=None,
        mode=split,
    )
    train_idx, val_idx, test_idx = split_dataset_indices(
        dataset_size=len(dataset),
        train_split=getattr(config, "TRAIN_SPLIT", 0.8),
        val_split=getattr(config, "VAL_SPLIT", 0.1),
        test_split=getattr(config, "TEST_SPLIT", 0.1),
        random_seed=seed,
    )
    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    if split not in split_indices:
        raise ValueError(f"Unsupported split '{split}'. Choose from train/val/test.")
    indices = split_indices[split]
    if max_samples is not None:
        indices = indices[:max_samples]

    processor, model = _load_model(model_name, torch_device)

    smape_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0
    invalid = 0

    for idx in tqdm(indices, desc=split, leave=False):
        image, raw_text, price, _ = dataset[int(idx)]
        prompt = _build_prompt(prompt_template, raw_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = _prepare_inputs(processor, messages).to(model.device)

        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        response = processor.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True,
        ).strip()
        pred = _extract_price(response)
        if pred is None:
            invalid += 1
            continue

        target = float(price.item())
        diff = pred - target
        mae_sum += abs(diff)
        mse_sum += diff * diff
        denom = (abs(pred) + abs(target)) / 2.0
        smape_sum += abs(diff) / max(denom, 1e-8)
        count += 1

    if count == 0:
        raise RuntimeError("No valid predictions parsed; metrics cannot be computed.")

    mse = mse_sum / count
    rmse = mse ** 0.5
    metrics = {
        "smape": smape_sum / count,
        "mae": mae_sum / count,
        "mse": mse,
        "rmse": rmse,
        "num_samples": len(indices),
        "num_valid": count,
        "num_invalid": invalid,
    }

    print(f"\n{split.capitalize()} metrics:")
    print(f"  SMAPE: {metrics['smape']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    if invalid:
        print(f"  Invalid predictions: {invalid}")

    run_name = getattr(config, "WANDB_RUN_NAME", None) or f"qwen-eval-{split}"
    wandb_payload = {
        "model_name": model_name,
        "split": split,
        "prompt_template": prompt_template,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "device": str(torch_device),
        "max_samples": max_samples,
        "csv_path": str(csv_path),
        "images_dir": str(images_dir),
        "seed": seed,
    }
    if wandb.run is None:
        _init_wandb(wandb_payload, run_name)

    wandb.log({
        f"{split}/loss": metrics["smape"],
        f"{split}/smape": metrics["smape"],
        f"{split}/mae": metrics["mae"],
        f"{split}/mse": metrics["mse"],
        f"{split}/rmse": metrics["rmse"],
        f"{split}/num_samples": metrics["num_samples"],
        f"{split}/num_valid": metrics["num_valid"],
        f"{split}/num_invalid": metrics["num_invalid"],
    })

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/Qwen2.5-VL-7B-Instruct on the dataset."
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--csv-path", default=str(config.CSV_PATH))
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    parser.add_argument("--model-name", default=getattr(config, "QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--prompt-template", default=getattr(config, "QWEN_PROMPT_TEMPLATE", "{text}"))
    parser.add_argument("--max-new-tokens", type=int, default=getattr(config, "QWEN_MAX_NEW_TOKENS", 40))
    parser.add_argument("--temperature", type=float, default=getattr(config, "QWEN_TEMPERATURE", 0.0))
    parser.add_argument("--top-p", type=float, default=getattr(config, "QWEN_TOP_P", 1.0))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=getattr(config, "RANDOM_SEED", 42))

    args = parser.parse_args()

    evaluate_qwen(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split=args.split,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
