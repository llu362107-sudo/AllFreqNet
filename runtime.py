"""Shared command-line, checkpoint and inference helpers."""

import csv
import inspect
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.common import MODALITIES
from model import AllFreqNet


def add_data_arguments(parser):
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--raw-shape", nargs=2, type=int, default=(512, 512), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--num-workers", type=int, default=0, help="0 is portable on Windows")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    return device


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(path, training=False):
    if not Path(path).is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # PyTorch 1.x's restricted unpickler cannot read optimizer float metadata.
        legacy_training = training and int(torch.__version__.split(".")[0]) < 2
        kwargs["weights_only"] = not legacy_training
        if legacy_training:
            warnings.warn("PyTorch 1.x resumes training via pickle. Resume only your own trusted training_state.pt files.")
    else:
        warnings.warn("This PyTorch version lacks weights_only loading. Load only trusted checkpoints.")
    return torch.load(str(path), **kwargs)


def extract_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Expected a state_dict or a checkpoint containing model/state_dict")
    return {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint.items()}


def load_generator(path, device):
    model = AllFreqNet().to(device)
    model.load_state_dict(extract_state(load_checkpoint(path)), strict=True)
    return model.eval()


def predict(model, image):
    """Pad the right/bottom to a multiple of eight, then crop to the input size."""
    height, width = image.shape[-2:]
    pad_h = max(16, (height + 7) // 8 * 8) - height
    pad_w = max(16, (width + 7) // 8 * 8) - width
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return model(image)[..., :height, :width]


def mean_metrics(rows):
    result = {"count": len(rows)}
    for key in ("psnr", "ssim", "rmse"):
        values = [row[key] for row in rows if not math.isnan(row[key])]
        result[key] = float(np.mean(values)) if values else float("nan")
        result[key + "_valid_count"] = len(values)
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["modality", "name", "psnr", "ssim", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    def sanitize(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(k): sanitize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(x) for x in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
