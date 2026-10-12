"""Paired evaluation shared by training validation and test.py."""

from pathlib import Path

import torch
from tqdm import tqdm

from data.common import save_image, transformData
from evaluation.evaluation_metric import compute_measure
from runtime import mean_metrics, predict


@torch.no_grad()
def evaluate(model, loader, device, output_dir=None, output_format="npy"):
    model.eval()
    transform = transformData()
    rows = []
    for index, (lq, hq, modalities, names) in enumerate(tqdm(loader, desc="Evaluate")):
        modality, name = modalities[0], names[0]
        restored = transform.postprocess(predict(model, lq.to(device)), modality).cpu()
        target = transform.postprocess(hq, modality)
        psnr, ssim, rmse = compute_measure(restored, target, target.max() - target.min())
        rows.append({"modality": modality, "name": name, "psnr": psnr, "ssim": ssim, "rmse": rmse})
        if output_dir:
            item = loader.dataset.items[index]
            save_image(restored, Path(output_dir) / modality / (name + "." + output_format), item["lq"])
    if not rows:
        raise ValueError("Evaluation contains no samples")
    summary = {"overall": mean_metrics(rows)}
    for modality in sorted({row["modality"] for row in rows}):
        summary[modality] = mean_metrics([row for row in rows if row["modality"] == modality])
    return rows, summary
