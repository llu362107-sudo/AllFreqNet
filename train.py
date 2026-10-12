"""Train AllFreqNet jointly on the selected restoration tasks."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.MedicalDataUniform import DataSampler, Test_Data, Train_Data
from evaluation.runner import evaluate
from model import AllFreqNet
from runtime import (add_data_arguments, extract_state, load_checkpoint, resolve_device,
                     seed_worker, set_seeds, write_csv, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/AllFreqNet"))
    parser.add_argument("--iterations", type=int, default=200000)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--classification-weight", type=float, default=0.001)
    parser.add_argument("--moe-weight", type=float, default=0.001)
    parser.add_argument("--val-limit", type=int, default=None, help="Optional positive limit per modality")
    parser.add_argument("--seed", type=int, default=42)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--weights", type=Path, help="Initialize model weights only")
    group.add_argument("--resume", type=Path, help="Continue a training_state.pt checkpoint")
    args = parser.parse_args()
    if min(args.iterations, args.val_interval, args.batch_size) <= 0 or args.num_workers < 0:
        parser.error("Iterations, validation interval and batch size must be positive; workers must be nonnegative")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        parser.error("Require 0 <= min-lr <= lr and lr > 0")
    if args.moe_weight < 0 or args.classification_weight < 0:
        parser.error("Loss weights must be nonnegative")
    return args


def save_weights(model, path):
    # Keep the model on its current device and preserve the legacy state_dict format.
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(path))


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    training = Train_Data(args.data_root, args.modalities, args.patch_size, args.raw_shape)
    validation = Test_Data(args.data_root, args.modalities, use_num=args.val_limit, raw_shape=args.raw_shape)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=args.num_workers)
    sampler = DataSampler(train_loader)
    model = AllFreqNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.min_lr)
    start, best = 0, -float("inf")
    if args.weights:
        model.load_state_dict(extract_state(load_checkpoint(args.weights)), strict=True)
    if args.resume:
        checkpoint = load_checkpoint(args.resume, training=True)
        if not all(key in checkpoint for key in ("model", "optimizer", "scheduler", "iteration", "config")):
            raise ValueError("--resume needs training_state.pt; use --weights for a plain state_dict")
        # Changing these fields would make the resumed schedule or objective inconsistent.
        for key in ("iterations", "lr", "min_lr", "modalities", "batch_size", "patch_size",
                    "classification_weight", "moe_weight"):
            if checkpoint["config"].get(key) != vars(args)[key]:
                raise ValueError("Resume configuration differs for {}".format(key))
        model.load_state_dict(extract_state(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["iteration"], checkpoint["best_psnr"]
    if start >= args.iterations:
        raise ValueError("Checkpoint has already reached --iterations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    write_json(args.output_dir / "config.json", vars(args))
    print("Device: {}; training slices: {}; validation slices: {}".format(device, len(training), len(validation)))
    l1, ce = nn.L1Loss(), nn.CrossEntropyLoss()
    progress = tqdm(range(start + 1, args.iterations + 1), desc="Train")
    for iteration in progress:
        model.train()
        lq, target, labels = next(sampler)
        lq, target, labels = lq.to(device), target.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        restored, moe_loss, class_logits = model(lq)
        pixel_loss, class_loss = l1(restored, target), ce(class_logits, labels)
        loss = pixel_loss + args.moe_weight * moe_loss + args.classification_weight * class_loss
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite training loss at iteration {}".format(iteration))
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=float(loss.detach()))
        if iteration % args.val_interval == 0 or iteration == args.iterations:
            rows, summary = evaluate(model, val_loader, device)
            summary["iteration"] = iteration
            write_csv(args.output_dir / "validation" / ("{}.csv".format(iteration)), rows)
            write_json(args.output_dir / "validation" / ("{}.json".format(iteration)), summary)
            score = summary["overall"]["psnr"]
            save_weights(model, model_dir / ("Generator_iteration_{}.pth".format(iteration)))
            if not math.isnan(score) and score >= best:
                best = score
                save_weights(model, model_dir / "Generator_best.pth")
                write_json(args.output_dir / "best.json", summary)
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "iteration": iteration, "best_psnr": best,
                        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                       str(model_dir / "training_state.pt"))


if __name__ == "__main__":
    main()
