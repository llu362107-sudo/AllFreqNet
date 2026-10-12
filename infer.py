"""Restore LQ slices without requiring HQ targets."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.MedicalDataUniform import InferenceData
from data.common import save_image, transformData
from runtime import add_data_arguments, load_generator, predict, resolve_device, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference"))
    parser.add_argument("--output-format", choices=("npy", "nii", "nii.gz"), default="npy")
    args = parser.parse_args()
    device = resolve_device(args.device)
    dataset = InferenceData(args.data_root, args.modalities, args.split, args.raw_shape)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = load_generator(args.weights, device)
    transform = transformData()
    for index, (lq, modalities, names) in enumerate(tqdm(loader, desc="Restore")):
        modality, name = modalities[0], names[0]
        result = transform.postprocess(predict(model, lq.to(device)), modality)
        path = args.output_dir / modality / (name + "." + args.output_format)
        save_image(result.cpu(), path, dataset.items[index]["lq"])
    write_json(args.output_dir / "config.json", vars(args))
    print("Restored {} slices to {}".format(len(dataset), args.output_dir))


if __name__ == "__main__":
    main()
