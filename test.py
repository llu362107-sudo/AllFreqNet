"""Evaluate a checkpoint on paired restoration slices."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data.MedicalDataUniform import Test_Data
from evaluation.runner import evaluate
from runtime import add_data_arguments, load_generator, resolve_device, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/test"))
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-format", choices=("npy", "nii", "nii.gz"), default="npy")
    args = parser.parse_args()
    device = resolve_device(args.device)
    dataset = Test_Data(args.data_root, args.modalities, target_folder=args.split, raw_shape=args.raw_shape)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = load_generator(args.weights, device)
    rows, summary = evaluate(model, loader, device,
                             args.output_dir / "images" if args.save_images else None, args.output_format)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "config.json", vars(args))
    for modality in args.modalities:
        print("{}: {}".format(modality, summary[modality]))


if __name__ == "__main__":
    main()
