"""Restore an uncompressed, single-frame CT DICOM folder (CT or MA task)."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from tqdm import tqdm

from data.common import transformData
from runtime import load_generator, predict, resolve_device


def save_derived(reference, image, path, series_uid):
    ds = copy.deepcopy(reference)
    if not getattr(ds, "file_meta", None):
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SeriesInstanceUID, ds.SOPInstanceUID = series_uid, generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.SeriesDescription = (str(getattr(ds, "SeriesDescription", ""))[:40] + " AllFreqNet")
    ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.PixelData = np.rint(image).clip(-32768, 32767).astype("<i2").tobytes()
    ds[0x7FE00010].VR = "OW"
    for key in ("SmallestImagePixelValue", "LargestImagePixelValue", "PixelPaddingValue", "PixelPaddingRangeLimit"):
        if key in ds:
            del ds[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--modality", choices=("CT", "MA"), default="CT")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if source == destination or source in destination.parents:
        parser.error("Choose an output directory outside the input directory")
    if not source.is_dir():
        raise FileNotFoundError("Input directory not found: {}".format(source))
    paths = sorted(p for p in (source.rglob("*") if args.recursive else source.iterdir()) if p.is_file())
    device = resolve_device(args.device)
    model = load_generator(args.weights, device)
    transform, series, count = transformData(), {}, 0
    used_outputs = set()
    for path in tqdm(paths, desc="DICOM"):
        try:
            ds = pydicom.dcmread(str(path))
        except pydicom.errors.InvalidDicomError:
            continue
        if "PixelData" not in ds:
            continue
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            raise ValueError("Decompress DICOM before inference: {}".format(path))
        if str(getattr(ds, "Modality", "")) != "CT":
            raise ValueError("Expected CT DICOM: {}".format(path))
        image = ds.pixel_array.astype(np.float32)
        if image.ndim != 2:
            raise ValueError("Expected single-frame 2D DICOM: {}".format(path))
        image = image * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        tensor = torch.from_numpy(transform.normalize(image, args.modality))[None, None].to(device)
        result = transform.postprocess(predict(model, tensor), args.modality).cpu().numpy()[0, 0]
        original_series = str(getattr(ds, "SeriesInstanceUID", path.parent))
        if original_series not in series:
            series[original_series] = generate_uid()
        output = destination / path.relative_to(source).with_suffix(".dcm")
        if output in used_outputs or output.exists():
            raise FileExistsError("Output collision: {}".format(output))
        used_outputs.add(output)
        save_derived(ds, result, output, series[original_series])
        count += 1
    if not count:
        raise ValueError("No readable CT DICOM slices found")
    print("Restored {} DICOM slices to {}".format(count, destination))


if __name__ == "__main__":
    main()
