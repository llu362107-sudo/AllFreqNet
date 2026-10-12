"""Image I/O and the intensity conventions used by version3."""

import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

MODALITIES = ("CT", "MA", "PET", "MRI")
CLASS_LABELS = {"PET": 0, "CT": 1, "MRI": 2, "MA": 3}
EXTENSIONS = (".nii.gz", ".nii", ".raw", ".bin", ".npy", ".dcm")


def image_stem(path):
    name = Path(path).name
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def extension(path):
    return ".nii.gz" if str(path).lower().endswith(".nii.gz") else Path(path).suffix.lower()


def as_slice(array, path="image"):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError("{}: expected a 2D slice, got {}. Split volumes into slices first.".format(path, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or infinity".format(path))
    return np.ascontiguousarray(array)


def read_image(path, raw_shape=(512, 512)):
    path = Path(path)
    ext = extension(path)
    if ext in (".nii", ".nii.gz"):
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    elif ext == ".npy":
        array = np.load(str(path), allow_pickle=False)
    elif ext == ".raw":
        array = np.fromfile(str(path), dtype="<f4")
        if array.size != int(np.prod(raw_shape)):
            raise ValueError("{}: RAW size does not match --raw-shape {}".format(path, raw_shape))
        array = array.reshape(raw_shape)
    elif ext == ".bin":
        # Legacy version3 files are pickled arrays/tensors. Use trusted files only.
        with path.open("rb") as stream:
            array = pickle.load(stream)
    elif ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        array = ds.pixel_array.astype(np.float32)
        array = array * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    else:
        raise ValueError("Unsupported image format: {}".format(path))
    return as_slice(array, path)


def save_image(array, path, reference=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = as_slice(array)
    if extension(path) == ".npy":
        np.save(str(path), array)
    elif extension(path) in (".nii", ".nii.gz"):
        ref = None
        if reference and extension(reference) in (".nii", ".nii.gz"):
            ref = sitk.ReadImage(str(reference))
            if ref.GetDimension() == 3 and ref.GetSize()[2] == 1:
                array = array[None]
        image = sitk.GetImageFromArray(array)
        if ref is not None and image.GetSize() == ref.GetSize():
            image.CopyInformation(ref)
        sitk.WriteImage(image, str(path))
    else:
        raise ValueError("Outputs support .npy, .nii and .nii.gz. Use infer_dicom.py for DICOM export.")


class transformData:
    """Normalize to [0, 1]; preserve version3's evaluation clipping."""

    data_range = {"CT": (-1024.0, 3071.0), "MA": (-1024.0, 3071.0),
                  "PET": (0.0, 20.0), "MRI": (0.0, 4095.0)}
    test_trucate_data_range = {"CT": (-1000.0, 1000.0), "MA": (-1000.0, 1000.0),
                              "PET": (0.0, 20.0), "MRI": (0.0, 4095.0)}

    @staticmethod
    def truncate(image, d_min, d_max):
        return image.clamp(d_min, d_max) if torch.is_tensor(image) else np.clip(image, d_min, d_max)

    def normalize(self, image, modality):
        low, high = self.data_range[modality]
        return (self.truncate(image, low, high) - low) / (high - low)

    def denormalize(self, image, modality):
        low, high = self.data_range[modality]
        return self.truncate(image * (high - low) + low, low, high)

    def truncate_test(self, image, modality):
        return self.truncate(image, *self.test_trucate_data_range[modality])

    def postprocess(self, image, modality):
        return self.truncate_test(self.denormalize(image, modality), modality)
