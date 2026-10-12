"""Regression checks for data shape, pairing, metrics and network contracts."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from data.MedicalDataUniform import Test_Data, Train_Data
from data.common import CLASS_LABELS, read_image, save_image, transformData
from evaluation.evaluation_metric import compute_measure
from model import AllFreqNet
from runtime import predict


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def put_pair(self, modality, split, name, data):
        for quality in ("LQ", "HQ"):
            path = self.root / modality / split / quality / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".bin"):
                with path.open("wb") as stream:
                    pickle.dump(torch.from_numpy(data)[None], stream)
            elif name.endswith(".raw"):
                data.astype("<f4").tofile(str(path))
            else:
                np.save(str(path), data)

    def test_shapes_for_legacy_tensor_bin_and_raw(self):
        image = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
        self.put_pair("MA", "validation", "a.raw", image)
        self.put_pair("MA", "validation", "b.bin", image)
        dataset = Test_Data(self.root, ["MA"], raw_shape=(16, 24))
        self.assertEqual(len(dataset), 2)
        for lq, hq, _, _ in dataset:
            self.assertEqual(tuple(lq.shape), (1, 16, 24))
            self.assertEqual(tuple(hq.shape), (1, 16, 24))

    def test_validation_count_resets_for_each_modality(self):
        image = np.zeros((16, 24), dtype=np.float32)
        for modality, count in (("CT", 1), ("MRI", 3)):
            for index in range(count):
                self.put_pair(modality, "validation", "{}.npy".format(index), image)
        self.assertEqual(len(Test_Data(self.root, ["CT", "MRI"])), 4)
        self.assertEqual(len(Test_Data(self.root, ["CT", "MRI"], use_num=2)), 3)
        with self.assertRaises(ValueError):
            Test_Data(self.root, ["CT", "MRI"], use_num=0)

    def test_missing_pair_and_empty_folder_fail(self):
        folder = self.root / "CT" / "test" / "LQ"
        folder.mkdir(parents=True)
        with self.assertRaises(ValueError):
            Test_Data(self.root, ["CT"], target_folder="test")
        np.save(str(folder / "a.npy"), np.zeros((16, 16), dtype=np.float32))
        with self.assertRaises(FileNotFoundError):
            Test_Data(self.root, ["CT"], target_folder="test")

    def test_paired_crop_and_checkpoint_class_mapping(self):
        image = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
        for modality, label in CLASS_LABELS.items():
            self.put_pair(modality, "train", "a.npy", image)
            lq, hq, found = Train_Data(self.root, [modality], patch_size=16)[0]
            self.assertEqual(found, label)
            torch.testing.assert_close(lq, hq)
            self.assertEqual(tuple(lq.shape), (1, 16, 16))

    def test_nifti_metadata_survives_singleton_slice_export(self):
        original = sitk.GetImageFromArray(np.ones((1, 16, 24), dtype=np.float32))
        original.SetSpacing((0.7, 0.8, 2.5))
        original.SetOrigin((2, 4, 6))
        source, destination = self.root / "source.nii.gz", self.root / "output.nii.gz"
        sitk.WriteImage(original, str(source))
        save_image(read_image(source), destination, source)
        output = sitk.ReadImage(str(destination))
        self.assertEqual(output.GetSize(), original.GetSize())
        np.testing.assert_allclose(output.GetSpacing(), original.GetSpacing())
        np.testing.assert_allclose(output.GetOrigin(), original.GetOrigin())

    def test_volume_is_not_silently_reshaped(self):
        path = self.root / "volume.npy"
        np.save(str(path), np.ones((3, 16, 16), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "2D slice"):
            read_image(path)

    def test_normalization_roundtrip(self):
        transform = transformData()
        for modality, (low, high) in transform.data_range.items():
            value = torch.linspace(low, high, 24).reshape(1, 4, 6)
            torch.testing.assert_close(transform.denormalize(transform.normalize(value, modality), modality), value)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_model_forward_backward_and_padding(self):
        torch.manual_seed(42)
        model = AllFreqNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 25114700)
        restored, balancing, logits = model(torch.rand(2, 1, 16, 24))
        self.assertEqual(tuple(restored.shape), (2, 1, 16, 24))
        self.assertEqual(tuple(logits.shape), (2, 4))
        loss = restored.abs().mean() + 0.001 * balancing + 0.001 * logits.square().mean()
        self.assertTrue(torch.isfinite(loss).all().item())
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all().item())
        model.eval()
        with torch.no_grad():
            result = predict(model, torch.rand(1, 1, 17, 19))
        self.assertEqual(tuple(result.shape), (1, 1, 17, 19))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_rectangular_image_metrics(self):
        image = torch.linspace(0, 1, 16 * 24).reshape(16, 24)
        psnr, ssim, rmse = compute_measure(image, image, image.max() - image.min())
        self.assertEqual(psnr, float("inf"))
        self.assertAlmostEqual(ssim, 1.0, places=6)
        self.assertEqual(rmse, 0)
        psnr, ssim, rmse = compute_measure(image + 0.1, image, 1.0)
        self.assertAlmostEqual(psnr, 20, places=4)
        self.assertAlmostEqual(rmse, 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
