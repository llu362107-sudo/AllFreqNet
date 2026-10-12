"""Deterministic file pairing with a fixed, checkpoint-compatible class mapping."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import CLASS_LABELS, EXTENSIONS, extension, image_stem, read_image, transformData


def collect_items(root_dir, modalities, split, paired):
    if not modalities or len(modalities) != len(set(modalities)):
        raise ValueError("Specify one or more distinct modalities")
    items = []
    for modality in modalities:
        if modality not in CLASS_LABELS:
            raise ValueError("Unknown modality: {}".format(modality))
        folder = Path(root_dir) / modality / split / "LQ"
        if not folder.is_dir():
            raise FileNotFoundError("Missing LQ folder: {}".format(folder))
        paths = sorted(p for p in folder.iterdir() if p.is_file() and extension(p) in EXTENSIONS)
        if not paths:
            raise ValueError("No supported image files in {}".format(folder))
        seen = set()
        for path in paths:
            name = image_stem(path)
            if name in seen:
                raise ValueError("Duplicate output name {} in {}".format(name, folder))
            seen.add(name)
            hq = folder.parent / "HQ" / path.name
            if paired and not hq.is_file():
                raise FileNotFoundError("Missing paired HQ image: {}".format(hq))
            items.append({"lq": path, "hq": hq if paired else None, "modality": modality, "name": name})
    return items


class SliceData(Dataset):
    def __init__(self, root_dir, modality_list, split, paired=True, raw_shape=(512, 512), use_num=None):
        self.items = collect_items(root_dir, modality_list, split, paired)
        if use_num is not None:
            if use_num <= 0:
                raise ValueError("use_num must be positive or None (all samples)")
            limited = []
            for modality in modality_list:
                matches = [item for item in self.items if item["modality"] == modality]
                limited.extend(matches[:use_num])
            self.items = limited
        self.raw_shape = tuple(raw_shape)
        self.transform = transformData()
        self.length = len(self.items)

    def __len__(self):
        return self.length

    def load_item(self, index):
        item = self.items[index]
        lq = self.transform.normalize(read_image(item["lq"], self.raw_shape), item["modality"])
        lq = torch.from_numpy(lq).float().unsqueeze(0)
        hq = None
        if item["hq"] is not None:
            hq = self.transform.normalize(read_image(item["hq"], self.raw_shape), item["modality"])
            hq = torch.from_numpy(hq).float().unsqueeze(0)
            if lq.shape != hq.shape:
                raise ValueError("LQ/HQ shapes differ for {}".format(item["lq"]))
        return lq, hq, item


class Train_Data(SliceData):
    def __init__(self, root_dir, modality_list=("CT", "MA", "PET", "MRI"), patch_size=128, raw_shape=(512, 512)):
        super().__init__(root_dir, modality_list, "train", raw_shape=raw_shape)
        if patch_size < 16 or patch_size % 8:
            raise ValueError("patch_size must be a multiple of 8 and at least 16")
        self.patch_size = patch_size

    def __getitem__(self, index):
        lq, hq, item = self.load_item(index)
        height, width = lq.shape[-2:]
        size = self.patch_size
        if height < size or width < size:
            raise ValueError("Training slice {} is smaller than patch_size {}".format(item["lq"], size))
        top, left = random.randint(0, height - size), random.randint(0, width - size)
        return lq[:, top:top + size, left:left + size], hq[:, top:top + size, left:left + size], CLASS_LABELS[item["modality"]]


class Test_Data(SliceData):
    def __init__(self, root_dir, modality_list=("CT", "MA", "PET", "MRI"), use_num=None,
                 target_folder="validation", raw_shape=(512, 512)):
        super().__init__(root_dir, modality_list, target_folder, raw_shape=raw_shape, use_num=use_num)

    def __getitem__(self, index):
        lq, hq, item = self.load_item(index)
        return lq, hq, item["modality"], item["name"]


class InferenceData(SliceData):
    def __init__(self, root_dir, modality_list, target_folder="test", raw_shape=(512, 512)):
        super().__init__(root_dir, modality_list, target_folder, paired=False, raw_shape=raw_shape)

    def __getitem__(self, index):
        lq, _, item = self.load_item(index)
        return lq, item["modality"], item["name"]


class DataSampler:
    def __init__(self, dataloader):
        if len(dataloader) == 0:
            raise ValueError("No complete training batch; reduce --batch-size or add samples")
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)
