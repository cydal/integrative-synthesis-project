import json
import pathlib

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 32
RANDOM_STATE = 42

CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"
with open(CHECKPOINT_DIR / "class_names.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)


class HFImageDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        image = row["image"].convert("RGB")
        label = row["label"]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

aug_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])


def load_gtsrb():
    hf_train = load_dataset("tanganke/gtsrb", split="train")
    hf_test = load_dataset("tanganke/gtsrb", split="test")
    return hf_train, hf_test


def train_val_split(n_total, val_fraction=0.15, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_val = int(val_fraction * n_total)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx


def labels_for_split(hf_train, idx):
    return np.array([hf_train[i]["label"] for i in idx])
