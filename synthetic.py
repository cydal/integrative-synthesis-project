
import json

import torch
from torch.utils.data import Dataset

from data import CLASS_NAMES, NUM_CLASSES

BASE_COUNT_PER_CLASS = 100
WEAK_CLASS_EXTRA = 300


class SyntheticImageDataset(Dataset):
    """Wraps generated (image, label) tensors but returns plain int labels, matching
    HFImageDataset's label type so ConcatDataset collation stays consistent."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def build_class_counts(weak_classes_path="weak_classes.json"):
    with open(weak_classes_path) as f:
        weak = json.load(f)["weakest_10_classes"]
    weak_set = set(weak)
    counts = {}
    for name in CLASS_NAMES:
        counts[name] = BASE_COUNT_PER_CLASS + (WEAK_CLASS_EXTRA if name in weak_set else 0)
    return counts


def generate_synthetic_dataset(cvae, device, counts_per_class=None, seed=None):
    if counts_per_class is None:
        counts_per_class = build_class_counts()
    if seed is not None:
        torch.manual_seed(seed)

    cvae.eval()
    images, labels = [], []
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    with torch.no_grad():
        for name, count in counts_per_class.items():
            c = name_to_idx[name]
            batch_labels = torch.full((count,), c, dtype=torch.long)
            samples = cvae.sample(batch_labels, device).cpu().clamp(0, 1)
            images.append(samples)
            labels.append(batch_labels)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return SyntheticImageDataset(images, labels)
