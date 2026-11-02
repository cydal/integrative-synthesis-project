"""Identify which GTSRB classes the existing P4/P6 classifier struggles with most.
"""
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CHECKPOINT_DIR, CLASS_NAMES, HFImageDataset, base_transform, load_gtsrb
from models import TrafficSignCNN

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

t0 = time.time()
_, hf_test = load_gtsrb()
test_set = HFImageDataset(hf_test, transform=base_transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
print(f"Loaded test set: {len(test_set)} images in {time.time() - t0:.1f}s")

model = TrafficSignCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT_DIR / "p4_reused_cnn.pt", map_location=device))
model.to(device).eval()

all_preds, all_labels = [], []
t0 = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
print(f"Inference over test set took {time.time() - t0:.1f}s")

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

per_class_acc = {}
for c in range(len(CLASS_NAMES)):
    mask = all_labels == c
    n = mask.sum()
    if n == 0:
        continue
    acc = (all_preds[mask] == c).mean()
    per_class_acc[CLASS_NAMES[c]] = {"accuracy": float(acc), "n_test": int(n)}

overall_acc = (all_preds == all_labels).mean()
print(f"Overall test accuracy (sanity check vs P4/P6): {overall_acc:.4f}")

ranked = sorted(per_class_acc.items(), key=lambda kv: kv[1]["accuracy"])
print("\nWeakest 10 classes:")
for name, stats in ranked[:10]:
    print(f"  {stats['accuracy']:.3f}  (n={stats['n_test']:4d})  {name}")

weak_classes = [name for name, _ in ranked[:10]]
with open("weak_classes.json", "w") as f:
    json.dump({
        "overall_test_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "weakest_10_classes": weak_classes,
    }, f, indent=2)
print("\nSaved weak_classes.json")
