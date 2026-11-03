"""Stage B: controlled ablation comparing three ways of training the Project 4 CNN
architecture on GTSRB. 
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data import (CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, HFImageDataset,
                   aug_transform, base_transform, load_gtsrb, train_val_split)
from models import ConditionalVAE, TrafficSignCNN, train_cnn
from synthetic import build_class_counts, generate_synthetic_dataset

EPOCHS = 10
BATCH_SIZE = 128
SEEDS = [42, 43, 44]
LATENT_DIM = 32

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

t0 = time.time()
hf_train, hf_test = load_gtsrb()
train_full_base = HFImageDataset(hf_train, transform=base_transform)
train_full_aug = HFImageDataset(hf_train, transform=aug_transform)
test_set = HFImageDataset(hf_test, transform=base_transform)

train_idx, val_idx = train_val_split(len(train_full_base), seed=RANDOM_STATE)
val_ds = Subset(train_full_base, val_idx)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
print(f"Data ready in {time.time()-t0:.1f}s | train={len(train_idx)} val={len(val_idx)} test={len(test_set)}")

cvae = ConditionalVAE(LATENT_DIM, NUM_CLASSES).to(device)
cvae.load_state_dict(torch.load("checkpoints/cvae.pt", map_location=device))
cvae.eval()
class_counts = build_class_counts()
print(f"Synthetic counts/class: base=100, weak classes get +300 "
      f"(total synthetic per generation: {sum(class_counts.values())})")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = float((all_preds == all_labels).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    return acc, macro_f1


results = {"real_only": [], "classic_aug": [], "cvae_synthetic": []}

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")

    # Arm 1: real only
    train_ds = Subset(train_full_base, train_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    torch.manual_seed(seed)
    model = TrafficSignCNN(num_classes=NUM_CLASSES)
    t0 = time.time()
    train_cnn(model, train_loader, val_loader, EPOCHS, device)
    acc, f1 = evaluate(model, test_loader)
    print(f"[real_only]      seed={seed}  test_acc={acc:.4f}  macro_f1={f1:.4f}  ({time.time()-t0:.1f}s)")
    results["real_only"].append({"seed": seed, "test_acc": acc, "macro_f1": f1})
    if seed == SEEDS[0]:
        torch.save(model.state_dict(), "checkpoints/real_only_seed42.pt")

    # Arm 2: classic augmentation (Project 4's recipe)
    train_ds = Subset(train_full_aug, train_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    torch.manual_seed(seed)
    model = TrafficSignCNN(num_classes=NUM_CLASSES)
    t0 = time.time()
    train_cnn(model, train_loader, val_loader, EPOCHS, device)
    acc, f1 = evaluate(model, test_loader)
    print(f"[classic_aug]     seed={seed}  test_acc={acc:.4f}  macro_f1={f1:.4f}  ({time.time()-t0:.1f}s)")
    results["classic_aug"].append({"seed": seed, "test_acc": acc, "macro_f1": f1})

    # Arm 3: real + CVAE synthetic, concentrated on weak classes
    synthetic_ds = generate_synthetic_dataset(cvae, device, class_counts, seed=seed)
    combined_ds = ConcatDataset([Subset(train_full_base, train_idx), synthetic_ds])
    train_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)
    torch.manual_seed(seed)
    model = TrafficSignCNN(num_classes=NUM_CLASSES)
    t0 = time.time()
    train_cnn(model, train_loader, val_loader, EPOCHS, device)
    acc, f1 = evaluate(model, test_loader)
    print(f"[cvae_synthetic]  seed={seed}  test_acc={acc:.4f}  macro_f1={f1:.4f}  ({time.time()-t0:.1f}s)")
    results["cvae_synthetic"].append({"seed": seed, "test_acc": acc, "macro_f1": f1})

    if seed == SEEDS[0]:
        torch.save(model.state_dict(), "checkpoints/cvae_synthetic_seed42.pt")

summary = {}
for arm, runs in results.items():
    accs = np.array([r["test_acc"] for r in runs])
    f1s = np.array([r["macro_f1"] for r in runs])
    summary[arm] = {
        "test_acc_mean": float(accs.mean()), "test_acc_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
        "runs": runs,
    }
    print(f"\n{arm}: test_acc={accs.mean():.4f}+/-{accs.std():.4f}  "
          f"macro_f1={f1s.mean():.4f}+/-{f1s.std():.4f}")

with open("ablation_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved ablation_results.json")
