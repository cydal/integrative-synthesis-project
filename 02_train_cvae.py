"""Stage A: train a class-conditional VAE on GTSRB 
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import HFImageDataset, NUM_CLASSES, RANDOM_STATE, base_transform, load_gtsrb, train_val_split
from models import ConditionalVAE, cvae_loss

LATENT_DIM = 32
BETA = 0.5
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("CVAE_EPOCHS", 60))

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}, epochs: {EPOCHS}")

t0 = time.time()
hf_train, _ = load_gtsrb()
train_full = HFImageDataset(hf_train, transform=base_transform)
train_idx, val_idx = train_val_split(len(train_full))
train_ds = Subset(train_full, train_idx)
val_ds = Subset(train_full, val_idx)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Loaded + split data in {time.time() - t0:.1f}s "
      f"(train={len(train_ds)}, val={len(val_ds)})")

torch.manual_seed(RANDOM_STATE)
model = ConditionalVAE(LATENT_DIM, NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_epoch(loader, train: bool):
    model.train(train)
    total_recon, total_kl, n_batches = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        y_onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        recon, mu, logvar = model(images, y_onehot)
        recon_loss, kl_loss = cvae_loss(recon, images, mu, logvar, BETA)
        loss = recon_loss + BETA * kl_loss
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1
    return total_recon / n_batches, total_kl / n_batches


history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
t0 = time.time()
for epoch in range(EPOCHS):
    e0 = time.time()
    train_recon, train_kl = run_epoch(train_loader, train=True)
    with torch.no_grad():
        val_recon, val_kl = run_epoch(val_loader, train=False)
    history["train_recon"].append(train_recon)
    history["train_kl"].append(train_kl)
    history["val_recon"].append(val_recon)
    history["val_kl"].append(val_kl)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1}/{EPOCHS}  train_recon={train_recon:.2f}  train_kl={train_kl:.2f}  "
              f"val_recon={val_recon:.2f}  val_kl={val_kl:.2f}  ({time.time()-e0:.1f}s/epoch)")

total_time = time.time() - t0
print(f"\nTotal CVAE training time: {total_time:.1f}s for {EPOCHS} epochs "
      f"({total_time/EPOCHS:.2f}s/epoch avg)")

torch.save(model.state_dict(), "checkpoints/cvae.pt")
np.save("cvae_history.npy", history)
print("Saved checkpoints/cvae.pt and cvae_history.npy")

if "--preview" in sys.argv:
    model.eval()
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    classes_to_show = list(range(8)) * 4
    for i, (ax, c) in enumerate(zip(axes.flat, classes_to_show)):
        with torch.no_grad():
            sample = model.sample(torch.tensor([c]), device)
        ax.imshow(sample.squeeze(0).permute(1, 2, 0).cpu())
        ax.axis("off")
    plt.suptitle("CVAE Sample Preview")
    plt.tight_layout()
    plt.savefig("figure_cvae_preview.png", dpi=150)
    print("Saved figure_cvae_preview.png")
