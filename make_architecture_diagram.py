"""Renders the P7 integrated-system architecture diagram: how P4 (CNN),
P5 (VAE), and P6 (agent) combine into one pipeline."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec="#2b4a7a", fontsize=9, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, linespacing=1.6)
    return b


def arrow(x1, y1, x2, y2, color="#333333", style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                         linewidth=lw, connectionstyle=connectionstyle, mutation_scale=14)
    ax.add_patch(a)


ax.text(6, 10.6, "Integrated System: AI-based Traffic Sign Recognition & Safety Validation",
        ha="center", fontsize=13, weight="bold")

# Inputs
box(0.3, 8.7, 2.6, 1.1, "GTSRB train split\n(real images, 43 classes)", fc="#f5f5f5", ec="#555555")
box(0.3, 7.0, 2.6, 1.2, "Weak-class analysis\n(P4/P6 CNN, reused\nas a building block)", fc="#f5f5f5", ec="#555555")

# Stage A
box(3.4, 7.3, 2.9, 2.2,
    "Stage A (extends P5)\nClass-conditional VAE\ntrained on GTSRB\n\nUnlike P5's unconditional\nMTSD VAE, labels\nare available here",
    fc="#f3edff", ec="#5a3a9b", weight="bold")
box(6.8, 7.6, 2.9, 1.5, "Synthetic sign crops,\nweighted toward the\n10 weakest classes", fc="#f5f5f5", ec="#555555")

arrow(2.9, 9.25, 3.4, 8.9)
arrow(2.9, 7.6, 3.4, 8.0)
arrow(6.3, 8.4, 6.8, 8.4)

# Stage B
box(0.3, 5.0, 9.4, 1.8,
    "Stage B (extends P4)\nControlled ablation: TrafficSignCNN retrained on 3 training conditions x 3 seeds,\n"
    "everything else held constant\n"
    "real_only (baseline)  |  classic_aug (P4's recipe)  |  cvae_synthetic (real + Stage A crops)",
    fc="#e8f0ff", ec="#2b4a7a", weight="bold", fontsize=9.5)

arrow(1.6, 7.0, 1.6, 6.8)
arrow(8.25, 7.6, 7.5, 6.8, connectionstyle="arc3,rad=-0.2")

box(0.3, 3.5, 9.4, 1.0,
    "Best classifier selected on held-out GTSRB test set (test accuracy + macro-F1)",
    fc="#f5f5f5", ec="#555555")
arrow(5.0, 5.0, 5.0, 4.5)

# Stage C
box(0.3, 1.2, 9.4, 1.9,
    "Stage C (extends P6)\nSafetyAgent (unmodified): deterministic safeguards + Claude reasoning over\n"
    "classifier output + session memory -> ACCEPT or ESCALATE,\n"
    "evaluated on a stratified test sample",
    fc="#fff6da", ec="#9a7b00", weight="bold", fontsize=9.5)
arrow(5.0, 3.5, 5.0, 3.1)

box(0.3, 0.1, 9.4, 0.9,
    "Integrated evaluation: accept/escalate rate, integrated accuracy, safety-critical misses accepted",
    fc="#f5f5f5", ec="#555555")
arrow(5.0, 1.2, 5.0, 1.0)

plt.tight_layout()
plt.savefig("figure0_architecture.png", dpi=150)
print("saved figure0_architecture.png")
