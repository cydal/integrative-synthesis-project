"""Stage C: wrap the real_only and cvae_synthetic classifiers (Stage B, seed 42) in
Project 6's unmodified SafetyAgent 
"""
import json
import random

import numpy as np

from data import CLASS_NAMES, load_gtsrb
from safety_agent import ClassifierTool, SafetyAgent, SessionMemory

SAMPLE_PER_CLASS = 10
SEED = 42

_, hf_test = load_gtsrb()
labels = np.array(hf_test["label"])

rng = random.Random(SEED)
sample_indices = []
for c in range(len(CLASS_NAMES)):
    class_idx = np.where(labels == c)[0].tolist()
    rng.shuffle(class_idx)
    sample_indices.extend(class_idx[:SAMPLE_PER_CLASS])
print(f"Stratified sample size: {len(sample_indices)} cases across {len(CLASS_NAMES)} classes")

CONFIGS = {
    "real_only": "checkpoints/real_only_seed42.pt",
    "cvae_synthetic": "checkpoints/cvae_synthetic_seed42.pt",
}

all_results = {}
for arm_name, checkpoint_path in CONFIGS.items():
    print(f"\n=== Agent eval: {arm_name} ===")
    classifier = ClassifierTool(checkpoint_path, CLASS_NAMES)
    agent = SafetyAgent(classifier, memory=SessionMemory())

    for n, idx in enumerate(sample_indices):
        row = hf_test[idx]
        image = row["image"]
        true_label = CLASS_NAMES[row["label"]]
        entry = agent.review(image, case_id=f"{arm_name}_{idx}", true_label=true_label)
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(sample_indices)} cases processed")

    log = agent.log
    n_total = len(log)
    n_accept = sum(1 for e in log if e["final_decision"] == "ACCEPT")
    n_escalate = n_total - n_accept
    accept_correct = sum(
        1 for e in log
        if e["final_decision"] == "ACCEPT" and e["classifier_output"]["predicted_class"] == e["true_label"]
    )
    accept_incorrect_cases = [
        e for e in log
        if e["final_decision"] == "ACCEPT" and e["classifier_output"]["predicted_class"] != e["true_label"]
    ]
    safety_critical_misses = [
        e for e in accept_incorrect_cases
        if e["true_label"] in {
            "Stop", "Yield", "No entry", "Right-of-way at the next intersection",
            "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
            "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
            "Speed limit (100km/h)", "Speed limit (120km/h)",
        }
    ]

    summary = {
        "n_total": n_total,
        "n_accept": n_accept,
        "n_escalate": n_escalate,
        "accept_rate": n_accept / n_total,
        "accept_accuracy": accept_correct / n_accept if n_accept else None,
        "n_accept_incorrect": len(accept_incorrect_cases),
        "n_safety_critical_misses_accepted": len(safety_critical_misses),
        "llm_overridden_count": sum(1 for e in log if e["llm_overridden"]),
    }
    print(json.dumps(summary, indent=2))
    all_results[arm_name] = {"summary": summary, "log": log}

with open("agent_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved agent_eval_results.json")
