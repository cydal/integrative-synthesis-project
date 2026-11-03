# integrative-synthesis-project

Integrated system: **AI-based Traffic Sign Recognition & Safety Validation**. Combines
P4 (CNN classifier), P5 (VAE generative augmentation), and P6 (agentic
safety escalation) into one new pipeline.

**Dataset:** [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html), loaded from the
[`tanganke/gtsrb`](https://huggingface.co/datasets/tanganke/gtsrb) Hugging Face mirror (same
source as P4). Everything in this project, including the new conditional generative
model, is trained and evaluated on this one dataset so results are directly comparable to
P4's baseline.

## Pipeline

1. **`01_analyze_baseline.py`** — reuses the P4/6 trained CNN checkpoint (copied into
   `checkpoints/p4_reused_cnn.pt`) as a building block to find the 10 weakest GTSRB classes on
   the held-out test set.
2. **`02_train_cvae.py`** — Stage A. Trains a class-conditional VAE on GTSRB (extends P5's
   unconditional VAE: GTSRB's 43 classes have thousands of examples each).
3. **`03_ablation_train.py`** — Stage B. Retrains the P4 CNN architecture under 3
   training conditions x 3 seeds, holding architecture/epochs/batch size/optimizer/split
   constant: `real_only` (baseline), `classic_aug` (P4's existing augmentation recipe),
   `cvae_synthetic` (real data + Stage A synthetic crops weighted toward the weakest classes).
   Writes `ablation_results.json` and `checkpoints/{real_only,cvae_synthetic}_seed42.pt`.
4. **`04_agent_eval.py`** — Stage C. Wraps the `real_only` and `cvae_synthetic` classifiers in
   P6's unmodified `SafetyAgent` (`safety_agent.py`) and runs both over the same
   stratified test sample (up to 10 cases/class), to evaluate the *integrated* system, not just
   the bare classifier.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic API key for Stage C (either exported in the shell or in a local `.env`
file, never committed — see `.gitignore`):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Running the notebook

```bash
jupyter nbconvert --to notebook --execute --inplace integrated_pipeline.ipynb
```

## Report

`module_summary.pdf` is the Synthesis Paper 
