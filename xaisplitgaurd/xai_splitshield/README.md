# XAI-SplitShield: Explainability-Driven Mitigation of Backdoor Attacks in Split Learning

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of:  
**"XAI-SplitShield: Explainability-Driven Mitigation of Backdoor Attacks in Split Learning"**  
*Submitted to IEEE S&P 2026 / CCS 2026*

---

## Overview

XAI-SplitShield is the first explainable AI (XAI)-driven framework for detecting and mitigating backdoor attacks in split learning (SL). It operates exclusively on **smashed data** (intermediate activations), requiring no access to client-side weights, raw inputs, or labelled auxiliary data.

### Core Components

| Component | Description |
|-----------|-------------|
| `SDA` | Smashed-Data Attribution via DeepSHAP-SL + LRP |
| `AAS` | Attribution Anomaly Score for poisoning detection |
| `AGNS` | Attribution-Guided Neuron Suppression for neutralization |
| `AWSA` | Attribution-Weighted Secure Aggregation for multi-client SL |

---

## Installation

```bash
git clone https://github.com/anonymous/xai-splitshield
cd xai-splitshield
pip install -r requirements.txt
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended)

---

## Dataset Preparation

The training script auto-downloads CIFAR-10 and GTSRB on first use. The chest
X-ray dataset must be downloaded manually (see below).

### Pre-download all datasets

```bash
python scripts/download_datasets.py
```

This script applies several fallback strategies (SSL bypass, HTTP mirror,
direct urllib) to handle the well-known `cs.toronto.edu` SSL/HTTP issues.

### Troubleshooting: HTTP / SSL errors during download

The `cs.toronto.edu` server hosting CIFAR-10 has had repeated availability
issues since 2022 (expired SSL certs, intermittent HTTP 503 errors). The
codebase handles this through a **6-level fallback chain**:

1. Default torchvision download (HTTPS to `cs.toronto.edu`)
2. SSL verification disabled
3. Plain HTTP override
4. **Direct Parquet download from HuggingFace** (no `datasets` lib needed) — fixes HTTP 503
5. **GitHub-hosted `.tar.gz` mirror** — secondary fallback
6. **`datasets.load_dataset("uoft-cs/cifar10")`** — last resort

If you see `HTTP Error 503: Service Unavailable` or
`URLError: [SSL: CERTIFICATE_VERIFY_FAILED]`, the loader automatically falls
through to level 4, which downloads two Parquet files (~144 MB total) from
HuggingFace's Cloudflare CDN and converts them to the torchvision pickle-batch
format on disk. This path requires only `pyarrow` and `Pillow` (both in
`requirements.txt`).

### Manual fallback if all auto-download strategies fail

**CIFAR-10** (use the first mirror that works for you):
```bash
mkdir -p data/raw && cd data/raw
wget https://github.com/EN10/CIFAR/raw/master/cifar-10-python.tar.gz \
  || wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
# Result: data/raw/cifar-10-batches-py/
```

**GTSRB:** Auto-downloaded by torchvision. If that fails, download from
[sid.erda.dk](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/)
and extract into `data/raw/gtsrb/`.

**Chest X-Ray (manual only):** Download from
[Kaggle: paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and extract into `data/raw/chest_xray/` so the structure looks like:

```
data/raw/chest_xray/
├── train/{NORMAL,PNEUMONIA}/*.jpeg
└── test/{NORMAL,PNEUMONIA}/*.jpeg
```

---

## Quick Start

### Single-client SL with BadNets attack + XAI-SplitShield defense

```bash
python experiments/run_experiment.py \
    --dataset cifar10 \
    --attack badnets \
    --defense xai_splitshield \
    --split_layer 2 \
    --poison_rate 0.1 \
    --epochs 100
```

### Multi-client SL with WaNet attack + AWSA

```bash
python experiments/run_experiment.py \
    --dataset gtsrb \
    --attack wanet \
    --defense awsa \
    --num_clients 10 \
    --num_malicious 3 \
    --epochs 100
```

### Run all experiments from the paper

```bash
bash experiments/run_all.sh
```

---

## Project Structure

```
xai_splitshield/
├── configs/                  # YAML experiment configs
│   ├── base.yaml
│   ├── cifar10_badnets.yaml
│   ├── gtsrb_wanet.yaml
│   └── xray_blended.yaml
├── data/                     # Dataset loading & poisoning
│   ├── dataset_loader.py
│   └── poisoning.py
├── models/                   # Neural network architectures
│   ├── resnet.py             # ResNet-18 with configurable split
│   ├── client_model.py       # Client-side sub-network
│   └── server_model.py       # Server-side sub-network
├── attacks/                  # Backdoor attack implementations
│   ├── badnets.py
│   ├── blended.py
│   ├── wanet.py
│   └── lira.py
├── defense/                  # XAI-SplitShield defense modules
│   ├── sda.py                # Smashed-Data Attribution
│   ├── aas.py                # Attribution Anomaly Score
│   ├── agns.py               # Attribution-Guided Neuron Suppression
│   └── awsa.py               # Attribution-Weighted Secure Aggregation
├── experiments/              # Training & evaluation scripts
│   ├── run_experiment.py
│   ├── run_all.sh
│   └── evaluate.py
├── utils/                    # Logging, metrics, visualization
│   ├── metrics.py
│   ├── logger.py
│   └── visualization.py
├── tests/                    # Unit tests
│   ├── test_sda.py
│   ├── test_aas.py
│   └── test_attacks.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Reproducing Paper Results

### Table 2: Main Results (ASR & Clean Accuracy)
```bash
python experiments/evaluate.py --config configs/cifar10_badnets.yaml
python experiments/evaluate.py --config configs/gtsrb_wanet.yaml
python experiments/evaluate.py --config configs/xray_blended.yaml
```

### Ablation Study
```bash
python experiments/run_experiment.py --ablation --dataset cifar10 --attack wanet
```

### Adaptive Adversary Evaluation
```bash
python experiments/run_experiment.py --adaptive_attack --dataset cifar10
```

---

## Citation

```bibtex
@inproceedings{xai_splitshield_2026,
  title     = {XAI-SplitShield: Explainability-Driven Mitigation of Backdoor Attacks in Split Learning},
  author    = {Anonymous},
  booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
