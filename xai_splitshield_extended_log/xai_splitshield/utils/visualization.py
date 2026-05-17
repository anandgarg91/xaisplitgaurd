"""
utils/visualization.py
───────────────────────
Visualization utilities for XAI-SplitShield:
  - Attribution heatmaps (SHAP / LRP maps overlaid on input)
  - AAS score timeseries plots
  - Per-client trust weight bar charts (AWSA)
  - ASR / CA comparison tables rendered as matplotlib figures
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")             # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch import Tensor


# ── Color palette ─────────────────────────────────────────────────────────────
_PALETTE = {
    "clean":    "#2196F3",  # blue
    "poisoned": "#F44336",  # red
    "defense":  "#4CAF50",  # green
    "neutral":  "#9E9E9E",  # grey
}


# ── 1. Attribution Heatmap ────────────────────────────────────────────────────

def plot_attribution_heatmap(
    image: Tensor,
    phi: Tensor,
    title: str = "Attribution Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlays an attribution map on the original image using a red-blue colormap.

    Args:
        image:     Original image tensor (C, H, W) in [0, 1].
        phi:       Attribution tensor (C, H, W) — uses mean across channels.
        title:     Plot title.
        save_path: Optional filepath to save the figure.

    Returns:
        matplotlib Figure.
    """
    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    phi_np = phi.detach().cpu().mean(dim=0).numpy()  # (H, W)
    phi_np = (phi_np - phi_np.min()) / (phi_np.max() - phi_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(phi_np, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title("Attribution Map (φ)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(phi_np, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── 2. AAS Score Timeseries ───────────────────────────────────────────────────

def plot_aas_timeseries(
    aas_scores: List[float],
    is_poisoned_gt: List[bool],
    threshold: float,
    title: str = "Attribution Anomaly Score Over Training",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots AAS scores per batch over training, coloured by ground-truth poisoning.

    Args:
        aas_scores:     List of AAS values per batch.
        is_poisoned_gt: Ground-truth poison flag per batch.
        threshold:      Detection threshold tau.
        save_path:      Optional save path.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    steps = np.arange(len(aas_scores))
    scores = np.array(aas_scores)
    gt     = np.array(is_poisoned_gt, dtype=bool)

    # Clean batches
    ax.scatter(steps[~gt], scores[~gt], c=_PALETTE["clean"],
               s=12, alpha=0.6, label="Clean batch", zorder=2)
    # Poisoned batches
    ax.scatter(steps[gt], scores[gt], c=_PALETTE["poisoned"],
               s=18, marker="^", alpha=0.8, label="Poisoned batch", zorder=3)

    # Threshold line
    ax.axhline(threshold, color=_PALETTE["defense"], linewidth=2,
               linestyle="--", label=f"Threshold τ = {threshold:.3f}")

    ax.set_xlabel("Training Batch", fontsize=11)
    ax.set_ylabel("AAS Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── 3. AWSA Trust Weight Bar Chart ────────────────────────────────────────────

def plot_awsa_trust_weights(
    trust_report: Dict[int, dict],
    malicious_ids: Optional[List[int]] = None,
    title: str = "AWSA Client Trust Weights",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of per-client trust weights coloured by malicious/benign status.

    Args:
        trust_report:  Output of XAISplitShield.trust_report().
        malicious_ids: List of known malicious client IDs (for colouring).
        save_path:     Optional save path.
    """
    malicious_ids = set(malicious_ids or [])
    cids    = sorted(trust_report.keys())
    weights = [trust_report[c]["trust_weight"] for c in cids]
    colors  = [_PALETTE["poisoned"] if c in malicious_ids
               else _PALETTE["clean"] for c in cids]

    fig, ax = plt.subplots(figsize=(max(8, len(cids) * 0.6), 4))
    bars = ax.bar([f"C{c}" for c in cids], weights, color=colors, edgecolor="white")

    # Annotate EMA AAS above each bar
    for bar, cid in zip(bars, cids):
        ema = trust_report[cid]["ema_aas"]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{ema:.2f}", ha="center", va="bottom",
                fontsize=7, color="dimgray")

    patches = [
        mpatches.Patch(color=_PALETTE["clean"],    label="Benign client"),
        mpatches.Patch(color=_PALETTE["poisoned"], label="Malicious client"),
    ]
    ax.legend(handles=patches, fontsize=10)
    ax.set_ylabel("Trust Weight $w_i(t)$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(weights) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── 4. ASR / CA Comparison Bar Chart ─────────────────────────────────────────

def plot_asr_ca_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "asr",
    title: str = "Attack Success Rate Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing ASR (or CA) across defenses.

    Args:
        results:  {defense_name: {"asr": ..., "ca": ...}}
        metric:   "asr" or "ca".
        save_path: Optional save path.
    """
    defenses = list(results.keys())
    values   = [results[d].get(metric, 0.0) * 100 for d in defenses]
    colors   = [_PALETTE["defense"] if "XAI" in d or "ours" in d.lower()
                else _PALETTE["neutral"] for d in defenses]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(defenses, values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ylabel = "ASR (%)" if metric == "asr" else "Clean Accuracy (%)"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── 5. Training Curves ────────────────────────────────────────────────────────

def plot_training_curves(
    train_acc:  List[float],
    val_acc:    List[float],
    train_loss: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots loss and accuracy curves over epochs."""
    epochs = np.arange(1, len(train_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, color=_PALETTE["poisoned"], label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_acc],
             color=_PALETTE["clean"], label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in val_acc],
             color=_PALETTE["defense"], linestyle="--", label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
