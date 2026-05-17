"""
scripts/plot_real_results.py
─────────────────────────────
Generates publication-quality plots from XAI-SplitShield experimental logs.

INPUTS — reads from the standard results tree produced by run_experiment.py:

    results/
    └── <run_name>/                  e.g. cifar10_badnets_xai_splitshield_split2_s42
        ├── config.json              full hyperparameter snapshot
        ├── metrics.jsonl            one JSON record per epoch
        └── summary.json             final CA + ASR + config

    Each line in metrics.jsonl contains:
        epoch, phase, loss, train_acc, val_acc, asr, aas_mean,
        precision, recall, f1

OUTPUTS — saved to <out_dir>/ (default: results/plots/):

    fig_training_curves.png      Train/val accuracy + loss over epochs (G2 detection quality)
    fig_aas_evolution.png        AAS over rounds with detection threshold (G2)
    fig_detection_metrics.png    Precision/recall/F1/FPR over rounds (G2)
    fig_asr_vs_byzantine.png     CA & ASR vs malicious fraction (G3) — needs multiple runs
    fig_defense_comparison.png   ASR / CA bar chart per defense (overall comparison)
    fig_attack_comparison.png    ASR for each attack with vs. without defense
    summary_table.csv            Aggregated mean±std table across seeds

USAGE:
    # After running experiments to populate ./results/
    python scripts/plot_real_results.py --log_dir ./results

    # Custom output directory
    python scripts/plot_real_results.py --log_dir ./results --out_dir ./paper_figs

    # Filter to a single dataset
    python scripts/plot_real_results.py --log_dir ./results --dataset cifar10

The script auto-skips figures whose required data is not present (e.g. if you've
only run a single seed, it still works — it just shows that single run instead
of mean ± std bands). Each plot prints a one-line note explaining what it shows
and which runs went into it.
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_DEFENSE   = "#1f77b4"  # blue
C_ATTACK    = "#d62728"  # red
C_BASELINE  = "#7f7f7f"  # grey
C_VAL       = "#2ca02c"  # green
C_AAS_LINE  = "#9467bd"  # purple
C_THRESHOLD = "#ff7f0e"  # orange


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

class RunResult:
    """One experimental run loaded from disk."""
    def __init__(self, run_dir: Path):
        self.run_dir   = run_dir
        self.run_name  = run_dir.name
        self.config    = self._load_json("config.json") or {}
        self.summary   = self._load_json("summary.json") or {}
        self.epochs_df = self._load_jsonl("metrics.jsonl")

        # Parse run_name for grouping
        # Format: <dataset>_<attack>_<defense>_split<N>_s<seed>
        m = re.match(r"(\w+?)_(\w+?)_(\w+?)_split(\d+)_s(-?\d+)", self.run_name)
        if m:
            self.dataset, self.attack, self.defense = m.group(1), m.group(2), m.group(3)
            self.split_layer = int(m.group(4))
            self.seed        = int(m.group(5))
        else:
            self.dataset = self.config.get("dataset", "unknown")
            self.attack  = self.config.get("attack",  "unknown")
            self.defense = self.config.get("defense", "unknown")
            self.split_layer = self.config.get("split_layer", 0)
            self.seed        = self.config.get("seed", 0)

    @property
    def final_ca(self) -> Optional[float]:
        return self.summary.get("final_ca")

    @property
    def final_asr(self) -> Optional[float]:
        return self.summary.get("final_asr")

    @property
    def num_malicious(self) -> int:
        return int(self.config.get("num_malicious", 0))

    @property
    def num_clients(self) -> int:
        return int(self.config.get("num_clients", 1))

    @property
    def malicious_fraction(self) -> float:
        return self.num_malicious / max(self.num_clients, 1)

    @property
    def group_key(self) -> str:
        """Identifier for aggregating across seeds."""
        return (f"{self.dataset}_{self.attack}_{self.defense}_"
                f"split{self.split_layer}_n{self.num_clients}_m{self.num_malicious}")

    def epoch_series(self, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (epochs, values) arrays for the given metric, train phase only."""
        if not self.epochs_df:
            return np.array([]), np.array([])
        epochs, vals = [], []
        for rec in self.epochs_df:
            if rec.get("phase", "train") != "train":
                continue
            if metric in rec and rec[metric] is not None:
                epochs.append(rec["epoch"])
                vals.append(rec[metric])
        return np.array(epochs), np.array(vals, dtype=float)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _load_json(self, name: str) -> Optional[dict]:
        p = self.run_dir / name
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_jsonl(self, name: str) -> List[dict]:
        p = self.run_dir / name
        if not p.exists():
            return []
        records = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def discover_runs(log_dir: Path,
                  dataset_filter: Optional[str] = None) -> List[RunResult]:
    """Walks log_dir for completed runs (any dir containing summary.json)."""
    runs = []
    for summary_path in log_dir.rglob("summary.json"):
        run = RunResult(summary_path.parent)
        if dataset_filter and run.dataset != dataset_filter:
            continue
        runs.append(run)
    return runs


def group_by_condition(runs: List[RunResult]) -> Dict[str, List[RunResult]]:
    """Groups runs by everything except seed → list of seed-replicates."""
    groups = defaultdict(list)
    for r in runs:
        groups[r.group_key].append(r)
    return dict(groups)


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(runs: List[RunResult], out_dir: Path) -> None:
    """
    Train/val accuracy + loss curves, averaged over seeds with shaded
    bands for variance. One subplot per (dataset, attack, defense) group.
    """
    groups = group_by_condition(runs)
    valid_groups = {k: v for k, v in groups.items()
                    if any(len(r.epochs_df) > 0 for r in v)}
    if not valid_groups:
        print("  [skip] training_curves: no metrics.jsonl files with epoch data")
        return

    n = len(valid_groups)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (key, group_runs) in enumerate(sorted(valid_groups.items())):
        ax = axes[idx // cols][idx % cols]
        ax2 = ax.twinx()    # secondary axis for loss

        # Pool all seeds together by epoch
        epochs_all, train_acc_all, val_acc_all, loss_all = [], [], [], []
        for r in group_runs:
            e1, ta = r.epoch_series("train_acc")
            e2, va = r.epoch_series("val_acc")
            e3, lo = r.epoch_series("loss")
            if len(e1) > 0:
                epochs_all.append(e1); train_acc_all.append(ta)
            if len(e2) > 0:
                val_acc_all.append(va)
            if len(e3) > 0:
                loss_all.append(lo)

        if not epochs_all:
            ax.axis("off"); continue

        epochs = epochs_all[0]
        # Stack and reduce — pad to longest length with NaN
        def stack(arrs):
            if not arrs:
                return None
            L = max(len(a) for a in arrs)
            mat = np.full((len(arrs), L), np.nan)
            for i, a in enumerate(arrs):
                mat[i, :len(a)] = a
            return mat

        ta_mat = stack(train_acc_all)
        va_mat = stack(val_acc_all)
        lo_mat = stack(loss_all)

        if ta_mat is not None:
            mu, sd = np.nanmean(ta_mat, 0), np.nanstd(ta_mat, 0)
            xs = np.arange(1, len(mu) + 1)
            ax.plot(xs, mu, color=C_DEFENSE, lw=2, label="Train acc")
            if len(group_runs) > 1:
                ax.fill_between(xs, mu - sd, mu + sd, color=C_DEFENSE, alpha=0.2)
        if va_mat is not None:
            mu, sd = np.nanmean(va_mat, 0), np.nanstd(va_mat, 0)
            xs = np.arange(1, len(mu) + 1)
            ax.plot(xs, mu, color=C_VAL, lw=2, label="Val acc")
            if len(group_runs) > 1:
                ax.fill_between(xs, mu - sd, mu + sd, color=C_VAL, alpha=0.2)
        if lo_mat is not None:
            mu = np.nanmean(lo_mat, 0)
            xs = np.arange(1, len(mu) + 1)
            ax2.plot(xs, mu, color=C_ATTACK, lw=1.5, ls="--", alpha=0.75,
                     label="Loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax2.set_ylabel("Loss", color=C_ATTACK)
        ax2.tick_params(axis="y", labelcolor=C_ATTACK)
        ax.set_title(key + f"  (n={len(group_runs)} seed{'s' if len(group_runs)>1 else ''})",
                     fontsize=10)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for j in range(len(valid_groups), rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_training_curves.png", facecolor="white")
    plt.close()
    print(f"  ✓ fig_training_curves.png  ({len(valid_groups)} groups)")


def plot_aas_evolution(runs: List[RunResult], out_dir: Path) -> None:
    """AAS metric over training rounds. Defense runs only."""
    aas_runs = [r for r in runs if r.defense == "xai_splitshield"]
    if not aas_runs:
        print("  [skip] aas_evolution: no xai_splitshield runs found")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = 0
    for r in aas_runs:
        e, aas = r.epoch_series("aas_mean")
        if len(aas) == 0:
            continue
        label = f"{r.dataset}/{r.attack} (seed {r.seed})"
        ax.plot(e, aas, lw=1.8, alpha=0.85, label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        print("  [skip] aas_evolution: no aas_mean column in logs")
        return

    # Threshold heuristic — pull from config if available, else use 0.55
    tau = aas_runs[0].config.get("aas_threshold", 0.55)
    ax.axhline(tau, color=C_THRESHOLD, ls="--", lw=2,
               label=f"Detection threshold τ = {tau}")

    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"AAS$(\mathcal{B}_t)$")
    ax.set_title("G2 — AAS Evolution During Training (real experimental data)")
    ax.legend(loc="best", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_aas_evolution.png", facecolor="white")
    plt.close()
    print(f"  ✓ fig_aas_evolution.png  ({plotted} runs)")


def plot_detection_metrics(runs: List[RunResult], out_dir: Path) -> None:
    """Precision / recall / F1 over training rounds, averaged across seeds."""
    aas_runs = [r for r in runs if r.defense == "xai_splitshield"]
    if not aas_runs:
        print("  [skip] detection_metrics: no defense runs")
        return

    # First pass: check if any of these metrics are present
    has_any = False
    for r in aas_runs:
        for mkey in ("precision", "recall", "f1"):
            _, v = r.epoch_series(mkey)
            if len(v) > 0:
                has_any = True; break
        if has_any: break
    if not has_any:
        print("  [skip] detection_metrics: precision/recall/f1 not in logs")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("precision", "Precision", C_DEFENSE),
        ("recall",    "Recall",    C_VAL),
        ("f1",        "F1 Score",  C_AAS_LINE),
    ]

    for ax, (mkey, title, color) in zip(axes, metric_specs):
        groups = group_by_condition(aas_runs)
        for grp_idx, (gname, grp_runs) in enumerate(sorted(groups.items())):
            arrs = []
            for r in grp_runs:
                _, vals = r.epoch_series(mkey)
                if len(vals) > 0:
                    arrs.append(vals)
            if not arrs:
                continue
            L = max(len(a) for a in arrs)
            mat = np.full((len(arrs), L), np.nan)
            for i, a in enumerate(arrs):
                mat[i, :len(a)] = a
            mu = np.nanmean(mat, 0); sd = np.nanstd(mat, 0)
            xs = np.arange(1, L + 1)
            label = "/".join(gname.split("_")[:3])
            ax.plot(xs, mu, lw=2, label=label)
            if len(arrs) > 1:
                ax.fill_between(xs, mu - sd, mu + sd, alpha=0.15)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Batch-level {title}")
        ax.set_ylim(0, 1.05)
        # Only add legend if at least one line was drawn
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("G2 — AAS Detection Quality Over Training",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_detection_metrics.png", facecolor="white")
    plt.close()
    print("  ✓ fig_detection_metrics.png")


def plot_asr_vs_byzantine(runs: List[RunResult], out_dir: Path) -> None:
    """
    G3 plot: final ASR & CA as a function of f/N (malicious fraction).
    Needs at least 3 different num_malicious values to draw a meaningful curve.
    """
    # Group by (dataset, attack, defense) and aggregate across seeds AND f/N values
    by_setting = defaultdict(list)
    for r in runs:
        if r.num_clients < 2:
            continue
        key = (r.dataset, r.attack, r.defense)
        by_setting[key].append(r)

    if not by_setting:
        print("  [skip] asr_vs_byzantine: no multi-client runs (num_clients<2)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    any_plotted = False
    for (ds, atk, dfn), group_runs in sorted(by_setting.items()):
        # Group by malicious_fraction, aggregate seeds
        by_frac = defaultdict(lambda: {"ca": [], "asr": []})
        for r in group_runs:
            frac = r.malicious_fraction
            if r.final_ca  is not None: by_frac[frac]["ca"].append(r.final_ca)
            if r.final_asr is not None: by_frac[frac]["asr"].append(r.final_asr)

        fracs = sorted(by_frac.keys())
        if len(fracs) < 2:
            continue
        ca_mu  = [np.mean(by_frac[f]["ca"])  if by_frac[f]["ca"]  else np.nan for f in fracs]
        ca_sd  = [np.std (by_frac[f]["ca"])  if len(by_frac[f]["ca"]) > 1 else 0 for f in fracs]
        asr_mu = [np.mean(by_frac[f]["asr"]) if by_frac[f]["asr"] else np.nan for f in fracs]
        asr_sd = [np.std (by_frac[f]["asr"]) if len(by_frac[f]["asr"]) > 1 else 0 for f in fracs]

        lbl = f"{ds}/{atk}/{dfn}"
        axes[0].errorbar(fracs, ca_mu,  yerr=ca_sd,  marker="o", lw=2, capsize=4, label=lbl)
        axes[1].errorbar(fracs, asr_mu, yerr=asr_sd, marker="^", lw=2, capsize=4, label=lbl)
        any_plotted = True

    if not any_plotted:
        plt.close()
        print("  [skip] asr_vs_byzantine: need ≥2 distinct f/N values per condition")
        return

    for ax, ylab, title in [(axes[0], "Clean Accuracy (%)", "(a) Clean Accuracy vs Byzantine Fraction"),
                             (axes[1], "Attack Success Rate (%)", "(b) ASR vs Byzantine Fraction")]:
        ax.axvline(1/3, color="gray", ls=":", lw=1.5, alpha=0.7)
        ax.set_xlabel(r"Malicious-client fraction $f/N$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("G3 — Theorem 2: Convergence Under Byzantine Adversaries",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_asr_vs_byzantine.png", facecolor="white")
    plt.close()
    print("  ✓ fig_asr_vs_byzantine.png")


def plot_defense_comparison(runs: List[RunResult], out_dir: Path) -> None:
    """Bar chart: final ASR & CA broken down by defense (averaged over seeds & attacks)."""
    if not runs:
        return

    # Aggregate: (dataset, defense) → list of final_asr / final_ca
    table = defaultdict(lambda: {"asr": [], "ca": []})
    for r in runs:
        if r.final_ca  is not None: table[(r.dataset, r.defense)]["ca"].append(r.final_ca)
        if r.final_asr is not None: table[(r.dataset, r.defense)]["asr"].append(r.final_asr)

    if not table:
        print("  [skip] defense_comparison: no summary.json files with metrics")
        return

    datasets   = sorted({k[0] for k in table.keys()})
    defenses   = sorted({k[1] for k in table.keys()})
    if len(defenses) < 2:
        print(f"  [skip] defense_comparison: only one defense found ({defenses})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(datasets))
    w = 0.8 / len(defenses)

    for i, d in enumerate(defenses):
        asr_mu = [np.mean(table[(ds, d)]["asr"]) if table[(ds, d)]["asr"] else np.nan
                  for ds in datasets]
        ca_mu  = [np.mean(table[(ds, d)]["ca"])  if table[(ds, d)]["ca"]  else np.nan
                  for ds in datasets]
        asr_sd = [np.std(table[(ds, d)]["asr"])  if len(table[(ds, d)]["asr"]) > 1 else 0
                  for ds in datasets]
        ca_sd  = [np.std(table[(ds, d)]["ca"])   if len(table[(ds, d)]["ca"])  > 1 else 0
                  for ds in datasets]
        axes[0].bar(x + i*w - 0.4 + w/2, ca_mu,  w, yerr=ca_sd,  label=d, capsize=3)
        axes[1].bar(x + i*w - 0.4 + w/2, asr_mu, w, yerr=asr_sd, label=d, capsize=3)

    for ax, ylab, title in [(axes[0], "Clean Accuracy (%)", "(a) Clean Accuracy by Defense"),
                             (axes[1], "Attack Success Rate (%)", "(b) ASR by Defense")]:
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Dataset")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_defense_comparison.png", facecolor="white")
    plt.close()
    print(f"  ✓ fig_defense_comparison.png  ({len(defenses)} defenses × {len(datasets)} datasets)")


def plot_attack_comparison(runs: List[RunResult], out_dir: Path) -> None:
    """Per-attack ASR with vs without defense, grouped bars."""
    table = defaultdict(lambda: {"with_def": [], "without_def": []})
    for r in runs:
        if r.final_asr is None: continue
        key = (r.dataset, r.attack)
        if r.defense == "none":
            table[key]["without_def"].append(r.final_asr)
        else:
            table[key]["with_def"].append(r.final_asr)

    keys = sorted([k for k in table if table[k]["with_def"] or table[k]["without_def"]])
    if not keys:
        print("  [skip] attack_comparison: no runs with final_asr")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.5), 5))
    x = np.arange(len(keys))
    w = 0.36

    no_def_mu  = [np.mean(table[k]["without_def"]) if table[k]["without_def"] else 0 for k in keys]
    no_def_sd  = [np.std(table[k]["without_def"])  if len(table[k]["without_def"]) > 1 else 0 for k in keys]
    def_mu     = [np.mean(table[k]["with_def"])    if table[k]["with_def"] else 0 for k in keys]
    def_sd     = [np.std(table[k]["with_def"])     if len(table[k]["with_def"]) > 1 else 0 for k in keys]

    ax.bar(x - w/2, no_def_mu, w, yerr=no_def_sd, label="No defense",
           color=C_ATTACK, capsize=3)
    ax.bar(x + w/2, def_mu,    w, yerr=def_sd,    label="XAI-SplitShield",
           color=C_DEFENSE, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n{atk}" for ds, atk in keys], fontsize=9)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate: Defense vs No Defense")
    ax.legend(loc="best")
    ax.set_ylim(0, max(max(no_def_mu, default=0), max(def_mu, default=0)) * 1.15 + 5)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_attack_comparison.png", facecolor="white")
    plt.close()
    print(f"  ✓ fig_attack_comparison.png  ({len(keys)} dataset×attack pairs)")


def write_summary_table(runs: List[RunResult], out_dir: Path) -> None:
    """CSV: per-condition mean ± std for CA and ASR."""
    import csv
    groups = group_by_condition(runs)
    rows = []
    for key in sorted(groups):
        rs  = groups[key]
        cas = [r.final_ca  for r in rs if r.final_ca  is not None]
        asr = [r.final_asr for r in rs if r.final_asr is not None]
        if not cas and not asr: continue
        rows.append({
            "condition":      key,
            "n_seeds":        len(rs),
            "ca_mean":        f"{np.mean(cas):.2f}" if cas else "",
            "ca_std":         f"{np.std(cas):.2f}"  if len(cas) > 1 else "",
            "asr_mean":       f"{np.mean(asr):.2f}" if asr else "",
            "asr_std":        f"{np.std(asr):.2f}"  if len(asr) > 1 else "",
        })
    if not rows:
        print("  [skip] summary_table: no completed runs"); return

    path = out_dir / "summary_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ summary_table.csv  ({len(rows)} conditions)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--log_dir",  type=str, default="./results",
                   help="Root directory containing per-run subdirectories")
    p.add_argument("--out_dir",  type=str, default=None,
                   help="Where to save figures (default: <log_dir>/plots)")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Filter to a single dataset (cifar10, gtsrb, xray)")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: log_dir not found: {log_dir}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (log_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {log_dir} for experiment runs...")
    runs = discover_runs(log_dir, dataset_filter=args.dataset)
    if not runs:
        print(f"ERROR: no runs found in {log_dir}.")
        print("       Each run must have a summary.json file.")
        return 1

    print(f"Found {len(runs)} runs across {len({r.group_key for r in runs})} conditions.")
    print(f"Writing figures to {out_dir}\n")

    plot_training_curves(runs, out_dir)
    plot_aas_evolution(runs, out_dir)
    plot_detection_metrics(runs, out_dir)
    plot_asr_vs_byzantine(runs, out_dir)
    plot_defense_comparison(runs, out_dir)
    plot_attack_comparison(runs, out_dir)
    write_summary_table(runs, out_dir)

    print(f"\nAll plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
